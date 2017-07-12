/*
 * Copyright (C) 2016-2017 Waiman Long <longman@xxxxxxxxxx>
 *
 * This microbenchmark simulates how the use of different futex types can
 * affect the actual performanace of userspace locking primitives like mutex.
 *
 * The raw throughput of the futex lock and unlock calls is not a good
 * indication of actual throughput of the mutex code as it may not really
 * need to call into the kernel. Therefore, 3 sets of simple mutex lock and
 * unlock functions are written to implenment a mutex lock using the
 * wait-wake (2 versions) and PI futexes respectively. These functions serve
 * as the basis for measuring the locking throughput.
 */

#include <pthread.h>

#include <signal.h>
#include <string.h>
#include "../util/stat.h"
#include "../perf-sys.h"
#include <subcmd/parse-options.h>
#include <linux/compiler.h>
#include <linux/kernel.h>
#include <errno.h>
#include "bench.h"
#include "futex.h"

#include <err.h>
#include <stdlib.h>
#include <sys/time.h>

#define CACHELINE_SIZE		64
#define WORKERBUFSIZE 128*1024
#define gettid()		syscall(SYS_gettid)
#define __cacheline_aligned	__attribute__((__aligned__(CACHELINE_SIZE)))

typedef u32 futex_t;
typedef void (*lock_fn_t)(futex_t *futex, int tid);
typedef void (*unlock_fn_t)(futex_t *futex, int tid);

typedef struct mcs_lock_t mcs_lock_t;
struct mcs_lock_t
{
    mcs_lock_t *next;
    int spin;
};
typedef struct mcs_lock_t *mcs_lock;

mcs_lock mcs_cnt_lock = NULL;


/*
 * Statistical count list
 */
enum {
	STAT_OPS,	/* # of exclusive locking operations	*/
	STAT_LOCKS,	/* # of exclusive lock slowpath count	*/
	STAT_UNLOCKS,	/* # of exclusive unlock slowpath count	*/
	STAT_SLEEPS,	/* # of exclusive lock sleeps		*/
	STAT_EAGAINS,	/* # of EAGAIN errors			*/
	STAT_WAKEUPS,	/* # of wakeups (unlock return)		*/
	STAT_TIMEOUTS,	/* # of exclusive lock timeouts		*/
	STAT_LOCKERRS,	/* # of exclusive lock errors		*/
	STAT_UNLKERRS,	/* # of exclusive unlock errors		*/
	STAT_MCS_LOCKS,	    /* # for mcs only */
	STAT_MCS_UNLOCKS,	/* # for mcs only */
	STAT_NUM	/* Total # of statistical count		*/
};

/*
 * Syscall time list
 */
enum {
	TIME_LOCK,	/* Total exclusive lock syscall time	*/
	TIME_UNLK,	/* Total exclusive unlock syscall time	*/
	TIME_NUM,
};

struct worker {
	futex_t *futex;
	pthread_t thread;

	/*
	 * Per-thread operation statistics
	 */
	u32 stats[STAT_NUM];

	/*
	 * Lock/unlock times
	 */
	u64 times[TIME_NUM];
  u8  *buf;    // allocated to be local by each thread
  u32 nextindex;
  u32 dummy;
  mcs_lock_t  mcs_me;
  
} __cacheline_aligned;

/*
 * Global cache-aligned futex
 */
static futex_t __cacheline_aligned global_futex;
static futex_t *pfutex = &global_futex;

static __thread futex_t thread_id;	/* Thread ID */
static __thread int counter;		/* Sleep counter */

static struct worker *worker, *worker_alloc;
static unsigned int nsecs = 10;
static unsigned int timeout;
static bool verbose, done, fshared, exit_now, timestat;
static unsigned int ncpus, nthreads;
static int flags;
static const char *ftype;
static int loadlat = 1;
static int locklat = 1;
static int wratio;
static int bufstep = 64;
static int cmpxchg_limit = 1;
struct timeval start, end, runtime;
struct timespec *ptospec = NULL;
struct timespec tospec;
static unsigned int worker_start;
static unsigned int threads_starting;
static unsigned int threads_stopping;
static struct stats throughput_stats;
static lock_fn_t mutex_lock_fn;
static unlock_fn_t mutex_unlock_fn;

/*
 * Glibc mutex
 */
static pthread_mutex_t __cacheline_aligned mutex;
static pthread_mutexattr_t mutex_attr;
static bool mutex_inited, mutex_attr_inited;

/*
 * Compute the syscall time in ns.
 */
static void compute_systime(int tid, int item, struct timespec *begin)
{
	struct timespec etime;

	clock_gettime(CLOCK_REALTIME, &etime);
	worker[tid].times[item] += (etime.tv_sec  - begin->tv_sec)*1000000000 +
				    etime.tv_nsec - begin->tv_nsec;
}

static inline double stat_percent(struct worker *w, int top, int bottom)
{
	return (double)w->stats[top] * 100 / w->stats[bottom];
}

/*
 * Macro for syscall time computation
 * The variables ret and tid must exist in the parent scope.
 */
#define FUTEX_CALL(func, item, ...)				\
	if (unlikely(timestat)) {				\
		struct timespec stime;				\
		clock_gettime(CLOCK_REALTIME, &stime);		\
		ret = func(__VA_ARGS__);			\
		compute_systime(tid, item, &stime);		\
	} else {						\
		ret = func(__VA_ARGS__);			\
	}

/*
 * Inline functions to update the statistical counts
 *
 * Enable statistics collection may sometimes impact the locking rates
 * to be measured. So we can specify the DISABLE_STAT macro to disable
 * statistic counts collection for all except the core locking rate counts.
 *
 * #define DISABLE_STAT
 */
#ifndef DISABLE_STAT
static inline void stat_add(int tid, int item, int num)
{
	worker[tid].stats[item] += num;
}

static inline void stat_inc(int tid, int item)
{
	stat_add(tid, item, 1);
}
#else
static inline void stat_add(int tid __maybe_unused, int item __maybe_unused,
			    int num __maybe_unused)
{
}

static inline void stat_inc(int tid __maybe_unused, int item __maybe_unused)
{
}
#endif

/*
 * The latency values within a lock critical section (load) and between locking
 * operations is in term of the number of cpu_relax() calls that are being
 * issued.
 */
static const struct option mutex_options[] = {
	OPT_INTEGER ('d', "locklat",	&locklat,  "Specify inter-locking latency (default = 1)"),
	OPT_STRING  ('f', "ftype",	&ftype,    "type", "Specify futex type: WW, PI, GC, all (default)"),
	OPT_INTEGER ('l', "loadlat",	&loadlat,  "Specify load latency (default = 1)"),
	OPT_UINTEGER('r', "runtime",	&nsecs,    "Specify runtime (in seconds, default = 10s)"),
	OPT_BOOLEAN ('S', "shared",	&fshared,  "Use shared futexes instead of private ones"),
	OPT_BOOLEAN ('s', "timestat",	&timestat, "Track lock/unlock syscall times"),
	OPT_UINTEGER('T', "timeout",	&timeout,  "Specify timeout value (in us, default = no timeout)"),
	OPT_UINTEGER('t', "threads",	&nthreads, "Specify number of threads, default = # of CPUs"),
	OPT_BOOLEAN ('v', "verbose",	&verbose,  "Verbose mode: display thread-level details"),
	OPT_INTEGER ('w', "wait-ratio", &wratio,   "Specify <n>/1024 of load is 1us sleep, default = 0"),
	OPT_INTEGER ('b', "bufstep",    &bufstep,  "Step size in bytes thru buffer during delays, default = 64"),
	OPT_INTEGER ('x', "cmpxchg-limit",    &cmpxchg_limit,  "limit to number of cmpxchgs in nokernel_lock"),
	OPT_END()
};

static const char * const bench_futex_mutex_usage[] = {
	"perf bench futex mutex <options>",
	NULL
};

/*
 * GCC atomic builtins are only available on gcc 4.7 and higher.
 */
#if GCC_VERSION >= 40700

#define smp_load_acquire(p)	__atomic_load_n(p, __ATOMIC_ACQUIRE)
#define smp_store_release(p,v)	__atomic_store_n(p, v, __ATOMIC_RELEASE)
#define atomic_add_return(p,v)	__atomic_add_fetch(p, v, __ATOMIC_SEQ_CST)
#define atomic_add_acquire(p,v)	__atomic_add_fetch(p, v, __ATOMIC_ACQUIRE)
#define atomic_add_release(p,v)	__atomic_add_fetch(p, v, __ATOMIC_RELEASE)
#define atomic_add_relaxed(p,v)	__atomic_add_fetch(p, v, __ATOMIC_RELAXED)
#define atomic_dec_release(p)	__atomic_sub_fetch(p, 1, __ATOMIC_RELEASE)
#define atomic_xchg(p,v)	__atomic_exchange_n(p, v, __ATOMIC_SEQ_CST)
#define atomic_xchg_acquire(p,v)			\
	__atomic_exchange_n(p, v, __ATOMIC_ACQUIRE)
#define atomic_xchg_release(p,v)			\
	__atomic_exchange_n(p, v, __ATOMIC_RELEASE)
#define atomic_xchg_relaxed(p,v)			\
	__atomic_exchange_n(p, v, __ATOMIC_RELAXED)

#define atomic_cmpxchg(p,po,n)				\
	__atomic_compare_exchange_n(p, po, n, 0,	\
	__ATOMIC_SEQ_CST, __ATOMIC_RELAXED)
#define atomic_cmpxchg_acquire(p,po,n)			\
	__atomic_compare_exchange_n(p, po, n, 0,	\
	__ATOMIC_ACQUIRE, __ATOMIC_RELAXED)
#define atomic_cmpxchg_release(p,po,n)			\
	__atomic_compare_exchange_n(p, po, n, 0,	\
	__ATOMIC_RELEASE, __ATOMIC_RELAXED)
#define atomic_cmpxchg_relaxed(p,po,n)			\
	__atomic_compare_exchange_n(p, po, n, 0,	\
	__ATOMIC_RELAXED, __ATOMIC_RELAXED)

#else /* GCC_VERSION >= 40700 */

#define smp_load_acquire(p)	\
({				\
	typeof(*p) __v = *p;	\
	__sync_synchronize();	\
	__v;			\
})

#define smp_store_release(p,v)	\
do {				\
	__sync_synchronize();	\
	*(p) = v;		\
} while (0)

#define atomic_cmpxchg(p,po,n)				\
({							\
	typeof(*po) __o = *po, __c;			\
	bool __r;					\
	__c = __sync_val_compare_and_swap(p, __o, n);	\
	__r = (__c == __o);				\
	if (!__r)					\
		*po = __c;				\
	__r;						\
})

#define atomic_add_return(p,v)	 __sync_add_and_fetch(p, v)
#define atomic_add_acquire(p,v)	atomic_add_return(p,v)
#define atomic_add_release(p,v)	atomic_add_return(p,v)
#define atomic_add_relaxed(p,v)	atomic_add_return(p,v)
#define atomic_dec_release(p)	atomic_add_return(p,-1)
#define atomic_xchg(p,v)	__sync_lock_test_and_set(p, v)

#define atomic_xchg_acquire(p,v)	atomic_xchg(p,v)
#define atomic_xchg_release(p,v)	atomic_xchg(p,v)
#define atomic_xchg_relaxed(p,v)	atomic_xchg(p,v)
#define atomic_cmpxchg_acquire(p,po,n)	atomic_cmpxchg(p,po,n)
#define atomic_cmpxchg_release(p,po,n)	atomic_cmpxchg(p,po,n)
#define atomic_cmpxchg_relaxed(p,po,n)	atomic_cmpxchg(p,po,n)

#endif /* GCC_VERSION >= 40700 */

#define atomic_inc_return(p)	atomic_add_return(p, 1)
#define atomic_dec_return(p)	atomic_add_return(p, -1)

/**********************[ MUTEX lock/unlock functions ]*********************/

/*
 * Wait-wake futex lock/unlock functions (Glibc implementation)
 * futex value: 0 - unlocked
 *		1 - locked
 *		2 - locked with waiters (contended)
 */
static void ww_mutex_lock(futex_t *futex, int tid)
{
	futex_t val = 0;
	int ret;

	if (atomic_cmpxchg_acquire(futex, &val, 1))
		return;

	stat_inc(tid, STAT_LOCKS);
	for (;;) {
		if (val != 2) {
			/*
			 * Force value to 2 to indicate waiter
			 */
			val = atomic_xchg_acquire(futex, 2);
			if (val == 0)
				return;
		}
		FUTEX_CALL(futex_wait, TIME_LOCK, futex, 2, ptospec, flags);

		if (ret < 0) {
			if (errno == EAGAIN)
				stat_inc(tid, STAT_EAGAINS);
			else if (errno == ETIMEDOUT)
				stat_inc(tid, STAT_TIMEOUTS);
			else
				stat_inc(tid, STAT_LOCKERRS);
		}

		val = *futex;
	}
}

static void ww_mutex_unlock(futex_t *futex, int tid)
{
	futex_t val;
	int ret;

	val = atomic_xchg_release(futex, 0);

	if (val == 2) {
		stat_inc(tid, STAT_UNLOCKS);
		FUTEX_CALL(futex_wake, TIME_UNLK, futex, 1, flags);

		if (ret < 0)
			stat_inc(tid, STAT_UNLKERRS);
		else
			stat_add(tid, STAT_WAKEUPS, ret);
	}
}

/*
 * Alternate wait-wake futex lock/unlock functions with thread_id lock word
 */
static void ww2_mutex_lock(futex_t *futex, int tid)
{
	futex_t val = 0;
	int ret;

	if (atomic_cmpxchg_acquire(futex, &val, thread_id))
		return;

	stat_inc(tid, STAT_LOCKS);
	for (;;) {
		/*
		 * Set the FUTEX_WAITERS bit, if not set yet.
		 */
		while (!(val & FUTEX_WAITERS)) {
			if (!val) {
				if (atomic_cmpxchg_acquire(futex, &val,
							   thread_id))
					return;
				continue;
			}
			if (atomic_cmpxchg_acquire(futex, &val,
						   val | FUTEX_WAITERS)) {
				val |= FUTEX_WAITERS;
				break;
			}
		}

		FUTEX_CALL(futex_wait, TIME_LOCK, futex, val, ptospec, flags);
		if (ret < 0) {
			if (errno == EAGAIN)
				stat_inc(tid, STAT_EAGAINS);
			else if (errno == ETIMEDOUT)
				stat_inc(tid, STAT_TIMEOUTS);
			else
				stat_inc(tid, STAT_LOCKERRS);
		}

		val = *futex;
	}
}

static void ww2_mutex_unlock(futex_t *futex, int tid)
{
	futex_t val;
	int ret;

	val = atomic_xchg_release(futex, 0);

	if ((val & FUTEX_TID_MASK) != thread_id)
		stat_inc(tid, STAT_UNLKERRS);

	if (val & FUTEX_WAITERS) {
		stat_inc(tid, STAT_UNLOCKS);
		FUTEX_CALL(futex_wake, TIME_UNLK, futex, 1, flags);
		if (ret < 0)
			stat_inc(tid, STAT_UNLKERRS);
		else
			stat_add(tid, STAT_WAKEUPS, ret);
	}
}


/*
 * Version of ww_mutex_lock where we read the futex first
 * and so might avoid the first cmpxchg
 */

static void ww3_mutex_lock(futex_t *futex, int tid)
{
	futex_t val = 0;
	int ret;

	if (*futex != 0) {
	  goto slowpath;
	}
	if (atomic_cmpxchg_acquire(futex, &val, 1))
		return;

slowpath:
	stat_inc(tid, STAT_LOCKS);
	for (;;) {
		if (val != 2) {
			/*
			 * Force value to 2 to indicate waiter
			 */
			val = atomic_xchg_acquire(futex, 2);
			if (val == 0)
				return;
		}
		FUTEX_CALL(futex_wait, TIME_LOCK, futex, 2, ptospec, flags);

		if (ret < 0) {
			if (errno == EAGAIN)
				stat_inc(tid, STAT_EAGAINS);
			else if (errno == ETIMEDOUT)
				stat_inc(tid, STAT_TIMEOUTS);
			else
				stat_inc(tid, STAT_LOCKERRS);
		}

		val = *futex;
	}
}


/*
 * Version of ww_mutex_lock where we just always call the futex
 * (this is not really a mutex but we wanted to time the futex path)
 */

static void nousermode_mutex_lock(futex_t *futex, int tid)
{
	int ret;

	if (tid & 1) return;
	
	stat_inc(tid, STAT_LOCKS);
	*futex = 0;
	FUTEX_CALL(futex_wait, TIME_LOCK, futex, 0, ptospec, flags);
	
	if (ret < 0) {
	  if (errno == EAGAIN)
		stat_inc(tid, STAT_EAGAINS);
	  else if (errno == ETIMEDOUT)
		stat_inc(tid, STAT_TIMEOUTS);
	  else
		stat_inc(tid, STAT_LOCKERRS);
	}
}

static void nousermode_mutex_unlock(futex_t *futex, int tid)
{
	int ret;

	if ((tid & 1) == 0) return;
	
	*futex = 0;
	stat_inc(tid, STAT_UNLOCKS);
	FUTEX_CALL(futex_wake, TIME_UNLK, futex, 1, flags);
	
	if (ret < 0)
	  stat_inc(tid, STAT_UNLKERRS);
	else
	  stat_add(tid, STAT_WAKEUPS, ret);
}



/*
 * Version of ww_mutex_lock where we just try the cmpxchg in user mode
 * but never call the futex_wait nor futex_wake.  The number of times we try the
 * cmpxchg is configurable.  (Note that this is not really a mutex
 * but we wanted to time the usermode path under contention)
 */
static void nokernel_mutex_lock(futex_t *futex, int tid)
{
  futex_t val;
	int tries = 0;
	
	while(1) {
	  val = 0;
	  if (atomic_cmpxchg_acquire(futex, &val, 1))
		break;
	  
	  tries++;
	  if (tries >= cmpxchg_limit)
		break;
	}
	
	if (tries > 0) {
	  stat_add(tid, STAT_LOCKS, tries);
	}

	if (tries >= cmpxchg_limit) {
	  stat_inc(tid, STAT_LOCKERRS);
	}
	
}

static void nokernel_mutex_unlock(futex_t *futex, int tid)
{
	futex_t val;

	val = atomic_xchg_release(futex, 0);

	if (val != 0) {
		stat_inc(tid, STAT_UNLOCKS);
	}
}



/*
 * PI futex lock/unlock functions
 */
static void pi_mutex_lock(futex_t *futex, int tid)
{
	futex_t val = 0;
	int ret;

	if (atomic_cmpxchg_acquire(futex, &val, thread_id))
		return;

	/*
	 * Retry if an error happens
	 */
	stat_inc(tid, STAT_LOCKS);
	for (;;) {
		FUTEX_CALL(futex_lock_pi, TIME_LOCK, futex, ptospec, flags);
		if (likely(ret >= 0))
			break;
		if (errno == ETIMEDOUT)
			stat_inc(tid, STAT_TIMEOUTS);
		else
			stat_inc(tid, STAT_LOCKERRS);
	}
}

static void pi_mutex_unlock(futex_t *futex, int tid)
{
	futex_t val = thread_id;
	int ret;

	if (atomic_cmpxchg_release(futex, &val, 0))
		return;

	stat_inc(tid, STAT_UNLOCKS);
	FUTEX_CALL(futex_unlock_pi, TIME_UNLK, futex, flags);
	if (likely(ret < 0))
		stat_inc(tid, STAT_UNLKERRS);
	else
		stat_add(tid, STAT_WAKEUPS, ret);
}

/*
 * Null futex lock/unlock functions
 */
static void null_mutex_lock(futex_t *futex __maybe_unused, int tid __maybe_unused)
{
}

static void null_mutex_unlock(futex_t *futex __maybe_unused, int tid __maybe_unused)
{
}

/*
 * Glibc mutex lock and unlock function
 */
static void gc_mutex_lock(futex_t *futex __maybe_unused,
			  int tid __maybe_unused)
{
	pthread_mutex_lock(&mutex);
}

static void gc_mutex_unlock(futex_t *futex __maybe_unused,
			    int tid __maybe_unused)
{
	pthread_mutex_unlock(&mutex);
}


/*************************
 *   MCS lock routines
 ************************/
static inline void *xchg_64(void *ptr, void *x)
{
    __asm__ __volatile__("xchgq %0,%1"
                :"=r" ((unsigned long long) x)
                :"m" (*(volatile long long *)ptr), "0" ((unsigned long long) x)
                :"memory");

    return x;
}

#define cmpxchg64(P, O, N) __sync_val_compare_and_swap((P), (O), (N))

static inline void lock_mcs(futex_t *futex __maybe_unused, int tid)
{
    mcs_lock_t *tail;
	mcs_lock *m;      // global lock
	mcs_lock_t *me;   // 
	struct worker *w = &worker[tid];
	
	m = &mcs_cnt_lock;
	me = &w->mcs_me;
    me->next = NULL;
    me->spin = 0;

    tail = xchg_64(m, me);
    
    /* No one there? */
    if (!tail) return;

    /* Someone there, need to link in */
	stat_inc(tid, STAT_MCS_LOCKS);
    tail->next = me;

    /* Make sure we do the above setting of next. */
    barrier();
    
    /* Spin on my spin variable */
    while (!me->spin) cpu_relax();
    
    return;
}

static inline void unlock_mcs(futex_t *futex __maybe_unused, int tid)
{
	mcs_lock *m;      // global lock
	mcs_lock_t *me;   // 

	struct worker *w = &worker[tid];
	
	m = &mcs_cnt_lock;
	me = &w->mcs_me;

    /* No successor yet? */
    if (!me->next)
    {
        /* Try to atomically unlock */
        if (cmpxchg64(m, me, NULL) == me) return;

		stat_inc(tid, STAT_MCS_UNLOCKS);
	
        /* Wait for successor to appear */
        while (!me->next) cpu_relax();
    }

    /* Unlock next one */
    me->next->spin = 1; 
}



/**************************************************************************/

static inline void csdelay(int n, int tid)
{
  struct worker *w = &worker[tid];
  u32 sum = 0;
  while (n-- > 0) {
	sum += w->buf[w->nextindex];
	w->nextindex += bufstep;
	if (w->nextindex > WORKERBUFSIZE) {
	  w->nextindex = 0;
	}
  }
  w->dummy = sum;

#if 0
  volatile int i = 0;
  volatile __maybe_unused int j;
	while (n > 0) {
	  i--;
	  n -= 2;
	}
#endif
}

/*
 * Load function
 */
static inline void load(int tid  __maybe_unused)
{
	/*
	 * Optionally does a 1us sleep instead if wratio is defined and
	 * is within bound.
	 */
#if 0
  if (wratio && (((counter++ + tid) & 0x3ff) < wratio)) {
		usleep(1);
		return;
	}
#endif
  
  csdelay(loadlat, tid);
}


static void toggle_done(int sig __maybe_unused,
			siginfo_t *info __maybe_unused,
			void *uc __maybe_unused)
{
	/* inform all threads that we're done for the day */
	done = true;
	gettimeofday(&end, NULL);
	timersub(&end, &start, &runtime);
	if (sig)
		exit_now = true;
}

static void *mutex_workerfn(void *arg)
{
	long tid = (long)arg;
	struct worker *w = &worker[tid];
	lock_fn_t lock_fn = mutex_lock_fn;
	unlock_fn_t unlock_fn = mutex_unlock_fn;
	// u32 ops_count = 0;

	thread_id = gettid();
	counter = 0;

	// allocate my buffer used in csdelay
	// and touch it
	w->buf = malloc(WORKERBUFSIZE);
	for (int n=0; n<WORKERBUFSIZE; n+=64) {
	  w->buf[n] = 0;
	}
	
	atomic_dec_return(&threads_starting);

	/*
	 * Busy wait until asked to start
	 */
	while (!worker_start)
		cpu_relax();

	do {
		lock_fn(w->futex, tid);
		load(tid);
		unlock_fn(w->futex, tid);
		w->stats[STAT_OPS]++;	/* One more locking operation */
		csdelay(locklat, tid);
	}  while (!done);
	
	if (verbose)
		printf("[thread %3ld (%d)] exited.\n", tid, thread_id);
	atomic_inc_return(&threads_stopping);
	return NULL;
}

static void create_threads(struct worker *w, pthread_attr_t *thread_attr,
			   void *(*workerfn)(void *arg), long tid)
{
	cpu_set_t cpu;

	/*
	 * Bind each thread to a CPU
	 */
	CPU_ZERO(&cpu);
	CPU_SET(tid % ncpus, &cpu);
	w->futex = pfutex;
	w->nextindex = 0;

#if 0
	if (pthread_attr_setaffinity_np(thread_attr, sizeof(cpu_set_t), &cpu))
		err(EXIT_FAILURE, "pthread_attr_setaffinity_np");
#endif
	
	if (pthread_create(&w->thread, thread_attr, workerfn, (void *)tid))
		err(EXIT_FAILURE, "pthread_create");
}

static int futex_mutex_type(const char **ptype)
{
	const char *type = *ptype;

	if (!strcasecmp(type, "WW")) {
		*ptype = "WW";
		mutex_lock_fn = ww_mutex_lock;
		mutex_unlock_fn = ww_mutex_unlock;
	} else if (!strcasecmp(type, "WW2")) {
		*ptype = "WW2";
		mutex_lock_fn = ww2_mutex_lock;
		mutex_unlock_fn = ww2_mutex_unlock;
	} else if (!strcasecmp(type, "WW3")) {
		*ptype = "WW3";
		mutex_lock_fn = ww3_mutex_lock;
		mutex_unlock_fn = ww_mutex_unlock;
	} else if (!strcasecmp(type, "NOU")) {
		*ptype = "NOU";
		mutex_lock_fn = nousermode_mutex_lock;
		mutex_unlock_fn = nousermode_mutex_unlock;
	} else if (!strcasecmp(type, "NOK")) {
		*ptype = "NOK";
		mutex_lock_fn = nokernel_mutex_lock;
		mutex_unlock_fn = nokernel_mutex_unlock;
	} else if (!strcasecmp(type, "PI")) {
		*ptype = "PI";
		mutex_lock_fn = pi_mutex_lock;
		mutex_unlock_fn = pi_mutex_unlock;
	} else if (!strcasecmp(type, "GC")) {
		pthread_mutexattr_t *attr = NULL;

		*ptype = "GC";
		mutex_lock_fn = gc_mutex_lock;
		mutex_unlock_fn = gc_mutex_unlock;
		/*
		 * Initialize pthread mutex
		 */
		if (fshared) {
			attr = &mutex_attr;
			pthread_mutexattr_init(attr);
			mutex_attr_inited = true;
			pthread_mutexattr_setpshared(attr, true);
		}
		pthread_mutex_init(&mutex, attr);
		mutex_inited = true;
	} else if (!strcasecmp(type, "QS")) {
		*ptype = "QS";
		mutex_lock_fn = lock_mcs;
		mutex_unlock_fn = unlock_mcs;

		mcs_cnt_lock = NULL;      // global control lock

	} else if (!strcasecmp(type, "NONE")) {
		*ptype = "NONE";
		mutex_lock_fn = null_mutex_lock;
		mutex_unlock_fn = null_mutex_unlock;
	} else {
		return -1;
	}
	return 0;
}

static void futex_test_driver(const char *futex_type,
			      int (*proc_type)(const char **ptype),
			      void *(*workerfn)(void *arg))
{
	u64 us;
	int i, j;
	struct worker total;
	double avg, stddev;
	pthread_attr_t thread_attr;

	/*
	 * There is an extra blank line before the error counts to highlight
	 * them.
	 */
	const char *desc[STAT_NUM] = {
		[STAT_OPS]	 = "Total exclusive locking ops",
		[STAT_LOCKS]	 = "Exclusive lock slowpaths",
		[STAT_UNLOCKS]	 = "Exclusive unlock slowpaths",
		[STAT_SLEEPS]	 = "Exclusive lock sleeps",
		[STAT_WAKEUPS]	 = "Process wakeups",
		[STAT_EAGAINS]	 = "EAGAIN lock errors",
		[STAT_TIMEOUTS]	 = "Exclusive lock timeouts",
		[STAT_LOCKERRS]  = "\nExclusive lock errors",
		[STAT_UNLKERRS]  = "\nExclusive unlock errors",
		[STAT_MCS_LOCKS]  = "\nMCS lock slowpaths",
		[STAT_MCS_UNLOCKS]  = "\nMCS unlock slowpaths",
	};

	if (exit_now)
		return;

	if (proc_type(&futex_type) < 0) {
		fprintf(stderr, "Unknown futex type '%s'!\n", futex_type);
		exit(1);
	}

	printf("\n=====================================\n");
	printf("[PID %d]: %d threads doing %s futex lockings (load=%d) for %d secs.\n\n",
	       getpid(), nthreads, futex_type, loadlat, nsecs);

	init_stats(&throughput_stats);

	*pfutex = 0;
	done = false;
	threads_starting = nthreads;
	pthread_attr_init(&thread_attr);

	for (i = 0; i < (int)nthreads; i++)
		create_threads(&worker[i], &thread_attr, workerfn, i);

	while (threads_starting)
		usleep(1);

	gettimeofday(&start, NULL);

	/*
	 * Start the test
	 *
	 * Unlike the other futex benchmarks, this one uses busy waiting
	 * instead of pthread APIs to make sure that all the threads (except
	 * the one that shares CPU with the parent) will start more or less
	 * simultaineously.
	 */
	atomic_inc_return(&worker_start);
	sleep(nsecs);
	toggle_done(0, NULL, NULL);

	/*
	 * In verbose mode, we check if all the threads have been stopped
	 * after 1ms and report the status if some are still running.
	 */
	if (verbose) {
		usleep(1000);
		if (threads_stopping != nthreads) {
			printf("%d threads still running 1ms after timeout"
				" - futex = 0x%x\n",
				nthreads - threads_stopping, *pfutex);
			/*
			 * If the threads are still running after 10s,
			 * go directly to statistics printing and exit.
			 */
			for (i = 10; i > 0; i--) {
				sleep(1);
				if (threads_stopping == nthreads)
					break;
			}
			if (!i) {
				printf("*** Threads waiting ABORTED!! ***\n\n");
				goto print_stat;
			}
		}
	}

	for (i = 0; i < (int)nthreads; i++) {
		int ret = pthread_join(worker[i].thread, NULL);

		if (ret)
			err(EXIT_FAILURE, "pthread_join");
	}

print_stat:
	pthread_attr_destroy(&thread_attr);

	us = runtime.tv_sec * 1000000 + runtime.tv_usec;
	memset(&total, 0, sizeof(total));
	for (i = 0; i < (int)nthreads; i++) {
		/*
		 * Get a rounded estimate of the # of locking ops/sec.
		 */
		u64 tp = (u64)worker[i].stats[STAT_OPS] * 1000000 / us;

		for (j = 0; j < STAT_NUM; j++)
			total.stats[j] += worker[i].stats[j];

		update_stats(&throughput_stats, tp);
		if (verbose)
			printf("[thread %3d] futex: %p [ %'ld ops/sec ]\n",
			       i, worker[i].futex, (long)tp);
	}

	avg    = avg_stats(&throughput_stats);
	stddev = stddev_stats(&throughput_stats);

	printf("Locking statistics:\n");
	printf("%-28s = %'.2fs\n", "Test run time", (double)us/1000000);
	for (i = 0; i < STAT_NUM; i++)
		if (total.stats[i])
			printf("%-28s = %'d\n", desc[i], total.stats[i]);

	if (timestat && total.times[TIME_LOCK]) {
		printf("\nSyscall times:\n");
		if (total.stats[STAT_LOCKS])
			printf("Avg exclusive lock syscall   = %'ldns\n",
			    total.times[TIME_LOCK]/total.stats[STAT_LOCKS]);
		if (total.stats[STAT_UNLOCKS])
			printf("Avg exclusive unlock syscall = %'ldns\n",
			    total.times[TIME_UNLK]/total.stats[STAT_UNLOCKS]);
	}

	printf("\nPercentages:\n");
	if (total.stats[STAT_LOCKS])
		printf("Exclusive lock slowpaths     = %.1f%%\n",
			stat_percent(&total, STAT_LOCKS, STAT_OPS));
	if (total.stats[STAT_UNLOCKS])
		printf("Exclusive unlock slowpaths   = %.1f%%\n",
			stat_percent(&total, STAT_UNLOCKS, STAT_OPS));
	if (total.stats[STAT_EAGAINS])
		printf("EAGAIN lock errors           = %.1f%%\n",
			stat_percent(&total, STAT_EAGAINS, STAT_LOCKS));
	if (total.stats[STAT_WAKEUPS])
		printf("Process wakeups              = %.1f%%\n",
			stat_percent(&total, STAT_WAKEUPS, STAT_UNLOCKS));

	printf("\nPer-thread Locking Rates:\n");
	printf("Avg = %'d ops/sec (+- %.2f%%)\n", (int)(avg + 0.5),
		rel_stddev_stats(stddev, avg));
	printf("Min = %'d ops/sec\n", (int)throughput_stats.min);
	printf("Max = %'d ops/sec\n", (int)throughput_stats.max);

	if (*pfutex != 0)
		printf("\nResidual futex value = 0x%x\n", *pfutex);

	/* Clear the workers area */
	memset(worker, 0, sizeof(*worker) * nthreads);

	if (mutex_inited)
		pthread_mutex_destroy(&mutex);
	if (mutex_attr_inited)
		pthread_mutexattr_destroy(&mutex_attr);
	mutex_inited  = mutex_attr_inited  = false;
}

int bench_futex_mutex(int argc, const char **argv,
		      const char *prefix __maybe_unused)
{
	struct sigaction act;

	argc = parse_options(argc, argv, mutex_options,
			     bench_futex_mutex_usage, 0);
	if (argc)
		goto err;

	// hack to support NOU "lock" type which needs a timeout
	// otherwise it hangs
	if (!strcmp(ftype, "NOU") && timeout == 0) {
	  timeout = 100000;
	}
	ncpus = sysconf(_SC_NPROCESSORS_ONLN);

	sigfillset(&act.sa_mask);
	act.sa_sigaction = toggle_done;
	sigaction(SIGINT, &act, NULL);

	if (!nthreads)
		nthreads = ncpus;

	/*
	 * Since the allocated memory buffer may not be properly cacheline
	 * aligned, we need to allocate one more than needed and manually
	 * adjust array boundary to be cacheline aligned.
	 */
	worker_alloc = calloc(nthreads + 1, sizeof(*worker));
	if (!worker_alloc)
		err(EXIT_FAILURE, "calloc");
	worker = (void *)((unsigned long)&worker_alloc[1] &
					~(CACHELINE_SIZE - 1));

	if (!fshared)
		flags = FUTEX_PRIVATE_FLAG;

	if (timeout) {
		/*
		 * Convert timeout value in us to timespec.
		 */
		tospec.tv_sec  = timeout / 1000000;
		tospec.tv_nsec = (timeout % 1000000) * 1000;
		ptospec        = &tospec;
	}

	if (!ftype || !strcmp(ftype, "all")) {
		futex_test_driver("WW", futex_mutex_type, mutex_workerfn);
		futex_test_driver("PI", futex_mutex_type, mutex_workerfn);
		futex_test_driver("GC", futex_mutex_type, mutex_workerfn);
	} else {
		futex_test_driver(ftype, futex_mutex_type, mutex_workerfn);
	}
	free(worker_alloc);
	return 0;
err:
	usage_with_options(bench_futex_mutex_usage, mutex_options);
	exit(EXIT_FAILURE);
}
