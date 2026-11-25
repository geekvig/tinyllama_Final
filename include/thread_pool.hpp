#pragma once
// =====================================================================
//  thread_pool.hpp  --  persistent worker threads + parallel_for
// =====================================================================
//  Creating a std::thread costs tens of microseconds. A transformer
//  does hundreds of matrix multiplies per token, so spawning threads
//  per call would waste more time than the math. Instead we create the
//  workers ONCE and park them on a condition variable. A "parallel
//  region" then only costs a wake-up.  (ggml's ggml_threadpool does
//  the same thing.)
//
//  How a region runs:
//    1. parallel_for(n, fn) cuts [0, n) into T contiguous chunks.
//    2. The caller publishes {fn, n, chunk size} under the mutex and
//       wakes every worker (cv_work_).
//    3. Everyone -- workers AND the caller ("thread 0") -- loops:
//       grab the next unclaimed chunk, run fn(begin, end).
//    4. Whoever finishes the last chunk signals cv_done_; the caller
//       waits for that, then returns.
//
//  Rules that keep this simple and correct:
//    * Nested parallel_for (e.g. attention parallelises over heads and
//      something inside also calls parallel_for) runs INLINE. Only the
//      outermost level fans out -> no oversubscription, no deadlock.
//    * Exceptions inside a chunk are captured and re-thrown on the
//      calling thread after all chunks finish.
//    * Every output element is computed by exactly one chunk with the
//      same summation order as serial code -> results are bit-identical
//      for any thread count. tests/run_tests.cpp checks this.
//
//  Design doc: Chapter 7 ("Making it fast: threads").
// =====================================================================

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tinyml {

class ThreadPool {
public:
    static constexpr int kMaxTasks = 64; // upper bound on chunks per region

    explicit ThreadPool(int n_workers) {
        workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; ++i) workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int num_workers() const { return (int)workers_.size(); }

    // Run fn(begin, end) over [0, n), split into at most max_tasks chunks,
    // each at least min_per_task long.
    void run(int64_t n, const std::function<void(int64_t, int64_t)>& fn,
             int64_t min_per_task, int max_tasks) {
        if (n <= 0) return;
        min_per_task = std::max<int64_t>(1, min_per_task);
        int64_t by_grain = (n + min_per_task - 1) / min_per_task;
        int tasks = (int)std::min<int64_t>({(int64_t)max_tasks, by_grain, (int64_t)kMaxTasks, n});

        if (tasks <= 1 || in_parallel_region() || workers_.empty()) {
            fn(0, n);                       // serial / nested: just do it here
            return;
        }

        std::lock_guard<std::mutex> submit_guard(submit_m_); // one region at a time
        const int64_t chunk = (n + tasks - 1) / tasks;
        tasks = (int)((n + chunk - 1) / chunk);

        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            chunk_ = chunk;
            n_tasks_ = tasks;
            next_task_ = 0;
            done_ = 0;
            error_ = nullptr;
        }
        cv_work_.notify_all();

        drain(); // the caller helps instead of sitting idle

        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return done_ == n_tasks_; });
        fn_ = nullptr;
        n_tasks_ = 0;
        next_task_ = 0;
        std::exception_ptr err = error_;
        error_ = nullptr;
        lk.unlock();
        if (err) std::rethrow_exception(err);
    }

    static bool in_parallel_region() { return tl_depth_() > 0; }

private:
    static int& tl_depth_() {
        thread_local int depth = 0;   // one counter PER THREAD
        return depth;
    }

    // Claim and run chunks until none are left.
    void drain() {
        for (;;) {
            const std::function<void(int64_t, int64_t)>* fn;
            int64_t b, e;
            {
                std::lock_guard<std::mutex> lk(m_);
                if (next_task_ >= n_tasks_) return;
                int t = next_task_++;
                fn = fn_;
                b = (int64_t)t * chunk_;
                e = std::min<int64_t>(n_, b + chunk_);
            }
            ++tl_depth_();
            try {
                (*fn)(b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lk(m_);
                if (!error_) error_ = std::current_exception();
            }
            --tl_depth_();
            {
                std::lock_guard<std::mutex> lk(m_);
                if (++done_ == n_tasks_) cv_done_.notify_all();
            }
        }
    }

    void worker_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [&] { return stop_ || next_task_ < n_tasks_; });
                if (stop_) return;
            }
            drain();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;          // protects everything below
    std::mutex submit_m_;   // serialises whole regions
    std::condition_variable cv_work_, cv_done_;
    bool stop_ = false;

    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t n_ = 0, chunk_ = 1;
    int n_tasks_ = 0, next_task_ = 0, done_ = 0;
    std::exception_ptr error_;
};

namespace detail {
inline int hardware_threads() {
    int n = (int)std::thread::hardware_concurrency();
    return std::max(1, std::min(n, ThreadPool::kMaxTasks));
}
inline std::atomic<int>& thread_limit() {
    static std::atomic<int> limit = [] {
        int n = hardware_threads();
        if (const char* env = std::getenv("TINYML_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) n = v;
        }
        return std::atomic<int>(std::max(1, std::min(n, ThreadPool::kMaxTasks)));
    }();
    return limit;
}
} // namespace detail

// One pool for the whole process, created on first use.
// It owns (hardware threads - 1) workers; the caller is the extra one.
inline ThreadPool& global_pool() {
    static ThreadPool pool(detail::hardware_threads() - 1);
    return pool;
}

// Cap on how many chunks one parallel region is split into (1 = serial).
inline void set_num_threads(int n) {
    detail::thread_limit().store(std::max(1, std::min(n, ThreadPool::kMaxTasks)));
}
inline int get_num_threads() { return detail::thread_limit().load(); }

// RAII: temporarily change the thread count (tests/benchmarks).
class ScopedThreads {
public:
    explicit ScopedThreads(int n) : prev_(get_num_threads()) { set_num_threads(n); }
    ~ScopedThreads() { set_num_threads(prev_); }
    ScopedThreads(const ScopedThreads&) = delete;
    ScopedThreads& operator=(const ScopedThreads&) = delete;
private:
    int prev_;
};

// parallel_for(n, fn, min_per_task, max_tasks):
//   calls fn(begin, end) on disjoint chunks that exactly cover [0, n).
template <typename F>
inline void parallel_for(int64_t n, F&& fn, int64_t min_per_task = 1, int max_tasks = 0) {
    if (max_tasks <= 0) max_tasks = get_num_threads();
    if (max_tasks <= 1 || n <= 1) {
        if (n > 0) fn((int64_t)0, n);
        return;
    }
    std::function<void(int64_t, int64_t)> f = std::forward<F>(fn);
    global_pool().run(n, f, min_per_task, max_tasks);
}

} // namespace tinyml
