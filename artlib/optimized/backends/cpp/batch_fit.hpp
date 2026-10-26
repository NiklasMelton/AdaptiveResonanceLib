#pragma once

// Per-sample work in the ARTMAP training loops.
//
// Training is sequential: each sample can change the weights that the next one
// is scored against. What can run in parallel is the work for one sample:
// scoring it against every category. That is the reduction predict runs (see
// batch_predict.hpp), on the same panel layout and with the same kernels, so
// each category's sum adds the features in the original order and the results
// are bit-identical to the old scalar loops. Here it runs for one sample at a
// time, with the panels split across a pool of threads that lives for the whole
// fit, so the per-sample hand-off costs microseconds instead of a thread start.

#include "batch_predict.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace art_fit {

using art_predict::kLanes;
using art_predict::Panels;

// Each thread that scores a sample should get at least this many lane updates
// (categories x features x streams), about 2-4 microseconds of work. Handing a
// sample to spinning pool threads and waiting for them costs well under that;
// measured on an M3 Pro, smaller shares (more threads) slowed medium models
// down at 12 threads.
constexpr std::size_t kWorkPerThread = std::size_t{1} << 14;

// Tasks per thread when a sample is split, so faster cores can take more.
constexpr std::size_t kTasksPerThread = 4;

// Spin-wait iterations before an idle pool thread starts yielding the core.
constexpr std::size_t kSpinsBeforeYield = std::size_t{1} << 14;

// Thread limit for fit: ``requested`` when positive, else ARTLIB_NUM_THREADS
// when set to a positive integer, else a default from the CPUs this process may
// run on.
//
// On an unrestricted machine the default is half the hardware threads. Unlike
// predict, every sample waits for its slowest thread, so leaving cores free does
// better: it skips SMT siblings and, on Apple chips, about matches the
// performance cores (6 of 12 on an M3 Pro was fastest; 12 was 15-60% slower).
//
// When the affinity mask is restricted (Slurm, taskset, a container cpuset),
// the scheduler has handed out exactly those CPUs, so fit uses all of them. On
// 16 allocated cores of a 40-core Xeon node, 16 threads trained 6.8x faster than
// one, while the old default (half of 40 = 20 threads) was up to 2x slower.
inline std::size_t resolve_threads(int requested) {
    if (requested > 0)
        return static_cast<std::size_t>(requested);
    if (const std::size_t env = art_predict::env_threads())
        return env;
    const std::size_t allowed = art_predict::available_cpus();
    if (allowed < art_predict::hardware_threads())
        return allowed;
    return std::max<std::size_t>(1, allowed / 2);
}

inline void spin_pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}

// Threads that stay alive between samples. run() hands them a batch of tasks
// and waits for it; between batches they spin, then yield, until the next one
// or stop(). Helper threads start on first use, so a fit that never has enough
// work per sample never starts one.
class Pool {
public:
    explicit Pool(std::size_t threads = 1) : limit_(std::max<std::size_t>(1, threads)) {}
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() { stop(); }

    // Most threads a batch may use, including the calling one.
    std::size_t limit() const { return limit_; }
    void set_limit(std::size_t threads) {
        stop();
        limit_ = std::max<std::size_t>(1, threads);
    }

    // Runs task(t) for every t in [0, n_tasks) on up to ``threads`` threads,
    // including the calling one, and returns when all are done. Tasks are
    // handed out in order from a shared counter. An exception thrown by a task
    // is rethrown here after the batch finishes.
    template <typename Task> void run(std::size_t n_tasks, std::size_t threads, Task& task) {
        threads = std::min({threads, limit_, n_tasks});
        if (threads > 1)
            threads = std::min(threads, start(threads - 1) + 1);
        if (threads <= 1) {
            for (std::size_t t = 0; t < n_tasks; ++t)
                task(t);
            return;
        }
        const std::size_t helpers = threads - 1;
        call_ = [](void* context, std::size_t t) { (*static_cast<Task*>(context))(t); };
        context_ = &task;
        n_tasks_ = n_tasks;
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        ++generation_;
        state_.store((generation_ << kGenerationShift) | helpers, std::memory_order_release);
        drain();
        while (done_.load(std::memory_order_acquire) != helpers)
            spin_pause();
        if (error_) {
            std::exception_ptr error = error_;
            error_ = nullptr;
            std::rethrow_exception(error);
        }
    }

    // Ends the helper threads. The next run() starts them again if needed.
    void stop() {
        if (workers_.empty())
            return;
        ++generation_;
        state_.store((generation_ << kGenerationShift) | kStop, std::memory_order_release);
        for (auto& worker : workers_)
            worker.join();
        workers_.clear();
    }

private:
    // state_ packs the helper count for the current batch (low 16 bits), the
    // stop flag, and a generation number that changes with every batch.
    static constexpr std::uint64_t kHelperMask = 0xFFFF;
    static constexpr std::uint64_t kStop = std::uint64_t{1} << 16;
    static constexpr int kGenerationShift = 17;

    // Starts helper threads until there are ``helpers`` of them, or until one
    // fails to start. Returns how many there are.
    std::size_t start(std::size_t helpers) {
        helpers = std::min<std::size_t>(helpers, kHelperMask);
        while (workers_.size() < helpers) {
            const std::size_t id = workers_.size();
            const std::uint64_t seen = generation_;
            try {
                workers_.emplace_back([this, id, seen] { work(id, seen); });
            } catch (...) {
                break;
            }
        }
        return workers_.size();
    }

    void work(std::size_t id, std::uint64_t seen) {
        for (;;) {
            std::uint64_t state;
            std::size_t spins = 0;
            while (((state = state_.load(std::memory_order_acquire)) >> kGenerationShift) == seen) {
                if (++spins < kSpinsBeforeYield)
                    spin_pause();
                else
                    std::this_thread::yield();
            }
            seen = state >> kGenerationShift;
            if (state & kStop)
                return;
            if (id >= (state & kHelperMask))
                continue; // not needed for this batch
            drain();
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    void drain() {
        const std::size_t n_tasks = n_tasks_;
        for (;;) {
            const std::size_t t = next_.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks)
                return;
            try {
                call_(context_, t);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex_);
                if (!error_)
                    error_ = std::current_exception();
            }
        }
    }

    std::size_t limit_;
    std::vector<std::thread> workers_;
    std::uint64_t generation_ = 0; // written only by the owning thread
    std::atomic<std::uint64_t> state_{0};
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> done_{0};
    void (*call_)(void*, std::size_t) = nullptr;
    void* context_ = nullptr;
    std::size_t n_tasks_ = 0;
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

// Calls range(begin, end) over chunks covering [0, n). ``work`` estimates the
// total cost in lane updates; the chunks run on the pool only when every
// thread gets at least kWorkPerThread of it. ``range`` must then be safe to
// call from several threads at once for different chunks.
template <typename Range>
void parallel_range(std::size_t n, std::size_t work, Pool& pool, Range range) {
    if (n == 0)
        return;
    const std::size_t threads =
        std::max<std::size_t>(1, std::min({pool.limit(), work / kWorkPerThread, n}));
    if (threads <= 1) {
        range(std::size_t{0}, n);
        return;
    }
    const std::size_t chunk = std::max<std::size_t>(1, n / (threads * kTasksPerThread));
    auto task = [&](std::size_t t) { range(t * chunk, std::min(n, (t + 1) * chunk)); };
    pool.run((n + chunk - 1) / chunk, threads, task);
}

// Scores one sample x against every category in ``weights``:
//     acc_c = 0.0;  for j in [0, dim): acc_c = step_j(acc_c, x[j], w_c at j)
// adding features in order, where step_j is ``body`` for j < split and ``tail``
// for j >= split (see art_predict::accumulate). Then calls finish(p, acc) for
// every panel p, where acc[b] is the result for category p * kLanes + b; lanes
// past the last category hold values to ignore. Large models split the panels
// across the pool, so ``finish`` must be safe to call from several threads for
// different panels.
template <typename Body, typename Tail, typename Finish>
void score_sample(const double* x, const Panels& weights, std::size_t split, Body body, Tail tail,
                  Finish finish, Pool& pool) {
    const std::size_t dim = weights.dim;
    const std::size_t stride = weights.feature_stride();
    const std::size_t js = std::min(split, dim);
    const std::size_t n_panels = weights.panels();
    parallel_range(n_panels, n_panels * kLanes * dim * weights.streams, pool,
                   [&](std::size_t p0, std::size_t p1) {
                       for (std::size_t p = p0; p < p1; ++p) {
                           double acc[kLanes] = {};
                           art_predict::split_kernel<1>(&x, weights.panel(p), stride, 0, js, dim,
                                                        acc, kLanes, body, tail);
                           finish(p, acc);
                       }
                   });
}

} // namespace art_fit
