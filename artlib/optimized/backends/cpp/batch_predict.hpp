#pragma once

// Batched prediction shared by the ARTMAP backends.
//
// Prediction scores every test sample against every category. Each sample is
// independent, so samples are split into blocks that run on separate threads.
//
// Within a block, the per-category reductions (fuzzy AND sums, squared
// distances) are computed for several categories at once. Each category keeps
// its own accumulator and adds the features in the original order j = 0, 1, ...,
// so the results are bit-identical to the scalar loops they replace. The speed
// comes from running independent sums side by side (which the compiler turns
// into SIMD lanes), not from reordering any single sum.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace art_predict {

// Categories per panel. Eight doubles fill one 64-byte cache line.
constexpr std::size_t kLanes = 8;

// Samples per register tile. kTile * kLanes accumulators must stay in registers:
// 32 on AArch64 (16 of its 32 vector registers), 16 on x86-64 (8 of its 16).
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr std::size_t kTile = 4;
#else
constexpr std::size_t kTile = 2;
#endif

// Each panel slice (kLanes categories x a block of features) should stay in L1.
constexpr std::size_t kPanelSliceBytes = 32 * 1024;

// Most samples per thread task. A task reads all weights once, so larger blocks
// mean less weight traffic; the block's accumulators (block x categories) stay small.
constexpr std::size_t kMaxBlock = 64;

// Below this many lane updates (samples x categories x features) the work is too
// small to be worth starting a thread for.
constexpr std::size_t kWorkPerThread = std::size_t{1} << 22;

// Hardware threads in the machine (at least 1).
inline std::size_t hardware_threads() {
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? hardware : 1;
}

// CPUs this process may run on. On Linux that is the CPU affinity mask, which
// Slurm, taskset and container cpusets restrict; hardware_concurrency() counts
// every CPU in the machine, so a job given 16 of 40 cores would start 40 threads.
// Elsewhere (and if the mask cannot be read) it is the hardware thread count.
inline std::size_t available_cpus() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0)
            return static_cast<std::size_t>(count);
    }
#endif
    return hardware_threads();
}

// Positive ARTLIB_NUM_THREADS, or 0 when unset or invalid.
inline std::size_t env_threads() {
    if (const char* env = std::getenv("ARTLIB_NUM_THREADS")) {
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end != env && value > 0)
            return static_cast<std::size_t>(value);
    }
    return 0;
}

// Worker count: ``requested`` when positive, else the ARTLIB_NUM_THREADS
// environment variable when set to a positive integer, else every CPU this
// process may run on.
inline std::size_t resolve_threads(int requested) {
    if (requested > 0)
        return static_cast<std::size_t>(requested);
    if (const std::size_t env = env_threads())
        return env;
    return available_cpus();
}

// Caps the worker count so each thread gets a useful share of the work.
inline std::size_t useful_threads(std::size_t threads, std::size_t work) {
    return std::max<std::size_t>(1, std::min(threads, work / kWorkPerThread));
}

// Runs task(t) for every t in [0, n_tasks) on up to ``threads`` threads,
// including the calling one. Tasks are handed out in order from a shared
// counter, so faster cores take more of them. The first exception thrown by a
// task stops the remaining tasks and is rethrown here.
template <typename Task> void parallel_tasks(std::size_t n_tasks, std::size_t threads, Task task) {
    if (n_tasks == 0)
        return;
    threads = std::min(threads, n_tasks);
    if (threads <= 1) {
        for (std::size_t t = 0; t < n_tasks; ++t)
            task(t);
        return;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&]() {
        for (;;) {
            if (failed.load(std::memory_order_relaxed))
                return;
            const std::size_t t = next.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks)
                return;
            try {
                task(t);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error)
                    error = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (std::size_t i = 0; i + 1 < threads; ++i) {
        try {
            pool.emplace_back(work);
        } catch (...) {
            break; // could not start another thread; the ones running will finish the work
        }
    }
    work();
    for (auto& thread : pool)
        thread.join();
    if (error)
        std::rethrow_exception(error);
}

// Splits ``n`` samples into blocks of at most kMaxBlock, small enough that every
// thread gets at least one block.
inline std::size_t block_size(std::size_t n, std::size_t threads) {
    const std::size_t per_thread = (n + threads - 1) / threads;
    return std::max<std::size_t>(1, std::min(kMaxBlock, per_thread));
}

// Category weights in panels of kLanes categories. Within a panel, the values
// for feature j are stored together, one stream after another:
//   data[((panel * dim + j) * streams + stream) * kLanes + lane]
// so the kernel loads kLanes consecutive doubles per feature and stream. A
// model with two per-feature arrays (Gaussian mean and inverse variance) uses
// two streams. Lanes past the last category are zero and never read back.
struct Panels {
    std::size_t count = 0;   // categories
    std::size_t dim = 0;     // features per stream
    std::size_t streams = 1; // per-feature arrays per category
    std::vector<double> data;

    std::size_t panels() const { return (count + kLanes - 1) / kLanes; }
    std::size_t padded() const { return panels() * kLanes; }
    std::size_t feature_stride() const { return streams * kLanes; }
    const double* panel(std::size_t p) const { return data.data() + p * dim * feature_stride(); }

    // Removes every category and sets the shape.
    void reset(std::size_t n_features, std::size_t n_streams) {
        count = 0;
        dim = n_features;
        streams = n_streams;
        data.clear();
    }

    // Adds a category with every value zero and returns its index. A full last
    // panel gets a new one after it; std::vector keeps spare capacity, so this is
    // amortized O(dim).
    std::size_t append() {
        if (count % kLanes == 0)
            data.resize(data.size() + dim * feature_stride(), 0.0);
        return count++;
    }

    // Stream s of category c: feature j is at values(c, s)[j * feature_stride()].
    double* values(std::size_t c, std::size_t s) {
        return data.data() + (c / kLanes) * dim * feature_stride() + s * kLanes + c % kLanes;
    }
    const double* values(std::size_t c, std::size_t s) const {
        return data.data() + (c / kLanes) * dim * feature_stride() + s * kLanes + c % kLanes;
    }

    // ``source(c, s)`` returns a pointer to ``dim`` doubles: stream s of category c.
    template <typename Source>
    void pack(std::size_t n_categories, std::size_t n_features, std::size_t n_streams,
              Source source) {
        count = n_categories;
        dim = n_features;
        streams = n_streams;
        data.assign(panels() * dim * feature_stride(), 0.0);
        for (std::size_t c = 0; c < count; ++c) {
            double* base = data.data() + (c / kLanes) * dim * feature_stride() + c % kLanes;
            for (std::size_t s = 0; s < streams; ++s) {
                const double* values = source(c, s);
                double* out = base + s * kLanes;
                for (std::size_t j = 0; j < dim; ++j)
                    out[j * feature_stride()] = values[j];
            }
        }
    }
};

// Applies features [j0, j1) of S samples to one panel. ``acc`` holds S rows of
// kLanes accumulators, ``ld`` doubles apart. ``step(acc, x, w, lane)`` returns
// the updated accumulator, where ``w`` points at feature j of the panel (stream
// s of lane b is w[s * kLanes + b]).
template <std::size_t S, typename Step>
inline void panel_kernel(const double* const* rows, const double* panel, std::size_t stride,
                         std::size_t j0, std::size_t j1, double* acc, std::size_t ld, Step step) {
    double a[S][kLanes];
    for (std::size_t s = 0; s < S; ++s)
        for (std::size_t b = 0; b < kLanes; ++b)
            a[s][b] = acc[s * ld + b];
    for (std::size_t j = j0; j < j1; ++j) {
        const double* w = panel + j * stride;
        for (std::size_t s = 0; s < S; ++s) {
            const double x = rows[s][j];
            for (std::size_t b = 0; b < kLanes; ++b)
                a[s][b] = step(a[s][b], x, w, b);
        }
    }
    for (std::size_t s = 0; s < S; ++s)
        for (std::size_t b = 0; b < kLanes; ++b)
            acc[s * ld + b] = a[s][b];
}

// panel_kernel with ``body`` on features [j0, js) and ``tail`` on [js, j1).
template <std::size_t S, typename Body, typename Tail>
inline void split_kernel(const double* const* rows, const double* panel, std::size_t stride,
                         std::size_t j0, std::size_t js, std::size_t j1, double* acc,
                         std::size_t ld, Body body, Tail tail) {
    if (j0 < js)
        panel_kernel<S>(rows, panel, stride, j0, js, acc, ld, body);
    if (js < j1)
        panel_kernel<S>(rows, panel, stride, js, j1, acc, ld, tail);
}

// Where the scalar distance loops this kernel replaces switch to fused
// multiply-adds. Clang (-O3, the default -ffp-contract=on) compiles
//     for (j = 0; j < dim; ++j) { d = x[j] - w[j]; s += d * d; }
// into an 8-wide body that squares with vector multiplies and then adds the
// products to s one at a time, in order (so the multiply and add are separate
// roundings), followed by a scalar loop over the last dim % 8 features that uses
// a fused multiply-add. Distance kernels reproduce that split, writing the
// multiply and the add as separate statements before this index and as one
// contractible expression after it, so their results stay bit-identical to the
// scalar loops that fit and predict used before. Where the target has no FMA, or the
// compiler fuses across statements (GCC's default), both halves compile alike
// and still match the scalar loop built by the same compiler.
inline std::size_t fused_tail_start(std::size_t dim) { return dim - dim % 8; }

// For every sample i in [0, n) (row i starts at X + i * x_stride) and category c,
// computes
//     acc = 0.0;  for j in [0, dim): acc = step_j(acc, x[i][j], w_c at j)
// adding features in order, where step_j is ``body`` for j < split and ``tail``
// for j >= split. It then calls finish(i, acc_row), where acc_row[c] is the
// result for category c. Samples run in parallel blocks; ``finish`` must be safe
// to call from several threads for different samples.
template <typename Body, typename Tail, typename Finish>
void accumulate(const double* X, std::size_t n, std::size_t x_stride, const Panels& weights,
                std::size_t threads, std::size_t split, Body body, Tail tail, Finish finish) {
    if (n == 0 || weights.count == 0)
        return;
    const std::size_t dim = weights.dim;
    const std::size_t stride = weights.feature_stride();
    const std::size_t ld = weights.padded();
    const std::size_t n_panels = weights.panels();
    const std::size_t feature_block =
        std::max<std::size_t>(16, kPanelSliceBytes / (stride * sizeof(double)));

    threads = useful_threads(threads, n * weights.count * std::max<std::size_t>(dim, 1));
    const std::size_t block = block_size(n, threads);
    const std::size_t n_blocks = (n + block - 1) / block;

    parallel_tasks(n_blocks, threads, [&](std::size_t task) {
        const std::size_t i0 = task * block;
        const std::size_t m = std::min(block, n - i0);
        std::vector<double> acc(m * ld, 0.0);
        std::vector<const double*> rows(m);
        for (std::size_t i = 0; i < m; ++i)
            rows[i] = X + (i0 + i) * x_stride;

        for (std::size_t j0 = 0; j0 < dim; j0 += feature_block) {
            const std::size_t j1 = std::min(dim, j0 + feature_block);
            const std::size_t js = std::min(std::max(split, j0), j1);
            for (std::size_t p = 0; p < n_panels; ++p) {
                const double* panel = weights.panel(p);
                double* out = acc.data() + p * kLanes;
                std::size_t i = 0;
                for (; i + kTile <= m; i += kTile)
                    split_kernel<kTile>(rows.data() + i, panel, stride, j0, js, j1, out + i * ld,
                                        ld, body, tail);
                for (; i < m; ++i)
                    split_kernel<1>(rows.data() + i, panel, stride, j0, js, j1, out + i * ld, ld,
                                    body, tail);
            }
        }
        for (std::size_t i = 0; i < m; ++i)
            finish(i0 + i, acc.data() + i * ld);
    });
}

// Index of the highest score, scanning c = 0, 1, ... and replacing the best only
// on a strictly greater score, so ties go to the lowest index. ``score(c)``
// computes the score of category c.
template <typename Score> std::size_t argmax_first(std::size_t count, Score score) {
    std::size_t best = 0;
    double best_score = score(0);
    for (std::size_t c = 1; c < count; ++c) {
        const double value = score(c);
        if (value > best_score) {
            best = c;
            best_score = value;
        }
    }
    return best;
}

// Binary vectors packed 64 features per word, for intersection counts by popcount.
inline std::uint32_t popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint32_t>(__builtin_popcountll(value));
#else
    value = value - ((value >> 1) & 0x5555555555555555ULL);
    value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<std::uint32_t>((value * 0x0101010101010101ULL) >> 56);
#endif
}

inline std::size_t bit_words(std::size_t bits) { return (bits + 63) / 64; }

// Packs ``bits`` values (nonzero means 1) into ``out``, which must hold
// bit_words(bits) words.
template <typename T> void pack_bits(const T* values, std::size_t bits, std::uint64_t* out) {
    const std::size_t words = bit_words(bits);
    std::fill(out, out + words, std::uint64_t{0});
    for (std::size_t j = 0; j < bits; ++j)
        out[j / 64] |= static_cast<std::uint64_t>(values[j] != 0) << (j % 64);
}

inline std::uint32_t and_count(const std::uint64_t* a, const std::uint64_t* b, std::size_t words) {
    std::uint32_t count = 0;
    for (std::size_t k = 0; k < words; ++k)
        count += popcount64(a[k] & b[k]);
    return count;
}

} // namespace art_predict
