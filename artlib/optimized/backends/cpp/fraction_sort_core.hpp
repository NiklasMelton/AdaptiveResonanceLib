#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace fracsort {

// Wide multiplication type: uint32 -> uint64 (exact), uint64 -> __int128 (exact)
template <typename T>
struct WideMul;

template <>
struct WideMul<uint32_t> {
    using wide_t = uint64_t;
    static inline wide_t mul(uint32_t a, uint32_t b) {
        return static_cast<wide_t>(a) * static_cast<wide_t>(b);
    }
};

template <>
struct WideMul<uint64_t> {
    using wide_t = __int128;
    static inline wide_t mul(uint64_t a, uint64_t b) {
        return static_cast<wide_t>(a) * static_cast<wide_t>(b);
    }
};

template <typename T>
static inline bool frac_less_idx(const T* num, const T* den, size_t i, size_t j) {
    using W = typename WideMul<T>::wide_t;

    // Compare num[i]/den[i] < num[j]/den[j]  <=>  num[i]*den[j] < num[j]*den[i]
    const W left  = WideMul<T>::mul(num[i], den[j]);
    const W right = WideMul<T>::mul(num[j], den[i]);

    if (left < right) return true;
    if (left > right) return false;
    return i < j; // tie-break by index => strict total order
}

template <typename T>
inline void argsort_inplace(const T* num, const T* den, size_t n, size_t* idx) {
    std::sort(idx, idx + n, [&](size_t i, size_t j) {
        return frac_less_idx<T>(num, den, i, j);
    });
}

} // namespace fracsort
