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
struct Item {
    T num;
    T den;
    size_t idx; // original index for tie-break
};

template <typename T>
static inline bool frac_greater_item(const Item<T>& a, const Item<T>& b) noexcept {
    using W = typename WideMul<T>::wide_t;

    // Optional fast paths (always correct)
    if (a.den == b.den) {
        if (a.num != b.num) return a.num > b.num; // same den => compare num
        return a.idx < b.idx;                     // exact tie => lowest index first
    }
    if (a.num == b.num) {
        // If num == 0, all fractions are exactly 0 regardless of denominator.
        if (a.num == 0) return a.idx < b.idx;

        // Otherwise, same positive num: larger fraction has smaller denominator.
        if (a.den != b.den) return a.den < b.den;
        return a.idx < b.idx;
    }

    // Exact comparison: a.num/a.den > b.num/b.den  <=>  a.num*b.den > b.num*a.den
    const W left  = WideMul<T>::mul(a.num, b.den);
    const W right = WideMul<T>::mul(b.num, a.den);

    if (left > right) return true;   // descending
    if (left < right) return false;
    if (a.den != b.den) return a.den > b.den; // ties => lowest original index first
    return a.idx < b.idx;
}

template <typename T>
inline void argsort_items_inplace(Item<T>* items, size_t n) {
    std::sort(items, items + n, [](const Item<T>& a, const Item<T>& b) noexcept {
        return frac_greater_item<T>(a, b);
    });
}

template <typename T>
inline size_t fracargmax_items(const Item<T>* items, size_t n) {
    // Returns the original index (items[k].idx) of the maximum fraction.
    // Uses the same ordering as frac_greater_item:
    //   - larger num/den
    //   - tie: larger den
    //   - tie: lower original index
    if (n == 0) return 0; // caller should ensure n>0; keeps function total.

    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (frac_greater_item<T>(items[i], items[best])) {
            best = i;
        }
    }
    return items[best].idx;
}


} // namespace fracsort
