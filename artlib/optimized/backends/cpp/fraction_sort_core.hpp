#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>
#include <type_traits>

// Enable uint64 fraction support only when we have a real 128-bit integer type.
#if !defined(_MSC_VER) && defined(__SIZEOF_INT128__)
  #define FRACSORT_HAS_U64 1
#else
  #define FRACSORT_HAS_U64 0
#endif


namespace fracsort {

template <typename T>
struct WideMul;

// Wide multiplication type: uint32 -> uint64 (exact), uint64 -> __int128 (exact)
template <>
struct WideMul<std::uint32_t> {
    using wide_t = std::uint64_t;
    static inline wide_t mul(std::uint32_t a, std::uint32_t b) noexcept {
        return static_cast<wide_t>(a) * static_cast<wide_t>(b);
    }
};

#if FRACSORT_HAS_U64
template <>
struct WideMul<std::uint64_t> {
    using wide_t = unsigned __int128;
    static inline wide_t mul(std::uint64_t a, std::uint64_t b) noexcept {
        return static_cast<wide_t>(a) * static_cast<wide_t>(b);
    }
};
#endif

template <typename T>
struct Item {
    T num;
    T den;
    T rnum;     // reduced numerator
    T rden;     // reduced denominator
    size_t idx; // original index for tie-break
};

template <typename T>
static inline void reduce_item_inplace(Item<T>& it) noexcept {
    // Define a canonical representation for zero.
    // (Any 0/den is the same value; pick 0/1 for reduced form.)
    if (it.num == 0) {
        it.rnum = 0;
        it.rden = 1;
        return;
    }

    // Assume den > 0 (as your code implicitly does). If den can be 0, handle it separately.
    T g = std::gcd(it.num, it.den);
    it.rnum = it.num / g;
    it.rden = it.den / g;
}

template <typename T>
inline void reduce_items_inplace(Item<T>* items, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) reduce_item_inplace(items[i]);
}


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

    // Fast exact equality check via reduced canonical form
    if (a.rnum == b.rnum && a.rden == b.rden) {
        // equal value => tie-break by largest original denominator, then index
        if (a.den != b.den) return a.den > b.den;
        return a.idx < b.idx;
    }


    // Exact comparison on reduced forms:
    // a.rnum/a.rden > b.rnum/b.rden  <=>  a.rnum*b.rden > b.rnum*a.rden
    const W left  = WideMul<T>::mul(a.rnum, b.rden);
    const W right = WideMul<T>::mul(b.rnum, a.rden);

    if (left > right) return true;   // descending
    if (left < right) return false;

    // Exact-value tie: break by largest *original* denominator, then by idx if denom also equal
    if (a.den != b.den) return a.den > b.den;
    return a.idx < b.idx;

}

template <typename T>
static inline bool frac_value_greater_item(const Item<T>& a, const Item<T>& b) noexcept {
    using W = typename WideMul<T>::wide_t;

    // Same reduced value => order by idx only (do NOT involve den here)
    if (a.rnum == b.rnum && a.rden == b.rden) {
        return a.idx < b.idx;
    }

    const W left  = WideMul<T>::mul(a.rnum, b.rden);
    const W right = WideMul<T>::mul(b.rnum, a.rden);

    if (left > right) return true;
    if (left < right) return false;

    // Should be unreachable if reduced forms differ, but keep it deterministic
    return a.idx < b.idx;
}

template <typename T>
static inline bool tie_den_desc_idx_asc(const Item<T>& a, const Item<T>& b) noexcept {
    if (a.den != b.den) return a.den > b.den; // larger denominator first
    return a.idx < b.idx;                    // then lower index
}



template <typename T>
inline void argsort_items_inplace(Item<T>* items, size_t n) {
    reduce_items_inplace(items, n);

    // 1) Pre-sort by tie-break keys (den desc, then idx asc)
    std::sort(items, items + n, [](const Item<T>& a, const Item<T>& b) noexcept {
        if (a.den != b.den) return a.den > b.den;
        return a.idx < b.idx;
    });

    // 2) Stable sort by primary key (value) only.
    //    Stability preserves the den/idx order within equal-value groups.
    std::stable_sort(items, items + n, [](const Item<T>& a, const Item<T>& b) noexcept {
        using W = typename WideMul<T>::wide_t;

        // Compare reduced values (descending)
        const W left  = WideMul<T>::mul(a.rnum, b.rden);
        const W right = WideMul<T>::mul(b.rnum, a.rden);

        return left > right;
    });
}




template <typename T>
inline size_t fracargmax_items(Item<T>* items, size_t n) {
    if (n == 0) return 0;
    reduce_items_inplace(items, n); // NEW
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (frac_greater_item<T>(items[i], items[best])) best = i;
    }
    return items[best].idx;
}



} // namespace fracsort
