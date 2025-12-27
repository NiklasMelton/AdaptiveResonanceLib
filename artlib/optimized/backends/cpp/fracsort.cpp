#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "fraction_sort_core.hpp"

namespace py = pybind11;

template <typename T>
static py::array_t<py::ssize_t> fracsort_impl(py::array num_in, py::array den_in) {
    if (num_in.ndim() != 1 || den_in.ndim() != 1) {
        throw std::runtime_error("num and den must be 1D arrays");
    }
    if (num_in.shape(0) != den_in.shape(0)) {
        throw std::runtime_error("num and den must have the same length");
    }
    if (!(num_in.flags() & py::array::c_style) || !(den_in.flags() & py::array::c_style)) {
        throw std::runtime_error("num and den must be C-contiguous arrays");
    }

    const size_t n = static_cast<size_t>(num_in.shape(0));
    const auto nbuf = num_in.request();
    const auto dbuf = den_in.request();

    const auto* num = static_cast<const T*>(nbuf.ptr);
    const auto* den = static_cast<const T*>(dbuf.ptr);

    // Enforce denominators >= 1 (i.e., nonzero)
    for (size_t i = 0; i < n; ++i) {
        if (den[i] == 0) {
            throw std::runtime_error("All denominators must be >= 1");
        }
    }

    std::vector<fracsort::Item<T>> items;
    items.resize(n);
    for (size_t i = 0; i < n; ++i) {
        items[i] = fracsort::Item<T>{num[i], den[i], i};
    }

    {
        py::gil_scoped_release release;
        fracsort::argsort_items_inplace<T>(items.data(), n);
    }


    py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
    auto outbuf = out.mutable_unchecked<1>();
    for (size_t k = 0; k < n; ++k) {
        outbuf(static_cast<py::ssize_t>(k)) = static_cast<py::ssize_t>(items[k].idx);
    }

    return out;
}

template <typename T>
static py::ssize_t fracargmax_impl(py::array num_in, py::array den_in) {
    if (num_in.ndim() != 1 || den_in.ndim() != 1) {
        throw std::runtime_error("num and den must be 1D arrays");
    }
    if (num_in.shape(0) != den_in.shape(0)) {
        throw std::runtime_error("num and den must have the same length");
    }
    if (!(num_in.flags() & py::array::c_style) || !(den_in.flags() & py::array::c_style)) {
        throw std::runtime_error("num and den must be C-contiguous arrays");
    }

    const size_t n = static_cast<size_t>(num_in.shape(0));
    if (n == 0) {
        throw std::runtime_error("num and den must be non-empty");
    }

    const auto nbuf = num_in.request();
    const auto dbuf = den_in.request();

    const auto* num = static_cast<const T*>(nbuf.ptr);
    const auto* den = static_cast<const T*>(dbuf.ptr);

    // Enforce denominators >= 1 (i.e., nonzero)
    for (size_t i = 0; i < n; ++i) {
        if (den[i] == 0) {
            throw std::runtime_error("All denominators must be >= 1");
        }
    }

    std::vector<fracsort::Item<T>> items;
    items.resize(n);
    for (size_t i = 0; i < n; ++i) {
        items[i] = fracsort::Item<T>{num[i], den[i], i};
    }

    size_t best_idx = 0;
    {
        py::gil_scoped_release release;
        best_idx = fracsort::fracargmax_items<T>(items.data(), n);
    }

    return static_cast<py::ssize_t>(best_idx);
}


static py::array_t<py::ssize_t> fracsort_dispatch(py::array num_in, py::array den_in) {
    const auto dt_num = num_in.dtype();
    const auto dt_den = den_in.dtype();

    if (!dt_num.is(dt_den)) {
        throw std::runtime_error("num and den must have the same dtype (uint32 or uint64)");
    }

    if (dt_num.is(py::dtype::of<uint32_t>())) {
        return fracsort_impl<uint32_t>(num_in, den_in);
    }
    if (dt_num.is(py::dtype::of<uint64_t>())) {
        return fracsort_impl<uint64_t>(num_in, den_in);
    }

    throw std::runtime_error("num and den must have dtype uint32 or uint64");
}

static py::ssize_t fracargmax_dispatch(py::array num_in, py::array den_in) {
    const auto dt_num = num_in.dtype();
    const auto dt_den = den_in.dtype();

    if (!dt_num.is(dt_den)) {
        throw std::runtime_error("num and den must have the same dtype (uint32 or uint64)");
    }

    if (dt_num.is(py::dtype::of<uint32_t>())) {
        return fracargmax_impl<uint32_t>(num_in, den_in);
    }
    if (dt_num.is(py::dtype::of<uint64_t>())) {
        return fracargmax_impl<uint64_t>(num_in, den_in);
    }

    throw std::runtime_error("num and den must have dtype uint32 or uint64");
}


PYBIND11_MODULE(fracsort, m) {
    m.doc() = "Division-free fraction argsort via cross-multiplication (uint32 fast path; uint64 supported).";
    m.def("fracsort", &fracsort_dispatch, py::arg("num"), py::arg("den"),
          "Return indices that sort by num[i]/den[i] descending (no division).\n"
          "Ties: larger denominator first, then lower index.\n"
          "Requires: num, den are 1D C-contiguous arrays with dtype uint32 or uint64; den[i] >= 1.");
    m.def("fracargmax", &fracargmax_dispatch, py::arg("num"), py::arg("den"),
          "Return the index i that maximizes num[i]/den[i] (no division).\n"
          "Ties: larger denominator first, then lower index.\n"
          "Requires: num, den are 1D C-contiguous arrays with dtype uint32 or uint64; den[i] >= 1.");

    m.def("reset_sort_stats", &fracsort::reset_stats);

    m.def("get_sort_stats", []() {
        auto snap = fracsort::get_stats_snapshot();
        py::dict d;
        d["cmp_calls"]      = snap.cmp_calls;
        d["left_eq_right"]  = snap.left_eq_right;
        d["num_eq_num"]     = snap.num_eq_num;
        d["num_is_zero"]    = snap.num_is_zero;
        return d;
    });

}
