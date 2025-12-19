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

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t{0});

    {
        py::gil_scoped_release release;
        fracsort::argsort_inplace<T>(num, den, n, idx.data());
    }

    py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
    auto outbuf = out.mutable_unchecked<1>();
    for (size_t k = 0; k < n; ++k) {
        outbuf(static_cast<py::ssize_t>(k)) = static_cast<py::ssize_t>(idx[k]);
    }
    return out;
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

PYBIND11_MODULE(fracsort, m) {
    m.doc() = "Division-free fraction argsort via cross-multiplication (uint32 fast path; uint64 supported).";
    m.def("fracsort", &fracsort_dispatch, py::arg("num"), py::arg("den"),
          "Return indices that sort by num[i]/den[i] ascending (no division), ties by lowest index.\n"
          "Requires: num, den are 1D C-contiguous arrays with dtype uint32 or uint64; den[i] >= 1.");
}
