// cppART1.cpp
// ----------------------------------------------------------
//  C++ accelerated ART1 (pybind11)
//  - supports incremental training via external weights
//  - binary inputs in {0,1} (NO complement coding)
//  - weights stored as length 2*dim: [w_bu (float), w_td (binary 0/1)]
//  - parameters: rho (vigilance), L (uncommitted node bias, >= 1)
// ----------------------------------------------------------
#include "art_core.hpp"
#include "native_arrays.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class NativeART1 {
public:
    struct Cluster {
        std::vector<double> weight; // size 2*dim: [bu..., td...]
    };

    NativeART1(double rho, double L, py::object weights = py::none()) : rho_(rho), L_(L), dim_(0) {
        if (rho_ < 0.0 || rho_ > 1.0)
            throw std::invalid_argument("rho must be in [0,1]");
        if (L_ < 1.0)
            throw std::invalid_argument("L must be >= 1");

        const bool have_W = !weights.is_none();
        if (have_W) {
            py::list w_list = weights.cast<py::list>();
            const std::size_t n_clusters = w_list.size();

            clusters_.resize(n_clusters);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto wb = w_arr.request();
                if (wb.ndim != 1)
                    throw std::invalid_argument("each weight must be 1-D");

                const int len = static_cast<int>(wb.shape[0]);
                if (len % 2 != 0)
                    throw std::invalid_argument("weight length must be even (2*dim)");

                clusters_[k].weight.resize(static_cast<std::size_t>(len));
                std::memcpy(clusters_[k].weight.data(), wb.ptr, sizeof(double) * len);

                if (dim_ == 0)
                    dim_ = len / 2;
                if (len != 2 * dim_)
                    throw std::invalid_argument("inconsistent weight dimension across clusters");
            }
        }
    }

    // FIT: returns (labels, weights_vec<ndarray>)
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>> fit(py::array_t<double> X) {
        auto xb = X.request();
        if (xb.ndim != 2)
            throw std::invalid_argument("X must be 2-D [n_samples, n_features]");

        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("X feature dimension mismatch with existing model");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> labels(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const double* row = Xptr + i * n_features;

            // validate binary {0,1}
            for (int j = 0; j < dim_; ++j) {
                const double v = row[j];
                if (!(v == 0.0 || v == 1.0)) {
                    throw std::invalid_argument("ART1 requires binary inputs in {0,1}");
                }
            }

            // copy sample into uint8-like semantics (still doubles for speed/simple)
            std::vector<double> sample(row, row + dim_);
            labels[i] = step_fit(sample);
        }

        auto labels_out = art_bind::output(labels);

        auto weight_out = art_bind::pack_weights<double>(
            clusters_, [](const Cluster& cluster) -> const auto& { return cluster.weight; });

        return {labels_out, weight_out};
    }

    // PREDICT: returns y
    py::array_t<int> predict(py::array_t<double> X) {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        if (xb.ndim != 2)
            throw std::invalid_argument("X must be 2-D [n_samples, n_features]");

        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("X feature dimension mismatch with model");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const double* row = Xptr + i * dim_;

            // validate binary
            for (int j = 0; j < dim_; ++j) {
                const double v = row[j];
                if (!(v == 0.0 || v == 1.0)) {
                    throw std::invalid_argument("ART1 requires binary inputs in {0,1}");
                }
            }
            art_core::NoLabels unused_labels;
            auto state = art_core::state_view(clusters_, unused_labels, dim_);
            const int best_id = static_cast<int>(art_core::predict_one(
                state, [&](std::size_t c) { return category_choice(row, clusters_[c].weight); },
                std::greater<double>{}));
            y[i] = best_id;
        }

        py::array_t<int> y_out(y.size());
        std::memcpy(y_out.mutable_data(), y.data(), sizeof(int) * y.size());
        return y_out;
    }

private:
    double rho_;
    double L_;
    int dim_;
    std::vector<Cluster> clusters_;

    // category choice: dot(i, w_bu)
    double category_choice(const double* sample, const std::vector<double>& w) const {
        double s = 0.0;
        // w_bu is first dim_
        for (int j = 0; j < dim_; ++j) {
            s += sample[j] * w[j];
        }
        return s;
    }

    // match criterion: count_nonzero(i & w_td) / dim
    double match(const double* sample, const std::vector<double>& w) const {
        int count = 0;
        // w_td is second dim_
        for (int j = 0; j < dim_; ++j) {
            const int i_bit = (sample[j] != 0.0);
            const int td_bit = (w[dim_ + j] != 0.0);
            if (i_bit & td_bit)
                ++count;
        }
        return static_cast<double>(count) / static_cast<double>(dim_);
    }

    // update rule:
    // w_td_new = i & w_td
    // count = nnz(w_td_new)
    // w_bu_new = (L / (L - 1 + count)) * w_td_new
    std::vector<double> update_weight(const double* sample, const std::vector<double>& w) const {
        std::vector<double> out(2 * dim_, 0.0);

        // compute new top-down, count nnz
        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int td_new = ((sample[j] != 0.0) & (w[dim_ + j] != 0.0));
            out[dim_ + j] = static_cast<double>(td_new);
            if (td_new)
                ++count;
        }

        // scaling for bottom-up
        const double denom = (L_ - 1.0 + static_cast<double>(count));
        const double sf = (denom > 0.0) ? (L_ / denom) : 0.0;

        for (int j = 0; j < dim_; ++j) {
            out[j] = sf * out[dim_ + j];
        }

        return out;
    }

    // new cluster weight:
    // w_td = i
    // w_bu = (L / (L - 1 + dim)) * i
    std::vector<double> new_weight(const double* sample) const {
        std::vector<double> w(2 * dim_, 0.0);

        const double sf = L_ / (L_ - 1.0 + static_cast<double>(dim_));

        for (int j = 0; j < dim_; ++j) {
            const double bit = (sample[j] != 0.0) ? 1.0 : 0.0;
            w[dim_ + j] = bit; // top-down
            w[j] = sf * bit;   // bottom-up
        }

        return w;
    }

    int step_fit(const std::vector<double>& sample_vec) {
        const double* row = sample_vec.data();
        art_core::NoLabels unused_labels;
        auto state = art_core::state_view(clusters_, unused_labels, dim_);
        return static_cast<int>(art_core::fit_one(
            state, [&](std::size_t c) { return category_choice(row, clusters_[c].weight); },
            [&](std::size_t c) { return match(row, clusters_[c].weight); },
            [](const auto& scores, const auto&) { return art_core::descending_order(scores); },
            [&](double value) { return value >= rho_; }, [](std::size_t) { return true; },
            [](double) { return true; },
            [&](std::size_t c) { clusters_[c].weight = update_weight(row, clusters_[c].weight); },
            [&]() {
                const auto index = clusters_.size();
                clusters_.push_back({new_weight(row)});
                return index;
            }));
    }
};

// convenience wrappers
auto native_fit(py::object X, double rho, double L, py::object weights = py::none()) {
    auto data = art_bind::matrix<double>(X);

    NativeART1 model(rho, L, art_bind::weights<double>(weights));
    return model.fit(data);
}

auto native_predict(py::object X, double rho, double L, py::object weights = py::none()) {
    auto data = art_bind::matrix<double>(X);
    NativeART1 model(rho, L, art_bind::weights<double>(weights));
    return model.predict(data);
}

// pybind11 module
PYBIND11_MODULE(cppART1, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("rho"), py::arg("L"),
          py::arg("weights") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("L"),
          py::arg("weights") = py::none());
}
