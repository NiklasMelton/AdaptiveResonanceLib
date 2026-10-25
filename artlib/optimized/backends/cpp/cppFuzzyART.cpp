// cppFuzzyART.cpp
// ----------------------------------------------------------
//  C++ accelerated Fuzzy  ART  (pybind11)
//  ‑ supports incremental training via external weights
//  ‑ real‑valued inputs in [0,1] (already normalised & complement‑coded)
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

// ────────────────────────────────────────────────────────────
class NativeFuzzyART {
public:
    struct Cluster {
        std::vector<double> weight; // 2d components (complement‑coded)
        double sum = 0.0;           // cached sum of weight, refreshed on every change
    };

    NativeFuzzyART(double rho, double alpha, double beta, py::object weights = py::none())
        : rho_(rho), alpha_(alpha), beta_(beta), dim_original_(0) {
        const bool have_W = !weights.is_none();

        if (have_W) {
            py::list w_list = weights.cast<py::list>();

            const std::size_t n_clusters = w_list.size();

            clusters_.resize(n_clusters);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto w_b = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::invalid_argument("each weight must be 1-D");
                if (w_b.shape[0] < 2 || w_b.shape[0] % 2 != 0)
                    throw std::invalid_argument("weight length must be positive and even");
                clusters_[k].weight.resize(static_cast<std::size_t>(w_b.shape[0]));
                std::memcpy(clusters_[k].weight.data(), w_b.ptr, sizeof(double) * w_b.shape[0]);
                clusters_[k].sum = weight_sum(clusters_[k].weight);

                if (dim_original_ == 0)
                    dim_original_ = static_cast<int>(w_b.shape[0] / 2);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>> fit(py::array_t<double> X) {
        auto xb = X.request();

        if (xb.ndim != 2)
            throw std::invalid_argument("X must be 2-D");
        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (n_features % 2 != 0)
            throw std::invalid_argument("X must be complement coded");

        if (dim_original_ == 0) {
            dim_original_ = n_features / 2;
        }
        if (n_features != 2 * dim_original_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i)
            labels_a[i] = step_fit(Xptr + static_cast<std::size_t>(i) * n_features);

        /* -------- pack output -------- */
        auto labels_out = art_bind::output(labels_a);

        auto weight_out = art_bind::pack_weights<double>(
            clusters_, [](const Cluster& cluster) -> const auto& { return cluster.weight; });

        return {labels_out, weight_out};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a)
    // ────────────────────────────────────────────────────────
    py::array_t<int> predict(py::array_t<double> X) {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        const int n = static_cast<int>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_original_ == 0)
            dim_original_ = dim / 2;
        if (dim % 2 != 0 || dim != 2 * dim_original_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y_a(n);

        for (int i = 0; i < n; ++i) {
            const double* row = Xptr + i * dim;
            art_core::NoLabels unused_labels;
            auto state = art_core::state_view(clusters_, unused_labels, dim_original_);
            const int best_id = static_cast<int>(art_core::predict_one(
                state, [&](std::size_t c) { return category_choice(row, clusters_[c]); },
                std::greater<double>{})); // fallback
            y_a[i] = best_id;
        }

        py::array_t<int> ya_out(y_a.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        return ya_out;
    }

private:
    /* ───── hyper‑parameters ───── */
    double rho_, alpha_, beta_;

    /* ───── state ───── */
    int dim_original_;
    std::vector<Cluster> clusters_;

    /* ───── per-sample scratch, reused between samples ───── */
    std::vector<double> and_sums_;
    art_core::LazySearchScratch<double> scratch_;

    /* ─────────────────────────────────────────────────── */

    static double weight_sum(const std::vector<double>& w) {
        return std::accumulate(w.begin(), w.end(), 0.0);
    }

    // fuzzy  AND L1
    static double l1_and(const double* x, const std::vector<double>& w) {
        double s = 0.0;
        for (std::size_t j = 0; j < w.size(); ++j)
            s += std::min(x[j], w[j]);
        return s;
    }

    /* ─────────────────────────────────────────────────── */
    double category_choice(const double* sample, const Cluster& cluster) const {
        return l1_and(sample, cluster.weight) / (alpha_ + cluster.sum);
    }

    void update_weight(const double* i, Cluster& cluster) const {
        auto& w = cluster.weight;
        for (std::size_t j = 0; j < w.size(); ++j)
            w[j] = beta_ * std::min(i[j], w[j]) + (1.0 - beta_) * w[j];
        cluster.sum = weight_sum(w);
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const double* sample) {
        art_core::NoLabels unused_labels;
        auto state = art_core::state_view(clusters_, unused_labels, dim_original_);
        and_sums_.resize(clusters_.size());
        return static_cast<int>(art_core::fit_one_lazy(
            state,
            [&](std::size_t c) {
                // The fuzzy AND sum is shared by choice and match.
                and_sums_[c] = l1_and(sample, clusters_[c].weight);
                return and_sums_[c] / (alpha_ + clusters_[c].sum);
            },
            [&](std::size_t c) { return and_sums_[c] / static_cast<double>(dim_original_); },
            [&](double value) { return value >= rho_; }, [](std::size_t) { return true; },
            [](double) { return true; },
            [&](std::size_t c) { update_weight(sample, clusters_[c]); },
            [&]() {
                const auto index = clusters_.size();
                Cluster cluster;
                cluster.weight.assign(sample, sample + 2 * dim_original_);
                cluster.sum = weight_sum(cluster.weight);
                clusters_.push_back(std::move(cluster));
                return index;
            },
            scratch_));
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, double rho, double alpha, double beta,
                py::object weights = py::none()) {
    auto data = art_bind::matrix<double>(X);

    NativeFuzzyART model(rho, alpha, beta, art_bind::weights<double>(weights));
    return model.fit(data);
}

auto native_predict(py::object X, double rho, double alpha, double beta,
                    py::object weights = py::none()) {
    auto data = art_bind::matrix<double>(X);
    NativeFuzzyART model(rho, alpha, beta, art_bind::weights<double>(weights));
    return model.predict(data);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppFuzzyART, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("rho"), py::arg("alpha"), py::arg("beta"),
          py::arg("weights") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("weights") = py::none());
}
