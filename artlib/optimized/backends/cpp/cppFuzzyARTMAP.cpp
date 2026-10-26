// cppFuzzyARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Fuzzy  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in [0,1] (already normalised & complement‑coded)
// ----------------------------------------------------------
#include "art_core.hpp"
#include "batch_predict.hpp"
#include "native_arrays.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ────────────────────────────────────────────────────────────
class NativeFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<double> weight; // 2d components (complement‑coded)
        double sum = 0.0;           // cached sum of weight, refreshed on every change
    };

    NativeFuzzyARTMAP(double rho, double alpha, double beta, const std::string& MT, double epsilon,
                      py::object weights = py::none(), py::object cluster_labels = py::none())
        : base_rho_(rho), alpha_(alpha), beta_(beta), epsilon_(epsilon), match_tracking_(MT),
          dim_original_(0), rho_(rho) {
        const bool have_W = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_W != have_cl) {
            throw std::invalid_argument("Provide BOTH 'weights' and 'cluster_labels' or neither.");
        }

        if (have_W) {
            py::list w_list = weights.cast<py::list>();
            py::array_t<int> cl = cluster_labels.cast<py::array_t<int>>();
            auto cl_b = cl.request();
            if (cl_b.ndim != 1)
                throw std::invalid_argument("cluster_labels must be 1‑D");

            const std::size_t n_clusters = w_list.size();
            if (static_cast<std::size_t>(cl_b.shape[0]) != n_clusters)
                throw std::invalid_argument("weights / cluster_labels size mismatch");

            clusters_.resize(n_clusters);
            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

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
                cluster_map_[static_cast<int>(k)] = cl_ptr[k];

                if (dim_original_ == 0)
                    dim_original_ = static_cast<int>(w_b.shape[0] / 2);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>, cluster_labels)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>>
    fit(py::array_t<double> X, py::array_t<int> y) {
        auto xb = X.request();
        auto yb = y.request();

        if (xb.ndim != 2 || yb.ndim != 1)
            throw std::invalid_argument("X must be 2-D and y must be 1-D");
        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_original_ == 0) {
            dim_original_ = n_features / 2;
        }
        if (n_features % 2 != 0 || n_features != 2 * dim_original_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const int* yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i)
            labels_a[i] = step_fit(Xptr + static_cast<std::size_t>(i) * n_features, yptr[i]);

        /* -------- pack output -------- */
        auto labels_out = art_bind::output(labels_a);

        auto weight_out = art_bind::pack_weights<double>(
            clusters_, [](const Cluster& cluster) -> const auto& { return cluster.weight; });

        std::vector<int> clabels_vec(clusters_.size());
        for (const auto& kv : cluster_map_)
            clabels_vec[kv.first] = kv.second;
        auto clabels_out = art_bind::output(clabels_vec);

        return {labels_out, weight_out, clabels_out};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a, y_b)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<double> X,
                                                           int n_threads = 0) {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        const std::size_t n = static_cast<std::size_t>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_original_ == 0)
            dim_original_ = dim / 2;
        if (dim % 2 != 0 || dim != 2 * dim_original_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const std::size_t count = clusters_.size();
        const std::size_t width = static_cast<std::size_t>(dim);

        art_predict::Panels panels;
        panels.pack(count, width, 1,
                    [&](std::size_t c, std::size_t) { return clusters_[c].weight.data(); });
        std::vector<double> denominators(count);
        std::vector<int> labels(count);
        for (std::size_t c = 0; c < count; ++c) {
            denominators[c] = alpha_ + clusters_[c].sum;
            labels[c] = cluster_map_.at(static_cast<int>(c));
        }

        std::vector<int> y_a(n), y_b(n);
        {
            py::gil_scoped_release release;
            // Fuzzy AND sum; there is no multiply, so the whole range is one step.
            auto step = [](double acc, double x, const double* w, std::size_t lane) {
                return acc + std::min(x, w[lane]);
            };
            art_predict::accumulate(Xptr, n, width, panels, art_predict::resolve_threads(n_threads),
                                    width, step, step, [&](std::size_t i, const double* and_sums) {
                                        const std::size_t best =
                                            art_predict::argmax_first(count, [&](std::size_t c) {
                                                return and_sums[c] / denominators[c];
                                            });
                                        y_a[i] = static_cast<int>(best);
                                        y_b[i] = labels[best];
                                    });
        }
        return {art_bind::output(y_a), art_bind::output(y_b)};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, beta_, epsilon_;
    std::string match_tracking_;

    /* ───── state ───── */
    int dim_original_;
    double rho_;
    std::vector<Cluster> clusters_;
    std::unordered_map<int, int> cluster_map_;

    /* ───── per-sample scratch, reused between samples ───── */
    std::vector<double> and_sums_;
    art_core::LazySearchScratch<double> scratch_;

    /* ─────────────────────────────────────────────────── */
    void reset_rho() { rho_ = base_rho_; }

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
    int step_fit(const double* sample, int c_b) {
        reset_rho();
        const auto mode = art_core::parse_match_tracking(match_tracking_);

        auto state = art_core::state_view(clusters_, cluster_map_, dim_original_);
        and_sums_.resize(clusters_.size());
        return static_cast<int>(art_core::fit_one_lazy(
            state,
            [&](std::size_t c) {
                // The fuzzy AND sum is shared by choice and match.
                and_sums_[c] = l1_and(sample, clusters_[c].weight);
                return and_sums_[c] / (alpha_ + clusters_[c].sum);
            },
            [&](std::size_t c) { return and_sums_[c] / static_cast<double>(dim_original_); },
            [&](double value) { return art_core::passes_match(value, rho_, mode); },
            [&](std::size_t c) {
                auto found = cluster_map_.find(static_cast<int>(c));
                return found == cluster_map_.end() || found->second == c_b;
            },
            [&](double value) { return art_core::track_match(value, rho_, epsilon_, mode); },
            [&](std::size_t c) {
                update_weight(sample, clusters_[c]);
                cluster_map_[static_cast<int>(c)] = c_b;
            },
            [&]() {
                const auto index = clusters_.size();
                Cluster cluster;
                cluster.weight.assign(sample, sample + 2 * dim_original_);
                cluster.sum = weight_sum(cluster.weight);
                clusters_.push_back(std::move(cluster));
                cluster_map_[static_cast<int>(index)] = c_b;
                return index;
            },
            scratch_));
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha, double beta,
                const std::string& MT, double epsilon, py::object weights = py::none(),
                py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, art_bind::weights<double>(weights),
                            art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target);
}

auto native_predict(py::object X, double rho, double alpha, double beta, const std::string& MT,
                    double epsilon, py::object weights = py::none(),
                    py::object cluster_labels = py::none(), int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    NativeFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, art_bind::weights<double>(weights),
                            art_bind::cluster_labels(cluster_labels));
    return model.predict(data, n_threads);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppFuzzyARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0);
}
