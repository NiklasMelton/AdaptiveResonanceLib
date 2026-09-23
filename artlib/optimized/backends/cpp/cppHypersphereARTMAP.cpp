// cppHypersphereARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Hypersphere  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in ℝ^d  (NO complement coding here)
// ----------------------------------------------------------
#include "art_core.hpp"
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
class NativeHypersphereARTMAP {
public:
    // *** weight: centroid (d doubles)  +  radius (1 double) ***
    struct Cluster {
        std::vector<double> weight; // length d+1
    };

    NativeHypersphereARTMAP(double rho, double alpha, double beta, double r_hat,
                            const std::string& MT, double epsilon, py::object weights = py::none(),
                            py::object cluster_labels = py::none())
        : base_rho_(rho), alpha_(alpha), beta_(beta), r_hat_(r_hat), epsilon_(epsilon),
          match_tracking_(MT), dim_(0), rho_(rho) {
        if (r_hat_ <= 0.0)
            throw std::invalid_argument("'r_hat' must be > 0");

        const bool have_W = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_W != have_cl)
            throw std::invalid_argument("Provide BOTH 'weights' and 'cluster_labels' or neither.");

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
                    throw std::invalid_argument("each weight must be 1‑D");
                if (w_b.shape[0] < 2)
                    throw std::invalid_argument("weight must be at least 2 elements (d ≥1)");

                clusters_[k].weight.resize(static_cast<std::size_t>(w_b.shape[0]));
                std::memcpy(clusters_[k].weight.data(), w_b.ptr, sizeof(double) * w_b.shape[0]);
                cluster_map_[static_cast<int>(k)] = cl_ptr[k];

                if (dim_ == 0)
                    dim_ = static_cast<int>(w_b.shape[0] - 1);
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
            throw std::invalid_argument("X must be 2‑D, y must be 1‑D");
        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const int* yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> sample(Xptr + i * n_features, Xptr + (i + 1) * n_features);
            labels_a[i] = step_fit(sample, yptr[i]);
        }

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
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<double> X) {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        const int n = static_cast<int>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_ == 0)
            dim_ = dim;
        if (dim != dim_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y_a(n), y_b(n);

        for (int i = 0; i < n; ++i) {
            const double* row = Xptr + i * dim;

            auto state = art_core::state_view(clusters_, cluster_map_, dim_);
            const int best_id = static_cast<int>(art_core::predict_one(
                state, [&](std::size_t c) { return category_choice(row, clusters_[c].weight); },
                std::greater<double>{})); // fallback
            y_a[i] = best_id;
            y_b[i] = cluster_map_.at(best_id);
        }

        py::array_t<int> ya_out(y_a.size()), yb_out(y_b.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        std::memcpy(yb_out.mutable_data(), y_b.data(), sizeof(int) * y_b.size());
        return {ya_out, yb_out};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, beta_, r_hat_, epsilon_;
    std::string match_tracking_;

    /* ───── state ───── */
    int dim_;
    double rho_;
    std::vector<Cluster> clusters_;
    std::unordered_map<int, int> cluster_map_;

    /* ─────────────────────────────────────────────────── */
    void reset_rho() { rho_ = base_rho_; }

    // *** Euclidean distance helper (√(∑(x-c)^2)) ***
    double euclidean(const double* x, const std::vector<double>& w_centroid) const {
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) {
            double d = x[j] - w_centroid[j];
            s += d * d;
        }
        return std::sqrt(s);
    }

    /* ─────────────────────────────────────────────────── */
    double category_choice(const double* sample, const std::vector<double>& w) const {
        const double radius = w.back();
        const double i_rad = euclidean(sample, w);
        const double max_r = std::max(radius, i_rad);
        return (r_hat_ - max_r) / (r_hat_ - radius + alpha_);
    }

    double match(const double* sample, const std::vector<double>& w) const {
        const double radius = w.back();
        const double i_rad = euclidean(sample, w);
        const double max_r = std::max(radius, i_rad);
        return 1.0 - (max_r / r_hat_);
    }

    std::vector<double> update_weight(const std::vector<double>& i, const std::vector<double>& w,
                                      const std::vector<double>& cache) const {
        // cache = {i_radius, max_radius}
        double i_radius = cache[0];
        double max_r = cache[1];
        double radius = w.back();

        double radius_new = radius + (beta_ / 2.0) * (max_r - radius);

        std::vector<double> out(dim_ + 1);
        for (int j = 0; j < dim_; ++j) {
            double centroid = w[j];
            out[j] = centroid + (beta_ / 2.0) * (i[j] - centroid) *
                                    (1.0 - (std::min(radius, i_radius) / (i_radius + alpha_)));
        }
        out.back() = radius_new;
        return out;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const std::vector<double>& sample, int c_b) {
        reset_rho();
        const auto mode = art_core::parse_match_tracking(match_tracking_);
        std::vector<std::array<double, 2>> caches(clusters_.size());
        auto state = art_core::state_view(clusters_, cluster_map_, dim_);
        return static_cast<int>(art_core::fit_one(
            state,
            [&](std::size_t c) {
                return ([&] {
                    const double radius = clusters_[c].weight.back();
                    const double input_radius = euclidean(sample.data(), clusters_[c].weight);
                    const double max_radius = std::max(radius, input_radius);
                    caches[c] = {input_radius, max_radius};
                    return (r_hat_ - max_radius) / (r_hat_ - radius + alpha_);
                }());
            },
            [&](std::size_t c) { return 1.0 - caches[c][1] / r_hat_; },
            [](const auto& scores, const auto&) { return art_core::descending_order(scores); },
            [&](double value) { return art_core::passes_match(value, rho_, mode); },
            [&](std::size_t c) {
                auto found = cluster_map_.find(static_cast<int>(c));
                return found == cluster_map_.end() || found->second == c_b;
            },
            [&](double value) { return art_core::track_match(value, rho_, epsilon_, mode); },
            [&](std::size_t c) {
                clusters_[c].weight =
                    update_weight(sample, clusters_[c].weight, {caches[c][0], caches[c][1]});
                cluster_map_[static_cast<int>(c)] = c_b;
            },
            [&]() {
                const auto index = clusters_.size();
                clusters_.push_back({new_weight(sample)});
                cluster_map_[static_cast<int>(index)] = c_b;
                return index;
            }));
    }

    std::vector<double> new_weight(const std::vector<double>& i) const {
        std::vector<double> w(i);
        w.push_back(0.0); // initial radius
        return w;
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha, double beta, double r_hat,
                const std::string& MT, double epsilon, py::object weights = py::none(),
                py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeHypersphereARTMAP model(rho, alpha, beta, r_hat, MT, epsilon,
                                  art_bind::weights<double>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target);
}

auto native_predict(py::object X, double rho, double alpha, double beta, double r_hat,
                    const std::string& MT, double epsilon, py::object weights = py::none(),
                    py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<double>(X);
    NativeHypersphereARTMAP model(rho, alpha, beta, r_hat, MT, epsilon,
                                  art_bind::weights<double>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.predict(data);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppHypersphereARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("r_hat"), py::arg("MT"), py::arg("epsilon"),
          py::arg("weights") = py::none(), py::arg("cluster_labels") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("r_hat"), py::arg("MT"), py::arg("epsilon"),
          py::arg("weights") = py::none(), py::arg("cluster_labels") = py::none());
}
