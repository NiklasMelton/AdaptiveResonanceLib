// cppGaussianARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Gaussian  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in ℝd  (no complement coding)
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
class NativeGaussianARTMAP {
public:
    /* ------------------------------------------------------------------
       Weight layout per cluster
         ┌── mean          (d)  ─┐
         │   σ (std‑dev)   (d)  │
       w = inv_σ²          (d)  │   ← store 1/σ² for speed
           √det(Σ)         (1)  │
           n (sample cnt)  (1)  ┘            size = 3d + 2
       ------------------------------------------------------------------ */
    struct Cluster {
        std::vector<double> weight;
    };

    NativeGaussianARTMAP(double rho, double alpha, py::array_t<double> sigma_init,
                         const std::string& MT, double epsilon, py::object weights = py::none(),
                         py::object cluster_labels = py::none())
        : base_rho_(rho), alpha_(alpha), epsilon_(epsilon), match_tracking_(MT), dim_(0),
          rho_(rho) {
        // copy sigma_init to std::vector
        auto s_b = sigma_init.request();
        if (s_b.ndim != 1)
            throw std::invalid_argument("'sigma_init' must be 1‑D");
        sigma_init_.resize(s_b.shape[0]);
        std::memcpy(sigma_init_.data(), s_b.ptr, sizeof(double) * s_b.shape[0]);

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
                if (w_b.shape[0] < 5 || (w_b.shape[0] - 2) % 3 != 0)
                    throw std::invalid_argument("invalid Gaussian weight length");

                clusters_[k].weight.resize(w_b.shape[0]);
                std::memcpy(clusters_[k].weight.data(), w_b.ptr, sizeof(double) * w_b.shape[0]);
                cluster_map_[static_cast<int>(k)] = cl_ptr[k];

                if (dim_ == 0) {
                    if ((w_b.shape[0] - 2) % 3 != 0)
                        throw std::invalid_argument("invalid weight length");
                    dim_ = static_cast<int>((w_b.shape[0] - 2) / 3);
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT  → (labels_a, weights[], cluster_labels)
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

        // total sample count for p(c_j)
        double total_n = 0.0;
        for (const auto& c : clusters_)
            total_n += c.weight.back();

        for (int i = 0; i < n; ++i) {
            const double* row = Xptr + i * dim;

            auto state = art_core::state_view(clusters_, cluster_map_, dim_);
            const int best_id = static_cast<int>(art_core::predict_one(
                state,
                [&](std::size_t c) { return category_choice(row, clusters_[c].weight, total_n); },
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
    double base_rho_, alpha_, epsilon_;
    std::string match_tracking_;

    /* ───── state ───── */
    int dim_;
    double rho_;
    std::vector<double> sigma_init_;
    std::vector<Cluster> clusters_;
    std::unordered_map<int, int> cluster_map_;

    /* ─────────────────────────────────────────────────── */
    void reset_rho() { rho_ = base_rho_; }

    // helper: exp(‑½ (x‑μ)ᵀ Σ⁻¹ (x‑μ))
    double gaussian_exp(const double* x, const std::vector<double>& w) const {
        const double* mean = w.data();
        const double* inv_sig = w.data() + 2 * dim_;

        double q = 0.0;
        for (int j = 0; j < dim_; ++j) {
            double d = x[j] - mean[j];
            q += d * d * inv_sig[j];
        }
        return std::exp(-0.5 * q);
    }

    /* ─────────────────────────────────────────────────── */
    double category_choice(const double* sample, const std::vector<double>& w,
                           double total_n) const {
        double exp_term = gaussian_exp(sample, w);
        double sqrt_det = w[3 * dim_];
        double n_c = w[3 * dim_ + 1];
        double p_i_cj = exp_term / (alpha_ + sqrt_det);
        double p_cj = n_c / std::max(total_n, 1e-12);
        return p_i_cj * p_cj;
    }

    double match(const double* sample, const std::vector<double>& w) const {
        return gaussian_exp(sample, w); // vigilance compares to exp term
    }

    std::vector<double> update_weight(const std::vector<double>& i,
                                      const std::vector<double>& w) const {
        /* unpack */
        const double* mean = w.data();
        const double* sigma = w.data() + dim_;
        double n = w[3 * dim_ + 1];

        double n_new = n + 1.0;
        std::vector<double> mean_new(dim_), sigma_new(dim_);

        for (int j = 0; j < dim_; ++j) {
            mean_new[j] = (1.0 - 1.0 / n_new) * mean[j] + (1.0 / n_new) * i[j];
            double sigma2_old = sigma[j] * sigma[j];
            double sigma2_new =
                (1.0 - 1.0 / n_new) * sigma2_old + (1.0 / n_new) * std::pow(mean_new[j] - i[j], 2);
            sigma_new[j] = std::sqrt(sigma2_new);
        }

        /* recompute inv_sig, sqrt(det Σ) */
        std::vector<double> inv_sig_new(dim_);
        double det = 1.0;
        for (int j = 0; j < dim_; ++j) {
            double s2 = sigma_new[j] * sigma_new[j];
            inv_sig_new[j] = 1.0 / s2;
            det *= s2;
        }
        double sqrt_det_new = std::sqrt(det);

        /* pack */
        std::vector<double> out;
        out.reserve(3 * dim_ + 2);
        out.insert(out.end(), mean_new.begin(), mean_new.end());
        out.insert(out.end(), sigma_new.begin(), sigma_new.end());
        out.insert(out.end(), inv_sig_new.begin(), inv_sig_new.end());
        out.push_back(sqrt_det_new);
        out.push_back(n_new);
        return out;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const std::vector<double>& sample, int c_b) {
        reset_rho();
        const auto mode = art_core::parse_match_tracking(match_tracking_);
        double total_n = 0.0;
        for (const auto& cluster : clusters_)
            total_n += cluster.weight.back();
        auto state = art_core::state_view(clusters_, cluster_map_, dim_);
        return static_cast<int>(art_core::fit_one(
            state,
            [&](std::size_t c) {
                return category_choice(sample.data(), clusters_[c].weight, total_n);
            },
            [&](std::size_t c) { return match(sample.data(), clusters_[c].weight); },
            [](const auto& scores, const auto&) { return art_core::descending_order(scores); },
            [&](double value) { return art_core::passes_match(value, rho_, mode); },
            [&](std::size_t c) {
                auto found = cluster_map_.find(static_cast<int>(c));
                return found == cluster_map_.end() || found->second == c_b;
            },
            [&](double value) { return art_core::track_match(value, rho_, epsilon_, mode); },
            [&](std::size_t c) {
                clusters_[c].weight = update_weight(sample, clusters_[c].weight);
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
        if (sigma_init_.size() != static_cast<std::size_t>(dim_))
            throw std::invalid_argument("sigma_init dimension mismatch");

        std::vector<double> inv_sig(dim_);
        double det = 1.0;
        for (int j = 0; j < dim_; ++j) {
            double s2 = sigma_init_[j] * sigma_init_[j];
            inv_sig[j] = 1.0 / s2;
            det *= s2;
        }
        double sqrt_det = std::sqrt(det);

        std::vector<double> w;
        w.reserve(3 * dim_ + 2);
        w.insert(w.end(), i.begin(), i.end());                     // mean
        w.insert(w.end(), sigma_init_.begin(), sigma_init_.end()); // σ
        w.insert(w.end(), inv_sig.begin(), inv_sig.end());         // inv σ²
        w.push_back(sqrt_det);
        w.push_back(1.0); // n
        return w;
    }

    /* ── match‑tracking helpers (unchanged) ── */
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha,
                py::array_t<double> sigma_init, const std::string& MT, double epsilon,
                py::object weights = py::none(), py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeGaussianARTMAP model(rho, alpha, art_bind::vector<double>(sigma_init, "sigma_init"), MT,
                               epsilon, art_bind::weights<double>(weights),
                               art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target);
}

auto native_predict(py::object X, double rho, double alpha, py::array_t<double> sigma_init,
                    const std::string& MT, double epsilon, py::object weights = py::none(),
                    py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<double>(X);
    NativeGaussianARTMAP model(rho, alpha, art_bind::vector<double>(sigma_init, "sigma_init"), MT,
                               epsilon, art_bind::weights<double>(weights),
                               art_bind::cluster_labels(cluster_labels));
    return model.predict(data);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppGaussianARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("alpha"),
          py::arg("sigma_init"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("sigma_init"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());
}
