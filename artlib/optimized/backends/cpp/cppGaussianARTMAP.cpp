// cppGaussianARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Gaussian  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in ℝd  (no complement coding)
// ----------------------------------------------------------
#include "art_core.hpp"
#include "batch_fit.hpp"
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

            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto w_b = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::invalid_argument("each weight must be 1‑D");
                if (w_b.shape[0] < 5 || (w_b.shape[0] - 2) % 3 != 0)
                    throw std::invalid_argument("invalid Gaussian weight length");

                const std::size_t width = static_cast<std::size_t>((w_b.shape[0] - 2) / 3);
                if (dim_ == 0) {
                    dim_ = static_cast<int>(width);
                    weights_.reset(width, 2);
                }
                if (width != weights_.dim)
                    throw std::invalid_argument("inconsistent weight dimensions");
                const double* w = static_cast<const double*>(w_b.ptr);
                const std::size_t c = weights_.append();
                double* mean = weights_.values(c, 0);
                double* inv_sig = weights_.values(c, 1);
                const std::size_t stride = weights_.feature_stride();
                for (std::size_t j = 0; j < width; ++j) {
                    mean[j * stride] = w[j];
                    inv_sig[j * stride] = w[2 * width + j];
                }
                sigma_.insert(sigma_.end(), w + width, w + 2 * width);
                sqrt_det_.push_back(w[3 * width]);
                n_.push_back(w[3 * width + 1]);
                labels_.push_back(cl_ptr[k]);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT  → (labels_a, weights[], cluster_labels)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>>
    fit(py::array_t<double> X, py::array_t<int> y, int n_threads = 0) {
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
        if (weights_.count == 0)
            weights_.reset(static_cast<std::size_t>(n_features), 2);

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const int* yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        if (n_samples > 0) {
            mode_ = art_core::parse_match_tracking(match_tracking_);
            prefilter_ = art_core::vigilance_only_rises(mode_, epsilon_);
            pool_.set_limit(art_fit::resolve_threads(n_threads));
            py::gil_scoped_release release;
            for (int i = 0; i < n_samples; ++i)
                labels_a[i] = step_fit(Xptr + static_cast<std::size_t>(i) * n_features, yptr[i]);
            pool_.stop();
        }

        /* -------- pack output -------- */
        const std::size_t width = weights_.dim;
        const std::size_t stride = weights_.feature_stride();
        std::vector<py::array_t<double>> weight_out;
        weight_out.reserve(weights_.count);
        for (std::size_t c = 0; c < weights_.count; ++c) {
            py::array_t<double> array(3 * width + 2);
            double* out = array.mutable_data();
            const double* mean = weights_.values(c, 0);
            const double* inv_sig = weights_.values(c, 1);
            const double* sigma = sigma_.data() + c * width;
            for (std::size_t j = 0; j < width; ++j) {
                out[j] = mean[j * stride];
                out[width + j] = sigma[j];
                out[2 * width + j] = inv_sig[j * stride];
            }
            out[3 * width] = sqrt_det_[c];
            out[3 * width + 1] = n_[c];
            weight_out.push_back(std::move(array));
        }

        return {art_bind::output(labels_a), weight_out, art_bind::output(labels_)};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a, y_b)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<double> X,
                                                           int n_threads = 0) {
        if (weights_.count == 0)
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        const std::size_t n = static_cast<std::size_t>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_ == 0)
            dim_ = dim;
        if (dim != dim_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const std::size_t count = weights_.count;
        const std::size_t width = static_cast<std::size_t>(dim);

        // total sample count for p(c_j)
        const double total_n = total_count();

        std::vector<double> denominators(count), priors(count);
        for (std::size_t c = 0; c < count; ++c) {
            denominators[c] = alpha_ + sqrt_det_[c];
            priors[c] = n_[c] / std::max(total_n, 1e-12);
        }

        std::vector<int> y_a(n), y_b(n);
        {
            py::gil_scoped_release release;
            art_predict::accumulate(
                Xptr, n, width, weights_, art_predict::resolve_threads(n_threads),
                art_predict::fused_tail_start(width), TermBody{}, TermTail{},
                [&](std::size_t i, const double* q) {
                    const std::size_t best = art_predict::argmax_first(count, [&](std::size_t c) {
                        const double p_i_cj = std::exp(-0.5 * q[c]) / denominators[c];
                        return p_i_cj * priors[c];
                    });
                    y_a[i] = static_cast<int>(best);
                    y_b[i] = labels_[best];
                });
        }
        return {art_bind::output(y_a), art_bind::output(y_b)};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, epsilon_;
    std::string match_tracking_;
    art_core::MatchTracking mode_ = art_core::MatchTracking::Plus;
    bool prefilter_ = false;

    /* ───── state ───── */
    int dim_;
    double rho_;
    std::vector<double> sigma_init_;
    art_predict::Panels weights_; // streams: mean, then inv_σ² (d doubles each)
    std::vector<double> sigma_;   // σ, d doubles per category, one after another
    std::vector<double> sqrt_det_, n_;
    std::vector<int> labels_; // category -> class label

    /* ───── new-category values from sigma_init, computed on first use ───── */
    bool have_init_ = false;
    std::vector<double> inv_sig_init_;
    double sqrt_det_init_ = 0.0;

    /* ───── per-sample scratch, reused between samples ───── */
    std::vector<double> scores_, exp_terms_;
    std::vector<std::size_t> order_;
    art_fit::Pool pool_;

    // Mahalanobis term (x‑μ)ᵀ Σ⁻¹ (x‑μ) of exp(‑½ (x‑μ)ᵀ Σ⁻¹ (x‑μ)), matching the
    // old scalar loop bit for bit; see art_predict::fused_tail_start for why
    // there are two steps.
    struct TermBody {
        double operator()(double acc, double x, const double* w, std::size_t lane) const {
            const double d = x - w[lane];
            const double term = d * d * w[art_predict::kLanes + lane];
            return acc + term;
        }
    };
    struct TermTail {
        double operator()(double acc, double x, const double* w, std::size_t lane) const {
            const double d = x - w[lane];
            return acc + d * d * w[art_predict::kLanes + lane];
        }
    };

    double total_count() const {
        double total_n = 0.0;
        for (double n : n_)
            total_n += n;
        return total_n;
    }

    /* ─────────────────────────────────────────────────── */
    void update_weight(const double* i, std::size_t c) {
        /* unpack */
        double* mean = weights_.values(c, 0);
        double* inv_sig = weights_.values(c, 1);
        double* sigma = sigma_.data() + c * static_cast<std::size_t>(dim_);
        const std::size_t stride = weights_.feature_stride();
        double n = n_[c];

        double n_new = n + 1.0;

        for (int j = 0; j < dim_; ++j) {
            const double mean_new = (1.0 - 1.0 / n_new) * mean[j * stride] + (1.0 / n_new) * i[j];
            double sigma2_old = sigma[j] * sigma[j];
            double sigma2_new =
                (1.0 - 1.0 / n_new) * sigma2_old + (1.0 / n_new) * std::pow(mean_new - i[j], 2);
            mean[j * stride] = mean_new;
            sigma[j] = std::sqrt(sigma2_new);
        }

        /* recompute inv_sig, sqrt(det Σ) */
        double det = 1.0;
        for (int j = 0; j < dim_; ++j) {
            double s2 = sigma[j] * sigma[j];
            inv_sig[j * stride] = 1.0 / s2;
            det *= s2;
        }
        sqrt_det_[c] = std::sqrt(det);
        n_[c] = n_new;
    }

    std::size_t create(const double* i) {
        if (sigma_init_.size() != static_cast<std::size_t>(dim_))
            throw std::invalid_argument("sigma_init dimension mismatch");
        if (!have_init_) {
            inv_sig_init_.resize(dim_);
            double det = 1.0;
            for (int j = 0; j < dim_; ++j) {
                double s2 = sigma_init_[j] * sigma_init_[j];
                inv_sig_init_[j] = 1.0 / s2;
                det *= s2;
            }
            sqrt_det_init_ = std::sqrt(det);
            have_init_ = true;
        }

        const std::size_t c = weights_.append();
        double* mean = weights_.values(c, 0);
        double* inv_sig = weights_.values(c, 1);
        const std::size_t stride = weights_.feature_stride();
        for (int j = 0; j < dim_; ++j) {
            mean[j * stride] = i[j];
            inv_sig[j * stride] = inv_sig_init_[j];
        }
        sigma_.insert(sigma_.end(), sigma_init_.begin(), sigma_init_.end());
        sqrt_det_.push_back(sqrt_det_init_);
        n_.push_back(1.0);
        return c;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const double* sample, int c_b) {
        rho_ = base_rho_;
        const std::size_t count = weights_.count;
        const double total_n = total_count();
        scores_.resize(count);
        exp_terms_.resize(count);
        art_fit::score_sample(
            sample, weights_, art_predict::fused_tail_start(weights_.dim), TermBody{}, TermTail{},
            [&](std::size_t p, const double* q) {
                const std::size_t first = p * art_fit::kLanes;
                const std::size_t last = std::min(count, first + art_fit::kLanes);
                for (std::size_t c = first; c < last; ++c) {
                    // The exponential term is both part of the choice and the match.
                    exp_terms_[c] = std::exp(-0.5 * q[c - first]);
                    double p_i_cj = exp_terms_[c] / (alpha_ + sqrt_det_[c]);
                    double p_cj = n_[c] / std::max(total_n, 1e-12);
                    scores_[c] = p_i_cj * p_cj;
                }
            },
            pool_);

        return static_cast<int>(art_core::fit_one_scored(
            count, scores_.data(), exp_terms_.data(), prefilter_,
            [&](double value) { return art_core::passes_match(value, rho_, mode_); },
            [&](std::size_t c) { return labels_[c] == c_b; },
            [&](double value) { return art_core::track_match(value, rho_, epsilon_, mode_); },
            [&](std::size_t c) {
                update_weight(sample, c);
                labels_[c] = c_b;
            },
            [&]() {
                const std::size_t c = create(sample);
                labels_.push_back(c_b);
                return c;
            },
            order_));
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha,
                py::array_t<double> sigma_init, const std::string& MT, double epsilon,
                py::object weights = py::none(), py::object cluster_labels = py::none(),
                int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeGaussianARTMAP model(rho, alpha, art_bind::vector<double>(sigma_init, "sigma_init"), MT,
                               epsilon, art_bind::weights<double>(weights),
                               art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target, n_threads);
}

auto native_predict(py::object X, double rho, double alpha, py::array_t<double> sigma_init,
                    const std::string& MT, double epsilon, py::object weights = py::none(),
                    py::object cluster_labels = py::none(), int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    NativeGaussianARTMAP model(rho, alpha, art_bind::vector<double>(sigma_init, "sigma_init"), MT,
                               epsilon, art_bind::weights<double>(weights),
                               art_bind::cluster_labels(cluster_labels));
    return model.predict(data, n_threads);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppGaussianARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("alpha"),
          py::arg("sigma_init"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0);

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("sigma_init"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0);
}
