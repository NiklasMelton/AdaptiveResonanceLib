// cppFuzzyARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Fuzzy  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in [0,1] (already normalised & complement‑coded)
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
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ────────────────────────────────────────────────────────────
class NativeFuzzyARTMAP {
public:
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

            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto w_b = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::invalid_argument("each weight must be 1-D");
                if (w_b.shape[0] < 2 || w_b.shape[0] % 2 != 0)
                    throw std::invalid_argument("weight length must be positive and even");
                const std::size_t width = static_cast<std::size_t>(w_b.shape[0]);
                if (dim_original_ == 0) {
                    dim_original_ = static_cast<int>(width / 2);
                    weights_.reset(width, 1);
                }
                if (width != weights_.dim)
                    throw std::invalid_argument("inconsistent weight dimensions");
                const double* w = static_cast<const double*>(w_b.ptr);
                const std::size_t c = weights_.append();
                double* out = weights_.values(c, 0);
                for (std::size_t j = 0; j < width; ++j)
                    out[j * weights_.feature_stride()] = w[j];
                sums_.push_back(std::accumulate(w, w + width, 0.0));
                labels_.push_back(cl_ptr[k]);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>, cluster_labels)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>>
    fit(py::array_t<double> X, py::array_t<int> y, int n_threads = 0) {
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
        if (weights_.count == 0)
            weights_.reset(static_cast<std::size_t>(n_features), 1);

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
            py::array_t<double> array(width);
            double* out = array.mutable_data();
            const double* w = weights_.values(c, 0);
            for (std::size_t j = 0; j < width; ++j)
                out[j] = w[j * stride];
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
        if (dim_original_ == 0)
            dim_original_ = dim / 2;
        if (dim % 2 != 0 || dim != 2 * dim_original_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const std::size_t count = weights_.count;
        const std::size_t width = static_cast<std::size_t>(dim);

        std::vector<double> denominators(count);
        for (std::size_t c = 0; c < count; ++c)
            denominators[c] = alpha_ + sums_[c];

        std::vector<int> y_a(n), y_b(n);
        {
            py::gil_scoped_release release;
            // Fuzzy AND sum; there is no multiply, so the whole range is one step.
            auto step = [](double acc, double x, const double* w, std::size_t lane) {
                return acc + std::min(x, w[lane]);
            };
            art_predict::accumulate(
                Xptr, n, width, weights_, art_predict::resolve_threads(n_threads), width, step,
                step, [&](std::size_t i, const double* and_sums) {
                    const std::size_t best = art_predict::argmax_first(
                        count, [&](std::size_t c) { return and_sums[c] / denominators[c]; });
                    y_a[i] = static_cast<int>(best);
                    y_b[i] = labels_[best];
                });
        }
        return {art_bind::output(y_a), art_bind::output(y_b)};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, beta_, epsilon_;
    std::string match_tracking_;
    art_core::MatchTracking mode_ = art_core::MatchTracking::Plus;
    bool prefilter_ = false;

    /* ───── state ───── */
    int dim_original_;
    double rho_;
    art_predict::Panels weights_; // 2d components per category (complement‑coded)
    std::vector<double> sums_;    // sum of each weight, refreshed on every change
    std::vector<int> labels_;     // category -> class label

    /* ───── per-sample scratch, reused between samples ───── */
    std::vector<double> scores_, matches_;
    std::vector<std::size_t> order_;
    art_fit::Pool pool_;

    /* ─────────────────────────────────────────────────── */
    void update_weight(const double* i, std::size_t c) {
        double* w = weights_.values(c, 0);
        const std::size_t stride = weights_.feature_stride();
        double sum = 0.0;
        for (std::size_t j = 0; j < weights_.dim; ++j) {
            double& value = w[j * stride];
            value = beta_ * std::min(i[j], value) + (1.0 - beta_) * value;
            sum += value;
        }
        sums_[c] = sum;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const double* sample, int c_b) {
        rho_ = base_rho_;
        const std::size_t count = weights_.count;
        scores_.resize(count);
        matches_.resize(count);
        // The fuzzy AND sum is shared by choice and match.
        auto step = [](double acc, double x, const double* w, std::size_t lane) {
            return acc + std::min(x, w[lane]);
        };
        art_fit::score_sample(
            sample, weights_, weights_.dim, step, step,
            [&](std::size_t p, const double* and_sums) {
                const std::size_t first = p * art_fit::kLanes;
                const std::size_t last = std::min(count, first + art_fit::kLanes);
                for (std::size_t c = first; c < last; ++c) {
                    scores_[c] = and_sums[c - first] / (alpha_ + sums_[c]);
                    matches_[c] = and_sums[c - first] / static_cast<double>(dim_original_);
                }
            },
            pool_);

        return static_cast<int>(art_core::fit_one_scored(
            count, scores_.data(), matches_.data(), prefilter_,
            [&](double value) { return art_core::passes_match(value, rho_, mode_); },
            [&](std::size_t c) { return labels_[c] == c_b; },
            [&](double value) { return art_core::track_match(value, rho_, epsilon_, mode_); },
            [&](std::size_t c) {
                update_weight(sample, c);
                labels_[c] = c_b;
            },
            [&]() {
                const std::size_t c = weights_.append();
                double* w = weights_.values(c, 0);
                for (std::size_t j = 0; j < weights_.dim; ++j)
                    w[j * weights_.feature_stride()] = sample[j];
                sums_.push_back(std::accumulate(sample, sample + weights_.dim, 0.0));
                labels_.push_back(c_b);
                return c;
            },
            order_));
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha, double beta,
                const std::string& MT, double epsilon, py::object weights = py::none(),
                py::object cluster_labels = py::none(), int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, art_bind::weights<double>(weights),
                            art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target, n_threads);
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
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0);

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0);

    // The thread counts fit and predict use when n_threads is 0, shared by every
    // native ARTMAP model. Private; used by the tests.
    m.def("_default_threads", []() {
        return py::make_tuple(art_fit::resolve_threads(0), art_predict::resolve_threads(0));
    });
}
