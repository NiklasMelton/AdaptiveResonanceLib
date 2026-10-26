// cppHypersphereARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Hypersphere  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in ℝ^d  (NO complement coding here)
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

namespace {
// Squared Euclidean distance ∑(x-c)^2, matching the old scalar loop bit for bit;
// see art_predict::fused_tail_start for why there are two steps.
struct SquareBody {
    double operator()(double acc, double x, const double* w, std::size_t lane) const {
        const double d = x - w[lane];
        const double square = d * d;
        return acc + square;
    }
};
struct SquareTail {
    double operator()(double acc, double x, const double* w, std::size_t lane) const {
        const double d = x - w[lane];
        return acc + d * d;
    }
};
} // namespace

// ────────────────────────────────────────────────────────────
class NativeHypersphereARTMAP {
public:
    // *** weight: centroid (d doubles)  +  radius (1 double) ***
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

            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto w_b = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::invalid_argument("each weight must be 1‑D");
                if (w_b.shape[0] < 2)
                    throw std::invalid_argument("weight must be at least 2 elements (d ≥1)");

                const std::size_t width = static_cast<std::size_t>(w_b.shape[0] - 1);
                if (dim_ == 0) {
                    dim_ = static_cast<int>(width);
                    centroids_.reset(width, 1);
                }
                if (width != centroids_.dim)
                    throw std::invalid_argument("inconsistent weight dimensions");
                const double* w = static_cast<const double*>(w_b.ptr);
                const std::size_t c = centroids_.append();
                double* out = centroids_.values(c, 0);
                for (std::size_t j = 0; j < width; ++j)
                    out[j * centroids_.feature_stride()] = w[j];
                radii_.push_back(w[width]);
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
            throw std::invalid_argument("X must be 2‑D, y must be 1‑D");
        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("feature dimension mismatch");
        if (centroids_.count == 0)
            centroids_.reset(static_cast<std::size_t>(n_features), 1);

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
        const std::size_t width = centroids_.dim;
        const std::size_t stride = centroids_.feature_stride();
        std::vector<py::array_t<double>> weight_out;
        weight_out.reserve(centroids_.count);
        for (std::size_t c = 0; c < centroids_.count; ++c) {
            py::array_t<double> array(width + 1);
            double* out = array.mutable_data();
            const double* w = centroids_.values(c, 0);
            for (std::size_t j = 0; j < width; ++j)
                out[j] = w[j * stride];
            out[width] = radii_[c];
            weight_out.push_back(std::move(array));
        }

        return {art_bind::output(labels_a), weight_out, art_bind::output(labels_)};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a, y_b)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<double> X,
                                                           int n_threads = 0) {
        if (centroids_.count == 0)
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        const std::size_t n = static_cast<std::size_t>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_ == 0)
            dim_ = dim;
        if (dim != dim_)
            throw std::invalid_argument("feature dimension mismatch");

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const std::size_t count = centroids_.count;
        const std::size_t width = static_cast<std::size_t>(dim);

        std::vector<double> denominators(count);
        for (std::size_t c = 0; c < count; ++c)
            denominators[c] = r_hat_ - radii_[c] + alpha_;

        std::vector<int> y_a(n), y_b(n);
        {
            py::gil_scoped_release release;
            art_predict::accumulate(
                Xptr, n, width, centroids_, art_predict::resolve_threads(n_threads),
                art_predict::fused_tail_start(width), SquareBody{}, SquareTail{},
                [&](std::size_t i, const double* squares) {
                    const std::size_t best = art_predict::argmax_first(count, [&](std::size_t c) {
                        const double i_rad = std::sqrt(squares[c]);
                        const double max_r = std::max(radii_[c], i_rad);
                        return (r_hat_ - max_r) / denominators[c];
                    });
                    y_a[i] = static_cast<int>(best);
                    y_b[i] = labels_[best];
                });
        }
        return {art_bind::output(y_a), art_bind::output(y_b)};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, beta_, r_hat_, epsilon_;
    std::string match_tracking_;
    art_core::MatchTracking mode_ = art_core::MatchTracking::Plus;
    bool prefilter_ = false;

    /* ───── state ───── */
    int dim_;
    double rho_;
    art_predict::Panels centroids_; // d doubles per category
    std::vector<double> radii_;
    std::vector<int> labels_; // category -> class label

    /* ───── per-sample scratch, reused between samples ───── */
    std::vector<double> scores_, matches_;
    std::vector<double> input_radii_, max_radii_;
    std::vector<std::size_t> order_;
    art_fit::Pool pool_;

    /* ─────────────────────────────────────────────────── */
    void update_weight(const double* i, std::size_t c) {
        const double i_radius = input_radii_[c];
        const double max_r = max_radii_[c];
        const double radius = radii_[c];

        const double radius_new = radius + (beta_ / 2.0) * (max_r - radius);

        // With alpha == 0 and the sample on the centroid this ratio is 0/0. Use its
        // limit as alpha -> 0, which is 0.
        const double denom = i_radius + alpha_;
        const double ratio = denom > 0.0 ? std::min(radius, i_radius) / denom : 0.0;

        double* w = centroids_.values(c, 0);
        const std::size_t stride = centroids_.feature_stride();
        for (int j = 0; j < dim_; ++j) {
            const double centroid = w[j * stride];
            w[j * stride] = centroid + (beta_ / 2.0) * (i[j] - centroid) * (1.0 - ratio);
        }
        radii_[c] = radius_new;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const double* sample, int c_b) {
        rho_ = base_rho_;
        const std::size_t count = centroids_.count;
        scores_.resize(count);
        matches_.resize(count);
        input_radii_.resize(count);
        max_radii_.resize(count);
        art_fit::score_sample(
            sample, centroids_, art_predict::fused_tail_start(centroids_.dim), SquareBody{},
            SquareTail{},
            [&](std::size_t p, const double* squares) {
                const std::size_t first = p * art_fit::kLanes;
                const std::size_t last = std::min(count, first + art_fit::kLanes);
                for (std::size_t c = first; c < last; ++c) {
                    const double radius = radii_[c];
                    const double input_radius = std::sqrt(squares[c - first]);
                    const double max_radius = std::max(radius, input_radius);
                    input_radii_[c] = input_radius;
                    max_radii_[c] = max_radius;
                    scores_[c] = (r_hat_ - max_radius) / (r_hat_ - radius + alpha_);
                    matches_[c] = 1.0 - max_radius / r_hat_;
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
                const std::size_t c = centroids_.append();
                double* w = centroids_.values(c, 0);
                for (std::size_t j = 0; j < centroids_.dim; ++j)
                    w[j * centroids_.feature_stride()] = sample[j];
                radii_.push_back(0.0); // initial radius
                labels_.push_back(c_b);
                return c;
            },
            order_));
    }
};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double alpha, double beta, double r_hat,
                const std::string& MT, double epsilon, py::object weights = py::none(),
                py::object cluster_labels = py::none(), int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeHypersphereARTMAP model(rho, alpha, beta, r_hat, MT, epsilon,
                                  art_bind::weights<double>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target, n_threads);
}

auto native_predict(py::object X, double rho, double alpha, double beta, double r_hat,
                    const std::string& MT, double epsilon, py::object weights = py::none(),
                    py::object cluster_labels = py::none(), int n_threads = 0) {
    auto data = art_bind::matrix<double>(X);
    NativeHypersphereARTMAP model(rho, alpha, beta, r_hat, MT, epsilon,
                                  art_bind::weights<double>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.predict(data, n_threads);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppHypersphereARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("r_hat"), py::arg("MT"), py::arg("epsilon"),
          py::arg("weights") = py::none(), py::arg("cluster_labels") = py::none(),
          py::arg("n_threads") = 0);

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("alpha"),
          py::arg("beta"), py::arg("r_hat"), py::arg("MT"), py::arg("epsilon"),
          py::arg("weights") = py::none(), py::arg("cluster_labels") = py::none(),
          py::arg("n_threads") = 0);
}
