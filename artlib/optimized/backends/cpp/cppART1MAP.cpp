// cppART1MAP.cpp
// ----------------------------------------------------------
//  C++ accelerated ART1MAP (pybind11)
//  - supports incremental training via external weights/map
//  - ART1 expects binary data; input dtype is int16.
//    We interpret x != 0 as 1, else 0.
//  - weights stored as length 2*dim: [w_bu (double), w_td (0/1)]
//  - match tracking modes identical to your Fuzzy ARTMAP scaffold
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
class NativeART1MAP {
public:
    struct Cluster {
        std::vector<double> weight; // size 2*dim: [bu..., td...]
    };

    NativeART1MAP(double rho, double L, const std::string& MT, double epsilon,
                  py::object weights = py::none(), py::object cluster_labels = py::none())
        : base_rho_(rho), L_(L), match_tracking_(MT), epsilon_(epsilon), dim_(0), rho_(rho) {
        if (base_rho_ < 0.0 || base_rho_ > 1.0)
            throw std::invalid_argument("rho must be in [0,1]");
        if (L_ < 1.0)
            throw std::invalid_argument("L must be >= 1");

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
                throw std::invalid_argument("cluster_labels must be 1-D");

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

                const int len = static_cast<int>(w_b.shape[0]);
                if (len % 2 != 0)
                    throw std::invalid_argument("weight length must be even (2*dim)");

                clusters_[k].weight.resize(static_cast<std::size_t>(len));
                std::memcpy(clusters_[k].weight.data(), w_b.ptr, sizeof(double) * len);

                if (dim_ == 0)
                    dim_ = len / 2;
                if (len != 2 * dim_)
                    throw std::invalid_argument("inconsistent weight dimensions");

                cluster_map_[static_cast<int>(k)] = cl_ptr[k];
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>, cluster_labels)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>>
    fit(py::array_t<std::int16_t> X, py::array_t<int> y) {
        auto xb = X.request();
        auto yb = y.request();

        if (xb.ndim != 2 || yb.ndim != 1)
            throw std::invalid_argument("X must be 2-D and y must be 1-D");

        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);
        if (static_cast<int>(yb.shape[0]) != n_samples)
            throw std::invalid_argument("X/y size mismatch");

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("X feature dimension mismatch with model");

        const std::int16_t* Xptr = static_cast<const std::int16_t*>(xb.ptr);
        const int* yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const std::int16_t* row = Xptr + i * dim_;
            labels_a[i] = step_fit(row, yptr[i]);
        }

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
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<std::int16_t> X) {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        if (xb.ndim != 2)
            throw std::invalid_argument("X must be 2-D");

        const int n_samples = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0)
            dim_ = n_features;
        if (n_features != dim_)
            throw std::invalid_argument("X feature dimension mismatch with model");

        const std::int16_t* Xptr = static_cast<const std::int16_t*>(xb.ptr);

        std::vector<int> y_a(n_samples), y_b(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const std::int16_t* row = Xptr + i * dim_;

            auto state = art_core::state_view(clusters_, cluster_map_, dim_);
            const int best_id = static_cast<int>(art_core::predict_one(
                state, [&](std::size_t c) { return category_choice(row, clusters_[c].weight); },
                std::greater<double>{}));
            y_a[i] = best_id;
            y_b[i] = cluster_map_.at(best_id);
        }

        py::array_t<int> ya_out(y_a.size()), yb_out(y_b.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        std::memcpy(yb_out.mutable_data(), y_b.data(), sizeof(int) * y_b.size());
        return {ya_out, yb_out};
    }

private:
    /* ───── hyper-parameters ───── */
    double base_rho_;
    double L_;
    std::string match_tracking_;
    double epsilon_;

    /* ───── state ───── */
    int dim_;
    double rho_;
    std::vector<Cluster> clusters_;
    std::unordered_map<int, int> cluster_map_;

    /* ─────────────────────────────────────────────────── */
    void reset_rho() { rho_ = base_rho_; }

    static inline int bit(const std::int16_t v) {
        // ART1 is binary; treat any nonzero int16 as 1.
        return (v != 0) ? 1 : 0;
    }

    // ART1 choice: T = dot(I, w_bu)
    double category_choice(const std::int16_t* sample, const std::vector<double>& w) const {
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) {
            float wf = static_cast<float>(w[j]); // emulate float32 rounding
            s += static_cast<double>(bit(sample[j])) * static_cast<double>(wf);
        }
        return s;
    }

    // ART1 match: M = |I & w_td| / dim
    double match(const std::int16_t* sample, const std::vector<double>& w) const {
        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int i_bit = bit(sample[j]);
            const int td_bit = (w[dim_ + j] != 0.0) ? 1 : 0;
            if (i_bit & td_bit)
                ++count;
        }
        return static_cast<double>(count) / static_cast<double>(dim_);
    }

    // update rule:
    // w_td_new = I & w_td
    // count = nnz(w_td_new)
    // w_bu_new = (L / (L - 1 + count)) * w_td_new
    std::vector<double> update_weight(const std::int16_t* sample,
                                      const std::vector<double>& w) const {
        std::vector<double> out(2 * dim_, 0.0);

        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int td_new = (bit(sample[j]) & ((w[dim_ + j] != 0.0) ? 1 : 0));
            out[dim_ + j] = static_cast<double>(td_new);
            if (td_new)
                ++count;
        }

        const double denom = (L_ - 1.0 + static_cast<double>(count));
        const double sf = (denom > 0.0) ? (L_ / denom) : 0.0;

        for (int j = 0; j < dim_; ++j) {
            out[j] = sf * out[dim_ + j];
        }

        return out;
    }

    // new cluster:
    // w_td = I
    // w_bu = (L / (L - 1 + dim)) * I
    std::vector<double> new_weight(const std::int16_t* sample) const {
        std::vector<double> w(2 * dim_, 0.0);
        const double sf = L_ / (L_ - 1.0 + static_cast<double>(dim_));

        for (int j = 0; j < dim_; ++j) {
            const double b = static_cast<double>(bit(sample[j]));
            w[dim_ + j] = b; // top-down
            w[j] = sf * b;   // bottom-up
        }

        return w;
    }

    int step_fit(const std::int16_t* sample, int c_b) {
        reset_rho();
        const auto mode = art_core::parse_match_tracking(match_tracking_);

        auto state = art_core::state_view(clusters_, cluster_map_, dim_);
        return static_cast<int>(art_core::fit_one(
            state, [&](std::size_t c) { return category_choice(sample, clusters_[c].weight); },
            [&](std::size_t c) { return match(sample, clusters_[c].weight); },
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
};

// ────────────────────────────────────────────────────────────
// convenience free-functions (fit / predict)
// ────────────────────────────────────────────────────────────
auto native_fit(py::object X, py::object y, double rho, double L, const std::string& MT,
                double epsilon, py::object weights = py::none(),
                py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<std::int16_t>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeART1MAP model(rho, L, MT, epsilon, art_bind::weights<double>(weights),
                        art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target);
}

auto native_predict(py::object X, double rho, double L, const std::string& MT, double epsilon,
                    py::object weights = py::none(), py::object cluster_labels = py::none()) {
    auto data = art_bind::matrix<std::int16_t>(X);
    NativeART1MAP model(rho, L, MT, epsilon, art_bind::weights<double>(weights),
                        art_bind::cluster_labels(cluster_labels));
    return model.predict(data);
}

// ────────────────────────────────────────────────────────────
// pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppART1MAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("L"),
          py::arg("MT"), py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("L"), py::arg("MT"),
          py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());
}
