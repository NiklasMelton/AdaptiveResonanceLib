#include "art_core.hpp"
#include "native_arrays.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Use the shared exact fraction ordering core.
#include "fraction_sort_core.hpp"

namespace py = pybind11;

class NativeBinaryFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<uint32_t> weight; // binary 0/1
    };

    NativeBinaryFuzzyARTMAP(double rho, std::string MT, uint32_t epsilon,
                            py::object weights = py::none(), py::object cluster_labels = py::none())
        : base_rho_(rho), match_tracking_(std::move(MT)), epsilon_(epsilon) {
        dim_original_ = 0;
        rho_w1_ = 0;
        rho_int_ = 0;

        const bool have_weights = !weights.is_none();
        const bool have_cluster_labels = !cluster_labels.is_none();

        // Must provide both or neither
        if (have_weights != have_cluster_labels) {
            throw std::invalid_argument(
                "You must provide BOTH 'weights' and 'cluster_labels' OR neither.");
        }

        if (!have_weights)
            return;

        py::list w_list = weights.cast<py::list>();
        py::array_t<int> c_array = cluster_labels.cast<py::array_t<int>>();
        py::buffer_info c_info = c_array.request();

        if (c_info.ndim != 1) {
            throw std::invalid_argument("cluster_labels must be a 1D array.");
        }

        const py::ssize_t n_clusters = w_list.size();
        if (c_info.shape[0] != n_clusters) {
            throw std::invalid_argument("Inconsistent sizes: weights has " +
                                        std::to_string(n_clusters) +
                                        " clusters, but cluster_labels has length " +
                                        std::to_string(c_info.shape[0]) + ".");
        }

        const int* c_ptr = static_cast<const int*>(c_info.ptr);

        clusters_.clear();
        clusters_.resize(static_cast<size_t>(n_clusters));
        w_count_cache_.clear();
        w_count_cache_.resize(static_cast<size_t>(n_clusters), 0u);
        cluster_map_.clear();

        uint32_t inferred_dim = 0;

        for (py::ssize_t i = 0; i < n_clusters; ++i) {
            py::array_t<int> w_array = w_list[i].cast<py::array_t<int>>();
            py::buffer_info w_info = w_array.request();

            if (w_info.ndim != 1) {
                throw std::invalid_argument("Each weight array must be 1D.");
            }

            const py::ssize_t weight_size_ssize = w_info.shape[0];
            if (weight_size_ssize <= 0) {
                throw std::invalid_argument("Weight arrays must be non-empty.");
            }

            if (weight_size_ssize % 2 != 0) {
                throw std::invalid_argument("Weight length must be even (2*dim_original).");
            }

            const uint32_t weight_size = static_cast<uint32_t>(weight_size_ssize);
            const uint32_t dim_here = weight_size / 2u;
            if (inferred_dim == 0)
                inferred_dim = dim_here;
            if (dim_here != inferred_dim) {
                throw std::invalid_argument("All weight vectors must have the same length.");
            }

            clusters_[static_cast<size_t>(i)].weight.resize(static_cast<size_t>(weight_size));
            const int* w_ptr = static_cast<const int*>(w_info.ptr);

            // Convert to uint32_t 0/1 and compute cached |w|
            uint32_t w_count = 0u;
            for (uint32_t j = 0u; j < weight_size; ++j) {
                const int v = w_ptr[j];
                // Preserve the existing nonzero-is-one input convention.
                const uint32_t u = (v != 0) ? 1u : 0u;
                clusters_[static_cast<size_t>(i)].weight[static_cast<size_t>(j)] = u;
                w_count += u;
            }
            w_count_cache_[static_cast<size_t>(i)] = w_count;

            cluster_map_[static_cast<uint32_t>(i)] = c_ptr[i];
        }

        // Now we can set dim_original_ and rho thresholds
        dim_original_ = inferred_dim;
        rho_w1_ = static_cast<uint32_t>(std::ceil(base_rho_ * static_cast<double>(dim_original_)));
        rho_int_ = rho_w1_;
    }

    // ============================================
    // FIT
    // ============================================
    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>>
    fit(py::array_t<int> X, py::array_t<int> y) {
        py::buffer_info x_buf = X.request();
        py::buffer_info y_buf = y.request();

        if (x_buf.ndim != 2)
            throw std::invalid_argument("X must be a 2D array.");
        if (y_buf.ndim != 1)
            throw std::invalid_argument("y must be a 1D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        if (num_samples_ssize < 0 || num_features_ssize <= 0) {
            throw std::invalid_argument("Invalid X shape.");
        }

        const uint32_t num_samples = static_cast<uint32_t>(num_samples_ssize);
        const uint32_t num_features = static_cast<uint32_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::invalid_argument("Number of features must be even (2*dim_original).");
        }

        // Infer dim_original_ if needed
        if (dim_original_ == 0u) {
            dim_original_ = num_features / 2u;
            rho_w1_ =
                static_cast<uint32_t>(std::ceil(base_rho_ * static_cast<double>(dim_original_)));
            rho_int_ = rho_w1_;
        }

        if (num_features != 2u * dim_original_) {
            throw std::invalid_argument("Number of features do not match existing weights.");
        }

        const int* x_ptr = static_cast<const int*>(x_buf.ptr);
        const int* y_ptr = static_cast<const int*>(y_buf.ptr);

        std::vector<int> labels_out_vec;
        labels_out_vec.resize(static_cast<size_t>(num_samples));

        for (uint32_t i = 0u; i < num_samples; ++i) {
            const int* row_ptr = x_ptr + static_cast<size_t>(i) * static_cast<size_t>(num_features);

            std::vector<uint32_t> sample;
            sample.resize(static_cast<size_t>(num_features));
            for (uint32_t j = 0u; j < num_features; ++j) {
                sample[static_cast<size_t>(j)] = (row_ptr[j] != 0) ? 1u : 0u;
            }

            const int c_b = y_ptr[i];
            const uint32_t chosen = step_fit(sample, c_b);
            labels_out_vec[static_cast<size_t>(i)] = static_cast<int>(chosen);
        }

        auto labels_py = art_bind::output(labels_out_vec);

        // Export weights as int arrays for compatibility
        auto weight_arrays = art_bind::pack_weights<int>(
            clusters_, [](const Cluster& cluster) -> const auto& { return cluster.weight; });

        // Export cluster labels (cluster_id -> label)
        std::vector<int> cluster_labels_vec;
        cluster_labels_vec.resize(clusters_.size(), 0);
        for (const auto& kv : cluster_map_) {
            const uint32_t cluster_id = kv.first;
            const int label = kv.second;
            if (cluster_id < cluster_labels_vec.size()) {
                cluster_labels_vec[static_cast<size_t>(cluster_id)] = label;
            }
        }

        auto cluster_labels_py = art_bind::output(cluster_labels_vec);

        return std::make_tuple(labels_py, weight_arrays, cluster_labels_py);
    }

    // ============================================
    // PREDICT
    // ============================================
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<int> X) {
        if (clusters_.empty()) {
            throw std::runtime_error("Cannot call predict() because the model has no clusters. "
                                     "Call fit() or provide existing weights.");
        }

        py::buffer_info x_buf = X.request();
        if (x_buf.ndim != 2)
            throw std::invalid_argument("X must be a 2D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        if (num_samples_ssize < 0 || num_features_ssize <= 0) {
            throw std::invalid_argument("Invalid X shape.");
        }

        const uint32_t num_samples = static_cast<uint32_t>(num_samples_ssize);
        const uint32_t num_features = static_cast<uint32_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::invalid_argument("Number of features must be even (2*dim_original).");
        }

        if (dim_original_ == 0u) {
            dim_original_ = num_features / 2u;
            rho_w1_ =
                static_cast<uint32_t>(std::ceil(base_rho_ * static_cast<double>(dim_original_)));
            rho_int_ = rho_w1_;
        }

        if (num_features != 2u * dim_original_) {
            throw std::invalid_argument("Number of features do not match existing weights.");
        }

        const int* x_ptr = static_cast<const int*>(x_buf.ptr);

        std::vector<int> pred_a_vec;
        std::vector<int> pred_b_vec;
        pred_a_vec.resize(static_cast<size_t>(num_samples));
        pred_b_vec.resize(static_cast<size_t>(num_samples));

        for (uint32_t i = 0u; i < num_samples; ++i) {
            const int* row_ptr = x_ptr + static_cast<size_t>(i) * static_cast<size_t>(num_features);

            std::vector<uint32_t> sample;
            sample.resize(static_cast<size_t>(num_features));
            for (uint32_t j = 0u; j < num_features; ++j) {
                sample[static_cast<size_t>(j)] = (row_ptr[j] != 0) ? 1u : 0u;
            }

            auto state = art_core::state_view(clusters_, cluster_map_, dim_original_);
            const size_t best_cluster = art_core::predict_one(
                state,
                [&](std::size_t c) {
                    const uint32_t count = intersection_count(sample, clusters_[c].weight);
                    fracsort::Item<uint32_t> item{count, std::max<uint32_t>(1u, w_count_cache_[c]),
                                                  0, 1, c};
                    fracsort::reduce_item_inplace(item);
                    return item;
                },
                [](const auto& a, const auto& b) { return fracsort::frac_greater_item(a, b); });

            pred_a_vec[static_cast<size_t>(i)] = static_cast<int>(best_cluster);

            pred_b_vec[static_cast<size_t>(i)] =
                cluster_map_.at(static_cast<uint32_t>(best_cluster));
        }

        py::array_t<int> pred_a_py(pred_a_vec.size());
        std::memcpy(pred_a_py.mutable_data(), pred_a_vec.data(), pred_a_vec.size() * sizeof(int));

        py::array_t<int> pred_b_py(pred_b_vec.size());
        std::memcpy(pred_b_py.mutable_data(), pred_b_vec.data(), pred_b_vec.size() * sizeof(int));

        return std::make_tuple(pred_a_py, pred_b_py);
    }

private:
    // Inputs/hyperparams
    double base_rho_; // float input (converted to int thresholds when dim known)
    std::string match_tracking_;
    uint32_t epsilon_; // integer epsilon

    // Derived/internals
    uint32_t dim_original_; // original dimension (half of input length)
    uint32_t rho_w1_;       // ceil(base_rho * dim_original_)
    uint32_t rho_int_;      // current vigilance threshold (integer)

    std::vector<Cluster> clusters_;
    std::vector<uint32_t> w_count_cache_;           // cached |w| per cluster
    std::unordered_map<uint32_t, int> cluster_map_; // cluster_id -> label
    std::vector<uint32_t> intersection_cache_;
    std::vector<fracsort::Item<uint32_t>> candidate_items_;
    std::vector<std::size_t> candidate_order_;
    art_core::SearchScratch<fracsort::Item<uint32_t>, uint32_t> search_scratch_;

    // ---- helpers ----

    void reset_rho() { rho_int_ = rho_w1_; }

    static uint32_t intersection_count(const std::vector<uint32_t>& i,
                                       const std::vector<uint32_t>& w) {
        // i and w are binary 0/1; intersection count is sum(i[j] & w[j])
        uint32_t c = 0u;
        const size_t n = i.size();
        for (size_t j = 0; j < n; ++j)
            c += (i[j] & w[j]);
        return c;
    }

    static uint32_t ones_count(const std::vector<uint32_t>& v) {
        uint32_t c = 0u;
        for (uint32_t x : v)
            c += x;
        return c;
    }

    bool validate_hypothesis(uint32_t cluster_id, int c_b) const {
        auto it = cluster_map_.find(cluster_id);
        return (it == cluster_map_.end()) || (it->second == c_b);
    }

    // Weight update: new_w = i & w (binary vectors)
    static void update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i) {
        const size_t n = w.size();
        for (size_t j = 0; j < n; ++j) {
            w[j] = (w[j] & i[j]);
        }
    }

    uint32_t step_fit(const std::vector<uint32_t>& sample, int c_b) {
        reset_rho();
        const auto mode = art_core::parse_match_tracking(match_tracking_);
        intersection_cache_.resize(clusters_.size());
        auto state = art_core::state_view(clusters_, cluster_map_, dim_original_);
        return static_cast<uint32_t>(art_core::fit_one_cached(
            state,
            [&](std::size_t c) {
                const auto count = intersection_count(sample, clusters_[c].weight);
                intersection_cache_[c] = count;
                return fracsort::Item<uint32_t>{count, std::max<uint32_t>(1u, w_count_cache_[c]), 0,
                                                1, c};
            },
            [&](std::size_t c) { return intersection_cache_[c]; },
            [&](const auto& scores, const auto& matches) -> const std::vector<std::size_t>& {
                candidate_items_.clear();
                candidate_items_.reserve(scores.size());
                candidate_order_.clear();
                for (std::size_t c = 0; c < scores.size(); ++c) {
                    if (mode == art_core::MatchTracking::Minus ||
                        art_core::passes_match(matches[c], rho_int_, mode)) {
                        candidate_items_.push_back(scores[c]);
                    }
                }
                if (!candidate_items_.empty()) {
                    fracsort::argsort_items_inplace<uint32_t>(candidate_items_.data(),
                                                              candidate_items_.size());
                }
                candidate_order_.reserve(candidate_items_.size());
                for (const auto& item : candidate_items_)
                    candidate_order_.push_back(item.idx);
                return candidate_order_;
            },
            [&](uint32_t value) { return art_core::passes_match(value, rho_int_, mode); },
            [&](std::size_t c) { return validate_hypothesis(static_cast<uint32_t>(c), c_b); },
            [&](uint32_t value) {
                return art_core::track_match(value, rho_int_, epsilon_, dim_original_, mode);
            },
            [&](std::size_t c) {
                update_inplace(clusters_[c].weight, sample);
                w_count_cache_[c] = intersection_cache_[c];
                cluster_map_[static_cast<uint32_t>(c)] = c_b;
            },
            [&]() {
                const auto index = clusters_.size();
                clusters_.push_back(Cluster{sample});
                w_count_cache_.push_back(ones_count(sample));
                cluster_map_[static_cast<uint32_t>(index)] = c_b;
                return index;
            },
            search_scratch_));
    }
};

// =======================================================================
// Free function for fit
// =======================================================================
std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>>
native_fit(py::object X, py::object y, double rho, std::string MT, uint32_t epsilon,
           py::object weights = py::none(), py::object cluster_labels = py::none()) {
    auto data = art_bind::binary_matrix<int>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeBinaryFuzzyARTMAP model(rho, MT, epsilon, art_bind::binary_weights<int>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target);
}

// =======================================================================
// Free function for predict
// =======================================================================
std::tuple<py::array_t<int>, py::array_t<int>>
native_predict(py::object X, double rho, std::string MT, uint32_t epsilon,
               py::object weights = py::none(), py::object cluster_labels = py::none()) {
    auto data = art_bind::binary_matrix<int>(X);
    NativeBinaryFuzzyARTMAP model(rho, MT, epsilon, art_bind::binary_weights<int>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.predict(data);
}

// =======================================================================
// PYBIND
// =======================================================================
PYBIND11_MODULE(cppBinaryFuzzyARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("MT"),
          py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(),
          R"doc(
Fit cppBinaryFuzzyARTMAP in a single function call.
Optionally re-initialize from existing weights/cluster_labels for partial fits.
Either provide BOTH 'weights' (a list of 1D arrays) and 'cluster_labels' (1D array)
or leave both as None.
)doc");

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("MT"),
          py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(),
          R"doc(
Predict labels using a temporary NativeBinaryFuzzyARTMAP model.

Parameters
----------
X : np.ndarray
    Data set (2D array).
rho : float
    Vigilance parameter (0.0 <= rho <= 1.0).
MT : str
    Match tracking mode.
epsilon : int
    Integer epsilon for match tracking adjustments.
weights : list of 1D np.ndarray, optional
cluster_labels : np.ndarray, optional

Returns
-------
pred_a : np.ndarray
    1D array of predicted cluster indices.
pred_b : np.ndarray
    1D array of final mapped labels (the "side-B" labels).
)doc");
}
