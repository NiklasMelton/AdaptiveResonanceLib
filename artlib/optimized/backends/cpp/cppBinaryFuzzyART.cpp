#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Reuse existing fraction ordering core (do NOT re-implement)
#include "fraction_sort_core.hpp"

namespace py = pybind11;

class cppBinaryFuzzyART {
public:
    struct Cluster {
        std::vector<uint32_t> weight; // binary 0/1
    };

    cppBinaryFuzzyART(
        double rho,
        py::object weights = py::none(),
        : rho_(rho),
    {
        dim_original_ = 0;
        rho_int_ = 0;

        const bool have_weights = !weights.is_none();

        if (!have_weights) return;

        py::list w_list = weights.cast<py::list>();
        py::array_t<int> c_array = cluster_labels.cast<py::array_t<int>>();
        py::buffer_info c_info = c_array.request();

        if (c_info.ndim != 1) {
            throw std::runtime_error("cluster_labels must be a 1D array.");
        }

        const py::ssize_t n_clusters = w_list.size();
        if (c_info.shape[0] != n_clusters) {
            throw std::runtime_error(
                "Inconsistent sizes: weights has " + std::to_string(n_clusters) +
                " clusters, but cluster_labels has length " +
                std::to_string(c_info.shape[0]) + ".");
        }

        const int* c_ptr = static_cast<const int*>(c_info.ptr);

        clusters_.clear();
        clusters_.resize(static_cast<size_t>(n_clusters));
        w_count_cache_.clear();
        w_count_cache_.resize(static_cast<size_t>(n_clusters), 0u);

        uint32_t inferred_dim = 0;

        for (py::ssize_t i = 0; i < n_clusters; ++i) {
            py::array_t<int> w_array = w_list[i].cast<py::array_t<int>>();
            py::buffer_info w_info = w_array.request();

            if (w_info.ndim != 1) {
                throw std::runtime_error("Each weight array must be 1D.");
            }

            const py::ssize_t weight_size_ssize = w_info.shape[0];
            if (weight_size_ssize <= 0) {
                throw std::runtime_error("Weight arrays must be non-empty.");
            }

            if (weight_size_ssize % 2 != 0) {
                throw std::runtime_error("Weight length must be even (2*dim_original).");
            }

            const uint32_t weight_size = static_cast<uint32_t>(weight_size_ssize);
            const uint32_t dim_here = weight_size / 2u;
            if (inferred_dim == 0) inferred_dim = dim_here;
            if (dim_here != inferred_dim) {
                throw std::runtime_error("All weight vectors must have the same length.");
            }

            clusters_[static_cast<size_t>(i)].weight.resize(static_cast<size_t>(weight_size));
            const int* w_ptr = static_cast<const int*>(w_info.ptr);

            // Convert to uint32_t 0/1 and compute cached |w|
            uint32_t w_count = 0u;
            for (uint32_t j = 0u; j < weight_size; ++j) {
                const int v = w_ptr[j];
                // You may tighten this to throw unless (v==0||v==1) if desired.
                const uint32_t u = (v != 0) ? 1u : 0u;
                clusters_[static_cast<size_t>(i)].weight[static_cast<size_t>(j)] = u;
                w_count += u;
            }
            w_count_cache_[static_cast<size_t>(i)] = w_count;

        }

        // Now we can set dim_original_ and rho thresholds
        dim_original_ = inferred_dim;
        rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
    }

    // ============================================
    // FIT
    // ============================================
    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>>
    fit(py::array_t<int> X) {
        py::buffer_info x_buf = X.request();

        if (x_buf.ndim != 2) throw std::runtime_error("X must be a 2D array.");
        if (y_buf.ndim != 1) throw std::runtime_error("y must be a 1D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        if (num_samples_ssize < 0 || num_features_ssize <= 0) {
            throw std::runtime_error("Invalid X shape.");
        }

        const uint32_t num_samples = static_cast<uint32_t>(num_samples_ssize);
        const uint32_t num_features = static_cast<uint32_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::runtime_error("Number of features must be even (2*dim_original).");
        }

        // Infer dim_original_ if needed
        if (dim_original_ == 0u) {
            dim_original_ = num_features / 2u;
            rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
        }

        if (num_features != 2u * dim_original_) {
            throw std::runtime_error("Number of features do not match existing weights.");
        }

        const int* x_ptr = static_cast<const int*>(x_buf.ptr);

        std::vector<int> labels_out_vec;
        labels_out_vec.resize(static_cast<size_t>(num_samples));

        for (uint32_t i = 0u; i < num_samples; ++i) {
            const int* row_ptr = x_ptr + static_cast<size_t>(i) * static_cast<size_t>(num_features);

            std::vector<uint32_t> sample;
            sample.resize(static_cast<size_t>(num_features));
            for (uint32_t j = 0u; j < num_features; ++j) {
                sample[static_cast<size_t>(j)] = (row_ptr[j] != 0) ? 1u : 0u;
            }

            const uint32_t chosen = step_fit(sample, c_b);
            labels_out_vec[static_cast<size_t>(i)] = static_cast<int>(chosen);
        }

        py::array_t<int> labels_py(labels_out_vec.size());
        std::memcpy(labels_py.mutable_data(),
                    labels_out_vec.data(),
                    labels_out_vec.size() * sizeof(int));

        // Export weights as int arrays for compatibility
        std::vector<py::array_t<int>> weight_arrays;
        weight_arrays.reserve(clusters_.size());
        for (const auto& cluster : clusters_) {
            py::array_t<int> arr(cluster.weight.size());
            int* out_ptr = arr.mutable_data();
            for (size_t j = 0; j < cluster.weight.size(); ++j) {
                out_ptr[j] = static_cast<int>(cluster.weight[j]);
            }
            weight_arrays.push_back(std::move(arr));
        }


        return std::make_tuple(labels_py, weight_arrays);
    }

    // ============================================
    // PREDICT
    // ============================================
    py::array_t<int>
    predict(py::array_t<int> X) {
        if (clusters_.empty()) {
            throw std::runtime_error(
                "Cannot call predict() because the model has no clusters. "
                "Call fit() or provide existing weights.");
        }

        py::buffer_info x_buf = X.request();
        if (x_buf.ndim != 2) throw std::runtime_error("X must be a 2D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        if (num_samples_ssize < 0 || num_features_ssize <= 0) {
            throw std::runtime_error("Invalid X shape.");
        }

        const uint32_t num_samples = static_cast<uint32_t>(num_samples_ssize);
        const uint32_t num_features = static_cast<uint32_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::runtime_error("Number of features must be even (2*dim_original).");
        }

        if (dim_original_ == 0u) {
            dim_original_ = num_features / 2u;
            rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
        }

        if (num_features != 2u * dim_original_) {
            throw std::runtime_error("Number of features do not match existing weights.");
        }

        const int* x_ptr = static_cast<const int*>(x_buf.ptr);

        std::vector<int> pred_a_vec;
        pred_a_vec.resize(static_cast<size_t>(num_samples));

        // Pre-allocate items buffer (size = n_clusters)
        std::vector<fracsort::Item<uint32_t>> items;
        items.resize(clusters_.size());

        for (uint32_t i = 0u; i < num_samples; ++i) {
            const int* row_ptr = x_ptr + static_cast<size_t>(i) * static_cast<size_t>(num_features);

            std::vector<uint32_t> sample;
            sample.resize(static_cast<size_t>(num_features));
            for (uint32_t j = 0u; j < num_features; ++j) {
                sample[static_cast<size_t>(j)] = (row_ptr[j] != 0) ? 1u : 0u;
            }

            // Build fraction items: num = |i & w|, den = max(1, |w|)
            for (size_t c = 0; c < clusters_.size(); ++c) {
                const uint32_t iw_count = intersection_count(sample, clusters_[c].weight);
                const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
                items[c] = fracsort::Item<uint32_t>{iw_count, den, 0, 1, c};
            }

            const size_t best_cluster = fracsort::fracargmax_items<uint32_t>(
                items.data(), items.size());

            pred_a_vec[static_cast<size_t>(i)] = static_cast<int>(best_cluster);

        }

        py::array_t<int> pred_a_py(pred_a_vec.size());
        std::memcpy(pred_a_py.mutable_data(),
                    pred_a_vec.data(),
                    pred_a_vec.size() * sizeof(int));


        return pred_a_py;
    }

private:
    // Inputs/hyperparams
    double rho_;           // float input (converted to int thresholds when dim known)

    // Derived/internals
    uint32_t dim_original_;     // original dimension (half of input length)
    uint32_t rho_int_;          // vigilance threshold (integer)

    std::vector<Cluster> clusters_;
    std::vector<uint32_t> w_count_cache_;              // cached |w| per cluster

    // ---- helpers ----


    static uint32_t intersection_count(const std::vector<uint32_t>& i,
                                       const std::vector<uint32_t>& w) {
        // i and w are binary 0/1; intersection count is sum(i[j] & w[j])
        uint32_t c = 0u;
        const size_t n = i.size();
        for (size_t j = 0; j < n; ++j) c += (i[j] & w[j]);
        return c;
    }

    static uint32_t ones_count(const std::vector<uint32_t>& v) {
        uint32_t c = 0u;
        for (uint32_t x : v) c += x;
        return c;
    }


    // Weight update: new_w = i & w (binary vectors)
    static void update_inplace(std::vector<uint32_t>& w,
                               const std::vector<uint32_t>& i) {
        const size_t n = w.size();
        for (size_t j = 0; j < n; ++j) {
            w[j] = (w[j] & i[j]);
        }
    }

    uint32_t step_fit(const std::vector<uint32_t>& sample) {

        // If no clusters => create first
        if (clusters_.empty()) {
            clusters_.push_back(Cluster{sample});
            w_count_cache_.push_back(ones_count(sample));
            return 0u;
        }

        const size_t n_clusters = clusters_.size();

        // Build candidate fraction items, with optional pre-MT filtering
        std::vector<fracsort::Item<uint32_t>> items;
        items.reserve(n_clusters);

        // Also store iw_count per cluster for later reuse (vigilance + update cache)
        std::vector<uint32_t> iw_counts;
        iw_counts.resize(n_clusters, 0u);

        for (size_t c = 0; c < n_clusters; ++c) {
            const uint32_t iw = intersection_count(sample, clusters_[c].weight);
            iw_counts[c] = iw;

            // Pre-match-tracking filtering (skip only when not MT-)
            bool mt_status = match_operator(MT_, iw, rho_int_);

            if (!mt_status) continue;

            const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
            items.push_back(fracsort::Item<uint32_t>{iw, den, 0, 1, c});
        }

        // Sort candidates by exact fraction ordering using existing core
        if (!items.empty()) {
            fracsort::argsort_items_inplace<uint32_t>(items.data(), items.size());
        }

        // Scan candidates in sorted order
        for (const auto& it : items) {
            const uint32_t idx = static_cast<uint32_t>(it.idx);
            const uint32_t iw_count = iw_counts[it.idx];

            // Check vigilance (integer)
            if (iw_count < rho_int_) {
                continue;
            }

            // Update cluster weight in-place
            update_inplace(clusters_[static_cast<size_t>(idx)].weight, sample);

            // New weight is exactly i&w, so its ones-count equals iw_count
            w_count_cache_[static_cast<size_t>(idx)] = iw_count;

            // Update label mapping
            cluster_map_[idx] = c_b;
            return idx;

        }

        // No existing cluster chosen => create new cluster
        const uint32_t new_cluster_id = static_cast<uint32_t>(clusters_.size());
        clusters_.push_back(Cluster{sample});
        w_count_cache_.push_back(ones_count(sample));
        return new_cluster_id;
    }
};

// =======================================================================
// Free function for fit
// =======================================================================
std::tuple<py::array_t<int>, std::vector<py::array_t<int>>>
FitBinaryFuzzyART(py::array_t<int> X,
                     double rho,
                     py::object weights = py::none())
{
    cppBinaryFuzzyART model(rho, weights);
    return model.fit(X);
}

// =======================================================================
// Free function for predict
// =======================================================================
std::tuple<py::array_t<int>>
PredictBinaryFuzzyART(py::array_t<int> X,
                         double rho,
                         py::object weights = py::none())
{
    cppBinaryFuzzyART model(rho, weights);
    return model.predict(X);
}

// =======================================================================
// PYBIND
// =======================================================================
PYBIND11_MODULE(cppBinaryFuzzyART, m) {
    py::class_<cppBinaryFuzzyART>(m, "cppBinaryFuzzyART")
        .def(py::init<double, py::object>(),
             py::arg("rho"),
             py::arg("weights") = py::none(),
             R"doc(
Construct a cppBinaryFuzzyART model.

Parameters
----------
rho : float
    Vigilance parameter (0.0 <= rho <= 1.0). Converted to integer rho_int once dim is known.
weights : list of 1D np.ndarray, optional
)doc")
        .def("fit", &cppBinaryFuzzyART::fit,
             py::arg("X"),
             R"doc(
Fit the model given data X and labels y.
Returns:
    (labels_out, weight_arrays, cluster_labels_out)
)doc")
        .def("predict", &cppBinaryFuzzyART::predict,
             py::arg("X"),
             R"doc(
Predict labels for X.
Returns:
    (pred_a, pred_b)
)doc")
        .def("__repr__", [](const cppBinaryFuzzyART&) {
            return "<cppBinaryFuzzyART model>";
        });

    m.def("FitBinaryFuzzyART",
          &FitBinaryFuzzyART,
          py::arg("X"),
          py::arg("rho"),
          py::arg("weights") = py::none(),
          R"doc(
Fit cppBinaryFuzzyART in a single function call.
Optionally re-initialize from existing weights/cluster_labels for partial fits.
Either provide BOTH 'weights' (a list of 1D arrays) and 'cluster_labels' (1D array)
or leave both as None.
)doc");

    m.def("PredictBinaryFuzzyART",
          &PredictBinaryFuzzyART,
          py::arg("X"),
          py::arg("rho"),
          py::arg("weights") = py::none(),
          R"doc(
Predict labels using a temporary cppBinaryFuzzyART model.

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
