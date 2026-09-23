#include "art_core.hpp"
#include "native_arrays.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

// Use the shared exact fraction ordering core.
#include "fraction_sort_core.hpp"

namespace py = pybind11;

class NativeBinaryFuzzyART {
public:
    struct Cluster {
        std::vector<uint32_t> weight; // binary 0/1
    };

    NativeBinaryFuzzyART(double rho, py::object weights = py::none()) : rho_(rho) {
        dim_original_ = 0;
        rho_int_ = 0;

        const bool have_weights = !weights.is_none();

        if (!have_weights)
            return;

        py::list w_list = weights.cast<py::list>();

        const py::ssize_t n_clusters = w_list.size();

        clusters_.clear();
        clusters_.resize(static_cast<size_t>(n_clusters));
        w_count_cache_.clear();
        w_count_cache_.resize(static_cast<size_t>(n_clusters), 0u);

        size_t inferred_dim = 0;

        for (py::ssize_t i = 0; i < n_clusters; ++i) {
            auto w_u8 = art_bind::binary_array<std::uint8_t>(w_list[i], "weights[i]");
            py::buffer_info w_info = w_u8.request();

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

            const size_t weight_size = static_cast<size_t>(weight_size_ssize);
            const size_t dim_here = weight_size / 2u;
            if (inferred_dim == 0)
                inferred_dim = dim_here;
            if (dim_here != inferred_dim) {
                throw std::invalid_argument("All weight vectors must have the same length.");
            }

            clusters_[static_cast<size_t>(i)].weight.resize(static_cast<size_t>(weight_size));
            const std::uint8_t* w_ptr = static_cast<const std::uint8_t*>(w_info.ptr);

            // Convert to uint32_t 0/1 and compute cached |w|
            uint32_t w_count = 0u;
            for (size_t j = 0; j < weight_size; ++j) {
                const std::uint8_t v = w_ptr[j];
                // Preserve the existing nonzero-is-one input convention.
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
    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>> fit(py::object X_obj) {

        auto X = art_bind::binary_array<std::uint8_t>(X_obj, "X");
        py::buffer_info x_buf = X.request();

        if (x_buf.ndim != 2)
            throw std::invalid_argument("X must be a 2D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        if (num_samples_ssize < 0 || num_features_ssize <= 0) {
            throw std::invalid_argument("Invalid X shape.");
        }

        const size_t num_samples = static_cast<size_t>(num_samples_ssize);
        const size_t num_features = static_cast<size_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::invalid_argument("Number of features must be even (2*dim_original).");
        }

        // Infer dim_original_ if needed
        if (dim_original_ == 0) {
            dim_original_ = num_features / 2;
            rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
        }

        if (num_features != 2 * dim_original_) {
            throw std::invalid_argument("Number of features do not match existing weights.");
        }

        const std::uint8_t* x_ptr = static_cast<const std::uint8_t*>(x_buf.ptr);

        std::vector<int> labels_out_vec;
        labels_out_vec.resize(num_samples);

        std::vector<uint32_t> sample(num_features);

        for (size_t i = 0u; i < num_samples; ++i) {
            const std::uint8_t* row_ptr = x_ptr + i * num_features;

            for (size_t j = 0u; j < num_features; ++j) {
                sample[j] = (row_ptr[j] != 0) ? 1u : 0u;
            }
            const uint32_t chosen = step_fit(sample);
            labels_out_vec[i] = static_cast<int>(chosen);
        }

        auto labels_py = art_bind::output(labels_out_vec);

        // Export weights as int arrays for compatibility
        auto weight_arrays = art_bind::pack_weights<int>(
            clusters_, [](const Cluster& cluster) -> const auto& { return cluster.weight; });

        return std::make_tuple(labels_py, weight_arrays);
    }

    // ============================================
    // PREDICT
    // ============================================
    py::array_t<int> predict(py::object X_obj) {

        if (clusters_.empty()) {
            throw std::runtime_error("Cannot call predict() because the model has no clusters. "
                                     "Call fit() or provide existing weights.");
        }
        auto X = art_bind::binary_array<std::uint8_t>(X_obj, "X");

        py::buffer_info x_buf = X.request();
        if (x_buf.ndim != 2)
            throw std::invalid_argument("X must be a 2D array.");

        const py::ssize_t num_samples_ssize = x_buf.shape[0];
        const py::ssize_t num_features_ssize = x_buf.shape[1];

        const size_t num_samples = static_cast<size_t>(num_samples_ssize);
        const size_t num_features = static_cast<size_t>(num_features_ssize);

        if (num_features % 2u != 0u) {
            throw std::invalid_argument("Number of features must be even (2*dim_original).");
        }

        if (dim_original_ == 0) {
            dim_original_ = num_features / 2;
            rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
        }

        if (num_features != 2 * dim_original_) {
            throw std::invalid_argument("Number of features do not match existing weights.");
        }

        const std::uint8_t* x_ptr = static_cast<const std::uint8_t*>(x_buf.ptr);

        std::vector<int> pred_a_vec;
        pred_a_vec.resize(num_samples);

        std::vector<uint32_t> sample(num_features);

        for (size_t i = 0u; i < num_samples; ++i) {
            const std::uint8_t* row_ptr = x_ptr + i * num_features;

            for (size_t j = 0u; j < num_features; ++j) {
                sample[j] = (row_ptr[j] != 0) ? 1u : 0u;
            }
            art_core::NoLabels unused_labels;
            auto state = art_core::state_view(clusters_, unused_labels, dim_original_);
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

            pred_a_vec[i] = static_cast<int>(best_cluster);
        }

        py::array_t<int> pred_a_py(pred_a_vec.size());
        std::memcpy(pred_a_py.mutable_data(), pred_a_vec.data(), pred_a_vec.size() * sizeof(int));

        return pred_a_py;
    }

private:
    // Inputs/hyperparams
    double rho_; // float input (converted to int thresholds when dim known)

    // Derived/internals
    size_t dim_original_; // original dimension (half of input length)
    uint32_t rho_int_;    // vigilance threshold (integer)

    std::vector<Cluster> clusters_;
    std::vector<uint32_t> w_count_cache_; // cached |w| per cluster
    std::vector<uint32_t> intersection_cache_;
    std::vector<fracsort::Item<uint32_t>> candidate_items_;
    std::vector<std::size_t> candidate_order_;
    art_core::SearchScratch<fracsort::Item<uint32_t>, uint32_t> search_scratch_;

    // ---- helpers ----

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

    // Weight update: new_w = i & w (binary vectors)
    static void update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i) {
        const size_t n = w.size();
        for (size_t j = 0; j < n; ++j) {
            w[j] = (w[j] & i[j]);
        }
    }

    uint32_t step_fit(const std::vector<uint32_t>& sample) {
        art_core::NoLabels unused_labels;
        intersection_cache_.resize(clusters_.size());
        auto state = art_core::state_view(clusters_, unused_labels, dim_original_);
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
                candidate_order_.clear();
                for (std::size_t c = 0; c < scores.size(); ++c) {
                    if (matches[c] >= rho_int_)
                        candidate_items_.push_back(scores[c]);
                }
                if (!candidate_items_.empty()) {
                    candidate_order_.push_back(fracsort::fracargmax_items<uint32_t>(
                        candidate_items_.data(), candidate_items_.size()));
                }
                return candidate_order_;
            },
            [&](uint32_t value) { return value >= rho_int_; }, [](std::size_t) { return true; },
            [](uint32_t) { return true; },
            [&](std::size_t c) {
                update_inplace(clusters_[c].weight, sample);
                w_count_cache_[c] = intersection_cache_[c];
            },
            [&]() {
                const auto index = clusters_.size();
                clusters_.push_back(Cluster{sample});
                w_count_cache_.push_back(ones_count(sample));
                return index;
            },
            search_scratch_));
    }
};

// =======================================================================
// Free function for fit
// =======================================================================
std::tuple<py::array_t<int>, std::vector<py::array_t<int>>>
native_fit(py::object X_obj, double rho, py::object weights = py::none()) {
    NativeBinaryFuzzyART model(rho, art_bind::binary_weights<std::uint8_t>(weights));
    return model.fit(X_obj);
}

// =======================================================================
// Free function for predict
// =======================================================================
py::array_t<int> native_predict(py::object X_obj, double rho, py::object weights = py::none()) {
    NativeBinaryFuzzyART model(rho, art_bind::binary_weights<std::uint8_t>(weights));
    return model.predict(X_obj);
}

// =======================================================================
// PYBIND
// =======================================================================
PYBIND11_MODULE(cppBinaryFuzzyART, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("rho"), py::arg("weights") = py::none(),
          R"doc(
Fit cppBinaryFuzzyART in a single function call.
Optionally re-initialize from existing weights for partial fits.
)doc");

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("weights") = py::none(),
          R"doc(
Predict labels using a temporary NativeBinaryFuzzyART model.

Parameters
----------
X : np.ndarray
    Data set (2D array).
rho : float
    Vigilance parameter (0.0 <= rho <= 1.0).
weights : list of 1D np.ndarray, optional

Returns
-------
pred_a : np.ndarray
    1D array of predicted cluster indices.
)doc");
}
