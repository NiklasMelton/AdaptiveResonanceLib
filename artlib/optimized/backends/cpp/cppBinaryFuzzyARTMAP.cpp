#include "art_core.hpp"
#include "batch_fit.hpp"
#include "batch_predict.hpp"
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
#include <vector>

// Use the shared exact fraction ordering core.
#include "fraction_sort_core.hpp"

namespace py = pybind11;

class NativeBinaryFuzzyARTMAP {
public:
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

        bits_.clear();
        w_count_cache_.clear();
        w_count_cache_.reserve(static_cast<size_t>(n_clusters));
        labels_.clear();
        labels_.reserve(static_cast<size_t>(n_clusters));

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
            if (inferred_dim == 0) {
                inferred_dim = dim_here;
                words_ = art_predict::bit_words(weight_size);
            }
            if (dim_here != inferred_dim) {
                throw std::invalid_argument("All weight vectors must have the same length.");
            }

            // Pack as bits (preserving the nonzero-is-one input convention) and
            // cache |w|.
            const int* w_ptr = static_cast<const int*>(w_info.ptr);
            bits_.resize(bits_.size() + words_);
            std::uint64_t* w = bits_.data() + bits_.size() - words_;
            art_predict::pack_bits(w_ptr, weight_size, w);
            w_count_cache_.push_back(ones_count(w));

            labels_.push_back(c_ptr[i]);
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
    fit(py::array_t<int> X, py::array_t<int> y, int n_threads = 0) {
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
        words_ = art_predict::bit_words(num_features);

        const int* x_ptr = static_cast<const int*>(x_buf.ptr);
        const int* y_ptr = static_cast<const int*>(y_buf.ptr);

        std::vector<int> labels_out_vec;
        labels_out_vec.resize(static_cast<size_t>(num_samples));

        if (num_samples > 0u) {
            mode_ = art_core::parse_match_tracking(match_tracking_);
            pool_.set_limit(art_fit::resolve_threads(n_threads));
            py::gil_scoped_release release;
            std::vector<std::uint64_t> sample(words_);
            for (uint32_t i = 0u; i < num_samples; ++i) {
                const int* row_ptr =
                    x_ptr + static_cast<size_t>(i) * static_cast<size_t>(num_features);
                art_predict::pack_bits(row_ptr, num_features, sample.data());

                const int c_b = y_ptr[i];
                const uint32_t chosen = step_fit(sample.data(), c_b);
                labels_out_vec[static_cast<size_t>(i)] = static_cast<int>(chosen);
            }
            pool_.stop();
        }

        auto labels_py = art_bind::output(labels_out_vec);

        // Export weights as int arrays for compatibility
        std::vector<py::array_t<int>> weight_arrays;
        weight_arrays.reserve(labels_.size());
        for (std::size_t c = 0; c < labels_.size(); ++c) {
            py::array_t<int> array(num_features);
            int* out = array.mutable_data();
            const std::uint64_t* w = bits_.data() + c * words_;
            for (uint32_t j = 0u; j < num_features; ++j)
                out[j] = static_cast<int>((w[j / 64] >> (j % 64)) & 1u);
            weight_arrays.push_back(std::move(array));
        }

        // Export cluster labels (cluster_id -> label)
        auto cluster_labels_py = art_bind::output(labels_);

        return std::make_tuple(labels_py, weight_arrays, cluster_labels_py);
    }

    // ============================================
    // PREDICT
    // ============================================
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<int> X, int n_threads = 0) {
        if (labels_.empty()) {
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
        const std::size_t count = labels_.size();
        const std::size_t width = static_cast<std::size_t>(num_features);
        const std::size_t words = words_;

        // Intersection counts by popcount over 64-feature words. Counts are
        // integers, so this matches a feature-by-feature count exactly.
        std::vector<uint32_t> denominators(count);
        for (std::size_t c = 0; c < count; ++c)
            denominators[c] = std::max<uint32_t>(1u, w_count_cache_[c]);

        std::vector<int> pred_a_vec(num_samples), pred_b_vec(num_samples);
        {
            py::gil_scoped_release release;
            const std::size_t n = num_samples;
            // Each category also costs a gcd and an exact fraction comparison.
            const std::size_t threads = art_predict::useful_threads(
                art_predict::resolve_threads(n_threads), n * count * (words + 16) * 8);
            const std::size_t block = art_predict::block_size(n, threads);
            art_predict::parallel_tasks((n + block - 1) / block, threads, [&](std::size_t task) {
                std::vector<std::uint64_t> sample_bits(words);
                const std::size_t end = std::min(n, (task + 1) * block);
                for (std::size_t i = task * block; i < end; ++i) {
                    art_predict::pack_bits(x_ptr + i * width, width, sample_bits.data());
                    auto choice = [&](std::size_t c) {
                        const uint32_t hits =
                            art_predict::and_count(sample_bits.data(), &bits_[c * words], words);
                        fracsort::Item<uint32_t> item{hits, denominators[c], 0, 1, c};
                        fracsort::reduce_item_inplace(item);
                        return item;
                    };
                    // Same scan as art_core::predict_one: replace only on a
                    // strictly better fraction, so ties keep the lowest index.
                    std::size_t best = 0;
                    auto best_item = choice(0);
                    for (std::size_t c = 1; c < count; ++c) {
                        const auto item = choice(c);
                        if (fracsort::frac_greater_item(item, best_item)) {
                            best = c;
                            best_item = item;
                        }
                    }
                    pred_a_vec[i] = static_cast<int>(best);
                    pred_b_vec[i] = labels_[best];
                }
            });
        }

        auto pred_a_py = art_bind::output(pred_a_vec);
        auto pred_b_py = art_bind::output(pred_b_vec);

        return std::make_tuple(pred_a_py, pred_b_py);
    }

private:
    // Inputs/hyperparams
    double base_rho_; // float input (converted to int thresholds when dim known)
    std::string match_tracking_;
    uint32_t epsilon_; // integer epsilon
    art_core::MatchTracking mode_ = art_core::MatchTracking::Plus;

    // Derived/internals
    uint32_t dim_original_; // original dimension (half of input length)
    uint32_t rho_w1_;       // ceil(base_rho * dim_original_)
    uint32_t rho_int_;      // current vigilance threshold (integer)

    // Binary weights, words_ 64-bit words per category (64 features per word,
    // unused high bits zero), one category after another.
    std::size_t words_ = 0;
    std::vector<std::uint64_t> bits_;
    std::vector<uint32_t> w_count_cache_; // cached |w| per cluster
    std::vector<int> labels_;             // cluster_id -> label
    std::vector<uint32_t> intersection_cache_;
    std::vector<fracsort::Item<uint32_t>> candidate_items_;
    std::vector<std::size_t> candidate_order_;
    art_core::SearchScratch<fracsort::Item<uint32_t>, uint32_t> search_scratch_;
    art_fit::Pool pool_;

    // ---- helpers ----

    void reset_rho() { rho_int_ = rho_w1_; }

    uint32_t ones_count(const std::uint64_t* bits) const {
        uint32_t c = 0u;
        for (std::size_t k = 0; k < words_; ++k)
            c += art_predict::popcount64(bits[k]);
        return c;
    }

    bool validate_hypothesis(uint32_t cluster_id, int c_b) const {
        return labels_[cluster_id] == c_b;
    }

    uint32_t step_fit(const std::uint64_t* sample, int c_b) {
        reset_rho();
        const auto mode = mode_;
        const std::size_t count = labels_.size();
        const std::size_t words = words_;
        intersection_cache_.resize(count);
        // A 64-bit word of AND and popcount costs about as much as 4 lane updates.
        art_fit::parallel_range(
            count, count * words * 4, pool_, [&](std::size_t first, std::size_t last) {
                for (std::size_t c = first; c < last; ++c)
                    intersection_cache_[c] =
                        art_predict::and_count(sample, &bits_[c * words], words);
            });
        // w_count_cache_ has one entry per category, which is all the search
        // needs from the weights.
        auto state = art_core::state_view(w_count_cache_, labels_, dim_original_);
        return static_cast<uint32_t>(art_core::fit_one_cached(
            state,
            [&](std::size_t c) {
                return fracsort::Item<uint32_t>{intersection_cache_[c],
                                                std::max<uint32_t>(1u, w_count_cache_[c]), 0, 1, c};
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
                // Weight update: new_w = i & w (binary vectors)
                std::uint64_t* w = bits_.data() + c * words;
                for (std::size_t k = 0; k < words; ++k)
                    w[k] &= sample[k];
                w_count_cache_[c] = intersection_cache_[c];
                labels_[c] = c_b;
            },
            [&]() {
                const auto index = labels_.size();
                bits_.insert(bits_.end(), sample, sample + words);
                w_count_cache_.push_back(ones_count(sample));
                labels_.push_back(c_b);
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
           py::object weights = py::none(), py::object cluster_labels = py::none(),
           int n_threads = 0) {
    auto data = art_bind::binary_matrix<int>(X);
    auto target = art_bind::labels<int>(y, data.shape(0));
    NativeBinaryFuzzyARTMAP model(rho, MT, epsilon, art_bind::binary_weights<int>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.fit(data, target, n_threads);
}

// =======================================================================
// Free function for predict
// =======================================================================
std::tuple<py::array_t<int>, py::array_t<int>>
native_predict(py::object X, double rho, std::string MT, uint32_t epsilon,
               py::object weights = py::none(), py::object cluster_labels = py::none(),
               int n_threads = 0) {
    auto data = art_bind::binary_matrix<int>(X);
    NativeBinaryFuzzyARTMAP model(rho, MT, epsilon, art_bind::binary_weights<int>(weights),
                                  art_bind::cluster_labels(cluster_labels));
    return model.predict(data, n_threads);
}

// =======================================================================
// PYBIND
// =======================================================================
PYBIND11_MODULE(cppBinaryFuzzyARTMAP, m) {
    m.def("fit", &native_fit, py::arg("X"), py::arg("y"), py::arg("rho"), py::arg("MT"),
          py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0,
          R"doc(
Fit cppBinaryFuzzyARTMAP in a single function call.
Optionally re-initialize from existing weights/cluster_labels for partial fits.
Either provide BOTH 'weights' (a list of 1D arrays) and 'cluster_labels' (1D array)
or leave both as None. n_threads caps the threads that score each sample; 0 (the
default) uses ARTLIB_NUM_THREADS when set, else every hardware thread.
)doc");

    m.def("predict", &native_predict, py::arg("X"), py::arg("rho"), py::arg("MT"),
          py::arg("epsilon"), py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none(), py::arg("n_threads") = 0,
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
n_threads : int, optional
    Worker threads. 0 (the default) uses ARTLIB_NUM_THREADS when set, else
    every hardware thread.

Returns
-------
pred_a : np.ndarray
    1D array of predicted cluster indices.
pred_b : np.ndarray
    1D array of final mapped labels (the "side-B" labels).
)doc");
}
