#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

class cppBinaryFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<int> weight;
    };

    cppBinaryFuzzyARTMAP(double rho,
                      double alpha,
                      std::string MT,
                      double epsilon,
                      py::object weights = py::none(),
                      py::object cluster_labels = py::none())
        : base_rho_(rho), alpha_(alpha), MT_(MT), epsilon_(epsilon)
    {
        // We do NOT know dim_original_ yet, so we simply set to 0 here.
        // We'll infer it in fit() from the data.
        dim_original_ = 0;
        rho_w1_       = 0;
        bool have_weights         = !weights.is_none();
        bool have_cluster_labels  = !cluster_labels.is_none();
        // Must provide both or neither:
        if (have_weights != have_cluster_labels) {
            throw std::invalid_argument(
                "You must provide BOTH 'weights' and 'cluster_labels' OR neither."
            );
        }
        // If the user passed existing data, initialize from them.
        if (have_weights) {
            // We expect 'weights' to be a list of 1D arrays
            // exactly matching how we produce `weight_arrays` in fit().
            py::list w_list = weights.cast<py::list>();

            // We also expect 'cluster_labels' to be a 1D int array
            py::array_t<int> c_array = cluster_labels.cast<py::array_t<int>>();
            py::buffer_info c_info   = c_array.request();

            if (c_info.ndim != 1) {
                throw std::runtime_error("cluster_labels must be a 1D array.");
            }

            py::ssize_t n_clusters = w_list.size();
            if (c_info.shape[0] != n_clusters) {
                throw std::runtime_error(
                    "Inconsistent sizes: weights has " + std::to_string(n_clusters)
                    + " clusters, but cluster_labels has length "
                    + std::to_string(c_info.shape[0]) + "."
                );
            }

            // Convert cluster_labels to a simple pointer
            const int* c_ptr = static_cast<const int*>(c_info.ptr);

            // Initialize clusters_ and cluster_map_
            clusters_.resize(n_clusters);

            for (py::ssize_t i = 0; i < n_clusters; ++i) {
                // Each entry in w_list should be a 1D array.
                py::array_t<int> w_array = w_list[i].cast<py::array_t<int>>();
                py::buffer_info w_info   = w_array.request();

                if (w_info.ndim != 1) {
                    throw std::runtime_error("Each weight array must be 1D.");
                }

                // Copy data into clusters_[i].weight
                py::ssize_t weight_size = w_info.shape[0];
                clusters_[i].weight.resize(static_cast<size_t>(weight_size));
                std::memcpy(
                    clusters_[i].weight.data(),
                    w_info.ptr,
                    static_cast<size_t>(weight_size) * sizeof(int)
                );

                // cluster_map_[i] = label
                cluster_map_[static_cast<int>(i)] = c_ptr[i];
            }
        }
    }
    // ============================================
    // FIT
    // ============================================
    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>>
    fit(py::array_t<int> X, py::array_t<int> y) {
        py::buffer_info x_buf = X.request(), y_buf = y.request();

        assert(x_buf.ndim == 2 && "X must be a 2D array");
        assert(y_buf.ndim == 1 && "y must be a 1D array");

        pybind11::ssize_t num_samples_ssize = x_buf.shape[0];
        pybind11::ssize_t num_features_ssize = x_buf.shape[1];

        // Ensure values fit in `int`
        assert(num_samples_ssize <= std::numeric_limits<int>::max());
        assert(num_features_ssize <= std::numeric_limits<int>::max());

        int num_samples = static_cast<int>(num_samples_ssize);
        int num_features = static_cast<int>(num_features_ssize);


        // If we haven't set dim_original_ yet, do so now:
        if (dim_original_ == 0) {
            dim_original_ = num_features / 2;  // typical assumption
            rho_w1_       = static_cast<int>(base_rho_ * dim_original_);
        }
        assert(
            num_features == 2*dim_original_ &&
            "Number of features do not match existing weights"
        );

        auto x_ptr = static_cast<int*>(x_buf.ptr);
        auto y_ptr = static_cast<int*>(y_buf.ptr);

        std::vector<int> labels(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            std::vector<int> sample(x_ptr + i * num_features, x_ptr + (i + 1) * num_features);
            int c_b = y_ptr[i];
            labels[i] = step_fit(sample, c_b);
        }

        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), labels.size() * sizeof(int));

        std::vector<py::array_t<int>> weight_arrays;
        weight_arrays.reserve(clusters_.size());
        for (const auto& cluster : clusters_) {
            py::array_t<int> arr(cluster.weight.size());
            std::memcpy(arr.mutable_data(), cluster.weight.data(), cluster.weight.size() * sizeof(int));
            weight_arrays.push_back(std::move(arr));
        }

        std::vector<int> cluster_labels(clusters_.size());
        for (const auto& [cluster_id, label] : cluster_map_) {
            cluster_labels[cluster_id] = label;
        }
        py::array_t<int> cluster_labels_out(cluster_labels.size());
        std::memcpy(cluster_labels_out.mutable_data(),
                    cluster_labels.data(),
                    cluster_labels.size() * sizeof(int));

        return std::make_tuple(labels_out, weight_arrays, cluster_labels_out);
    }

    // ============================================
    // PREDICT
    // ============================================
    std::tuple<py::array_t<int>, py::array_t<int>> predict(py::array_t<int> X)
    {
        // If we have no clusters, we can't predict
        if (clusters_.empty()) {
            throw std::runtime_error(
                "Cannot call predict() because the model has no clusters. "
                "Call fit() or provide existing weights."
            );
        }

        py::buffer_info x_buf = X.request();
        assert(x_buf.ndim == 2 && "X must be a 2D array");

        py::ssize_t num_samples_ssize  = x_buf.shape[0];
        py::ssize_t num_features_ssize = x_buf.shape[1];

        assert(num_samples_ssize  <= std::numeric_limits<int>::max());
        assert(num_features_ssize <= std::numeric_limits<int>::max());

        int num_samples  = static_cast<int>(num_samples_ssize);
        int num_features = static_cast<int>(num_features_ssize);

        // If dim_original_ is 0, attempt to infer it
        if (dim_original_ == 0) {
            dim_original_ = num_features / 2;
            rho_w1_       = static_cast<int>(base_rho_ * dim_original_);
        }

        assert(
            num_features == 2*dim_original_ &&
            "Number of features do not match existing weights"
        );

        auto x_ptr = static_cast<int*>(x_buf.ptr);

        std::vector<int> predictions_a(num_samples);
        std::vector<int> predictions_b(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            // Extract one sample
            const int* row_ptr = x_ptr + i * num_features;
            std::vector<int> sample(row_ptr, row_ptr + num_features);

            // We'll keep track of the best T so far
            double best_T = -1.0;
            int best_cluster = -1;

            for (size_t c = 0; c < clusters_.size(); ++c) {
                double T = category_choice_predict(sample, clusters_[c].weight);
                // If T is better, update
                if (T > best_T) {
                    best_T = T;
                    best_cluster = static_cast<int>(c);
                }
            }

            // The predicted label is cluster_map_[best_cluster]
            predictions_b[i] = cluster_map_.at(best_cluster);
            predictions_a[i] = best_cluster;
        }

        // Convert to a py::array
        py::array_t<int> pred_a_out(predictions_a.size());
        std::memcpy(pred_a_out.mutable_data(),
                    predictions_a.data(),
                    predictions_a.size() * sizeof(int));

        py::array_t<int> pred_b_out(predictions_b.size());
        std::memcpy(pred_b_out.mutable_data(),
                    predictions_b.data(),
                    predictions_b.size() * sizeof(int));

        return std::make_tuple(pred_a_out, pred_b_out);
    }

    /**
     * @brief Perform a single-sample "step_fit" but verbosely log each cluster
     *        check in order. Returns (final_cluster_id, log_list).
     *
     * @param sample A 1D numpy array of ints (the input pattern).
     * @param c_b    The label for this sample.
     *
     * @return (final_cluster_id, log_list)
     *         final_cluster_id = the cluster we ended up assigning or creating
     *         log_list         = python list of dicts describing each step
     */
    std::tuple<int, py::list>
    verbose_step_fit_single_sample(py::array_t<int> sample, int c_b)
    {
        py::list log_list;

        // ----------------------------------------------------
        // 1) Convert the input sample to std::vector<int>
        // ----------------------------------------------------
        py::buffer_info buf_info = sample.request();
        if (buf_info.ndim != 1) {
            throw std::runtime_error("`sample` must be a 1D array.");
        }
        auto s_ptr = static_cast<int*>(buf_info.ptr);
        std::vector<int> sample_vec(s_ptr, s_ptr + buf_info.shape[0]);

        // If dim_original_ is 0 but we have clusters, try to infer
        if (!clusters_.empty() && dim_original_ == 0) {
            const auto &w0 = clusters_[0].weight;
            dim_original_ = static_cast<int>(w0.size() / 2);
            rho_w1_       = static_cast<int>(base_rho_ * dim_original_);
        }

        // 2) If no clusters => create first
        if (clusters_.empty()) {
            double old_rho = rho_;
            reset_rho();

            py::dict entry;
            entry["stage"]          = "no_clusters_yet";
            entry["final_action"]   = "created_cluster_0";
            entry["rho_before"]     = old_rho;
            entry["rho_after"]      = rho_;
            log_list.append(entry);

            clusters_.push_back({ sample_vec });
            cluster_map_[0] = c_b;
            return std::make_tuple(0, log_list);
        }

        // 3) reset rho
        double old_rho = rho_;
        reset_rho(); // sets rho_ = base_rho_

        // 4) Gather T-values
        size_t n_clusters = clusters_.size();
        std::vector<double> T_values(n_clusters, std::nan(""));
        std::vector<int>    w1_values(n_clusters, 0);

        for (size_t i = 0; i < n_clusters; ++i) {
            int w1;
            double T = category_choice(sample_vec, clusters_[i].weight, w1, MT_);
            if (!std::isnan(T)) {
                T_values[i]  = T;
                w1_values[i] = w1;
            }
        }

        auto match_op = _match_tracking_operator(MT_);
        int final_cluster_id = -1;

        // 5) Repeatedly pick argmax until we find or run out
        while (true) {
            // find argmax
            int best_idx = -1;
            double best_val = -1.0;
            for (size_t i = 0; i < n_clusters; ++i) {
                double tv = T_values[i];
                if (!std::isnan(tv) && tv > best_val) {
                    best_val = tv;
                    best_idx = static_cast<int>(i);
                }
            }

            // If best_idx == -1 => all T are NaN => break => new cluster
            if (best_idx < 0) {
                break;
            }

            // build a log dict for this attempt
            py::dict entry;
            entry["cluster_index"]   = best_idx;
            entry["activation"]      = T_values[best_idx];
            entry["rho_before"]      = rho_;
            double w1 = static_cast<double>(w1_values[best_idx]);
            double vig_value = w1 / dim_original_;
            entry["vigilance_value"] = vig_value;

            bool pass_vigilance = match_op(vig_value, rho_);
            entry["passed_vigilance"] = pass_vigilance;

            if (pass_vigilance) {
                bool pass_hypothesis = validate_hypothesis(best_idx, c_b);
                entry["passed_hypothesis"] = pass_hypothesis;

                if (pass_hypothesis) {
                    // update cluster
                    clusters_[best_idx].weight = update(sample_vec, clusters_[best_idx].weight);
                    cluster_map_[best_idx]     = c_b;
                    final_cluster_id           = best_idx;
                    entry["rho_after"]         = rho_;
                    entry["final_action"]      = "updated_cluster";
                    log_list.append(entry);
                    break; // done
                } else {
                    // fails hypothesis => match tracking
                    bool keep_searching = _match_tracking(static_cast<int>(w1));
                    entry["rho_after"] = rho_;
                    entry["final_action"] = keep_searching
                                            ? "continue_search"
                                            : "stop_search";
                    log_list.append(entry);

                    if (!keep_searching) {
                        break;
                    }
                    // skip this cluster from further consideration
                    T_values[best_idx] = std::nan("");
                }
            } else {
                // skip cluster
                entry["passed_hypothesis"] = py::none();
                entry["rho_after"]         = rho_;
                entry["final_action"]      = "skip_cluster";
                log_list.append(entry);

                T_values[best_idx] = std::nan("");
            }
        }

        // 6) if still -1 => create a new cluster
        if (final_cluster_id < 0) {
            int new_id = static_cast<int>(clusters_.size());
            clusters_.push_back({ sample_vec });
            cluster_map_[new_id] = c_b;
            final_cluster_id = new_id;

            py::dict entry;
            entry["cluster_index"]     = new_id;
            entry["activation"]        = py::none();
            entry["vigilance_value"]   = py::none();
            entry["passed_vigilance"]  = py::none();
            entry["passed_hypothesis"] = py::none();
            entry["rho_before"]        = rho_;
            entry["rho_after"]         = rho_;
            entry["final_action"]      = "created_new_cluster";
            log_list.append(entry);
        }

        return std::make_tuple(final_cluster_id, log_list);
    }

private:
    double base_rho_;
    double rho_;
    double alpha_;
    int dim_original_;
    int rho_w1_;
    std::string MT_;
    double epsilon_;
    std::vector<Cluster> clusters_;
    std::unordered_map<int, int> cluster_map_;

    void reset_rho() {
        rho_ = base_rho_;
    }

    int step_fit(const std::vector<int>& sample, int c_b) {
        // 1) reset vigilance
        reset_rho();

        // 2) If no clusters => create first
        if (clusters_.empty()) {
            clusters_.push_back({sample});
            cluster_map_[0] = c_b;
            return 0;
        }

        // 3) Compute T-values for each cluster
        //    We'll store them in a vector of doubles, one per cluster.
        //    We'll also store w1 in a parallel vector so we can do vigilance checks.
        size_t n_clusters = clusters_.size();
        std::vector<double> T_values(n_clusters, std::nan(""));
        std::vector<int>    w1_values(n_clusters, 0);

        // For each cluster i, compute T. If T=NaN, that means we skip it outright
        for (size_t i = 0; i < n_clusters; ++i) {
            int w1;
            double T = category_choice(sample, clusters_[i].weight, w1, MT_);
            if (!std::isnan(T)) {
                T_values[i]  = T;
                w1_values[i] = w1;
            } else {
                T_values[i]  = std::nan("");
                w1_values[i] = w1; // w1 might be meaningless here
            }
        }

        auto match_op = _match_tracking_operator(MT_);

        // 4) Repeatedly pick argmax until we find a cluster or run out of T
        while (true) {
            // find argmax of T_values
            int best_idx = -1;
            double best_val = -1.0;
            for (size_t i = 0; i < n_clusters; ++i) {
                double tv = T_values[i];
                if (!std::isnan(tv) && tv > best_val) {
                    best_val = tv;
                    best_idx = static_cast<int>(i);
                }
            }

            // If best_idx == -1 => all T are NaN => break => we create new cluster
            if (best_idx < 0) {
                break;
            }

            // Check vigilance
            int w1  = w1_values[best_idx];
            double vig_value = static_cast<double>(w1) / dim_original_;
            bool pass_vigilance = match_op(vig_value, rho_);
            if (pass_vigilance) {
                // Check hypothesis
                if (validate_hypothesis(best_idx, c_b)) {
                    // update cluster
                    clusters_[best_idx].weight = update(sample, clusters_[best_idx].weight);
                    cluster_map_[best_idx]     = c_b;
                    return best_idx; // done
                } else {
                    // Fails hypothesis => do match_tracking
                    bool keep_searching = _match_tracking(w1);
                    if (!keep_searching) {
                        // we stop => break => new cluster
                        break;
                    }
                    // else continue searching => mark T[best_idx] = NaN so we skip this cluster
                    T_values[best_idx] = std::nan("");
                }
            } else {
                // fails vigilance => skip this cluster
                T_values[best_idx] = std::nan("");
            }
        }

        // 5) If we reach here => no existing cluster chosen => create new cluster
        int new_cluster_id = static_cast<int>(clusters_.size());
        clusters_.push_back({sample});
        cluster_map_[new_cluster_id] = c_b;
        return new_cluster_id;
    }


    double category_choice(const std::vector<int>& i, const std::vector<int>& w, int
    &w1, const std::string MT_) {
        w1 = 0;
        int sum_w = 0;
        for (size_t j = 0; j < i.size(); ++j) {
            w1 += (i[j] & w[j]);
            sum_w += w[j];
        }

        auto match_op = _match_tracking_operator(MT_);
        if ((MT_ == "MT-") || (match_op(w1, rho_w1_))) {
            return static_cast<double>(w1) / (alpha_ + sum_w);
        }
        return std::nan("");
    }

    // For predict only (no skip / no vigilance):
    double category_choice_predict(const std::vector<int>& sample,
                                   const std::vector<int>& w)
    {
        // Just compute the same formula T = |sample & w| / (alpha + |w|)
        int w1 = 0;
        int sum_w = 0;
        for (size_t j = 0; j < sample.size(); ++j) {
            w1    += (sample[j] & w[j]);
            sum_w += w[j];
        }
        return (double) w1 / (alpha_ + sum_w);
    }

    bool validate_hypothesis(int cluster_id, int c_b) {
        return cluster_map_.find(cluster_id) == cluster_map_.end() || cluster_map_[cluster_id] == c_b;
    }

    bool _match_tracking(int w1) {
        double M = static_cast<double>(w1) / dim_original_;

        // 1) Adjust rho_ based on method:
        if (MT_ == "MT+") {
            rho_ = M + epsilon_;
        } else if (MT_ == "MT-") {
            rho_ = M - epsilon_;
        } else if (MT_ == "MT0") {
            rho_ = M;
        } else if (MT_ == "MT1") {
            rho_ = std::numeric_limits<double>::infinity();
        } else if (MT_ == "MT~") {
            // do nothing (no change to rho_)
        } else {
            throw std::invalid_argument("Invalid Match Tracking Method: " + MT_);
        }

        // 2) If method == "MT1" or we exceed rho_ > 1.0, stop searching.
        //    Otherwise, keep searching.
        if (MT_ == "MT1" || rho_ > 1.0) {
            return false;
        } else {
            return true;
        }
    }

    std::function<bool(double, double)> _match_tracking_operator(std::string MT) {
        if (MT == "MT+" || MT == "MT-" || MT == "MT1") {
            return std::greater_equal<double>();  // Now operates on double
        } else if (MT == "MT0" || MT == "MT~") {
            return std::greater<double>();  // Now operates on double
        } else {
            throw std::invalid_argument("Invalid Match Tracking Method: " + MT);
        }
    }


    std::vector<int> update(const std::vector<int>& i, const std::vector<int>& w) {
        std::vector<int> new_w(w.size());
        for (size_t j = 0; j < i.size(); ++j) {
            new_w[j] = i[j] & w[j];
        }
        return new_w;
    }
};

// =======================================================================
// Free function for fit
// =======================================================================
// This free function creates a temporary cppBinaryFuzzyARTMAP model, runs fit,
// and returns exactly the same results you'd get from the class-based usage.
std::tuple<py::array_t<int>,
           std::vector<py::array_t<int>>,
           py::array_t<int>>
FitBinaryFuzzyARTMAP(py::array_t<int> X,
                     py::array_t<int> y,
                     double rho,
                     double alpha,
                     std::string MT,
                     double epsilon,
                     py::object weights = py::none(),
                     py::object cluster_labels = py::none())
{
    // Just construct the model with these parameters, then fit
    cppBinaryFuzzyARTMAP model(rho, alpha, MT, epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

// =======================================================================
// NEW: Free function for predict
// =======================================================================
std::tuple<py::array_t<int>,
           py::array_t<int>>
PredictBinaryFuzzyARTMAP(py::array_t<int> X,
                         py::object weights = py::none(),
                         py::object cluster_labels = py::none())
{
    // Construct ephemeral model
    // just need to provide some dummy parameters
    cppBinaryFuzzyARTMAP model(0.0, 0.0, "MT+", 0.0, weights, cluster_labels);
    // Then call predict
    return model.predict(X);
}

// =======================================================================
// NEW: Free function for verbose step_fit
// =======================================================================

std::tuple<int, py::list>
VerboseStepFitBinaryFuzzyARTMAP(py::array_t<int> sample,
                                int label,
                                double rho,
                                double alpha,
                                std::string MT,
                                double epsilon,
                                py::object weights,
                                py::object cluster_labels)
{
    // ephemeral model:
    cppBinaryFuzzyARTMAP model(rho, alpha, MT, epsilon, weights, cluster_labels);

    // call that public method
    return model.verbose_step_fit_single_sample(sample, label);
}

// =======================================================================
// PYBIND
// =======================================================================


PYBIND11_MODULE(cppBinaryFuzzyARTMAP, m) {
    py::class_<cppBinaryFuzzyARTMAP>(m, "cppBinaryFuzzyARTMAP")
        .def(py::init<double, double, std::string, double,
                      py::object, py::object>(),
             py::arg("rho"),
             py::arg("alpha"),
             py::arg("MT"),
             py::arg("epsilon"),
             py::arg("weights")        = py::none(),
             py::arg("cluster_labels") = py::none(),
             R"doc(
Construct a cppBinaryFuzzyARTMAP model.

Optionally provide:

    weights        -- a Python list of 1D numpy int arrays (one for each cluster)
    cluster_labels -- a 1D numpy int array mapping cluster_id -> label

Either provide both or neither for partial initialization vs. fresh model.
)doc")

        .def("fit", &cppBinaryFuzzyARTMAP::fit,
             py::arg("X"), py::arg("y"),
             R"doc(
Fit the model given data X and labels y.

Returns:
    (labels_out, weight_arrays, cluster_labels_out)
)doc")

        .def("__repr__",
             [](const cppBinaryFuzzyARTMAP &m) {
                 return "<cppBinaryFuzzyARTMAP model>";
             });

    // The free function that can also take optional weights, cluster_labels
    m.def("FitBinaryFuzzyARTMAP",
          &FitBinaryFuzzyARTMAP,
          py::arg("X"),
          py::arg("y"),
          py::arg("rho"),
          py::arg("alpha"),
          py::arg("MT"),
          py::arg("epsilon"),
          py::arg("weights")        = py::none(),
          py::arg("cluster_labels") = py::none(),
          R"doc(
Fit cppBinaryFuzzyARTMAP in a single function call.

Optionally re-initialize from existing weights/cluster_labels for partial fits.

Either provide BOTH 'weights' (a list of 1D arrays) and 'cluster_labels' (1D array)
or leave both as None.
)doc");

    // The free function for prediction
    m.def("PredictBinaryFuzzyARTMAP",
          &PredictBinaryFuzzyARTMAP,
          py::arg("X"),
          py::arg("weights")        = py::none(),
          py::arg("cluster_labels") = py::none(),
          R"doc(
Predict labels using a temporary cppBinaryFuzzyARTMAP model.

If no weights/cluster_labels are provided, the model has no clusters and will fail.

Returns:
    A 1D numpy array of predicted labels.
)doc");

    m.def("VerboseStepFitBinaryFuzzyARTMAP",
      &VerboseStepFitBinaryFuzzyARTMAP,
      py::arg("sample"),
      py::arg("label"),
      py::arg("rho"),
      py::arg("alpha"),
      py::arg("MT"),
      py::arg("epsilon"),
      py::arg("weights")        = py::none(),
      py::arg("cluster_labels") = py::none(),
      R"doc(
Constructs a temporary model from the given weights+labels, then calls
verbose_step_fit_single_sample(...) on that ephemeral model. Returns
(final_cluster_id, log_list) without modifying your real model.
)doc");

}
