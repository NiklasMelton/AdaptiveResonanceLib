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

class BinaryFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<int> weight;
    };

    BinaryFuzzyARTMAP(double rho,
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
        reset_rho();

        if (clusters_.empty()) {
            clusters_.push_back({sample});
            cluster_map_[0] = c_b;
            return 0;
        }

        std::vector<std::tuple<double, int, int>> T_values;

        for (size_t i = 0; i < clusters_.size(); ++i) {
            int w1;
            double T = category_choice(sample, clusters_[i].weight, w1, MT_);
            if (!std::isnan(T)) {
                T_values.emplace_back(T, static_cast<int>(i), w1);
            }
        }

        std::sort(T_values.begin(), T_values.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

        auto match_op = _match_tracking_operator(MT_);

        for (const auto& [T, c, w1] : T_values) {
            if (match_op(static_cast<double>(w1) / dim_original_, rho_)) {
                if (validate_hypothesis(c, c_b)) {
                    clusters_[c].weight = update(sample, clusters_[c].weight);
                    cluster_map_[c] = c_b;
                    return c;
                } else {
                    bool keep_searching = _match_tracking(w1);
                    if (!keep_searching) {
                        break;
                    }
                }
            }
        }

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

    bool validate_hypothesis(int cluster_id, int c_b) {
        return cluster_map_.find(cluster_id) == cluster_map_.end() || cluster_map_[cluster_id] == c_b;
    }

    bool _match_tracking(int w1) {
        double M = static_cast<double>(w1) / dim_original_;
        if (MT_ == "MT+") {
            rho_ = M + epsilon_;
            return true;
        } else if (MT_ == "MT-") {
            rho_ = M - epsilon_;
            return true;
        } else if (MT_ == "MT0") {
            rho_ = M;
            return true;
        } else if (MT_ == "MT1") {
            rho_ = std::numeric_limits<double>::infinity();
            return false;
        } else if (MT_ == "MT~") {
            return true;
        } else {
            throw std::invalid_argument("Invalid Match Tracking Method: " + MT_);
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
// Free function for fit, same as before
// =======================================================================
// This free function creates a temporary BinaryFuzzyARTMAP model, runs fit,
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
    BinaryFuzzyARTMAP model(rho, alpha, MT, epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

// =======================================================================
// NEW: Free function for predict
// =======================================================================
std::tuple<py::array_t<int>,
           py::array_t<int>>
PredictBinaryFuzzyARTMAP(py::array_t<int> X,
                         double rho,
                         double alpha,
                         std::string MT,
                         double epsilon,
                         py::object weights = py::none(),
                         py::object cluster_labels = py::none())
{
    // Construct ephemeral model
    BinaryFuzzyARTMAP model(rho, alpha, MT, epsilon, weights, cluster_labels);
    // Then call predict
    return model.predict(X);
}

PYBIND11_MODULE(BinaryFuzzyARTMAP, m) {
    py::class_<BinaryFuzzyARTMAP>(m, "BinaryFuzzyARTMAP")
        .def(py::init<double, double, std::string, double,
                      py::object, py::object>(),
             py::arg("rho"),
             py::arg("alpha"),
             py::arg("MT"),
             py::arg("epsilon"),
             py::arg("weights")        = py::none(),
             py::arg("cluster_labels") = py::none(),
             R"doc(
Construct a BinaryFuzzyARTMAP model.

Optionally provide:

    weights        -- a Python list of 1D numpy int arrays (one for each cluster)
    cluster_labels -- a 1D numpy int array mapping cluster_id -> label

Either provide both or neither for partial initialization vs. fresh model.
)doc")

        .def("fit", &BinaryFuzzyARTMAP::fit,
             py::arg("X"), py::arg("y"),
             R"doc(
Fit the model given data X and labels y.

Returns:
    (labels_out, weight_arrays, cluster_labels_out)
)doc")

        .def("__repr__",
             [](const BinaryFuzzyARTMAP &m) {
                 return "<BinaryFuzzyARTMAP model>";
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
Fit BinaryFuzzyARTMAP in a single function call.

Optionally re-initialize from existing weights/cluster_labels for partial fits.

Either provide BOTH 'weights' (a list of 1D arrays) and 'cluster_labels' (1D array)
or leave both as None.
)doc");

    // The free function for prediction
    m.def("PredictBinaryFuzzyARTMAP",
          &PredictBinaryFuzzyARTMAP,
          py::arg("X"),
          py::arg("rho"),
          py::arg("alpha"),
          py::arg("MT"),
          py::arg("epsilon"),
          py::arg("weights")        = py::none(),
          py::arg("cluster_labels") = py::none(),
          R"doc(
Predict labels using a temporary BinaryFuzzyARTMAP model.

If no weights/cluster_labels are provided, the model has no clusters and will fail.
If you want to do partial usage, provide both 'weights' and 'cluster_labels'.

Returns:
    A 1D numpy array of predicted labels.
)doc");
}
