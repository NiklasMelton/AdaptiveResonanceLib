#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace py = pybind11;

class BinaryFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<int> weight;
    };

    BinaryFuzzyARTMAP(double rho, double alpha, std::string MT, double epsilon)
        : base_rho_(rho), alpha_(alpha), MT_(MT), epsilon_(epsilon)
    {
        // We do NOT know dim_original_ yet, so we simply set to 0 here.
        // We'll infer it in fit() from the data.
        dim_original_ = 0;
        rho_w1_       = 0;
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>> fit(py::array_t<int> X, py::array_t<int> y) {
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

        dim_original_ = num_features / 2;
        rho_w1_       = static_cast<int>(base_rho_ * dim_original_);

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
            double T = category_choice(sample, clusters_[i].weight, w1);
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

    double category_choice(const std::vector<int>& i, const std::vector<int>& w, int &w1) {
        w1 = 0;
        int sum_w = 0;
        for (size_t j = 0; j < i.size(); ++j) {
            w1 += (i[j] & w[j]);
            sum_w += w[j];
        }

        auto match_op = _match_tracking_operator(MT_);
        if (match_op(w1, rho_w1_)) {
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

PYBIND11_MODULE(BinaryFuzzyARTMAP, m) {
    py::class_<BinaryFuzzyARTMAP>(m, "BinaryFuzzyARTMAP")
        .def(py::init<double, double, std::string, double>(),
             py::arg("rho"),
             py::arg("alpha"),
             py::arg("MT"),
             py::arg("epsilon"))

        .def("fit", &BinaryFuzzyARTMAP::fit)

        .def("__repr__",
            [](const BinaryFuzzyARTMAP &a) {
                return "<BinaryFuzzyARTMAP model>";
            }
        );
}
