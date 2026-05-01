// cppART1MAP.cpp
// ----------------------------------------------------------
//  C++ accelerated ART1MAP (pybind11)
//  - supports incremental training via external weights/map
//  - ART1 expects binary data; input dtype is int16.
//    We interpret x != 0 as 1, else 0.
//  - weights stored as length 2*dim: [w_bu (double), w_td (0/1)]
//  - match tracking modes identical to your Fuzzy ARTMAP scaffold
// ----------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ────────────────────────────────────────────────────────────
class cppART1MAP {
public:
    struct Cluster {
        std::vector<double> weight; // size 2*dim: [bu..., td...]
    };

    cppART1MAP(double rho,
               double L,
               const std::string& MT,
               double epsilon,
               py::object weights        = py::none(),
               py::object cluster_labels = py::none())
        : base_rho_(rho),
          L_(L),
          MT_(MT),
          epsilon_(epsilon),
          dim_(0),
          rho_(rho)
    {
        if (base_rho_ < 0.0 || base_rho_ > 1.0) throw std::invalid_argument("rho must be in [0,1]");
        if (L_ < 1.0) throw std::invalid_argument("L must be >= 1");

        const bool have_W  = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_W != have_cl) {
            throw std::invalid_argument("Provide BOTH 'weights' and 'cluster_labels' or neither.");
        }

        if (have_W) {
            py::list w_list = weights.cast<py::list>();
            py::array_t<int> cl = cluster_labels.cast<py::array_t<int>>();
            auto cl_b = cl.request();
            if (cl_b.ndim != 1) throw std::runtime_error("cluster_labels must be 1-D");

            const std::size_t n_clusters = w_list.size();
            if (static_cast<std::size_t>(cl_b.shape[0]) != n_clusters)
                throw std::runtime_error("weights / cluster_labels size mismatch");

            clusters_.resize(n_clusters);
            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto w_b = w_arr.request();
                if (w_b.ndim != 1) throw std::runtime_error("each weight must be 1-D");

                const int len = static_cast<int>(w_b.shape[0]);
                if (len % 2 != 0) throw std::runtime_error("weight length must be even (2*dim)");

                clusters_[k].weight.resize(static_cast<std::size_t>(len));
                std::memcpy(clusters_[k].weight.data(), w_b.ptr, sizeof(double) * len);

                if (dim_ == 0) dim_ = len / 2;
                if (len != 2 * dim_) throw std::runtime_error("inconsistent weight dimensions");

                cluster_map_[static_cast<int>(k)] = cl_ptr[k];
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>, cluster_labels)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>,
               std::vector<py::array_t<double>>,
               py::array_t<int>>
    fit(py::array_t<std::int16_t> X, py::array_t<int> y)
    {
        auto xb = X.request();
        auto yb = y.request();

        if (xb.ndim != 2 || yb.ndim != 1) throw std::runtime_error("X must be 2-D and y must be 1-D");

        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);
        if (static_cast<int>(yb.shape[0]) != n_samples) throw std::runtime_error("X/y size mismatch");

        if (dim_ == 0) dim_ = n_features;
        if (n_features != dim_) throw std::runtime_error("X feature dimension mismatch with model");

        const std::int16_t* Xptr = static_cast<const std::int16_t*>(xb.ptr);
        const int*          yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const std::int16_t* row = Xptr + i * dim_;
            labels_a[i] = step_fit(row, yptr[i]);
        }

        py::array_t<int> labels_out(labels_a.size());
        std::memcpy(labels_out.mutable_data(), labels_a.data(), labels_a.size() * sizeof(int));

        std::vector<py::array_t<double>> weight_out;
        weight_out.reserve(clusters_.size());
        for (const auto& c : clusters_) {
            py::array_t<double> w(c.weight.size());
            std::memcpy(w.mutable_data(), c.weight.data(), c.weight.size() * sizeof(double));
            weight_out.push_back(std::move(w));
        }

        std::vector<int> clabels_vec(clusters_.size());
        for (const auto& kv : cluster_map_) clabels_vec[kv.first] = kv.second;
        py::array_t<int> clabels_out(clabels_vec.size());
        std::memcpy(clabels_out.mutable_data(), clabels_vec.data(), clabels_vec.size() * sizeof(int));

        return {labels_out, weight_out, clabels_out};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a, y_b)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, py::array_t<int>>
    predict(py::array_t<std::int16_t> X)
    {
        if (clusters_.empty()) throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");

        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0) dim_ = n_features;
        if (n_features != dim_) throw std::runtime_error("X feature dimension mismatch with model");

        const std::int16_t* Xptr = static_cast<const std::int16_t*>(xb.ptr);

        std::vector<int> y_a(n_samples), y_b(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const std::int16_t* row = Xptr + i * dim_;

            int best_id = -1;
            double best_T = -1e300;

            for (std::size_t c = 0; c < clusters_.size(); ++c) {
                const double T = category_choice(row, clusters_[c].weight);
                if (T > best_T) {
                    best_T = T;
                    best_id = static_cast<int>(c);
                }
            }

            if (best_id < 0) best_id = 0;
            y_a[i] = best_id;
            y_b[i] = cluster_map_.at(best_id);
        }

        py::array_t<int> ya_out(y_a.size()), yb_out(y_b.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        std::memcpy(yb_out.mutable_data(), y_b.data(), sizeof(int) * y_b.size());
        return {ya_out, yb_out};
    }
    // ────────────────────────────────────────────────────────
    // category choice
    // ────────────────────────────────────────────────────────
    py::array_t<double> category_choice(py::array_t<std::int16_t> X)
    {
        if (clusters_.empty()) throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");

        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0) dim_ = n_features;
        if (n_features != dim_) throw std::runtime_error("X feature dimension mismatch with model");

        const std::int16_t* Xptr = static_cast<const std::int16_t*>(xb.ptr);
        const int K = static_cast<int>(clusters_.size());

        py::array_t<double> out({n_samples, K});
        auto outm = out.mutable_unchecked<2>();

        for (int i = 0; i < n_samples; ++i) {
            const std::int16_t* row = Xptr + i * dim_;
            for (int c = 0; c < K; ++c) {
                outm(i, c) = category_choice(row, clusters_[c].weight); // existing definition
            }
        }
        return out;
    }


private:
    /* ───── hyper-parameters ───── */
    double base_rho_;
    double L_;
    std::string MT_;
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
    double category_choice(const std::int16_t* sample,
                           const std::vector<double>& w) const
    {
        double s = 0.0;
        for (int j = 0; j < dim_; ++j) {
            float wf = static_cast<float>(w[j]);     // emulate float32 rounding
            s += static_cast<double>(bit(sample[j])) * static_cast<double>(wf);
        }
        return s;
    }

    // ART1 match: M = |I & w_td| / dim
    double match(const std::int16_t* sample,
                 const std::vector<double>& w) const
    {
        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int i_bit  = bit(sample[j]);
            const int td_bit = (w[dim_ + j] != 0.0) ? 1 : 0;
            if (i_bit & td_bit) ++count;
        }
        return static_cast<double>(count) / static_cast<double>(dim_);
    }

    // update rule:
    // w_td_new = I & w_td
    // count = nnz(w_td_new)
    // w_bu_new = (L / (L - 1 + count)) * w_td_new
    std::vector<double> update_weight(const std::int16_t* sample,
                                      const std::vector<double>& w) const
    {
        std::vector<double> out(2 * dim_, 0.0);

        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int td_new = (bit(sample[j]) & ((w[dim_ + j] != 0.0) ? 1 : 0));
            out[dim_ + j] = static_cast<double>(td_new);
            if (td_new) ++count;
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
    std::vector<double> new_weight(const std::int16_t* sample) const
    {
        std::vector<double> w(2 * dim_, 0.0);
        const double sf = L_ / (L_ - 1.0 + static_cast<double>(dim_));

        for (int j = 0; j < dim_; ++j) {
            const double b = static_cast<double>(bit(sample[j]));
            w[dim_ + j] = b;      // top-down
            w[j]        = sf * b; // bottom-up
        }

        return w;
    }

    int step_fit(const std::int16_t* sample, int c_b)
    {
        reset_rho();

        // first cluster?
        if (clusters_.empty()) {
            clusters_.push_back({ new_weight(sample) });
            cluster_map_[0] = c_b;
            return 0;
        }

        const std::size_t K = clusters_.size();
        std::vector<double> T(K), M(K);

        for (std::size_t k = 0; k < K; ++k) {
            T[k] = category_choice(sample, clusters_[k].weight);
            M[k] = match(sample, clusters_[k].weight);
        }

        auto op = _match_op(MT_);

        // candidate order: descending T, tie-break ascending index
        std::vector<int> order;
        order.reserve(K);
        for (std::size_t k = 0; k < K; ++k) order.push_back(static_cast<int>(k));

        std::sort(order.begin(), order.end(),
                  [&](int a, int b) {
                      const double Ta = T[a], Tb = T[b];
                      if (Ta != Tb) return Ta > Tb;
                      return a < b;
                  });

        for (int best : order) {
            // vigilance test
            if (!op(M[best], rho_)) continue;

            // map-field consistency test (ARTMAP)
            if (cluster_map_.count(best) && cluster_map_[best] != c_b) {
                // hypothesis violated -> match tracking
                if (!_match_tracking(M[best])) {
                    break; // stop searching -> new category
                }
                continue; // keep searching with updated rho_
            }

            // commit: learn ART1 weights and map label
            clusters_[best].weight = update_weight(sample, clusters_[best].weight);
            cluster_map_[best] = c_b;
            return best;
        }

        // create new
        const int new_id = static_cast<int>(clusters_.size());
        clusters_.push_back({ new_weight(sample) });
        cluster_map_[new_id] = c_b;
        return new_id;
    }

    /* ── match tracking helpers ── */
    bool _match_tracking(double M)
    {
        if (MT_ == "MT+")      rho_ = M + epsilon_;
        else if (MT_ == "MT-") rho_ = M - epsilon_;
        else if (MT_ == "MT0") rho_ = M;
        else if (MT_ == "MT1") rho_ = std::numeric_limits<double>::infinity();
        else if (MT_ == "MT~") {/* leave rho unchanged */}
        else throw std::invalid_argument("Invalid MT mode: " + MT_);

        // ART1 match M is in [0,1]; if rho_ exceeds 1 we force new cluster.
        return !(MT_ == "MT1" || rho_ > 1.0);
    }

    static std::function<bool(double,double)> _match_op(const std::string& MT)
    {
        if (MT == "MT+" || MT == "MT-" || MT == "MT1")
            return std::greater_equal<double>();
        if (MT == "MT0" || MT == "MT~")
            return std::greater<double>();
        throw std::invalid_argument("Invalid MT mode");
    }
};

// ────────────────────────────────────────────────────────────
// convenience free-functions (fit / predict)
// ────────────────────────────────────────────────────────────
auto FitART1MAP(py::array_t<std::int16_t> X,
                py::array_t<int>          y,
                double                    rho,
                double                    L,
                const std::string&        MT,
                double                    epsilon,
                py::object                weights = py::none(),
                py::object                cluster_labels = py::none())
{
    cppART1MAP model(rho, L, MT, epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

auto PredictART1MAP(py::array_t<std::int16_t> X,
                    double                    rho,
                    double                    L,
                    const std::string&        MT,
                    double                    epsilon,
                    py::object                weights = py::none(),
                    py::object                cluster_labels = py::none())
{
    cppART1MAP model(rho, L, MT, epsilon, weights, cluster_labels);
    return model.predict(X);
}

auto CategoryChoiceART1MAP(py::array_t<std::int16_t> X,
                           double                    rho,
                           double                    L,
                           const std::string&        MT,
                           double                    epsilon,
                           py::object                weights = py::none(),
                           py::object                cluster_labels = py::none())
{
    cppART1MAP model(rho, L, MT, epsilon, weights, cluster_labels);
    return model.category_choice(X);
}


// ────────────────────────────────────────────────────────────
// pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppART1MAP, m) {
    py::class_<cppART1MAP>(m, "cppART1MAP")
        .def(py::init<double,double,std::string,double,py::object,py::object>(),
             py::arg("rho"), py::arg("L"),
             py::arg("MT"),  py::arg("epsilon"),
             py::arg("weights")        = py::none(),
             py::arg("cluster_labels") = py::none())
        .def("fit", &cppART1MAP::fit, py::arg("X"), py::arg("y"))
        .def("predict", &cppART1MAP::predict, py::arg("X"))
        .def("__repr__", [](const cppART1MAP&) { return "<cppART1MAP>"; });

    m.def("FitART1MAP", &FitART1MAP,
          py::arg("X"), py::arg("y"),
          py::arg("rho"), py::arg("L"),
          py::arg("MT"),  py::arg("epsilon"),
          py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());

    m.def("PredictART1MAP", &PredictART1MAP,
          py::arg("X"),
          py::arg("rho"), py::arg("L"),
          py::arg("MT"),  py::arg("epsilon"),
          py::arg("weights") = py::none(),
          py::arg("cluster_labels") = py::none());

    m.def("CategoryChoiceART1MAP", &CategoryChoiceART1MAP,
      py::arg("X"),
      py::arg("rho"), py::arg("L"),
      py::arg("MT"),  py::arg("epsilon"),
      py::arg("weights") = py::none(),
      py::arg("cluster_labels") = py::none());
}
