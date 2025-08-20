// cppFuzzyARTMAP.cpp
// ----------------------------------------------------------
//  C++ accelerated Fuzzy  ARTMAP  (pybind11)
//  ‑ supports incremental training via external weights/map
//  ‑ real‑valued inputs in [0,1] (already normalised & complement‑coded)
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
class cppFuzzyARTMAP {
public:
    struct Cluster {
        std::vector<double> weight;          // 2d components (complement‑coded)
    };

    cppFuzzyARTMAP(double rho,
                   double alpha,
                   double beta,
                   const std::string& MT,
                   double epsilon,
                   py::object weights         = py::none(),
                   py::object cluster_labels  = py::none())
        : base_rho_(rho),
          alpha_(alpha),
          beta_(beta),
          MT_(MT),
          epsilon_(epsilon),
          dim_original_(0),
          rho_(rho)
    {
        const bool have_W  = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_W != have_cl) {
            throw std::invalid_argument(
                "Provide BOTH 'weights' and 'cluster_labels' or neither.");
        }

        if (have_W) {
            py::list      w_list = weights.cast<py::list>();
            py::array_t<int> cl  = cluster_labels.cast<py::array_t<int>>();
            auto            cl_b = cl.request();
            if (cl_b.ndim != 1)
                throw std::runtime_error("cluster_labels must be 1‑D");

            const std::size_t n_clusters = w_list.size();
            if (static_cast<std::size_t>(cl_b.shape[0]) != n_clusters)
                throw std::runtime_error("weights / cluster_labels size mismatch");

            clusters_.resize(n_clusters);
            const int* cl_ptr = static_cast<const int*>(cl_b.ptr);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto                w_b   = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::runtime_error("each weight must be 1‑D");
                clusters_[k].weight.resize(static_cast<std::size_t>(w_b.shape[0]));
                std::memcpy(clusters_[k].weight.data(),
                            w_b.ptr,
                            sizeof(double) * w_b.shape[0]);
                cluster_map_[static_cast<int>(k)] = cl_ptr[k];

                if (dim_original_ == 0) dim_original_ = static_cast<int>(w_b.shape[0] / 2);
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
    fit(py::array_t<double> X, py::array_t<int> y)
    {
        auto xb = X.request();
        auto yb = y.request();

        assert(xb.ndim == 2 && yb.ndim == 1);
        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_original_ == 0) {
            dim_original_ = n_features / 2;
        }
        assert(n_features == 2 * dim_original_);

        const double* Xptr = static_cast<const double*>(xb.ptr);
        const int*    yptr = static_cast<const int*>(yb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> sample(Xptr + i * n_features,
                                       Xptr + (i + 1) * n_features);
            labels_a[i] = step_fit(sample, yptr[i]);
        }

        /* -------- pack output -------- */
        py::array_t<int> labels_out(labels_a.size());
        std::memcpy(labels_out.mutable_data(),
                    labels_a.data(),
                    labels_a.size() * sizeof(int));

        std::vector<py::array_t<double>> weight_out;
        weight_out.reserve(clusters_.size());
        for (const auto& c : clusters_) {
            py::array_t<double> w(c.weight.size());
            std::memcpy(w.mutable_data(),
                        c.weight.data(),
                        c.weight.size() * sizeof(double));
            weight_out.push_back(std::move(w));
        }

        std::vector<int> clabels_vec(clusters_.size());
        for (const auto& kv : cluster_map_)
            clabels_vec[kv.first] = kv.second;
        py::array_t<int> clabels_out(clabels_vec.size());
        std::memcpy(clabels_out.mutable_data(),
                    clabels_vec.data(),
                    clabels_vec.size() * sizeof(int));

        return {labels_out, weight_out, clabels_out};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a, y_b)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>, py::array_t<int>>
    predict(py::array_t<double> X)
    {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb       = X.request();
        const int n   = static_cast<int>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_original_ == 0) dim_original_ = dim / 2;
        assert(dim == 2 * dim_original_);

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y_a(n), y_b(n);

        for (int i = 0; i < n; ++i) {
            const double* row = Xptr + i * dim;

            int    best_id = -1;
            double best_T  = -1.0;

            for (std::size_t c = 0; c < clusters_.size(); ++c) {
                double T = category_choice(row, clusters_[c].weight);
                double M = match(row, clusters_[c].weight);

                if (M >= base_rho_ && T > best_T) {
                    best_T  = T;
                    best_id = static_cast<int>(c);
                }
            }
            if (best_id < 0) best_id = 0;               // fallback
            y_a[i] = best_id;
            y_b[i] = cluster_map_.at(best_id);
        }

        py::array_t<int> ya_out(y_a.size()), yb_out(y_b.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        std::memcpy(yb_out.mutable_data(), y_b.data(), sizeof(int) * y_b.size());
        return {ya_out, yb_out};
    }

private:
    /* ───── hyper‑parameters ───── */
    double base_rho_, alpha_, beta_, epsilon_;
    std::string MT_;

    /* ───── state ───── */
    int    dim_original_;
    double rho_;
    std::vector<Cluster>           clusters_;
    std::unordered_map<int, int>   cluster_map_;

    /* ─────────────────────────────────────────────────── */
    void reset_rho() { rho_ = base_rho_; }

    // fuzzy  AND L1
    static double l1_and(const std::vector<double>& a,
                         const std::vector<double>& b)
    {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j)
            s += std::min(a[j], b[j]);
        return s;
    }
    static double l1_and(const double* x,
                         const std::vector<double>& w,
                         int len)
    {
        double s = 0.0;
        for (int j = 0; j < len; ++j)
            s += std::min(x[j], w[j]);
        return s;
    }

    /* ─────────────────────────────────────────────────── */
    double category_choice(const double* sample,
                           const std::vector<double>& w) const
    {
        const int len = static_cast<int>(w.size());
        double num = l1_and(sample, w, len);
        double denom = alpha_ + std::accumulate(w.begin(), w.end(), 0.0);
        return num / denom;
    }

    double match(const double* sample,
                 const std::vector<double>& w) const
    {
        const int len = static_cast<int>(w.size());
        double num = l1_and(sample, w, len);
        return num / static_cast<double>(dim_original_);
    }

    std::vector<double> update_weight(const std::vector<double>& i,
                                      const std::vector<double>& w) const
    {
        std::vector<double> out(w.size());
        for (std::size_t j = 0; j < w.size(); ++j)
            out[j] = beta_ * std::min(i[j], w[j]) + (1.0 - beta_) * w[j];
        return out;
    }

    /* ─────────────────────────────────────────────────── */
    int step_fit(const std::vector<double>& sample, int c_b)
    {
        reset_rho();

        // first ever cluster?
        if (clusters_.empty()) {
            clusters_.push_back({sample});
            cluster_map_[0] = c_b;
            return 0;
        }

        const std::size_t K = clusters_.size();
        std::vector<double> T(K), M(K);
        std::vector<char>   valid(K, 1);

        for (std::size_t k = 0; k < K; ++k) {
            T[k] = category_choice(sample.data(), clusters_[k].weight);
            M[k] = match(sample.data(), clusters_[k].weight);
        }

        auto op = _match_op(MT_);

        while (true) {
            int best = -1; double bestT = -1.0;
            for (std::size_t k = 0; k < K; ++k)
                if (valid[k] && T[k] > bestT) {
                    bestT = T[k]; best = static_cast<int>(k);
                }
            if (best < 0) break;                   // none valid → new cluster

            if (!op(M[best], rho_)) {              // fails vigilance
                valid[best] = 0;                   // discard and continue
                continue;
            }

            if (cluster_map_.count(best) && cluster_map_[best] != c_b) {
                // hypothesis violated → match‑tracking
                if (!_match_tracking(M[best])) break;
                valid[best] = 0;
                continue;
            }

            /* commit to cluster 'best' */
            clusters_[best].weight = update_weight(sample, clusters_[best].weight);
            cluster_map_[best]     = c_b;
            return best;
        }

        /* create new */
        int new_id = static_cast<int>(clusters_.size());
        clusters_.push_back({sample});
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
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto FitFuzzyARTMAP(py::array_t<double>  X,
                    py::array_t<int>     y,
                    double               rho,
                    double               alpha,
                    double               beta,
                    const std::string&   MT,
                    double               epsilon,
                    py::object           weights = py::none(),
                    py::object           cluster_labels = py::none())
{
    cppFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

auto PredictFuzzyARTMAP(py::array_t<double> X,
                        double              rho,
                        double              alpha,
                        double              beta,
                        const std::string&  MT,
                        double              epsilon,
                        py::object          weights = py::none(),
                        py::object          cluster_labels = py::none())
{
    cppFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, weights, cluster_labels);
    return model.predict(X);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppFuzzyARTMAP, m) {
    py::class_<cppFuzzyARTMAP>(m, "cppFuzzyARTMAP")
        .def(py::init<double,double,double,std::string,double,
                      py::object,py::object>(),
             py::arg("rho"), py::arg("alpha"), py::arg("beta"),
             py::arg("MT"),  py::arg("epsilon"),
             py::arg("weights")        = py::none(),
             py::arg("cluster_labels") = py::none())
        .def("fit",    &cppFuzzyARTMAP::fit,    py::arg("X"), py::arg("y"))
        .def("predict",&cppFuzzyARTMAP::predict,py::arg("X"))
        .def("__repr__", [](const cppFuzzyARTMAP&){ return "<cppFuzzyARTMAP>"; });

    m.def("FitFuzzyARTMAP",    &FitFuzzyARTMAP,
          py::arg("X"), py::arg("y"),
          py::arg("rho"), py::arg("alpha"), py::arg("beta"),
          py::arg("MT"),  py::arg("epsilon"),
          py::arg("weights")=py::none(), py::arg("cluster_labels")=py::none());

    m.def("PredictFuzzyARTMAP",&PredictFuzzyARTMAP,
          py::arg("X"),
          py::arg("rho"), py::arg("alpha"), py::arg("beta"),
          py::arg("MT"),  py::arg("epsilon"),
          py::arg("weights")=py::none(), py::arg("cluster_labels")=py::none());
}
