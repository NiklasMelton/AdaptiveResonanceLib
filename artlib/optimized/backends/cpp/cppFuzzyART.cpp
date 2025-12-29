// cppFuzzyART.cpp
// ----------------------------------------------------------
//  C++ accelerated Fuzzy  ART  (pybind11)
//  ‑ supports incremental training via external weights
//  ‑ real‑valued inputs in [0,1] (already normalised & complement‑coded)
// ----------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <numeric>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ────────────────────────────────────────────────────────────
class cppFuzzyART {
public:
    struct Cluster {
        std::vector<double> weight;          // 2d components (complement‑coded)
    };

    cppFuzzyART(double rho,
                double alpha,
                double beta,
                py::object weights         = py::none())
        : rho_(rho),
          alpha_(alpha),
          beta_(beta),
          dim_original_(0)
    {
        const bool have_W  = !weights.is_none();

        if (have_W) {
            py::list      w_list = weights.cast<py::list>();

            const std::size_t n_clusters = w_list.size();

            clusters_.resize(n_clusters);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto                w_b   = w_arr.request();
                if (w_b.ndim != 1)
                    throw std::runtime_error("each weight must be 1‑D");
                clusters_[k].weight.resize(static_cast<std::size_t>(w_b.shape[0]));
                std::memcpy(clusters_[k].weight.data(),
                            w_b.ptr,
                            sizeof(double) * w_b.shape[0]);

                if (dim_original_ == 0) dim_original_ = static_cast<int>(w_b.shape[0] / 2);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // FIT
    // returns (labels_a, weights_vec<ndarray>)
    // ────────────────────────────────────────────────────────
    std::tuple<py::array_t<int>,
               std::vector<py::array_t<double>>
               >
    fit(py::array_t<double> X)
    {
        auto xb = X.request();

        assert(xb.ndim == 2);
        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        assert((n_features % 2) == 0 && "Data has not been complement coded");


        if (dim_original_ == 0) {
            dim_original_ = n_features / 2;
        }
        assert(n_features == 2 * dim_original_);

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> labels_a(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            std::vector<double> sample(Xptr + i * n_features,
                                       Xptr + (i + 1) * n_features);
            labels_a[i] = step_fit(sample);
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

        return {labels_out, weight_out};
    }

    // ────────────────────────────────────────────────────────
    // PREDICT  (returns y_a)
    // ────────────────────────────────────────────────────────
    py::array_t<int>
    predict(py::array_t<double> X)
    {
        if (clusters_.empty())
            throw std::runtime_error("Model has no clusters");

        auto xb       = X.request();
        const int n   = static_cast<int>(xb.shape[0]);
        const int dim = static_cast<int>(xb.shape[1]);
        if (dim_original_ == 0) dim_original_ = dim / 2;
        assert(dim == 2 * dim_original_);
        assert((dim % 2) == 0 && "Data has not been complement coded");


        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y_a(n);

        for (int i = 0; i < n; ++i) {
            const double* row = Xptr + i * dim;

            int    best_id = -1;
            double best_T  = -1.0;

            for (std::size_t c = 0; c < clusters_.size(); ++c) {
                double T = category_choice(row, clusters_[c].weight);

                if (T > best_T) {
                    best_T  = T;
                    best_id = static_cast<int>(c);
                }
            }
            if (best_id < 0) best_id = 0;               // fallback
            y_a[i] = best_id;
        }

        py::array_t<int> ya_out(y_a.size());
        std::memcpy(ya_out.mutable_data(), y_a.data(), sizeof(int) * y_a.size());
        return ya_out;
    }

private:
    /* ───── hyper‑parameters ───── */
    double rho_, alpha_, beta_;

    /* ───── state ───── */
    int    dim_original_;
    std::vector<Cluster>           clusters_;

    /* ─────────────────────────────────────────────────── */

    // fuzzy  AND L1
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
    int step_fit(const std::vector<double>& sample)
    {

        // first ever cluster?
        if (clusters_.empty()) {
            clusters_.push_back({sample});
            return 0;
        }

        const std::size_t K = clusters_.size();
        std::vector<double> T(K), M(K);

        for (std::size_t k = 0; k < K; ++k) {
            T[k] = category_choice(sample.data(), clusters_[k].weight);
            M[k] = match(sample.data(), clusters_[k].weight);
        }

        // Build candidate order once:
        // primary = descending T, secondary = ascending index
        std::vector<int> order;
        order.reserve(K);
        for (std::size_t k = 0; k < K; ++k) {
            order.push_back(static_cast<int>(k));
        }

        std::sort(order.begin(), order.end(),
                  [&](int a, int b) {
                      const double Ta = T[a];
                      const double Tb = T[b];
                      if (Ta != Tb) return Ta > Tb;   // descending T
                      return a < b;                   // ascending index
                  });

        for (int best : order) {
            if (M[best] < rho_) {
                continue; // fails vigilance -> discard and continue
            }

            auto new_w = update_weight(sample, clusters_[best].weight);
            clusters_[best].weight = new_w;

            return best;
        }

        /* create new */
        int new_id = static_cast<int>(clusters_.size());
        clusters_.push_back({sample});
        return new_id;
    }

};

// ────────────────────────────────────────────────────────────
//  convenience free‑functions  (fit / predict)
// ────────────────────────────────────────────────────────────
auto FitFuzzyART(py::array_t<double>  X,
                 double               rho,
                 double               alpha,
                 double               beta,
                 py::object           weights = py::none())
{
    cppFuzzyART model(rho, alpha, beta, weights);
    return model.fit(X);
}

auto PredictFuzzyART(py::array_t<double> X,
                     double              rho,
                     double              alpha,
                     double              beta,
                     py::object          weights = py::none())
{
    cppFuzzyART model(rho, alpha, beta, weights);
    return model.predict(X);
}

// ────────────────────────────────────────────────────────────
//  pybind11 module
// ────────────────────────────────────────────────────────────
PYBIND11_MODULE(cppFuzzyART, m) {
    py::class_<cppFuzzyART>(m, "cppFuzzyART")
        .def(py::init<double,double,double,py::object>(),
             py::arg("rho"), py::arg("alpha"), py::arg("beta"),
             py::arg("weights")        = py::none())
        .def("fit",    &cppFuzzyART::fit,    py::arg("X"))
        .def("predict",&cppFuzzyART::predict,py::arg("X"))
        .def("__repr__", [](const cppFuzzyART&){ return "<cppFuzzyART>"; });

    m.def("FitFuzzyART",    &FitFuzzyART,
          py::arg("X"),
          py::arg("rho"), py::arg("alpha"), py::arg("beta"),
          py::arg("weights")=py::none());

    m.def("PredictFuzzyART",&PredictFuzzyART,
          py::arg("X"),
          py::arg("rho"), py::arg("alpha"), py::arg("beta"),
          py::arg("weights")=py::none());
}
