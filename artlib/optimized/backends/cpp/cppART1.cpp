// cppART1.cpp
// ----------------------------------------------------------
//  C++ accelerated ART1 (pybind11)
//  - supports incremental training via external weights
//  - binary inputs in {0,1} (NO complement coding)
//  - weights stored as length 2*dim: [w_bu (float), w_td (binary 0/1)]
//  - parameters: rho (vigilance), L (uncommitted node bias, >= 1)
// ----------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace py = pybind11;

class cppART1 {
public:
    struct Cluster {
        std::vector<double> weight; // size 2*dim: [bu..., td...]
    };

    cppART1(double rho,
            double L,
            py::object weights = py::none())
        : rho_(rho), L_(L), dim_(0)
    {
        if (rho_ < 0.0 || rho_ > 1.0) throw std::runtime_error("rho must be in [0,1]");
        if (L_ < 1.0) throw std::runtime_error("L must be >= 1");

        const bool have_W = !weights.is_none();
        if (have_W) {
            py::list w_list = weights.cast<py::list>();
            const std::size_t n_clusters = w_list.size();

            clusters_.resize(n_clusters);

            for (std::size_t k = 0; k < n_clusters; ++k) {
                py::array_t<double> w_arr = w_list[k].cast<py::array_t<double>>();
                auto wb = w_arr.request();
                if (wb.ndim != 1) throw std::runtime_error("each weight must be 1-D");

                const int len = static_cast<int>(wb.shape[0]);
                if (len % 2 != 0) throw std::runtime_error("weight length must be even (2*dim)");

                clusters_[k].weight.resize(static_cast<std::size_t>(len));
                std::memcpy(clusters_[k].weight.data(), wb.ptr, sizeof(double) * len);

                if (dim_ == 0) dim_ = len / 2;
                if (len != 2 * dim_) throw std::runtime_error("inconsistent weight dimension across clusters");
            }
        }
    }

    // FIT: returns (labels, weights_vec<ndarray>)
    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>>
    fit(py::array_t<double> X)
    {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D [n_samples, n_features]");

        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0) dim_ = n_features;
        if (n_features != dim_) throw std::runtime_error("X feature dimension mismatch with existing model");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> labels(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const double* row = Xptr + i * n_features;

            // validate binary {0,1}
            for (int j = 0; j < dim_; ++j) {
                const double v = row[j];
                if (!(v == 0.0 || v == 1.0)) {
                    throw std::runtime_error("ART1 requires binary inputs in {0,1}");
                }
            }

            // copy sample into uint8-like semantics (still doubles for speed/simple)
            std::vector<double> sample(row, row + dim_);
            labels[i] = step_fit(sample);
        }

        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), sizeof(int) * labels.size());

        std::vector<py::array_t<double>> weight_out;
        weight_out.reserve(clusters_.size());
        for (const auto& c : clusters_) {
            py::array_t<double> w(c.weight.size());
            std::memcpy(w.mutable_data(), c.weight.data(), sizeof(double) * c.weight.size());
            weight_out.push_back(std::move(w));
        }

        return {labels_out, weight_out};
    }

    // PREDICT: returns y
    py::array_t<int>
    predict(py::array_t<double> X)
    {
        if (clusters_.empty()) throw std::runtime_error("Model has no clusters");

        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D [n_samples, n_features]");

        const int n_samples  = static_cast<int>(xb.shape[0]);
        const int n_features = static_cast<int>(xb.shape[1]);

        if (dim_ == 0) dim_ = n_features;
        if (n_features != dim_) throw std::runtime_error("X feature dimension mismatch with model");

        const double* Xptr = static_cast<const double*>(xb.ptr);

        std::vector<int> y(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            const double* row = Xptr + i * dim_;

            // validate binary
            for (int j = 0; j < dim_; ++j) {
                const double v = row[j];
                if (!(v == 0.0 || v == 1.0)) {
                    throw std::runtime_error("ART1 requires binary inputs in {0,1}");
                }
            }

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
            y[i] = best_id;
        }

        py::array_t<int> y_out(y.size());
        std::memcpy(y_out.mutable_data(), y.data(), sizeof(int) * y.size());
        return y_out;
    }

private:
    double rho_;
    double L_;
    int dim_;
    std::vector<Cluster> clusters_;

    // category choice: dot(i, w_bu)
    double category_choice(const double* sample, const std::vector<double>& w) const
    {
        double s = 0.0;
        // w_bu is first dim_
        for (int j = 0; j < dim_; ++j) {
            s += sample[j] * w[j];
        }
        return s;
    }

    // match criterion: count_nonzero(i & w_td) / dim
    double match(const double* sample, const std::vector<double>& w) const
    {
        int count = 0;
        // w_td is second dim_
        for (int j = 0; j < dim_; ++j) {
            const int i_bit  = (sample[j] != 0.0);
            const int td_bit = (w[dim_ + j] != 0.0);
            if (i_bit & td_bit) ++count;
        }
        return static_cast<double>(count) / static_cast<double>(dim_);
    }

    // update rule:
    // w_td_new = i & w_td
    // count = nnz(w_td_new)
    // w_bu_new = (L / (L - 1 + count)) * w_td_new
    std::vector<double> update_weight(const double* sample, const std::vector<double>& w) const
    {
        std::vector<double> out(2 * dim_, 0.0);

        // compute new top-down, count nnz
        int count = 0;
        for (int j = 0; j < dim_; ++j) {
            const int td_new = ((sample[j] != 0.0) & (w[dim_ + j] != 0.0));
            out[dim_ + j] = static_cast<double>(td_new);
            if (td_new) ++count;
        }

        // scaling for bottom-up
        const double denom = (L_ - 1.0 + static_cast<double>(count));
        const double sf = (denom > 0.0) ? (L_ / denom) : 0.0;

        for (int j = 0; j < dim_; ++j) {
            out[j] = sf * out[dim_ + j];
        }

        return out;
    }

    // new cluster weight:
    // w_td = i
    // w_bu = (L / (L - 1 + dim)) * i
    std::vector<double> new_weight(const double* sample) const
    {
        std::vector<double> w(2 * dim_, 0.0);

        const double sf = L_ / (L_ - 1.0 + static_cast<double>(dim_));

        for (int j = 0; j < dim_; ++j) {
            const double bit = (sample[j] != 0.0) ? 1.0 : 0.0;
            w[dim_ + j] = bit;      // top-down
            w[j]        = sf * bit; // bottom-up
        }

        return w;
    }

    int step_fit(const std::vector<double>& sample_vec)
    {
        const double* sample = sample_vec.data();

        // first ever cluster?
        if (clusters_.empty()) {
            clusters_.push_back({ new_weight(sample) });
            return 0;
        }

        const std::size_t K = clusters_.size();
        std::vector<double> T(K), M(K);

        for (std::size_t k = 0; k < K; ++k) {
            T[k] = category_choice(sample, clusters_[k].weight);
            M[k] = match(sample, clusters_[k].weight);
        }

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

        // resonance search
        for (int best : order) {
            if (M[best] < rho_) continue;

            clusters_[best].weight = update_weight(sample, clusters_[best].weight);
            return best;
        }

        // no resonant category -> create new
        const int new_id = static_cast<int>(clusters_.size());
        clusters_.push_back({ new_weight(sample) });
        return new_id;
    }
};

// convenience wrappers
auto FitART1(py::array_t<double> X,
             double rho,
             double L,
             py::object weights = py::none())
{
    cppART1 model(rho, L, weights);
    return model.fit(X);
}

auto PredictART1(py::array_t<double> X,
                 double rho,
                 double L,
                 py::object weights = py::none())
{
    cppART1 model(rho, L, weights);
    return model.predict(X);
}

// pybind11 module
PYBIND11_MODULE(cppART1, m) {
    py::class_<cppART1>(m, "cppART1")
        .def(py::init<double,double,py::object>(),
             py::arg("rho"), py::arg("L"),
             py::arg("weights") = py::none())
        .def("fit", &cppART1::fit, py::arg("X"))
        .def("predict", &cppART1::predict, py::arg("X"))
        .def("__repr__", [](const cppART1&) { return "<cppART1>"; });

    m.def("FitART1", &FitART1,
          py::arg("X"),
          py::arg("rho"), py::arg("L"),
          py::arg("weights") = py::none());

    m.def("PredictART1", &PredictART1,
          py::arg("X"),
          py::arg("rho"), py::arg("L"),
          py::arg("weights") = py::none());
}
