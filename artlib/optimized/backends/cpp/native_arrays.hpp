#pragma once

#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace art_bind {
namespace py = pybind11;

template <typename T> py::array_t<T> matrix(py::handle input, const char* name = "X") {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(input);
    if (!array || array.ndim() != 2 || array.shape(1) <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a 2-D array with features");
    }
    return array;
}

template <typename T> py::array_t<T> binary_array(py::handle input, const char* name) {
    py::array array = py::array::ensure(input);
    if (!array)
        throw std::invalid_argument(std::string(name) + " must be an array");
    const std::string kind = py::str(array.dtype().attr("kind"));
    if (kind != "b" && kind != "i" && kind != "u") {
        throw std::invalid_argument(std::string(name) + " must have bool or integer dtype");
    }
    return py::array_t<T, py::array::c_style | py::array::forcecast>(array);
}

template <typename T> py::array_t<T> binary_matrix(py::handle input) {
    auto array = binary_array<T>(input, "X");
    if (array.ndim() != 2 || array.shape(1) <= 0) {
        throw std::invalid_argument("X must be a 2-D array with features");
    }
    return array;
}

template <typename T> py::array_t<T> vector(py::handle input, const char* name) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(input);
    if (!array || array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    return array;
}

template <typename T> py::array_t<T> labels(py::handle input, py::ssize_t rows) {
    auto result = vector<T>(input, "y");
    if (result.shape(0) != rows)
        throw std::invalid_argument("X/y size mismatch");
    return result;
}

template <typename T> py::object weights(py::object input) {
    if (input.is_none())
        return input;
    if (!py::isinstance<py::iterable>(input))
        throw std::invalid_argument("weights must be an iterable of arrays");
    py::list result;
    py::ssize_t length = -1;
    for (py::handle item : input) {
        auto weight = vector<T>(item, "weight");
        if (weight.shape(0) <= 0)
            throw std::invalid_argument("weights must be non-empty");
        if (length >= 0 && weight.shape(0) != length) {
            throw std::invalid_argument("inconsistent weight dimensions");
        }
        length = weight.shape(0);
        result.append(weight);
    }
    return result;
}

template <typename T> py::object binary_weights(py::object input) {
    if (input.is_none())
        return input;
    if (!py::isinstance<py::iterable>(input))
        throw std::invalid_argument("weights must be an iterable of arrays");
    py::list result;
    py::ssize_t length = -1;
    for (py::handle item : input) {
        auto weight = binary_array<T>(item, "weight");
        if (weight.ndim() != 1 || weight.shape(0) <= 0) {
            throw std::invalid_argument("each weight must be a non-empty 1-D array");
        }
        if (length >= 0 && weight.shape(0) != length) {
            throw std::invalid_argument("inconsistent weight dimensions");
        }
        length = weight.shape(0);
        result.append(weight);
    }
    return result;
}

inline py::object cluster_labels(py::object input) {
    return input.is_none() ? input : py::object(vector<int>(input, "cluster_labels"));
}

template <typename T> py::array_t<T> output(const std::vector<T>& values) {
    py::array_t<T> result(values.size());
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return result;
}

template <typename T, typename Clusters, typename Accessor>
std::vector<py::array_t<T>> pack_weights(const Clusters& clusters, Accessor weight_of) {
    std::vector<py::array_t<T>> result;
    result.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        const auto& values = weight_of(cluster);
        py::array_t<T> array(values.size());
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if constexpr (std::is_same_v<T, Value>) {
            if (!values.empty()) {
                std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
            }
        } else {
            for (std::size_t index = 0; index < values.size(); ++index) {
                array.mutable_data()[index] = static_cast<T>(values[index]);
            }
        }
        result.push_back(std::move(array));
    }
    return result;
}

} // namespace art_bind
