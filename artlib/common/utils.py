"""General utilities used throughout ARTLib."""
import numpy as np
from numba import njit
from typing import Tuple, Optional, Mapping, Sequence, Union, Any
from numpy.typing import ArrayLike, NDArray
from artlib.optimized.backends.cpp.fracsort import fracsort as _fracsort
from artlib.optimized.backends.cpp.fracsort import fracargmax as _fracargmax

IndexableOrKeyable = Union[Mapping[Any, Any], Sequence[Any]]


def normalize(
    data: np.ndarray,
    d_max: Optional[np.ndarray] = None,
    d_min: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data column-wise between 0 and 1.

    Parameters
    ----------
    data : np.ndarray
        2D array of dataset (rows = samples, columns = features).
    d_max : np.ndarray, optional
        Maximum values for each column.
    d_min : np.ndarray, optional
        Minimum values for each column.

    Returns
    -------
    np.ndarray
        Normalized data.
    np.ndarray
        Maximum values for each column.
    np.ndarray
        Minimum values for each column.

    """
    if d_min is None:
        d_min = np.min(data, axis=0)
    if d_max is None:
        d_max = np.max(data, axis=0)

    # Avoid division by zero
    range_vals = d_max - d_min
    mask = range_vals == 0  # Identify columns where d_max == d_min

    # Normalize safely
    normalized = np.zeros_like(data, dtype=np.float64)  # Default all to zero
    normalized[:, ~mask] = (data[:, ~mask] - d_min[~mask]) / range_vals[~mask]
    return normalized, d_max, d_min


def de_normalize(data: np.ndarray, d_max: np.ndarray, d_min: np.ndarray) -> np.ndarray:
    """Restore column-wise normalized data to original scale.

    Parameters
    ----------
    data : np.ndarray
        Normalized data.
    d_max : np.ndarray
        Maximum values for each column.
    d_min : np.ndarray
        Minimum values for each column.

    Returns
    -------
    np.ndarray
        De-normalized data.

    """
    return data * (d_max - d_min) + d_min


def complement_code(data: np.ndarray) -> np.ndarray:
    """Complement code the data.

    Parameters
    ----------
    data : np.ndarray
        Dataset.

    Returns
    -------
    np.ndarray
        complement coded data.

    """
    cc_data = np.hstack([data, 1.0 - data])
    return cc_data


def de_complement_code(data: np.ndarray) -> np.ndarray:
    """Find the centroid of complement coded data.

    Parameters
    ----------
    data : np.ndarray
        Dataset.

    Returns
    -------
    np.ndarray
        De-complement coded data.

    """
    # Get the shape of the array
    n, total_columns = data.shape

    # Ensure the number of columns is even so that it can be split evenly
    assert total_columns % 2 == 0, "The number of columns must be even"

    # Calculate the number of columns in each resulting array
    m = total_columns // 2

    # Split the array into two arrays of shape n x m
    arr1 = data[:, :m]
    arr2 = 1 - data[:, m:]

    # Find the element-wise mean
    mean_array = (arr1 + arr2) / 2

    return mean_array


@njit
def l1norm(x: np.ndarray) -> float:
    """Get the L1 norm of a vector using Numba.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    float
        L1 norm.

    """
    return np.sum(np.abs(x))  # np.absolute is the same as np.abs


def l2norm2(data: np.ndarray) -> float:
    """Get the squared L2 norm of a vector.

    Parameters
    ----------
    data : np.ndarray
        Input vector.

    Returns
    -------
    float
        Squared L2 norm.

    """
    return float(np.matmul(data, data))


@njit
def fuzzy_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get the fuzzy AND operation between two vectors.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    np.ndarray
        Fuzzy AND result.

    """
    return np.minimum(x, y)


def fracsort(num: ArrayLike, den: ArrayLike) -> NDArray[np.intp]:
    """Get argsort indices for elementwise fractions ``num[i] / den[i]`` without
    division.

    This function returns an index array that sorts the rational values exactly using
    cross-multiplication in a compiled C++ backend (no division is performed). Ties
    are broken by the lowest original index.

    Parameters
    ----------
    num : ArrayLike
        1D array-like of nonnegative numerators. Must be convertible to a contiguous
        NumPy array with dtype ``np.uint32`` or ``np.uint64``.
    den : ArrayLike
        1D array-like of denominators with ``den[i] >= 1``. Must be convertible to a
        contiguous NumPy array with dtype ``np.uint32`` or ``np.uint64`` and have the
        same shape and dtype as ``num``.

    Returns
    -------
    NDArray[np.intp]
        Indices that sort ``num[i] / den[i]`` in ascending order, with ties broken by
        the lowest index.

    """
    return _fracsort(num, den)


def fracargmax(num: ArrayLike, den: ArrayLike) -> np.intp:
    """Get the index that maximizes the elementwise fractions ``num[i] / den[i]``
    without division.

    This function returns the index of the maximum rational value exactly using
    cross-multiplication in a compiled C++ backend (no division is performed). Ties
    are broken first by the larger denominator, then by the lowest original index.

    Parameters
    ----------
    num : ArrayLike
        1D array-like of nonnegative numerators. Must be convertible to a contiguous
        NumPy array with dtype ``np.uint32`` or ``np.uint64``.
    den : ArrayLike
        1D array-like of denominators with ``den[i] >= 1``. Must be convertible to a
        contiguous NumPy array with dtype ``np.uint32`` or ``np.uint64`` and have the
        same shape and dtype as ``num``.

    Returns
    -------
    np.intp
        Index ``i`` that maximizes ``num[i] / den[i]`` (descending). Ties are broken
        by larger denominator first, then the lowest index.

    """
    return _fracargmax(num, den)


def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of
            samples and m is the number of features.
        n_bits (int): Number of bits to use for thermometer encoding.

    Returns:
        np.ndarray: A thermometer-coded representation of the input data with
            shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize to [0, 1]
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    # Quantize into `n_bits` levels (instead of `2^n_bits` levels)
    quantized_data = np.floor(normalized_data * n_bits).astype(int)

    # Generate thermometer encoding: fill from left to right
    thermometer_encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n_bits):
        thermometer_encoded[:, :, i] = (quantized_data > i).astype(np.uint8)

    return thermometer_encoded.reshape(n, m * n_bits)
