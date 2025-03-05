"""General utilities used throughout ARTLib."""
import numpy as np
from numba import njit
from typing import Tuple, Optional, Mapping, Sequence, Union, Any

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
