"""General utilities used throughout ARTLib."""
import numpy as np
from numba import njit
from typing import Tuple, Optional, Mapping, Sequence, Union, Any

IndexableOrKeyable = Union[Mapping[Any, Any], Sequence[Any]]


@njit
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

    range_vals = d_max - d_min
    mask = (
        range_vals == 0
    )  # Identify columns with zero range to prevent division by zero

    normalized = np.empty_like(data, dtype=np.float64)  # Preallocate
    for i in range(data.shape[1]):
        if mask[i]:  # If max == min, set entire column to zero
            normalized[:, i] = 0.0
        else:
            normalized[:, i] = (data[:, i] - d_min[i]) / range_vals[i]

    return normalized, d_max, d_min


@njit
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


@njit
def compliment_code(data: np.ndarray) -> np.ndarray:
    """Compliment code the data.

    Parameters
    ----------
    data : np.ndarray
        Dataset.

    Returns
    -------
    np.ndarray
        Compliment coded data.

    """
    n, m = data.shape
    cc_data = np.empty((n, 2 * m), dtype=np.float64)

    for i in range(m):
        cc_data[:, i] = data[:, i]
        cc_data[:, i + m] = 1.0 - data[:, i]

    return cc_data


def de_compliment_code(data: np.ndarray) -> np.ndarray:
    """Find the centroid of compliment coded data with a shape assertion.

    Parameters
    ----------
    data : np.ndarray
        Dataset.

    Returns
    -------
    np.ndarray
        De-compliment coded data.

    """
    # Ensure the number of columns is even so that it can be split evenly
    if data.shape[1] % 2 != 0:
        raise ValueError("The number of columns must be even")

    return _de_compliment_code_numba(data)


@njit
def _de_compliment_code_numba(data: np.ndarray) -> np.ndarray:
    """Numba-optimized de-compliment coding function (assumes valid input)."""
    n, total_columns = data.shape
    m = total_columns // 2

    mean_array = np.empty((n, m), dtype=np.float64)
    for i in range(m):
        mean_array[:, i] = (data[:, i] + (1 - data[:, i + m])) / 2

    return mean_array


def l1norm(x: np.ndarray) -> float:
    """Get the L1 norm of a vector.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    float
        L1 norm.

    """
    return float(np.sum(np.abs(x)))


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
    return float(np.dot(data, data))


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
