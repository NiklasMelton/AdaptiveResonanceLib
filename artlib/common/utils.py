import numpy as np
from typing import Tuple, Optional


def normalize(data: np.ndarray, d_max: Optional[np.ndarray] = None, d_min: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data column-wise between 0 and 1.

    Parameters:
    - data: 2D array of data set (rows = samples, columns = features)
    - d_max: Optional, maximum values for each column
    - d_min: Optional, minimum values for each column

    Returns:
    - normalized: normalized data
    - d_max: maximum values for each column
    - d_min: minimum values for each column
    """
    if d_min is None:
        d_min = np.min(data, axis=0)
    if d_max is None:
        d_max = np.max(data, axis=0)

    normalized = (data - d_min) / (d_max - d_min)
    return normalized, d_max, d_min


def de_normalize(data: np.ndarray, d_max: np.ndarray, d_min: np.ndarray) -> np.ndarray:
    """
    Restore column-wise normalized data to original scale.

    Parameters:
    - data: normalized data
    - d_max: maximum values for each column
    - d_min: minimum values for each column

    Returns:
    - De-normalized data
    """
    return data * (d_max - d_min) + d_min

def compliment_code(data: np.ndarray) -> np.ndarray:
    """
    compliment code data

    Parameters:
    - data: data set

    Returns:
        compliment coded data
    """
    cc_data = np.hstack([data, 1.0-data])
    return cc_data

def de_compliment_code(data: np.ndarray) -> np.ndarray:
    """
    finds centroid of compliment coded data

    Parameters:
    - data: data set

    Returns:
        compliment coded data
    """
    # Get the shape of the array
    n, total_columns = data.shape

    # Ensure the number of columns is even so that it can be split evenly
    assert total_columns % 2 == 0, "The number of columns must be even"

    # Calculate the number of columns in each resulting array
    m = total_columns // 2

    # Split the array into two arrays of shape n x m
    arr1 = data[:, :m]
    arr2 = 1-data[:, m:]

    # Find the element-wise mean
    mean_array = (arr1 + arr2) / 2

    return mean_array

def l1norm(x: np.ndarray) -> float:
    """
    get l1 norm of a vector

    Parameters:
    - x: some vector

    Returns:
        l1 norm
    """
    return float(np.sum(np.absolute(x)))

def l2norm2(data: np.ndarray) -> float:
    """
    get (l2 norm)^2 of a vector

    Parameters:
    - x: some vector

    Returns:
        (l2 norm)^2
    """
    return float(np.matmul(data, data))

def fuzzy_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    get the fuzzy AND operation between two vectors

    Parameters:
    - a: some vector
    - b: some vector

    Returns:
        Fuzzy AND result

    """
    return np.minimum(x, y)
