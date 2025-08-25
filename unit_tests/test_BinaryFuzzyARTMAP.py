import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import BinaryFuzzyARTMAP

def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Binarizes each feature in the data using thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for thermometer encoding.

    Returns:
        np.ndarray: A thermometer-coded representation of the input data with shape (n, m * n_bits).
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

def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.9, "alpha": 1e-10}
    A = SimpleARTMAP(BinaryFuzzyART(**params))
    B = BinaryFuzzyARTMAP(**params)

    data = binarize_features_thermometer(data, n_bits=4)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    assert np.array_equal(X_A, X_B)



def test_consistency():
    data, target = make_blobs(
            n_samples=1500,
            centers=3,
            cluster_std=0.50,
            random_state=0,
            shuffle=False,
        )

    params = {"rho": 0.9, "alpha": 1e-10}
    A = SimpleARTMAP(BinaryFuzzyART(**params))
    B = BinaryFuzzyARTMAP(**params)

    data = binarize_features_thermometer(data, n_bits=4)
    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    assert np.array_equal(A.module_a.W, B.module_a.W)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)