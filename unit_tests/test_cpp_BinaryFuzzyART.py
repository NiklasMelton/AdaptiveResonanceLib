import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART as pyBinaryFuzzyART
from artlib.optimized.backends.cpp.BinaryFuzzyART import BinaryFuzzyART as \
    cppBinaryFuzzyART
from artlib.common.utils import binarize_features_thermometer

def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    data = binarize_features_thermometer(data, 4).astype(np.bool)
    params = {"rho": 0.8}
    A = pyBinaryFuzzyART(**params)
    B = cppBinaryFuzzyART(**params)

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
    data = binarize_features_thermometer(data, 4).astype(np.bool)
    params = {"rho":0.8}
    A = pyBinaryFuzzyART(**params)
    B = cppBinaryFuzzyART(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert np.array_equal(A.W, B.W)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)