import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.ART1 import ART1 as pyART1
from artlib.optimized.backends.cpp.ART1 import ART1 as cppART1
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
    params = {"rho": 0.8, "L": 1.0}
    A = pyART1(**params)
    B = cppART1(**params)

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
    params = {"rho":0.8, "L":1.0}
    A = pyART1(**params)
    B = cppART1(**params)

    X = A.prepare_data(data)

    A = A.fit(X)
    B = B.fit(X)

    assert np.array_equal(A.W, B.W)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)