import pytest
import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.FuzzyART import FuzzyART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.cpp_optimized.FuzzyARTMAP import FuzzyARTMAP


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.8, "alpha": 1e-10, "beta": 1.0}
    A = SimpleARTMAP(FuzzyART(**params))
    B = FuzzyARTMAP(**params)

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

    params = {"rho":0.8, "alpha":1e-10, "beta":1.0}
    A = SimpleARTMAP(FuzzyART(**params))
    B = FuzzyARTMAP(**params)

    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)