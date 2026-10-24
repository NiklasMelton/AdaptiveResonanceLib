import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import BinaryFuzzyARTMAP
from artlib.common.utils import binarize_features_thermometer


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.9}
    A = SimpleARTMAP(BinaryFuzzyART(**params))
    B = BinaryFuzzyARTMAP(**params)

    data = binarize_features_thermometer(data, n_bits=4).astype(np.bool)

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

    params = {"rho": 0.9}
    A = SimpleARTMAP(BinaryFuzzyART(**params))
    B = BinaryFuzzyARTMAP(**params)

    data = binarize_features_thermometer(data, n_bits=4).astype(np.bool)
    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    assert np.array_equal(A.module_a.W, B.module_a.W)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)

def test_binary_fuzzy_art_prepare_data_and_clone():
    from sklearn.base import clone

    model = BinaryFuzzyART(rho=0.5)
    X = model.prepare_data(np.array([[0, 1], [1, 0], [1, 1]]))
    assert X.dtype == np.bool_
    model.fit(X)
    assert model.get_params() == {"rho": 0.5}
    copy = clone(model)
    assert isinstance(copy, BinaryFuzzyART)
    assert copy.params["rho"] == 0.5
