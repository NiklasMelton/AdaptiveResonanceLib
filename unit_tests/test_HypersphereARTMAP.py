import numpy as np
from sklearn.datasets import make_blobs
from artlib.elementary.HypersphereART import HypersphereART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.optimized.backends.cpp.HypersphereARTMAP import HypersphereARTMAP


def test_prepare_data():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    params = {"rho": 0.7, "alpha": 1e-10, "beta": 1.0, "r_hat": 0.8}
    A = SimpleARTMAP(HypersphereART(**params))
    B = HypersphereARTMAP(**params)

    X_A = A.prepare_data(data)
    X_B = B.prepare_data(data)
    np.testing.assert_allclose(X_A, X_B, rtol=1e-7, atol=1e-9)



def test_consistency():
    data, target = make_blobs(
            n_samples=1500,
            centers=3,
            cluster_std=0.50,
            random_state=0,
            shuffle=False,
        )

    params = {"rho": 0.7, "alpha": 1e-10, "beta": 1.0, "r_hat": 0.8}
    A = SimpleARTMAP(HypersphereART(**params))
    B = HypersphereARTMAP(**params)

    X = A.prepare_data(data)

    A = A.fit(X, target)
    B = B.fit(X, target)

    np.testing.assert_allclose(A.module_a.W, B.module_a.W, rtol=1e-7, atol=1e-9)

    y_A = A.labels_
    y_B = B.labels_

    assert np.array_equal(y_A, y_B)

def test_zero_alpha_sample_on_centroid_matches_python():
    X = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [0.8, 0.8], [0.25, 0.2]])
    y = np.array([0, 0, 1, 1, 0])
    params = {"rho": 0.5, "alpha": 0.0, "beta": 1.0, "r_hat": 1.0}
    A = SimpleARTMAP(HypersphereART(**params))
    B = HypersphereARTMAP(**params)
    A.fit(X, y)
    B.fit(X, y)
    assert all(np.isfinite(w).all() for w in B.module_a.W)
    assert len(A.module_a.W) == len(B.module_a.W)
    for w_a, w_b in zip(A.module_a.W, B.module_a.W):
        np.testing.assert_allclose(w_a, w_b, rtol=1e-12, atol=1e-12)
