from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory
import numpy as np
from sklearn.datasets import fetch_openml
from time import perf_counter

def _load_mnist_numpy():
    """
    Load MNIST from OpenML using scikit-learn.
    Returns:
        X_train (60000, 784) float32 in [0,1]
        y_train (60000,) int
        X_test  (10000, 784) float32 in [0,1]
        y_test  (10000,) int
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_all = mnist["data"].astype(np.float32) / 255.0
    y_all = mnist["target"].astype(int)

    # Standard MNIST split: first 60k train, last 10k test
    X_train, y_train = X_all[:10000], y_all[:10000]
    X_test, y_test   = X_all[60000:], y_all[60000:]
    return X_train, y_train, X_test, y_test


def test_fuzzy_artmap_factories(capsys):
    def time_call(label, fn, *args, **kwargs):
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        dt = perf_counter() - t0
        # Ensure this shows even with pytest's output capturing
        with capsys.disabled():
            print(f"[TIMING] {label}: {dt:.3f} s", flush=True)
        return out

    RHO, ALPHA, BETA = 0.8, 1e-10, 1.0
    m1 = FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend="python")
    m2 = FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend="torch")
    m3 = FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend="c++")

    # === MNIST loading & combine-before-prepare ===
    X_train, y_train, X_test, y_test = _load_mnist_numpy()
    X = np.vstack([X_train, X_test])          # (70000, 784)
    y = np.concatenate([y_train, y_test])     # (70000,)
    n_train = X_train.shape[0]
    # === END ===

    x1 = m1.prepare_data(X)
    x2 = m2.prepare_data(X)
    x3 = m3.prepare_data(X)

    assert np.all(np.isclose(x1, x2)), "Torch prepared data doesnt match python prepared data"
    assert np.all(np.isclose(x1, x3)), "C++ prepared data doesnt match python prepared data"

    x1_train, x1_test = x1[:n_train], x1[n_train:]
    x2_train, x2_test = x2[:n_train], x2[n_train:]
    x3_train, x3_test = x3[:n_train], x3[n_train:]
    y_train, y_test   = y[:n_train], y[n_train:]

    # --- Timed fits ---
    m1 = time_call("fit (python)", m1.fit, x1_train, y_train)
    m2 = time_call("fit (torch)",  m2.fit, x2_train, y_train)
    m3 = time_call("fit (c++)",    m3.fit, x3_train, y_train)

    W1, W2, W3 = np.vstack(m1.module_a.W), np.vstack(m2.module_a.W), np.vstack(m3.module_a.W)
    assert np.all(np.isclose(W1, W2)), "Torch weights dont match python weights."
    assert np.all(np.isclose(W1, W3)), "C++ weights dont match python weights."

    # --- Timed predicts ---
    y1 = time_call("predict (python)", m1.predict, x1_test)
    y2 = time_call("predict (torch)",  m2.predict, x2_test)
    y3 = time_call("predict (c++)",    m3.predict, x3_test)

    assert np.all(y1 == y2), "Torch predictions dont match python predictions."
    assert np.all(y1 == y3), "C++ predictions dont match python predictions."
