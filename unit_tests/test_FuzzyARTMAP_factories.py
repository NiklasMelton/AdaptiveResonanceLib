from artlib.optimized.FuzzyARTMAP import FuzzyARTMAP
import numpy as np
def test_fuzzy_artmap_factories():
    RHO, ALPHA, BETA = 0.8, 1e-10, 1.0
    m1 = FuzzyARTMAP(RHO, ALPHA, BETA, backend="python")
    m2 = FuzzyARTMAP(RHO, ALPHA, BETA, backend="torch")
    m3 = FuzzyARTMAP(RHO, ALPHA, BETA, backend="c++")

    x1 = m1.prepare_data(X)
    x2 = m2.prepare_data(X)
    x3 = m3.prepare_data(X)

    assert np.all(np.isclose(x1, x2)), ("Torch prepared data doesnt match python "
                                        "prepared data")
    assert np.all(np.isclose(x1, x3)), ("c++ prepared data doesnt match python "
                                        "prepared data")

    x1_train = x1[:n_train]
    x2_train = x2[:n_train]
    x3_train = x3[:n_train]

    x1_test = x1[n_train:]
    x2_test = x2[n_train:]
    x3_test = x3[n_train:]

    y_train = y[:n_train]
    y_test = y[n_train:]

    m1 = m1.fit(x1_train, y_train)
    m2 = m1.fit(x2_train, y_train)
    m3 = m1.fit(x3_train, y_train)

    W1 = np.vstack(m1.W)
    W2 = np.vstack(m2.W)
    W3 = np.vstack(m3.W)
    assert np.all(np.isclose(W1,W2)), "Torch weights dont match python weights."
    assert np.all(np.isclose(W1,W3)), "C++ weights dont match python weights."

    y1 = m1.predict(x1_test)
    y2 = m1.predict(x2_test)
    y3 = m1.predict(x3_test)

    assert np.all(y1==y2), "Torch predictions dont match python predictions."
    assert np.all(y1==y3), "C++ predictions dont match python predictions."
