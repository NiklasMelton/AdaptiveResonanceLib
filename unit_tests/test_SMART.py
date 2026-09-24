import pytest
import numpy as np
from artlib.hierarchical.SMART import SMART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART
from artlib.common.utils import complement_code
from matplotlib.axes import Axes


# Fixture to initialize a SMART instance for testing
@pytest.fixture
def smart_model():
    base_params = {"alpha": 0.01, "beta": 1.0}
    rho_values = [0.2, 0.5, 0.7]
    return SMART(FuzzyART, rho_values, base_params)


def test_initialization(smart_model):
    # Test that the model initializes correctly
    assert len(smart_model.rho_values) == 3
    assert isinstance(smart_model.modules[0], BaseART)
    assert isinstance(smart_model.modules[1], BaseART)
    assert isinstance(smart_model.modules[2], BaseART)


def test_prepare_and_restore_data(smart_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 5)

    X_prep = smart_model.prepare_data(X)

    X_restored = smart_model.restore_data(X_prep)
    assert np.allclose(X_restored, X)


def test_fit(smart_model):
    # Test the fit method
    X = np.random.rand(10, 5)

    # Prepare data before fitting
    X_prep = smart_model.prepare_data(X)
    smart_model.fit(X_prep, max_iter=1)

    assert smart_model.modules[0].labels_.shape[0] == X.shape[0]


def test_partial_fit(smart_model):
    # Test the partial_fit method
    X = np.random.rand(10, 5)

    # Prepare data before partial fitting
    X_prep = smart_model.prepare_data(X)
    print(smart_model.n_modules)
    smart_model.partial_fit(X_prep)

    assert smart_model.modules[0].labels_.shape[0] == X.shape[0]


def test_smart_inherits_layer_aware_merge_and_move():
    X = complement_code(np.array([[0.0], [0.2], [0.8], [1.0]]))
    model = SMART(FuzzyART, [0.2, 0.6, 1.0], {"alpha": 0.01, "beta": 1.0}).fit(X)

    assert model.move_prototype(2, 0, 1, 1) == 1
    np.testing.assert_array_equal(model.labels_deep_[1], [1, 1, 1])

    assert model.merge(0, 0, 1) == 0
    assert model.modules[0].n_clusters == 1
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0])
    assert model.predict(X)[0].shape == (4,)

    model.partial_fit(X[:1])
    assert model.labels_deep_.shape == (5, 3)
