import pytest
import numpy as np
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART
from sklearn.utils.validation import NotFittedError


# Fixture to initialize a SimpleARTMAP instance for testing
@pytest.fixture
def simple_artmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    return SimpleARTMAP(module_a=module_a)


def test_initialization(simple_artmap_model):
    # Test that the model initializes correctly
    assert isinstance(simple_artmap_model.module_a, BaseART)


def test_get_params(simple_artmap_model):
    # Test the get_params method
    params = simple_artmap_model.get_params()
    assert "module_a" in params


def test_validate_data(simple_artmap_model):
    # Test the validate_data method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)
    X_prep = simple_artmap_model.prepare_data(X)

    X_valid, y_valid = simple_artmap_model.validate_data(X_prep, y)
    assert X_valid.shape == X_prep.shape
    assert y_valid.shape == y.shape

    # Test invalid input data
    X_invalid = np.random.rand(0, 5)
    with pytest.raises(ValueError):
        simple_artmap_model.validate_data(X_invalid, y)


def test_prepare_and_restore_data(simple_artmap_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 5)

    X_prep = simple_artmap_model.prepare_data(X)

    X_restored = simple_artmap_model.restore_data(X_prep)
    assert np.allclose(X_restored, X)


def test_fit(simple_artmap_model):
    # Test the fit method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.fit(X_prep, y, max_iter=1)

    assert simple_artmap_model.module_a.labels_.shape[0] == X.shape[0]


def test_partial_fit(simple_artmap_model):
    # Test the partial_fit method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before partial fitting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.partial_fit(X_prep, y)

    assert simple_artmap_model.module_a.labels_.shape[0] == X.shape[0]


def test_predict(simple_artmap_model):
    # Test the predict method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting and predicting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.fit(X_prep, y, max_iter=1)

    predictions = simple_artmap_model.predict(X_prep)
    assert predictions.shape[0] == X.shape[0]


def test_predict_ab(simple_artmap_model):
    # Test the predict_ab method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting and predicting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.fit(X_prep, y, max_iter=1)

    predictions_a, predictions_b = simple_artmap_model.predict_ab(X_prep)
    assert predictions_a.shape[0] == X.shape[0]
    assert predictions_b.shape[0] == X.shape[0]


def test_predict_not_fitted(simple_artmap_model):
    # Test that predict raises an error if the model is not fitted
    X = np.random.rand(10, 5)

    with pytest.raises(NotFittedError):
        simple_artmap_model.predict(X)


def test_step_fit(simple_artmap_model):
    # Test the step_fit method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.module_a.W = []

    # Run step_fit for the first sample
    c_a = simple_artmap_model.step_fit(X_prep[0], y[0])
    assert isinstance(c_a, int)  # Ensure the result is an integer cluster label


def test_step_pred(simple_artmap_model):
    # Test the step_pred method
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep = simple_artmap_model.prepare_data(X)
    simple_artmap_model.fit(X_prep, y, max_iter=1)

    c_a, c_b = simple_artmap_model.step_pred(X_prep[0])
    print(type(c_a), type(c_b))
    assert isinstance(c_a, (int, np.integer))
    assert isinstance(c_b, (int, np.integer))
