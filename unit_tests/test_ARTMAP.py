import pytest
import numpy as np
from artlib.supervised.ARTMAP import ARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART

# Fixture to initialize an ARTMAP instance for testing
@pytest.fixture
def artmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.5, 0.01, 1.0)
    return ARTMAP(module_a=module_a, module_b=module_b)

def test_initialization(artmap_model):
    # Test that the model initializes correctly
    assert isinstance(artmap_model.module_a, BaseART)
    assert isinstance(artmap_model.module_b, BaseART)

def test_get_params(artmap_model):
    # Test the get_params method
    params = artmap_model.get_params()
    assert "module_a" in params
    assert "module_b" in params

def test_labels_properties(artmap_model):
    # Test the labels properties
    artmap_model.module_a.labels_ = np.array([0, 1, 1])
    artmap_model.module_b.labels_ = np.array([1, 0, 0])

    assert np.array_equal(artmap_model.labels_a, artmap_model.module_a.labels_)
    assert np.array_equal(artmap_model.labels_b, artmap_model.module_b.labels_)
    assert artmap_model.labels_ab == {"A": artmap_model.module_a.labels_, "B": artmap_model.module_b.labels_}

def test_validate_data(artmap_model):
    # Test the validate_data method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)
    X_prep, y_prep = artmap_model.prepare_data(X, y)

    artmap_model.validate_data(X_prep, y_prep)

    # Test invalid input data
    X_invalid = np.random.rand(0, 5)
    with pytest.raises(AssertionError):
        artmap_model.validate_data(X_invalid, y)

def test_prepare_and_restore_data(artmap_model):
    # Test prepare_data and restore_data methods
    np.random.seed(42)
    X = np.random.rand(5, 5)
    y = np.random.rand(5, 5)

    X_prep, y_prep = artmap_model.prepare_data(X, y)

    X_restored, y_restored = artmap_model.restore_data(X_prep, y_prep)
    assert np.allclose(X_restored, X)
    assert np.allclose(y_restored, y)

def test_fit(artmap_model):
    # Test the fit method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    # Prepare data before fitting
    X_prep, y_prep = artmap_model.prepare_data(X, y)
    artmap_model.fit(X_prep, y_prep, max_iter=1)

    assert artmap_model.module_a.labels_.shape[0] == X.shape[0]
    assert artmap_model.module_b.labels_.shape[0] == y.shape[0]

def test_partial_fit(artmap_model):
    # Test the partial_fit method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    # Prepare data before partial fitting
    X_prep, y_prep = artmap_model.prepare_data(X, y)
    artmap_model.partial_fit(X_prep, y_prep)

    assert artmap_model.module_a.labels_.shape[0] == X.shape[0]
    assert artmap_model.module_b.labels_.shape[0] == y.shape[0]

def test_predict(artmap_model):
    # Test the predict method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    # Prepare data before fitting and predicting
    X_prep, y_prep = artmap_model.prepare_data(X, y)
    artmap_model.fit(X_prep, y_prep, max_iter=1)

    predictions = artmap_model.predict(X_prep)
    assert predictions.shape[0] == X.shape[0]

def test_predict_ab(artmap_model):
    # Test the predict_ab method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    # Prepare data before fitting and predicting
    X_prep, y_prep = artmap_model.prepare_data(X, y)
    artmap_model.fit(X_prep, y_prep, max_iter=1)

    predictions_a, predictions_b = artmap_model.predict_ab(X_prep)
    assert predictions_a.shape[0] == X.shape[0]
    assert predictions_b.shape[0] == X.shape[0]

def test_predict_regression(artmap_model):
    # Test the predict_regression method
    np.random.seed(42)
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 5)

    # Prepare data before fitting and predicting regression
    X_prep, y_prep = artmap_model.prepare_data(X, y)
    artmap_model.fit(X_prep, y_prep, max_iter=1)

    regression_preds = artmap_model.predict_regression(X_prep)
    assert regression_preds.shape[0] == X.shape[0]
