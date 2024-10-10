import pytest
import numpy as np
from typing import Optional
from artlib.biclustering.BARTMAP import BARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART

# Fixture to initialize a BARTMAP instance for testing
@pytest.fixture
def bartmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.5, 0.01, 1.0)
    return BARTMAP(module_a=module_a, module_b=module_b, eta=0.01)

def test_initialization(bartmap_model):
    # Test that the model initializes correctly
    assert bartmap_model.params["eta"] == 0.01
    assert isinstance(bartmap_model.module_a, BaseART)
    assert isinstance(bartmap_model.module_b, BaseART)

def test_validate_params():
    # Test the validate_params method
    valid_params = {"eta": 0.5}
    BARTMAP.validate_params(valid_params)

    invalid_params = {"eta": "invalid"}  # eta should be a float
    with pytest.raises(AssertionError):
        BARTMAP.validate_params(invalid_params)

def test_get_params(bartmap_model):
    # Test the get_params method
    params = bartmap_model.get_params()
    assert "eta" in params
    assert "module_a" in params
    assert "module_b" in params

def test_set_params(bartmap_model):
    # Test the set_params method
    bartmap_model.set_params(eta=0.7)
    assert bartmap_model.eta == 0.7

def test_step_fit(bartmap_model):
    # Test the step_fit method
    X = np.random.rand(10, 10)

    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)

    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    # init module A
    bartmap_model.module_a.W = []
    bartmap_model.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

    c_a = bartmap_model.step_fit(X_a, 0)
    assert isinstance(c_a, int)  # Ensure the result is an integer cluster label

def test_match_criterion_bin(bartmap_model):
    # Test the match_criterion_bin method
    X = np.random.rand(10, 10)

    bartmap_model.X = X

    X_a = bartmap_model.module_a.prepare_data(X)
    X_b = bartmap_model.module_b.prepare_data(X.T)

    bartmap_model.module_b = bartmap_model.module_b.fit(X_b, max_iter=1)

    # init module A
    bartmap_model.module_a.W = []
    bartmap_model.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
    c_a = bartmap_model.step_fit(X_a, 0)

    result = bartmap_model.match_criterion_bin(X, 9, 0, {"eta": 0.5})
    assert isinstance(result, bool)  # Ensure the result is a boolean

def test_fit(bartmap_model):
    # Test the fit method
    X = np.random.rand(10, 10)

    bartmap_model.fit(X, max_iter=1)

    # Check that rows_ and columns_ are set
    assert hasattr(bartmap_model, "rows_")
    assert hasattr(bartmap_model, "columns_")

    # Check that the rows and columns shapes match the expected size
    assert bartmap_model.rows_.shape[0] == bartmap_model.module_a.n_clusters * bartmap_model.module_b.n_clusters
    assert bartmap_model.columns_.shape[0] == bartmap_model.module_a.n_clusters * bartmap_model.module_b.n_clusters

