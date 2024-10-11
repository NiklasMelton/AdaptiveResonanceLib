import pytest
import numpy as np
from artlib.topological.TopoART import TopoART
from artlib.elementary.FuzzyART import FuzzyART

@pytest.fixture
def topoart_model():
    base_module = FuzzyART(0.5, 0.01, 1.0)
    return TopoART(base_module, beta_lower=0.5, tau=10, phi=5)

def test_initialization(topoart_model):
    # Test that the model initializes correctly
    assert isinstance(topoart_model.base_module, FuzzyART)
    assert topoart_model.params["beta_lower"] == 0.5
    assert topoart_model.params["tau"] == 10
    assert topoart_model.params["phi"] == 5

def test_validate_params():
    # Test the validate_params method
    valid_params = {"beta": 0.8, "beta_lower": 0.5, "tau": 10, "phi": 5}
    TopoART.validate_params(valid_params)

    invalid_params = {"beta": 0.4, "beta_lower": 0.5, "tau": 10, "phi": 5}  # beta must be >= beta_lower
    with pytest.raises(AssertionError):
        TopoART.validate_params(invalid_params)

def test_get_cluster_centers(topoart_model):
    # Test the get_cluster_centers method
    topoart_model.base_module.W = [np.array([0.5, 1.0, 0.5, 1.0]), np.array([0.1, 0.4, 0.5, 0.4])]
    topoart_model.base_module.d_min_ = np.array([0.0, 0.0])
    topoart_model.base_module.d_max_ = np.array([1.0, 1.0])

    centers = topoart_model.get_cluster_centers()
    print(centers)
    assert len(centers) == 2
    assert np.allclose(centers[0], np.array([0.5, 0.5]))
    assert np.allclose(centers[1], np.array([0.3, 0.5]))

def test_prepare_and_restore_data(topoart_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 2)

    X_prep = topoart_model.prepare_data(X)

    X_restored = topoart_model.restore_data(X_prep)
    assert np.allclose(X_restored, X)

def test_step_fit(topoart_model):
    # Test the step_fit method with base_module's internal methods
    X = np.random.rand(10, 2)
    X_prep = topoart_model.prepare_data(X)
    topoart_model.validate_data(X_prep)

    topoart_model.base_module.W = []
    label = topoart_model.step_fit(X_prep[0,:])

    assert isinstance(label, int)  # Ensure the result is an integer cluster label
    assert label == 0  # First label should be 0

    # Add more data and check the adjacency matrix and labels
    for i in range(1, 10):
        label = topoart_model.step_fit(X_prep[i,:])
        assert isinstance(label, int)

def test_adjacency_matrix(topoart_model):
    # Test that the adjacency matrix updates correctly
    np.random.seed(42)
    X = np.random.rand(10, 2)
    X_prep = topoart_model.prepare_data(X)
    topoart_model.validate_data(X_prep)

    topoart_model.base_module.W = []
    topoart_model.step_fit(X_prep[0,:])
    assert topoart_model.adjacency.shape == (1, 1)

    topoart_model.step_fit(X_prep[1,:])
    assert topoart_model.adjacency.shape == (1, 1)

    # Add more data and check the adjacency matrix
    topoart_model.step_fit(X_prep[2,:])
    assert topoart_model.adjacency.shape == (2, 2)

def test_prune(topoart_model):
    # Test the pruning mechanism
    np.random.seed(42)
    X = np.random.rand(10, 2)
    topoart_model.base_module.W = [np.random.rand(2) for _ in range(5)]
    topoart_model.weight_sample_counter_ = [2, 6, 6, 20, 25]  # Sample counter for pruning
    topoart_model._permanent_mask = np.zeros((5,), dtype=bool)
    topoart_model.adjacency = np.random.randint(0,10, (5,5))
    topoart_model.labels_ = np.random.randint(0,5,(10,))

    topoart_model.prune(X)
    assert len(topoart_model.W) == 4  # W should have 4 remaining weights after pruning
