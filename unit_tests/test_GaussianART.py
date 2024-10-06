import pytest
import numpy as np
from unittest.mock import MagicMock
from artlib.elementary.GaussianART import GaussianART

# Fixture to initialize a GaussianART instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    sigma_init = np.array([0.5, 0.5])
    alpha = 1e-5
    return GaussianART(rho=rho, sigma_init=sigma_init, alpha=alpha)

def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params['rho'] == 0.7
    assert np.array_equal(art_model.params['sigma_init'], np.array([0.5, 0.5]))
    assert art_model.params['alpha'] == 1e-5
    assert art_model.sample_counter_ == 0
    assert art_model.weight_sample_counter_ == []

def test_validate_params():
    # Test the validate_params method
    valid_params = {
        "rho": 0.5,
        "sigma_init": np.array([0.5, 0.5]),
        "alpha": 1e-5
    }
    GaussianART.validate_params(valid_params)

    invalid_params = {
        "rho": 1.5,  # Invalid vigilance parameter
        "sigma_init": np.array([0.5, 0.5]),
        "alpha": -1e-5  # Invalid alpha
    }
    with pytest.raises(AssertionError):
        GaussianART.validate_params(invalid_params)

def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 2.0, 2.5, 0.5, 0.6, 1.2, 1.0, 5])  # Mock weight vector
    art_model.W = [w]
    params = {
        "rho": 0.7,
        "alpha": 1e-5
    }

    activation, cache = art_model.category_choice(i, w, params)
    assert 'exp_dist_sig_dist' in cache
    assert isinstance(activation, float)

def test_match_criterion(art_model):
    # Test the match_criterion method
    cache = {"exp_dist_sig_dist": 0.8}
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 2.0, 2.5, 0.5, 0.6, 1.2, 1.0, 5])  # Mock weight vector
    params = {"rho": 0.7}

    match_criterion, new_cache = art_model.match_criterion(i, w, params, cache=cache)
    assert match_criterion == cache["exp_dist_sig_dist"]

def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 2.0, 2.5, 0.5, 0.6, 1.2, 1.0, 5])  # Mock weight vector
    params = {"alpha": 1e-5}

    updated_weight = art_model.update(i, w, params)
    assert updated_weight[-1] == 6  # Check that the sample count has been updated

def test_new_weight(art_model):
    # Test the new_weight method
    i = np.array([0.2, 0.3])
    params = {"sigma_init": np.array([0.5, 0.5])}

    new_weight = art_model.new_weight(i, params)
    assert len(new_weight) == 8  # Mean, sigma, inverse sigma, determinant, and count

def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.W = [np.array([0.2, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 5])]
    art_model.dim_ = 2
    centers = art_model.get_cluster_centers()
    assert len(centers) == 1
    assert np.array_equal(centers[0], np.array([0.2, 0.3]))


def test_fit(art_model):
    # Test fitting the model
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    X = art_model.prepare_data(X)
    art_model.fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were created

def test_partial_fit(art_model):
    # Test partial_fit method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.partial_fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were partially fit

def test_predict(art_model):
    # Test predict method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.W = [np.array([0.1, 0.2, 0.5, 0.5, 2.0, 2.0, 2.0, 1])]

    labels = art_model.predict(X)
    assert len(labels) == 2
