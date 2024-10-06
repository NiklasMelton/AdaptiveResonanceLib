import pytest
import numpy as np
from artlib.elementary.EllipsoidART import EllipsoidART

# Fixture to initialize an EllipsoidART instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    alpha = 1e-5
    beta = 0.1
    mu = 0.5
    r_hat = 1.0
    return EllipsoidART(rho=rho, alpha=alpha, beta=beta, mu=mu, r_hat=r_hat)

def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params['rho'] == 0.7
    assert art_model.params['alpha'] == 1e-5
    assert art_model.params['beta'] == 0.1
    assert art_model.params['mu'] == 0.5
    assert art_model.params['r_hat'] == 1.0

def test_validate_params():
    # Test the validate_params method
    valid_params = {
        "rho": 0.7,
        "alpha": 1e-5,
        "beta": 0.1,
        "mu": 0.5,
        "r_hat": 1.0
    }
    EllipsoidART.validate_params(valid_params)

    invalid_params = {
        "rho": 1.5,  # Invalid vigilance parameter
        "alpha": -1e-5,  # Invalid alpha
        "beta": 1.1,  # Invalid beta
        "mu": -0.5,  # Invalid mu
        "r_hat": -1.0  # Invalid r_hat
    }
    with pytest.raises(AssertionError):
        EllipsoidART.validate_params(invalid_params)

def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 0.5, 0.0, 0.0])  # Mock weight (centroid, major axis, and radius)
    params = {
        "rho": 0.7,
        "alpha": 1e-5,
        "mu": 0.5,
        "r_hat": 1.0
    }

    activation, cache = art_model.category_choice(i, w, params)
    assert 'dist' in cache
    assert isinstance(activation, float)

def test_match_criterion(art_model):
    # Test the match_criterion method
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 0.5, 0.0, 0.0])  # Mock weight (centroid, major axis, and radius)
    params = {"rho": 0.7, "r_hat": 1.0}
    cache = {"dist": 0.6}

    match_criterion, new_cache = art_model.match_criterion(i, w, params, cache=cache)
    expected_match_criterion = 1 - (0.0 + max(0.0, 0.6)) / 1.0
    assert match_criterion == pytest.approx(expected_match_criterion, rel=1e-6)


def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35, 0.5, 0.0, 0.0])  # Mock weight (centroid, major axis, and radius)
    params = {"beta": 0.1, "mu": 0.5, "r_hat": 1.0}
    cache = {"dist": 0.6}

    updated_weight = art_model.update(i, w, params, cache=cache)
    assert updated_weight[-1] >= w[-1]  # Ensure that the radius has not decreased


def test_new_weight(art_model):
    # Test the new_weight method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    params = {"r_hat": 1.0}

    new_weight = art_model.new_weight(i, params)
    assert len(new_weight) == 5  # Centroid (2), major axis (2), and radius (1)
    assert new_weight[-1] == 0.0  # Initial radius should be 0

def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.dim_ = 2
    art_model.W = [np.array([0.2, 0.3, 0.0, 0.0, 0.5])]
    centers = art_model.get_cluster_centers()
    assert len(centers) == 1
    assert np.array_equal(centers[0], np.array([0.2, 0.3]))

def test_fit(art_model):
    # Test fitting the model
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    X = art_model.prepare_data(X)
    art_model.fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were created

def test_partial_fit(art_model):
    # Test partial_fit method
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.partial_fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were partially fit

def test_predict(art_model):
    # Test predict method
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.W = [np.array([0.1, 0.2, 0.0, 0.0, 0.5])]

    labels = art_model.predict(X)
    assert len(labels) == 2
