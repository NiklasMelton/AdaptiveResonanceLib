import pytest
import numpy as np
from artlib.elementary.ART2 import ART2A


# Fixture to initialize an ART2A instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    alpha = 0.1
    beta = 0.5
    return ART2A(rho=rho, alpha=alpha, beta=beta)


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho"] == 0.7
    assert art_model.params["alpha"] == 0.1
    assert art_model.params["beta"] == 0.5


def test_validate_params():
    # Test the validate_params method
    valid_params = {"rho": 0.7, "alpha": 0.1, "beta": 0.5}
    ART2A.validate_params(valid_params)

    invalid_params = {
        "rho": -0.7,  # Invalid vigilance parameter
        "alpha": -0.1,  # Invalid alpha
        "beta": 1.5,  # Invalid beta
    }
    with pytest.raises(AssertionError):
        ART2A.validate_params(invalid_params)


def test_check_dimensions(art_model):
    # Test the check_dimensions method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)

    assert art_model.dim_ == 2


def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35])  # Mock weight
    params = {"rho": 0.7}

    activation, cache = art_model.category_choice(i, w, params)
    assert "activation" in cache
    assert isinstance(activation, float)


def test_match_criterion(art_model):
    # Test the match_criterion method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35])  # Mock weight
    params = {"alpha": 0.1}
    cache = {"activation": 0.5}

    match_criterion, new_cache = art_model.match_criterion(
        i, w, params, cache=cache
    )
    assert (
        match_criterion == 0.5
    )  # Since activation is higher than uncommitted activation


def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array([0.25, 0.35])  # Mock weight
    params = {"beta": 0.5}
    cache = {"activation": 0.5}

    updated_weight = art_model.update(i, w, params, cache=cache)
    assert len(updated_weight) == 2  # Check that the weight is updated
    assert np.allclose(
        updated_weight, (0.5 * i + 0.5 * w)
    )  # Check the update formula


def test_new_weight(art_model):
    # Test the new_weight method
    i = np.array([0.2, 0.3])
    params = {"beta": 0.5}

    new_weight = art_model.new_weight(i, params)
    assert len(new_weight) == 2  # Check that the weight has two dimensions
    assert np.array_equal(new_weight, i)


def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.W = [np.array([0.2, 0.3]), np.array([0.4, 0.5])]
    centers = art_model.get_cluster_centers()
    assert len(centers) == 2
    assert np.array_equal(centers[0], np.array([0.2, 0.3]))
    assert np.array_equal(centers[1], np.array([0.4, 0.5]))


def test_fit(art_model):
    # Test fitting the model
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    art_model.check_dimensions(X)
    art_model.fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were created


def test_partial_fit(art_model):
    # Test partial_fit method
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)
    art_model.partial_fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were partially fit


def test_predict(art_model):
    # Test predict method
    art_model.dim_ = 2
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)
    art_model.W = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    labels = art_model.predict(X)
    assert len(labels) == 2
