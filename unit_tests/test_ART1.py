import pytest
import numpy as np
from artlib.elementary.ART1 import ART1


# Fixture to initialize an ART1 instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    L = 2.0
    return ART1(rho=rho, L=L)


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho"] == 0.7
    assert art_model.params["L"] == 2.0


def test_validate_params():
    # Test the validate_params method
    valid_params = {"rho": 0.7, "L": 2.0}
    ART1.validate_params(valid_params)

    invalid_params = {
        "rho": -0.7,  # Invalid vigilance parameter
        "L": 0.5,  # Invalid L (must be >= 1)
    }
    with pytest.raises(AssertionError):
        ART1.validate_params(invalid_params)


def test_validate_data(art_model):
    # Test the validate_data method
    binary_data = np.array([[1, 0], [0, 1]])
    art_model.validate_data(binary_data)

    non_binary_data = np.array([[0.5, 1.0], [1.2, 0.3]])
    with pytest.raises(AssertionError):
        art_model.validate_data(non_binary_data)


def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([1, 0])
    w = np.array(
        [0.5, 0.5, 1, 0]
    )  # Mock weight with both bottom-up and top-down weights
    params = {"rho": 0.7}

    activation, _ = art_model.category_choice(i, w, params)
    assert isinstance(activation, float)
    assert activation == 0.5  # np.dot(i, [0.5, 0.5]) = 0.5


def test_match_criterion(art_model):
    # Test the match_criterion method
    art_model.dim_ = 2
    i = np.array([1, 0])
    w = np.array(
        [0.5, 0.5, 1, 0]
    )  # Mock weight with both bottom-up and top-down weights
    params = {"rho": 0.7}
    cache = {}

    match_criterion, _ = art_model.match_criterion(i, w, params, cache=cache)
    assert isinstance(match_criterion, float)
    assert (
        match_criterion == 1.0
    )  # Intersection of i and top-down weight w_td: [1, 0] matches exactly with i


def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([1, 0])
    w = np.array(
        [0.5, 0.5, 1, 1]
    )  # Mock weight with both bottom-up and top-down weights
    params = {"L": 2.0}

    updated_weight = art_model.update(i, w, params)
    assert len(updated_weight) == 4  # Bottom-up and top-down weights
    assert np.array_equal(
        updated_weight[2:], np.array([1, 0])
    )  # Top-down weights should match input i


def test_new_weight(art_model):
    # Test the new_weight method
    art_model.dim_ = 2
    i = np.array([1, 0])
    params = {"L": 2.0}

    new_weight = art_model.new_weight(i, params)
    assert len(new_weight) == 4  # Bottom-up and top-down weights
    assert np.array_equal(
        new_weight[2:], i
    )  # Top-down weights should be equal to input i


def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.dim_ = 2
    art_model.W = [np.array([0.5, 0.5, 1, 0]), np.array([0.3, 0.7, 0, 1])]
    centers = art_model.get_cluster_centers()
    assert len(centers) == 2
    assert np.array_equal(centers[0], np.array([1, 0]))
    assert np.array_equal(centers[1], np.array([0, 1]))


def test_fit(art_model):
    # Test fitting the model
    art_model.dim_ = 2
    X = np.array([[1, 0], [0, 1], [1, 1]])
    art_model.check_dimensions(X)
    art_model.fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were created


def test_partial_fit(art_model):
    # Test partial_fit method
    art_model.dim_ = 2
    X = np.array([[1, 0], [0, 1]])
    art_model.check_dimensions(X)
    art_model.partial_fit(X)

    assert len(art_model.W) > 0  # Ensure that clusters were partially fit


def test_predict(art_model):
    # Test predict method
    art_model.dim_ = 2
    X = np.array([[1, 0], [0, 1]])
    art_model.check_dimensions(X)
    art_model.W = [np.array([0.5, 0.5, 1, 0]), np.array([0.3, 0.7, 0, 1])]

    labels = art_model.predict(X)
    assert len(labels) == 2
