import pytest
import numpy as np
from artlib.elementary.QuadraticNeuronART import QuadraticNeuronART


# Fixture to initialize a QuadraticNeuronART instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    s_init = 0.5
    lr_b = 0.1
    lr_w = 0.1
    lr_s = 0.05
    return QuadraticNeuronART(
        rho=rho, s_init=s_init, lr_b=lr_b, lr_w=lr_w, lr_s=lr_s
    )


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho"] == 0.7
    assert art_model.params["s_init"] == 0.5
    assert art_model.params["lr_b"] == 0.1
    assert art_model.params["lr_w"] == 0.1
    assert art_model.params["lr_s"] == 0.05


def test_validate_params():
    # Test the validate_params method
    valid_params = {
        "rho": 0.7,
        "s_init": 0.5,
        "lr_b": 0.1,
        "lr_w": 0.1,
        "lr_s": 0.05,
    }
    QuadraticNeuronART.validate_params(valid_params)

    invalid_params = {
        "rho": 1.5,  # Invalid vigilance parameter
        "s_init": -0.5,  # Invalid s_init
        "lr_b": 1.1,  # Invalid learning rate for cluster mean
        "lr_w": -0.1,  # Invalid learning rate for cluster weights
        "lr_s": 1.5,  # Invalid learning rate for activation parameter
    }
    with pytest.raises(AssertionError):
        QuadraticNeuronART.validate_params(invalid_params)


def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array(
        [1.0, 0.0, 0.0, 1.0, 0.25, 0.35, 0.5]
    )  # Mock weight (matrix, centroid, and activation parameter)
    params = {"rho": 0.7, "s_init": 0.5}

    activation, cache = art_model.category_choice(i, w, params)
    assert "activation" in cache
    assert "l2norm2_z_b" in cache
    assert isinstance(activation, float)


def test_match_criterion(art_model):
    # Test the match_criterion method
    i = np.array([0.2, 0.3])
    w = np.array(
        [1.0, 0.0, 0.0, 1.0, 0.25, 0.35, 0.5]
    )  # Mock weight (matrix, centroid, and activation parameter)
    params = {"rho": 0.7}
    cache = {"activation": 0.8}

    match_criterion, new_cache = art_model.match_criterion(
        i, w, params, cache=cache
    )
    assert match_criterion == cache["activation"]


def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array(
        [1.0, 0.0, 0.0, 1.0, 0.25, 0.35, 0.5]
    )  # Mock weight (matrix, centroid, and activation parameter)
    params = {"lr_b": 0.1, "lr_w": 0.1, "lr_s": 0.05}
    cache = {
        "s": 0.5,
        "w": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b": np.array([0.25, 0.35]),
        "z": np.array([0.2, 0.3]),
        "activation": 0.8,
        "l2norm2_z_b": 0.02,
    }

    updated_weight = art_model.update(i, w, params, cache=cache)
    assert (
        len(updated_weight) == 7
    )  # Check that the weight has matrix, centroid, and activation parameter
    assert (
        updated_weight[-1] < 0.5
    )  # Check that the activation parameter has been updated


def test_new_weight(art_model):
    # Test the new_weight method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    params = {"s_init": 0.5}

    new_weight = art_model.new_weight(i, params)
    assert (
        len(new_weight) == 7
    )  # Weight matrix (4 values), centroid (2 values), and activation parameter (1 value)
    assert (
        new_weight[-1] == 0.5
    )  # Initial activation parameter should be s_init


def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.dim_ = 2
    art_model.W = [np.array([1.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.5])]
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
    art_model.W = [np.array([1.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.5])]

    labels = art_model.predict(X)
    assert len(labels) == 2
