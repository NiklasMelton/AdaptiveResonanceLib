import pytest
import numpy as np
from unittest.mock import MagicMock
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.utils import complement_code

# Assuming BaseART is imported and available in the current namespace


@pytest.fixture
def art_model():
    # Fixture that sets up the model before each test
    params = {
        "rho": 0.5,
        "alpha": 0.0,
        "beta": 1.0,
    }
    return FuzzyART(**params)


def test_initialization(art_model):
    # Test that the ART model initializes correctly
    assert art_model.params == {"rho": 0.5, "alpha": 0.0, "beta": 1.0}
    assert art_model.sample_counter_ == 0
    assert art_model.weight_sample_counter_ == []


def test_set_get_params(art_model):
    # Test set_params and get_params functions
    new_params = {"rho": 0.7, "alpha": 0.05, "beta": 0.9}
    art_model.set_params(**new_params)
    assert art_model.get_params() == new_params
    assert art_model.rho == 0.7
    assert art_model.alpha == 0.05
    assert art_model.beta == 0.9


def test_attribute_access(art_model):
    # Test dynamic attribute access and setting using params
    assert art_model.rho == 0.5
    art_model.rho = 0.8
    assert art_model.rho == 0.8


def test_invalid_attribute(art_model):
    # Test accessing an invalid attribute
    with pytest.raises(AttributeError):
        art_model.non_existing_attribute


def test_prepare_restore_data(art_model):
    # Test data normalization and denormalization
    X = np.array([[1, 2], [3, 4], [5, 6]])
    normalized_X = art_model.prepare_data(X)
    restored_X = art_model.restore_data(normalized_X)
    np.testing.assert_array_almost_equal(restored_X, X)


def test_check_dimensions(art_model):
    # Test check_dimensions with valid data
    X = np.array([[1, 2], [3, 4]])
    art_model.check_dimensions(X)
    assert art_model.dim_ == 2

    # Test check_dimensions with invalid data (mismatch)
    X_invalid = np.array([[1, 2, 3]])
    with pytest.raises(AssertionError):
        art_model.check_dimensions(X_invalid)


def test_match_tracking(art_model):
    # Test match tracking with different methods
    cache = {"match_criterion": 0.5}
    art_model._match_tracking(
        cache, epsilon=0.01, params=art_model.params, method="MT+"
    )
    assert art_model.rho == 0.51

    art_model._match_tracking(
        cache, epsilon=0.01, params=art_model.params, method="MT-"
    )
    assert art_model.rho == 0.49

    art_model._match_tracking(
        cache, epsilon=0.01, params=art_model.params, method="MT0"
    )
    assert art_model.rho == 0.5

    art_model._match_tracking(
        cache, epsilon=0.01, params=art_model.params, method="MT~"
    )
    assert art_model.rho == 0.5

    art_model._match_tracking(
        cache, epsilon=0.01, params=art_model.params, method="MT1"
    )
    assert np.isinf(art_model.rho)


def test_step_fit(art_model):
    # Test step_fit for creating new clusters
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.W = []
    art_model.new_weight = MagicMock(return_value=np.array([0.1, 0.2]))
    art_model.add_weight = MagicMock()
    art_model.update = MagicMock(return_value=np.array([0.15, 0.25]))
    art_model.category_choice = MagicMock(return_value=(1.0, None))
    art_model.match_criterion_bin = MagicMock(return_value=(True, {}))

    label = art_model.step_fit(X[0])
    assert label == 0
    art_model.add_weight.assert_called_once()


def test_partial_fit(art_model):
    # Test partial_fit
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.new_weight = MagicMock(return_value=np.array([0.1, 0.2]))
    art_model.add_weight = MagicMock()
    art_model.update = MagicMock(return_value=np.array([0.15, 0.25]))
    art_model.category_choice = MagicMock(return_value=(1.0, None))
    art_model.match_criterion_bin = MagicMock(return_value=(True, {}))

    art_model.partial_fit(X)
    art_model.add_weight.assert_called()


def test_predict(art_model):
    # Test predict function
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X = art_model.prepare_data(X)
    art_model.category_choice = MagicMock(return_value=(1.0, None))
    art_model.step_pred = MagicMock(return_value=0)

    labels = art_model.predict(X)
    np.testing.assert_array_equal(labels, [0, 0])


def test_clustering(art_model):
    new_params = {"rho": 0.9, "alpha": 0.05, "beta": 1.0}
    art_model.set_params(**new_params)

    data = np.array(
        [[0.0, 0.0], [0.0, 0.08], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    )
    data = art_model.prepare_data(data)
    labels = art_model.fit_predict(data)

    assert np.all(np.equal(labels, np.array([0, 0, 1, 2, 3])))


def test_validate_data(art_model):
    # Test validate_data with normalized data
    X = np.array([[-0.1, 0.2], [1.1, 0.4]])
    art_model.is_fitted_ = False
    with pytest.raises(AssertionError):
        art_model.validate_data(X)

def test_validate_data_again(art_model):
    # Test validate_data with normalized data
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.is_fitted_ = False
    X_cc = complement_code(X)
    art_model.validate_data(X_cc)  # Should pass without assertion error

    # Test validate_data with data out of bounds
    X_invalid = np.array([[-0.1, 0.2], [1.1, 0.4]])
    with pytest.raises(AssertionError):
        art_model.validate_data(X_invalid)


def test_set_data_bounds(art_model):
    # Test set_data_bounds with valid bounds
    lower_bounds = np.array([0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0])
    art_model.is_fitted_ = False
    art_model.set_data_bounds(lower_bounds, upper_bounds)
    assert np.all(art_model.d_min_ == lower_bounds)
    assert np.all(art_model.d_max_ == upper_bounds)

    # Test set_data_bounds after the model is fitted
    art_model.is_fitted_ = True
    with pytest.raises(ValueError, match="Cannot change data limits after fit."):
        art_model.set_data_bounds(lower_bounds, upper_bounds)
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    X_cc = complement_code(X)
    X_norm = art_model.prepare_data(X)
    assert np.all(X_norm == X_cc)


def test_find_data_bounds(art_model):
    # Test find_data_bounds with multiple data batches
    batch_1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    batch_2 = np.array([[0.0, 0.1], [0.5, 0.6]])
    lower_bounds, upper_bounds = art_model.find_data_bounds(batch_1, batch_2)
    np.testing.assert_array_equal(lower_bounds, np.array([0.0, 0.1]))
    np.testing.assert_array_equal(upper_bounds, np.array([0.5, 0.6]))


def test_prepare_data(art_model):
    # Test prepare_data with valid data
    X = np.array([[0.0, 0.5], [0.2, 1.0]])
    X_cc = complement_code(X)
    art_model.d_min_ = np.array([0.0, 0.0])
    art_model.d_max_ = np.array([1.0, 1.0])
    normalized_X = art_model.prepare_data(X)
    np.testing.assert_array_almost_equal(normalized_X, X_cc)  # Already normalized

    # Test prepare_data with data requiring normalization
    X = np.array([[1.0, 10.0], [5.0, 20.0]])
    art_model.d_min_ = np.array([1.0, 10.0])
    art_model.d_max_ = np.array([5.0, 20.0])
    normalized_X = art_model.prepare_data(X)
    expected_normalized_X_cc = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(normalized_X, expected_normalized_X_cc)