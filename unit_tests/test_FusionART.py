import pytest
import numpy as np
from artlib.common.BaseART import BaseART
from artlib.fusion.FusionART import FusionART
from artlib.elementary.FuzzyART import FuzzyART


@pytest.fixture
def fusionart_model():
    # Initialize FusionART with two FuzzyART modules
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.7, 0.01, 1.0)
    gamma_values = np.array([0.5, 0.5])
    channel_dims = [4, 4]
    return FusionART(modules=[module_a, module_b], gamma_values=gamma_values, channel_dims=channel_dims)


def test_initialization(fusionart_model):
    # Test that the model initializes correctly
    assert isinstance(fusionart_model.modules[0], BaseART)
    assert isinstance(fusionart_model.modules[1], BaseART)
    assert np.all(fusionart_model.params["gamma_values"] == np.array([0.5, 0.5]))
    assert fusionart_model.channel_dims == [4, 4]


def test_validate_params():
    # Test the validate_params method
    valid_params = {"gamma_values": np.array([0.5, 0.5])}
    FusionART.validate_params(valid_params)

    invalid_params = {"gamma_values": np.array([0.6, 0.6])}  # sum of gamma_values must be 1.0
    with pytest.raises(AssertionError):
        FusionART.validate_params(invalid_params)


def test_get_cluster_centers(fusionart_model):
    # Test the get_cluster_centers method
    fusionart_model.modules[0].W = [np.array([0.1, 0.4, 0.5, 0.4])]
    fusionart_model.modules[1].W = [np.array([0.2, 0.2, 0.2, 0.2])]

    centers = fusionart_model.get_cluster_centers()

    assert len(centers) == 1
    assert np.allclose(centers[0], np.array([0.3, 0.5, 0.5, 0.5]))


def test_prepare_and_restore_data(fusionart_model):
    # Test prepare_data and restore_data methods
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]

    X_prep = fusionart_model.prepare_data(X)
    assert X_prep.shape == (10, 8)

    X_restored = fusionart_model.restore_data(X_prep)
    assert np.allclose(X_restored[0], X[0])
    assert np.allclose(X_restored[1], X[1])


def test_fit(fusionart_model):
    # Test the fit method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    assert fusionart_model.labels_.shape[0] == X_prep.shape[0]


def test_partial_fit(fusionart_model):
    # Test the partial_fit method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.partial_fit(X_prep)

    assert fusionart_model.labels_.shape[0] == X_prep.shape[0]


def test_predict(fusionart_model):
    # Test the predict method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    predictions = fusionart_model.predict(X_prep)
    assert predictions.shape[0] == X_prep.shape[0]


def test_step_fit(fusionart_model):
    # Test the step_fit method with base_module's internal methods
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    # Prepare data before fitting
    fusionart_model.modules[0].W = []
    fusionart_model.modules[1].W = []

    # Run step_fit for the first sample
    label = fusionart_model.step_fit(X_prep[0])
    assert isinstance(label, int)  # Ensure the result is an integer cluster label


def test_step_pred(fusionart_model):
    # Test the step_pred method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    label = fusionart_model.step_pred(X_prep[0])
    assert isinstance(label, int)  # Ensure the result is an integer


def test_predict_regression(fusionart_model):
    # Test the predict_regression method
    X = [np.random.rand(10, 2), np.random.rand(10, 2)]
    X_prep = fusionart_model.prepare_data(X)

    fusionart_model.fit(X_prep, max_iter=1)

    predicted_regression = fusionart_model.predict_regression(X_prep)
    assert predicted_regression.shape[0] == X_prep.shape[0]


def test_join_channel_data(fusionart_model):
    # Test the join_channel_data method
    channel_1 = np.random.rand(10, 2)
    channel_2 = np.random.rand(10, 2)

    X = fusionart_model.join_channel_data([channel_1, channel_2])
    assert X.shape == (10, 4)
