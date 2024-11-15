import pytest
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART
from artlib.cvi.CVIART import CVIART


@pytest.fixture
def cviart_model():
    # Initialize CVIART with a FuzzyART base module and Calinski-Harabasz validity index
    base_module = FuzzyART(0.5, 0.01, 1.0)
    return CVIART(base_module=base_module, validity=CVIART.CALINSKIHARABASZ)


def test_cviart_initialization(cviart_model):
    # Test that the CVIART model initializes correctly
    assert isinstance(cviart_model.base_module, BaseART)
    assert cviart_model.params["validity"] == CVIART.CALINSKIHARABASZ


def test_cviart_validate_params():
    # Test the validate_params method
    base_module = FuzzyART(0.5, 0.01, 1.0)
    cviart_model = CVIART(base_module=base_module, validity=CVIART.SILHOUETTE)
    valid_params = dict(base_module.params)
    valid_params.update({"validity": CVIART.SILHOUETTE})
    cviart_model.validate_params(valid_params)

    invalid_params = {"validity": 999}  # Invalid validity index
    with pytest.raises(AssertionError):
        cviart_model.validate_params(invalid_params)


def test_cviart_prepare_and_restore_data(cviart_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 4)

    X_prep = cviart_model.prepare_data(X)

    X_restored = cviart_model.restore_data(X_prep)
    assert np.allclose(X_restored, X)


def test_cviart_fit(cviart_model):
    # Test the fit method of CVIART
    X = np.random.rand(10, 4)
    X_prep = cviart_model.prepare_data(X)

    cviart_model.fit(X_prep, max_iter=1)

    assert len(cviart_model.W) > 0
    assert cviart_model.labels_.shape[0] == X.shape[0]


def test_cviart_step_fit_not_implemented(cviart_model):
    # Test that the step_fit method raises NotImplementedError
    X = np.random.rand(10, 4)

    with pytest.raises(NotImplementedError):
        cviart_model.step_fit(X[0])


def test_cviart_CVI_match(cviart_model):
    # Test the CVI_match method with Calinski-Harabasz index
    X = np.random.rand(10, 4)
    cviart_model.data = X
    cviart_model.labels_ = np.random.randint(0, 2, size=(10,))

    x = X[0]
    w = np.random.rand(4)
    cviart_model.base_module.W = w

    # Test that CVI_match correctly works with Calinski-Harabasz
    result = cviart_model.CVI_match(
        x,
        w,
        1,
        cviart_model.params,
        {"validity": CVIART.CALINSKIHARABASZ, "index": 0},
        {},
    )
    assert isinstance(result, np.bool_)


def test_cviart_get_cluster_centers(cviart_model):
    # Test the get_cluster_centers method
    cviart_model.base_module.d_min_ = np.array([0.0, 0.0])
    cviart_model.base_module.d_max_ = np.array([1.0, 1.0])
    cviart_model.base_module.W = [np.array([0.1, 0.4, 0.5, 0.4])]

    centers = cviart_model.get_cluster_centers()

    assert len(centers) == 1
    assert np.allclose(centers[0], np.array([0.3, 0.5]))
