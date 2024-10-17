import pytest
import numpy as np
from artlib.elementary.FuzzyART import FuzzyART
from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART
from artlib.cvi.iCVIs.CalinkskiHarabasz import iCVI_CH


@pytest.fixture
def icvi_fuzzyart_model():
    # Initialize iCVIFuzzyART with Calinski-Harabasz validity index and offline mode
    return iCVIFuzzyART(
        rho=0.5,
        alpha=0.01,
        beta=1.0,
        validity=iCVIFuzzyART.CALINSKIHARABASZ,
        offline=True,
    )


def test_icvi_fuzzyart_initialization(icvi_fuzzyart_model):
    # Test that the model initializes correctly
    assert isinstance(icvi_fuzzyart_model, iCVIFuzzyART)
    assert (
        icvi_fuzzyart_model.params["validity"] == iCVIFuzzyART.CALINSKIHARABASZ
    )
    assert icvi_fuzzyart_model.offline is True


def test_icvi_fuzzyart_validate_params(icvi_fuzzyart_model):
    # Test if validity parameter is validated correctly
    assert "validity" in icvi_fuzzyart_model.params
    assert isinstance(icvi_fuzzyart_model.params["validity"], int)
    assert (
        icvi_fuzzyart_model.params["validity"] == iCVIFuzzyART.CALINSKIHARABASZ
    )


def test_icvi_fuzzyart_prepare_and_restore_data(icvi_fuzzyart_model):
    # Test prepare_data and restore_data methods
    X = np.random.rand(10, 4)

    X_prep = icvi_fuzzyart_model.prepare_data(X)
    X_restored = icvi_fuzzyart_model.restore_data(X_prep)

    assert np.allclose(X_restored, X)


def test_icvi_fuzzyart_fit_offline(icvi_fuzzyart_model):
    # Test the fit method in offline mode
    X = np.random.rand(10, 4)
    X_prep = icvi_fuzzyart_model.prepare_data(X)

    icvi_fuzzyart_model.fit(X_prep)

    assert len(icvi_fuzzyart_model.W) > 0
    assert icvi_fuzzyart_model.labels_.shape[0] == X.shape[0]


def test_icvi_fuzzyart_iCVI_match(icvi_fuzzyart_model):
    # Test the iCVI_match method with Calinski-Harabasz index
    X = np.random.rand(10, 4)
    X_prep = icvi_fuzzyart_model.prepare_data(X)
    icvi_fuzzyart_model.data = X_prep
    icvi_fuzzyart_model.labels_ = np.random.randint(0, 2, size=(10,))
    icvi_fuzzyart_model.iCVI = iCVI_CH(X[0])

    icvi_fuzzyart_model.fit(X_prep)

    x = X_prep[0]
    w = icvi_fuzzyart_model.W[0]

    # Test iCVI_match functionality
    result = icvi_fuzzyart_model.iCVI_match(
        x, w, 0, icvi_fuzzyart_model.params, {}
    )
    assert isinstance(result, np.bool_)


def test_icvi_fuzzyart_step_fit(icvi_fuzzyart_model):
    # Test step_fit method raises NotImplementedError from parent FuzzyART
    X = np.random.rand(10, 4)
    X_prep = icvi_fuzzyart_model.prepare_data(X)
    icvi_fuzzyart_model.W = []

    label = icvi_fuzzyart_model.step_fit(X_prep[0])
    assert label == 0


def test_icvi_fuzzyart_fit_with_custom_reset_function(icvi_fuzzyart_model):
    # Test the fit method with a custom match_reset_func
    X = np.random.rand(10, 4)
    X_prep = icvi_fuzzyart_model.prepare_data(X)

    def custom_reset_func(x, w, c_, params, cache):
        return True

    icvi_fuzzyart_model.fit(X_prep, match_reset_func=custom_reset_func)

    assert len(icvi_fuzzyart_model.W) > 0
    assert icvi_fuzzyart_model.labels_.shape[0] == X.shape[0]
