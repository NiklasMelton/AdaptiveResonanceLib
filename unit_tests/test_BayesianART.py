import pytest
import numpy as np
from artlib.elementary.BayesianART import BayesianART


# Fixture to initialize a BayesianART instance for testing
@pytest.fixture
def art_model():
    rho = 0.7
    cov_init = np.array([[1.0, 0.0], [0.0, 1.0]])  # Initial covariance matrix
    return BayesianART(rho=rho, cov_init=cov_init)


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho"] == 0.7
    assert np.array_equal(
        art_model.params["cov_init"], np.array([[1.0, 0.0], [0.0, 1.0]])
    )


def test_validate_params():
    # Test the validate_params method
    valid_params = {"rho": 0.7, "cov_init": np.array([[1.0, 0.0], [0.0, 1.0]])}
    BayesianART.validate_params(valid_params)

    invalid_params = {
        "rho": -0.5,  # Invalid vigilance parameter
        "cov_init": "not_a_matrix",  # Invalid covariance matrix
    }
    with pytest.raises(AssertionError):
        BayesianART.validate_params(invalid_params)


def test_check_dimensions(art_model):
    # Test the check_dimensions method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)

    assert art_model.dim_ == 2


def test_category_choice(art_model):
    # Test the category_choice method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array(
        [0.25, 0.35, 1.0, 0.0, 0.0, 1.0, 5]
    )  # Mock weight (mean, covariance, and sample count)
    art_model.W = [w]
    params = {"rho": 0.7}

    activation, cache = art_model.category_choice(i, w, params)
    assert "cov" in cache
    assert "det_cov" in cache
    assert isinstance(activation, float)


def test_match_criterion(art_model):
    # Test the match_criterion method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array(
        [0.25, 0.35, 1.0, 0.0, 0.0, 1.0, 5]
    )  # Mock weight (mean, covariance, and sample count)
    params = {"rho": 0.7}
    cache = {}

    match_criterion, new_cache = art_model.match_criterion(
        i, w, params, cache=cache
    )
    assert isinstance(match_criterion, float)


def test_update(art_model):
    # Test the update method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    w = np.array(
        [0.25, 0.35, 1.0, 0.0, 0.0, 1.0, 5]
    )  # Mock weight (mean, covariance, and sample count)
    params = {"rho": 0.7}
    cache = {}

    updated_weight = art_model.update(i, w, params, cache=cache)
    assert (
        len(updated_weight) == 7
    )  # Mean (2D), covariance (4 values), and sample count


def test_new_weight(art_model):
    # Test the new_weight method
    art_model.dim_ = 2
    i = np.array([0.2, 0.3])
    params = {"cov_init": np.array([[1.0, 0.0], [0.0, 1.0]])}

    new_weight = art_model.new_weight(i, params)
    assert (
        len(new_weight) == 7
    )  # Mean (2D), covariance (4 values), and sample count
    assert new_weight[-1] == 1  # Initial sample count should be 1


def test_get_cluster_centers(art_model):
    # Test getting cluster centers
    art_model.dim_ = 2
    art_model.W = [np.array([0.2, 0.3, 1.0, 0.0, 0.0, 1.0, 5])]
    centers = art_model.get_cluster_centers()
    assert len(centers) == 1
    assert np.array_equal(centers[0], np.array([0.2, 0.3]))


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
    art_model.W = [np.array([0.1, 0.2, 1.0, 0.0, 0.0, 1.0, 5])]

    labels = art_model.predict(X)
    assert len(labels) == 2



def _reference_choice(model, x, w):
    # Activation as originally written, with no caching.
    dim = model.dim_
    cov = w[dim:-1].reshape((dim, dim))
    dist = w[:dim] - x
    p_i_cj = np.exp(
        -0.5 * np.matmul(dist.T, np.matmul(np.linalg.inv(cov), dist))
    ) / np.sqrt((model.pi2**dim) * np.linalg.det(cov))
    return p_i_cj * (w[-1] / sum(w_[-1] for w_ in model.W))


def test_cached_choice_matches_reference(art_model):
    X = np.random.default_rng(0).random((30, 2))

    def check():
        for w in art_model.W:
            got, _ = art_model.category_choice(X[-1], w, art_model.params)
            assert got == _reference_choice(art_model, X[-1], w)

    art_model.fit(X[:20])
    check()
    art_model.partial_fit(X[20:25])
    check()
    # Learn a sample on an existing mean: counts change but not the length of W.
    n_before = len(art_model.W)
    art_model.step_fit(art_model.W[0][: art_model.dim_].copy())
    assert len(art_model.W) == n_before
    check()
    art_model.add_weight(art_model.new_weight(X[25], art_model.params))
    check()
    art_model.merge(0, len(art_model.W) - 1)
    check()
    art_model.W.append(art_model.new_weight(X[26], art_model.params))
    check()
    art_model.W = list(art_model.W[:-1])
    check()


def test_pickle_and_deepcopy_with_cache(art_model):
    import copy
    import pickle

    X = np.random.default_rng(1).random((20, 2))
    art_model.fit(X)
    expected = art_model.predict(X)
    for clone in (pickle.loads(pickle.dumps(art_model)), copy.deepcopy(art_model)):
        assert "_cov_cache_" not in clone.__dict__
        np.testing.assert_array_equal(clone.predict(X), expected)
