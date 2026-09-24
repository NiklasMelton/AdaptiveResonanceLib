import pytest
import numpy as np
from typing import Optional
from artlib.topological.DualVigilanceART import DualVigilanceART
from artlib.common.BaseART import BaseART
from artlib.common.utils import complement_code
from artlib.elementary.FuzzyART import FuzzyART


# Mock BaseART class for testing purposes
class MockBaseART(BaseART):
    def __init__(self):
        params = {"rho": 0.7}
        super().__init__(params)
        self.W = []
        self.labels_ = np.array([])
        self.dim_ = 2

    @staticmethod
    def validate_params(params: dict):
        pass

    def prepare_data(self, X: np.ndarray):
        return X

    def restore_data(self, X: np.ndarray):
        return X

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return i

    def add_weight(self, w: np.ndarray):
        self.W.append(w)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        return np.random.random(), {}

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
        op=None,
    ) -> tuple[bool, dict]:
        return True, {}

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        return w

    def get_cluster_centers(self) -> list:
        return [w for w in self.W]

    def check_dimensions(self, X: np.ndarray):
        assert X.shape[1] == self.dim_


@pytest.fixture
def art_model():
    base_module = MockBaseART()
    rho_lower_bound = 0.3
    return DualVigilanceART(base_module=base_module, rho_lower_bound=rho_lower_bound)


def test_initialization(art_model):
    # Test that the model initializes correctly
    assert art_model.params["rho_lower_bound"] == 0.3
    assert isinstance(art_model.base_module, BaseART)


def test_prepare_data(art_model):
    # Test the prepare_data method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    prepared_X = art_model.prepare_data(X)
    assert np.array_equal(prepared_X, X)


def test_restore_data(art_model):
    # Test the restore_data method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    restored_X = art_model.restore_data(X)
    assert np.array_equal(restored_X, X)


def test_get_params(art_model):
    # Test the get_params method
    params = art_model.get_params(deep=True)
    assert "rho_lower_bound" in params
    assert "base_module" in params


def test_n_clusters(art_model):
    # Test the n_clusters property
    assert art_model.n_clusters == 0  # No clusters initially
    art_model.map = {0: 0, 1: 1}
    assert art_model.n_clusters == 2  # Two clusters


def test_check_dimensions(art_model):
    # Test the check_dimensions method
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    art_model.check_dimensions(X)  # Should pass without assertion errors


def test_validate_params(art_model):
    # Test the validate_params method
    valid_params = {"rho_lower_bound": 0.3}
    art_model.validate_params(valid_params)

    invalid_params = {"rho_lower_bound": -0.3}  # Invalid rho_lower_bound
    with pytest.raises(AssertionError):
        art_model.validate_params(invalid_params)


def test_step_fit(art_model):
    # Test the step_fit method
    x = np.array([0.1, 0.2])
    cluster_label = art_model.step_fit(x)
    assert cluster_label == 0  # First sample should create a new cluster


def test_step_pred(art_model):
    # Test the step_pred method
    x = np.array([0.1, 0.2])
    art_model.step_fit(x)  # Create the first cluster
    cluster_label = art_model.step_pred(x)
    assert cluster_label == 0  # Predict should return the correct cluster


def test_get_cluster_centers(art_model):
    # Test the get_cluster_centers method
    art_model.step_fit(np.array([0.1, 0.2]))  # Create the first cluster
    centers = art_model.get_cluster_centers()
    assert len(centers) == 1
    assert np.array_equal(centers[0], np.array([0.1, 0.2]))


def fitted_dual(max_iter=1):
    X = complement_code(np.array([[0.0], [0.2], [0.8], [1.0]]))
    model = DualVigilanceART(
        FuzzyART(rho=1.0, alpha=0.01, beta=1.0), rho_lower_bound=0.6
    ).fit(X, max_iter=max_iter)
    assert model.map == {0: 0, 1: 0, 2: 1, 3: 1}
    return model, X


def test_merge_abstract_clusters_keeps_base_prototypes():
    model, X = fitted_dual()
    weights = [weight.copy() for weight in model.W]
    counts = model.base_module.weight_sample_counter_.copy()

    assert model.merge(1, 0) == 0

    assert model.map == {0: 0, 1: 0, 2: 0, 3: 0}
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0])
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    np.testing.assert_array_equal(model.prototype_labels_, [0, 1, 2, 3])
    assert model.base_module.weight_sample_counter_ == counts
    for actual, expected in zip(model.W, weights):
        np.testing.assert_array_equal(actual, expected)


def test_move_prototype_updates_historical_labels_and_removes_empty_source():
    model, X = fitted_dual()
    weights = [weight.copy() for weight in model.W]
    counts = model.base_module.weight_sample_counter_.copy()

    assert model.move_prototype(0, 1, 1) == 1
    assert model.map == {0: 0, 1: 1, 2: 1, 3: 1}
    np.testing.assert_array_equal(model.labels_, [0, 1, 1, 1])
    np.testing.assert_array_equal(model.predict(X), model.labels_)

    assert model.move_prototype(0, 0, 1) == 0
    assert model.n_clusters == 1
    assert model.map == {0: 0, 1: 0, 2: 0, 3: 0}
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0])
    assert model.base_module.weight_sample_counter_ == counts
    for actual, expected in zip(model.W, weights):
        np.testing.assert_array_equal(actual, expected)


def test_prototype_history_survives_epochs_partial_fit_and_fresh_fit():
    model, X = fitted_dual(max_iter=2)
    np.testing.assert_array_equal(model.prototype_labels_, [0, 1, 2, 3])

    model.partial_fit(X[:1])
    assert len(model.labels_) == len(model.prototype_labels_) == 5
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1, 0])

    model.fit(X[:2])
    assert model.map == {0: 0, 1: 0}
    np.testing.assert_array_equal(model.prototype_labels_, [0, 1])
    np.testing.assert_array_equal(model.labels_, [0, 0])
    assert len(model.base_module.weight_sample_counter_) == 2


def test_fit_gif_tracks_final_prototype_assignments(tmp_path):
    X = complement_code(np.array([[0.0, 0.0], [0.2, 0.2]]))
    model = DualVigilanceART(
        FuzzyART(rho=1.0, alpha=0.01, beta=1.0), rho_lower_bound=0.6
    )

    model.fit_gif(X, max_iter=2, filename=str(tmp_path / "dual.gif"), fps=1)

    assert len(model.prototype_labels_) == len(model.labels_) == 2
    np.testing.assert_array_equal(
        model.labels_, [model.map[proto] for proto in model.prototype_labels_]
    )


def test_invalid_dual_edits_leave_state_unchanged():
    model, _ = fitted_dual()
    old_map = model.map.copy()
    old_labels = model.labels_.copy()

    with pytest.raises(ValueError):
        model.merge(0, 0)
    with pytest.raises(TypeError):
        model.merge(True, 1)
    with pytest.raises(IndexError):
        model.move_prototype(0, 1, 2)
    with pytest.raises(ValueError):
        model.move_prototype(1, 0, 0)
    assert model.map == old_map
    np.testing.assert_array_equal(model.labels_, old_labels)

    del model._prototype_labels_
    with pytest.raises(RuntimeError, match="assignments are unavailable"):
        model.move_prototype(0, 1, 1)
    assert model.map == old_map
    np.testing.assert_array_equal(model.labels_, old_labels)
