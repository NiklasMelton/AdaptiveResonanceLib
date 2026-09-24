import pytest
import numpy as np
from artlib.hierarchical.DeepARTMAP import DeepARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.supervised.ARTMAP import ARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.BaseART import BaseART
from artlib.common.utils import complement_code


# Fixture to initialize a DeepARTMAP instance for testing
@pytest.fixture
def deep_artmap_model():
    module_a = FuzzyART(0.5, 0.01, 1.0)
    module_b = FuzzyART(0.7, 0.01, 1.0)
    return DeepARTMAP(modules=[module_a, module_b])


def test_initialization(deep_artmap_model):
    # Test that the model initializes correctly
    assert isinstance(deep_artmap_model.modules[0], BaseART)
    assert isinstance(deep_artmap_model.modules[1], BaseART)
    assert len(deep_artmap_model.modules) == 2


def test_get_params(deep_artmap_model):
    # Test the get_params method
    params = deep_artmap_model.get_params()
    assert "module_0" in params
    assert "module_1" in params


def test_set_params(deep_artmap_model):
    # Test the set_params method
    deep_artmap_model.set_params(module_0__rho=0.6)
    assert deep_artmap_model.modules[0].params["rho"] == 0.6


def test_validate_data(deep_artmap_model):
    # Test the validate_data method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)
    deep_artmap_model.validate_data(X, y)

    # Test invalid input data
    X_invalid = [np.random.rand(9, 5), np.random.rand(10, 5)]
    with pytest.raises(AssertionError):
        deep_artmap_model.validate_data(X_invalid, y)


def test_prepare_and_restore_data(deep_artmap_model):
    # Test prepare_data and restore_data methods
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    X_prep, _ = deep_artmap_model.prepare_data(X)

    X_restored, _ = deep_artmap_model.restore_data(X_prep)
    assert np.allclose(X_restored[0], X[0])
    assert np.allclose(X_restored[1], X[1])


def test_fit_supervised(deep_artmap_model):
    # Test the supervised fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    assert deep_artmap_model.layers[0].labels_.shape[0] == X[0].shape[0]


def test_fit_unsupervised(deep_artmap_model):
    # Test the unsupervised fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    assert deep_artmap_model.layers[0].labels_a.shape[0] == X[0].shape[0]


def test_partial_fit_supervised(deep_artmap_model):
    # Test the supervised partial_fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before partial fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.partial_fit(X_prep, y)

    assert deep_artmap_model.layers[0].labels_.shape[0] == X[0].shape[0]


def test_partial_fit_unsupervised(deep_artmap_model):
    # Test the unsupervised partial_fit method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before partial fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.partial_fit(X_prep)

    assert deep_artmap_model.layers[0].labels_a.shape[0] == X[0].shape[0]


def test_predict(deep_artmap_model):
    # Test the predict method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]

    # Prepare data before fitting and predicting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, max_iter=1)

    predictions = deep_artmap_model.predict(X_prep)
    assert predictions[-1].shape[0] == X[-1].shape[0]


def test_labels_deep(deep_artmap_model):
    # Test the labels_deep_ method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting and predicting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    labels_deep = deep_artmap_model.labels_deep_
    assert labels_deep.shape == (10, 3)


def prepared_edit_data():
    return complement_code(np.array([[0.0], [0.2], [0.8], [1.0]]))


def fitted_edit_hierarchy(rhos=(0.2, 0.6, 1.0), y=None):
    X = prepared_edit_data()
    modules = [FuzzyART(rho=rho, alpha=0.01, beta=1.0) for rho in rhos]
    model = DeepARTMAP(modules).fit([X] * len(modules), y)
    return model, X


def test_unsupervised_merge_reindexes_child_maps_at_intermediate_level():
    model, X = fitted_edit_hierarchy(rhos=(0.0, 0.6, 1.0))
    assert [module.n_clusters for module in model.modules] == [1, 2, 4]

    assert model.merge(1, 0, 1) == 0

    assert [module.n_clusters for module in model.modules] == [1, 1, 4]
    assert model.layers[0].map == {0: 0}
    assert set(model.layers[1].map.values()) == {0}
    np.testing.assert_array_equal(model.labels_deep_[:, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(model.predict(X)[1], [0, 0, 0, 0])

    model.partial_fit([X[:1]] * 3)
    assert model.labels_deep_.shape == (5, 3)
    assert all(len(layer.labels_) == 5 for layer in model.layers)


def test_unsupervised_move_propagates_upward_then_root_merge():
    model, X = fitted_edit_hierarchy()
    assert [module.n_clusters for module in model.modules] == [2, 2, 4]

    assert model.move_prototype(2, 0, 1, 1) == 1
    np.testing.assert_array_equal(model.labels_deep_[1], [1, 1, 1])
    np.testing.assert_array_equal(model.modules[1].labels_, [0, 0, 1, 1])
    np.testing.assert_array_equal(model.predict(X)[0], model.labels_)

    assert model.merge(0, 1, 0) == 0
    assert model.modules[0].n_clusters == 1
    np.testing.assert_array_equal(model.labels_deep_[:, 0], [0, 0, 0, 0])
    assert set(model.layers[0].map.values()) == {0}


def test_supervised_merge_and_move_existing_external_classes():
    y = np.array([10, 10, 20, 20])
    model, X = fitted_edit_hierarchy(rhos=(1.0, 1.0, 1.0), y=y)
    assert model.merge(0, 0, 1) == 0
    assert model.modules[0].n_clusters == 3
    assert model.layers[1].map == {0: 0, 1: 0, 2: 1, 3: 2}
    np.testing.assert_array_equal(model.labels_deep_[:, 0], y)
    np.testing.assert_array_equal(model.labels_deep_[:, 1], [0, 0, 1, 2])

    model, X = fitted_edit_hierarchy(rhos=(1.0, 1.0, 1.0), y=y)
    assert model.move_prototype(0, 10, 0, 20) == 20
    assert model.move_prototype(0, 10, 1, 20) == 20
    assert model.modules[0].n_clusters == 4
    np.testing.assert_array_equal(model.labels_, [20, 20, 20, 20])
    np.testing.assert_array_equal(model.layers[0].classes_, [20])
    np.testing.assert_array_equal(model.predict(X)[0], model.labels_)

    model.partial_fit([X[:1]] * 3, np.array([20]))
    assert model.labels_deep_.shape == (5, 4)
    np.testing.assert_array_equal(model.labels_deep_[:4, 0], [20, 20, 20, 20])


def test_hierarchy_rejects_cross_parent_merge_and_empty_parent_move():
    model, _ = fitted_edit_hierarchy()
    old_maps = [layer.map.copy() for layer in model.layers]
    old_labels = model.labels_deep_.copy()

    with pytest.raises(ValueError, match="same parent"):
        model.merge(1, 0, 1)
    with pytest.raises(ValueError, match="At least one"):
        model.move_prototype(1, 0, 0, 1)
    with pytest.raises(ValueError, match="no parent"):
        model.move_prototype(0, 0, 0, 1)
    with pytest.raises(IndexError, match="Module level"):
        model.merge(3, 0, 1)

    assert [layer.map for layer in model.layers] == old_maps
    np.testing.assert_array_equal(model.labels_deep_, old_labels)


def test_hierarchy_rejects_module_without_merge_support():
    class NoMergeFuzzyART(FuzzyART):
        merge = BaseART.merge

    X = prepared_edit_data()
    model = DeepARTMAP(
        [
            FuzzyART(rho=0.0, alpha=0.01, beta=1.0),
            NoMergeFuzzyART(rho=1.0, alpha=0.01, beta=1.0),
        ]
    ).fit([X, X])
    old_map = model.layers[0].map.copy()
    old_labels = model.labels_deep_.copy()

    with pytest.raises(NotImplementedError, match="does not support"):
        model.merge(1, 0, 1)

    assert model.modules[1].n_clusters == 4
    assert model.layers[0].map == old_map
    np.testing.assert_array_equal(model.labels_deep_, old_labels)


def test_partial_fit_preserves_supervised_string_labels():
    X = prepared_edit_data()
    y = np.array(["left", "left", "right", "right"])
    modules = [FuzzyART(rho=1.0, alpha=0.01, beta=1.0) for _ in range(2)]
    model = DeepARTMAP(modules).fit([X, X], y)

    model.partial_fit([X[:1], X[:1]], y[:1])

    np.testing.assert_array_equal(model.labels_, [*y, "left"])
    assert model.labels_deep_.shape == (5, 3)


def test_map_deep(deep_artmap_model):
    # Test the map_deep method
    X = [np.random.rand(10, 5), np.random.rand(10, 5)]
    y = np.random.randint(0, 2, size=10)

    # Prepare data before fitting
    X_prep, _ = deep_artmap_model.prepare_data(X)
    deep_artmap_model.fit(X_prep, y, max_iter=1)

    mapped_label = deep_artmap_model.map_deep(
        0, deep_artmap_model.layers[0].labels_a[0]
    )
    assert isinstance(mapped_label.tolist(), int)
