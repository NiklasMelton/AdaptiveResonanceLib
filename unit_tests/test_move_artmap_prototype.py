import numpy as np
import pytest

from artlib.common.utils import complement_code
from artlib.elementary.FuzzyART import FuzzyART
from artlib.supervised.ARTMAP import ARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP


def make_art_module():
    return FuzzyART(rho=1.0, alpha=0.01, beta=1.0)


def make_simple_artmap():
    X = complement_code(np.array([[0.0], [0.25], [0.75], [1.0]]))
    y = np.array([10, 20, 10, 30])
    model = SimpleARTMAP(make_art_module()).fit(X, y)
    return model, X, y


def make_artmap():
    X = complement_code(np.array([[0.0], [0.25], [0.75], [1.0]]))
    y = complement_code(np.array([[0.0], [0.5], [0.0], [1.0]]))
    model = ARTMAP(make_art_module(), make_art_module()).fit(X, y)
    return model, X


def test_simple_artmap_moves_prototype_to_existing_class():
    model, X, original_y = make_simple_artmap()
    old_weights = [weight.copy() for weight in model.module_a.W]
    old_counts = model.module_a.weight_sample_counter_.copy()
    old_a_labels = model.labels_a.copy()

    assert model.move_A_prototype(10, 2, 20) == 20

    assert model.map == {0: 10, 1: 20, 2: 20, 3: 30}
    np.testing.assert_array_equal(model.labels_, [10, 20, 20, 30])
    np.testing.assert_array_equal(model.classes_, [10, 20, 30])
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    np.testing.assert_array_equal(original_y, [10, 20, 10, 30])
    np.testing.assert_array_equal(model.labels_a, old_a_labels)
    assert model.module_a.weight_sample_counter_ == old_counts
    for actual, expected in zip(model.module_a.W, old_weights):
        np.testing.assert_array_equal(actual, expected)


def test_simple_artmap_accepts_sparse_new_class_and_empty_source():
    model, X, _ = make_simple_artmap()

    assert model.move_A_prototype(20, 1, 99) == 99

    assert model.map == {0: 10, 1: 99, 2: 10, 3: 30}
    assert model.n_clusters_b == 3
    np.testing.assert_array_equal(model.classes_, [10, 30, 99])
    np.testing.assert_array_equal(model.labels_, [10, 99, 10, 30])
    np.testing.assert_array_equal(model.predict(X), model.labels_)


def test_artmap_moves_only_a_mapping_to_existing_b_cluster():
    model, X = make_artmap()
    old_a_weights = [weight.copy() for weight in model.module_a.W]
    old_b_weights = [weight.copy() for weight in model.module_b.W]
    old_a_counts = model.module_a.weight_sample_counter_.copy()
    old_b_counts = model.module_b.weight_sample_counter_.copy()
    old_a_labels = model.labels_a.copy()
    old_b_labels = model.labels_b.copy()

    assert model.move_A_prototype(0, 2, 1) == 1

    assert model.map == {0: 0, 1: 1, 2: 1, 3: 2}
    np.testing.assert_array_equal(model.labels_, [0, 1, 1, 2])
    np.testing.assert_array_equal(model.predict(X), model.labels_)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    np.testing.assert_array_equal(model.labels_a, old_a_labels)
    np.testing.assert_array_equal(model.labels_b, old_b_labels)
    assert model.module_a.weight_sample_counter_ == old_a_counts
    assert model.module_b.weight_sample_counter_ == old_b_counts
    for actual, expected in zip(model.module_a.W, old_a_weights):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(model.module_b.W, old_b_weights):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "source, prototype, target, error",
    [
        (10, 2, 10, ValueError),
        (20, 2, 30, ValueError),
        (10, 8, 30, IndexError),
        (10, -1, 30, IndexError),
        (10, 2, -1, IndexError),
        (10, 2, True, TypeError),
        (10, 2.0, 30, TypeError),
    ],
)
def test_simple_artmap_rejects_invalid_move_without_changes(
    source, prototype, target, error
):
    model, _, _ = make_simple_artmap()
    old_map = model.map.copy()
    old_labels = model.labels_.copy()
    old_classes = model.classes_.copy()

    with pytest.raises(error):
        model.move_A_prototype(source, prototype, target)

    assert model.map == old_map
    np.testing.assert_array_equal(model.labels_, old_labels)
    np.testing.assert_array_equal(model.classes_, old_classes)


@pytest.mark.parametrize("source, prototype, target", [(0, 2, 3), (1, 1, 0)])
def test_artmap_rejects_new_target_or_empty_source(source, prototype, target):
    model, _ = make_artmap()
    old_map = model.map.copy()
    old_labels = model.labels_.copy()
    old_b_labels = model.labels_b.copy()

    with pytest.raises((IndexError, ValueError)):
        model.move_A_prototype(source, prototype, target)

    assert model.map == old_map
    np.testing.assert_array_equal(model.labels_, old_labels)
    np.testing.assert_array_equal(model.labels_b, old_b_labels)
