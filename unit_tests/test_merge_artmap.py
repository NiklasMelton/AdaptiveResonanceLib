import numpy as np
import pytest

from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.supervised.ARTMAP import ARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.common.utils import complement_code


def make_module(binary):
    if binary:
        module = BinaryFuzzyART(rho=0.5)
        weights = [
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
        ]
    else:
        module = FuzzyART(rho=0.5, alpha=0.01, beta=1.0)
        weights = [
            [0.2, 0.8, 0.7, 0.1],
            [0.4, 0.4, 0.4, 0.4],
            [0.6, 0.3, 0.2, 0.9],
        ]
    module.W = []
    for weight in weights:
        module.add_weight(np.array(weight))
    module.weight_sample_counter_ = [2, 3, 4]
    module.labels_ = np.array([0, 1, 2, 0])
    return module


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("artmap_type", [SimpleARTMAP, ARTMAP])
@pytest.mark.parametrize("target_idx, source_idx", [(0, 2), (2, 0)])
def test_merge_a_updates_module_and_map(binary, artmap_type, target_idx, source_idx):
    module_a = make_module(binary)
    if artmap_type is ARTMAP:
        model = ARTMAP(module_a, make_module(binary))
        model.map = {0: 0, 1: 1, 2: 0}
        model.labels_ = np.array([0, 1, 0, 0])
    else:
        model = SimpleARTMAP(module_a)
        model.map = {0: 10, 1: 20, 2: 10}
        model.labels_ = np.array([10, 20, 10, 10])
    old_labels_b = model.labels_.copy()
    old_weight_b = model.module_b.W[0].copy() if artmap_type is ARTMAP else None

    merged_idx = model.merge_A(target_idx, source_idx)

    assert merged_idx == (0 if target_idx == 0 else 1)
    assert model.n_clusters_a == 2
    if artmap_type is ARTMAP:
        expected_map = {0: 0, 1: 1} if target_idx == 0 else {0: 1, 1: 0}
    else:
        expected_map = {0: 10, 1: 20} if target_idx == 0 else {0: 20, 1: 10}
    assert model.map == expected_map
    np.testing.assert_array_equal(
        model.labels_a, [0, 1, 0, 0] if target_idx == 0 else [1, 0, 1, 1]
    )
    np.testing.assert_array_equal(model.labels_, old_labels_b)
    assert model.module_a.weight_sample_counter_ == (
        [6, 3] if target_idx == 0 else [3, 6]
    )
    if binary:
        assert model.module_a.w_count_cache == (
            [1, 2] if target_idx == 0 else [2, 1]
        )
    if artmap_type is ARTMAP:
        np.testing.assert_array_equal(model.module_b.W[0], old_weight_b)


@pytest.mark.parametrize("binary", [False, True])
def test_merge_a_rejects_conflicting_b_mappings_without_changes(binary):
    model = SimpleARTMAP(make_module(binary))
    model.map = {0: 10, 1: 20, 2: 10}
    before_weights = [w.copy() for w in model.module_a.W]

    with pytest.raises(ValueError, match="same B label"):
        model.merge_A(0, 1)

    assert model.map == {0: 10, 1: 20, 2: 10}
    assert model.module_a.n_clusters == 3
    for actual, expected in zip(model.module_a.W, before_weights):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("binary", [False, True])
def test_simple_artmap_merge_b_combines_class_labels(binary):
    model = SimpleARTMAP(make_module(binary))
    model.map = {0: 10, 1: 20, 2: 30}
    model.labels_ = np.array([10, 20, 30, 20])
    model.classes_ = np.array([10, 20, 30])
    old_labels_a = model.labels_a.copy()

    assert model.merge_B(10, 20) == 10

    assert model.map == {0: 10, 1: 10, 2: 30}
    np.testing.assert_array_equal(model.labels_b, [10, 10, 30, 10])
    np.testing.assert_array_equal(model.classes_, [10, 30])
    np.testing.assert_array_equal(model.labels_a, old_labels_a)
    assert model.n_clusters_b == 2


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("target_idx, source_idx", [(0, 2), (2, 0)])
def test_artmap_merge_b_updates_b_indices_and_labels(binary, target_idx, source_idx):
    model = ARTMAP(make_module(binary), make_module(binary))
    model.map = {0: 0, 1: 1, 2: 2}
    model.labels_ = model.module_b.labels_
    model.classes_ = np.array([0, 1, 2])
    old_labels_a = model.labels_a.copy()

    merged_idx = model.merge_B(target_idx, source_idx)

    assert merged_idx == (0 if target_idx == 0 else 1)
    assert model.module_b.n_clusters == 2
    assert model.map == (
        {0: 0, 1: 1, 2: 0} if target_idx == 0 else {0: 1, 1: 0, 2: 1}
    )
    np.testing.assert_array_equal(
        model.labels_b, [0, 1, 0, 0] if target_idx == 0 else [1, 0, 1, 1]
    )
    np.testing.assert_array_equal(model.labels_, model.labels_b)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.labels_a, old_labels_a)
    assert model.module_b.weight_sample_counter_ == (
        [6, 3] if target_idx == 0 else [3, 6]
    )
    if binary:
        assert model.module_b.w_count_cache == (
            [1, 2] if target_idx == 0 else [2, 1]
        )


def test_simple_artmap_merge_after_fit_keeps_predictions_and_labels_consistent():
    model = SimpleARTMAP(FuzzyART(rho=1.0, alpha=0.01, beta=1.0))
    X = complement_code(np.array([[0.0], [0.5], [1.0]]))
    y = np.array([10, 10, 20])
    model.fit(X, y)
    assert model.n_clusters_a == 3

    assert model.merge_A(0, 1) == 0
    assert model.merge_B(10, 20) == 10

    assert model.map == {0: 10, 1: 10}
    np.testing.assert_array_equal(model.labels_a, [0, 0, 1])
    np.testing.assert_array_equal(model.labels_b, [10, 10, 10])
    np.testing.assert_array_equal(model.classes_, [10])
    np.testing.assert_array_equal(model.predict(X), [10, 10, 10])
