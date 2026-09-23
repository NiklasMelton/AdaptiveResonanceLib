import numpy as np
import pytest

from artlib.elementary.ART1 import ART1
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.elementary.FuzzyART import FuzzyART
from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART


def test_merge_unimplemented_for_other_art_models():
    model = ART1(rho=0.5, L=2.0)

    with pytest.raises(NotImplementedError):
        model.merge(0, 1)


def test_icvi_fuzzy_art_merge_remains_unimplemented():
    model = iCVIFuzzyART(rho=0.5, alpha=0.0, beta=1.0, validity=1)

    with pytest.raises(NotImplementedError):
        model.merge(0, 1)


@pytest.mark.parametrize(
    "target_idx, source_idx, expected_labels, expected_target_idx",
    [
        (0, 2, [0, 1, 0, 0, 1], 0),
        (2, 0, [1, 0, 1, 1, 0], 1),
    ],
)
def test_fuzzy_art_merge_updates_model(
    target_idx, source_idx, expected_labels, expected_target_idx
):
    model = FuzzyART(rho=0.5, alpha=0.0, beta=0.5)
    model.W = [
        np.array([0.2, 0.8, 0.7, 0.1]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.6, 0.3, 0.2, 0.9]),
    ]
    model.weight_sample_counter_ = [2, 3, 4]
    model.labels_ = np.array([0, 1, 2, 0, 1])
    model.sample_counter_ = 9

    merged_idx = model.merge(target_idx, source_idx)

    assert merged_idx == expected_target_idx
    assert model.n_clusters == 2
    np.testing.assert_array_equal(model.W[merged_idx], [0.2, 0.3, 0.2, 0.1])
    np.testing.assert_array_equal(model.W[1 - merged_idx], [0.4, 0.4, 0.4, 0.4])
    assert model.weight_sample_counter_ == ([6, 3] if merged_idx == 0 else [3, 6])
    np.testing.assert_array_equal(model.labels_, expected_labels)
    assert model.sample_counter_ == 9


def test_binary_fuzzy_art_merge_updates_weight_count_cache():
    model = BinaryFuzzyART(rho=0.5)
    model.W = []
    model.add_weight(np.array([True, True, False, False]))
    model.add_weight(np.array([True, False, True, False]))
    model.add_weight(np.array([False, False, True, True]))
    model.weight_sample_counter_ = [2, 3, 4]
    model.labels_ = np.array([0, 1, 2, 1])

    merged_idx = model.merge(1, 0)

    assert merged_idx == 0
    assert model.n_clusters == 2
    np.testing.assert_array_equal(model.W[0], [True, False, False, False])
    assert model.W[0].dtype == np.bool_
    assert model.weight_sample_counter_ == [5, 4]
    assert model.w_count_cache == [1, 2]
    np.testing.assert_array_equal(model.labels_, [0, 0, 1, 0])

    assert model.merge(0, 1) == 0
    assert model.n_clusters == 1
    np.testing.assert_array_equal(model.W[0], [False, False, False, False])
    assert model.weight_sample_counter_ == [9]
    assert model.w_count_cache == [1]
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0])


def test_fuzzy_art_merge_rejects_invalid_indices_without_modifying_model():
    model = FuzzyART(rho=0.5, alpha=0.0, beta=1.0)
    model.W = [np.array([0.2, 0.8]), np.array([0.3, 0.7])]
    model.weight_sample_counter_ = [2, 3]
    model.labels_ = np.array([0, 1])

    for indices, error in [
        ((0, 0), ValueError),
        ((0, 2), IndexError),
        ((-1, 0), IndexError),
    ]:
        with pytest.raises(error):
            model.merge(*indices)

    np.testing.assert_array_equal(model.W, [[0.2, 0.8], [0.3, 0.7]])
    assert model.weight_sample_counter_ == [2, 3]
    np.testing.assert_array_equal(model.labels_, [0, 1])
