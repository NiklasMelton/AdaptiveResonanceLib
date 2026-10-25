"""Checks for the vectorized helpers in BARTMAP, DeepARTMAP, FALCON and CVIART."""
import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.datasets import make_blobs

from artlib import BARTMAP, CVIART, FALCON, FuzzyART
from artlib.hierarchical.DeepARTMAP import _map_labels


def test_pearsonr_rows_matches_scipy():
    rng = np.random.default_rng(0)
    A, b = rng.random((40, 13)), rng.random(13)
    expected = [pearsonr(a, b)[0] for a in A]
    np.testing.assert_allclose(BARTMAP._pearsonr_rows(A, b), expected, atol=1e-12)
    assert np.isnan(BARTMAP._pearsonr(np.ones(5), np.arange(5.0)))
    assert np.isnan(BARTMAP._pearsonr([1.0], [2.0]))


def test_bartmap_fits_non_square_data():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3, 20))
    X = np.vstack([base[i % 3] + 0.1 * rng.standard_normal(20) for i in range(45)])
    model = BARTMAP(FuzzyART(0.5, 1e-3, 1.0), FuzzyART(0.5, 1e-3, 1.0), eta=0.5)
    model.fit(X)
    assert model.row_labels_.shape == (45,)
    assert model.column_labels_.shape == (20,)
    assert model.rows_.shape == (model.n_row_clusters * model.n_column_clusters, 45)


def test_bartmap_compares_with_rows_of_the_candidate_cluster():
    rng = np.random.default_rng(2)
    X = rng.random((12, 8))
    model = BARTMAP(FuzzyART(0.5, 1e-3, 1.0), FuzzyART(0.5, 1e-3, 1.0), eta=0.0)
    model.X = X
    model.module_b = model.module_b.fit(model.module_b.prepare_data(X.T))
    model.module_a.labels_ = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model._rows_seen_ = np.arange(12) < 6  # rows 6.. not assigned yet
    members = model._row_members(3, 0)
    np.testing.assert_array_equal(np.flatnonzero(members), [0, 2, 4])
    cols = model.column_labels_ == 0
    expected = np.mean([pearsonr(X[3, cols], X[r, cols])[0] for r in (0, 2, 4)])
    assert np.isclose(model._average_pearson_corr(X, 3, 0, cluster_a=0), expected)


def test_map_labels_matches_dict_lookup():
    mapping = {0: 5, 1: 3, 2: 5, 3: 0}
    labels = np.array([3, 0, 2, 2, 1, 0])
    expected = np.asarray([mapping[int(c)] for c in labels])
    result = _map_labels(mapping, labels)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype
    with pytest.raises(KeyError):
        _map_labels(mapping, np.array([0, 7]))


def test_falcon_scores_all_actions_like_single_predicts():
    rng = np.random.default_rng(0)
    states = rng.random((200, 2))
    actions = rng.integers(0, 4, (200, 1)).astype(float) / 3
    rewards = (np.sin(3 * states[:, :1]) * actions + 1) / 2.5
    falcon = FALCON(
        FuzzyART(0.7, 1e-3, 1.0),
        FuzzyART(0.95, 1e-3, 1.0),
        FuzzyART(0.9, 1e-3, 1.0),
        channel_dims=[4, 2, 2],
    )
    modules = falcon.fusion_art.modules
    s = modules[0].prepare_data(states)
    a = modules[1].prepare_data(actions)
    r = modules[2].prepare_data(rewards)
    falcon.fit(s, a, r)
    reward_centers = falcon.fusion_art.get_channel_centers(2)
    for state in s[:10]:
        action_space, rewards_out = falcon.get_actions_and_rewards(state)
        prepared = modules[1].prepare_data(action_space)
        expected = []
        for action in prepared:
            data = falcon.fusion_art.join_channel_data(
                [state.reshape(1, -1), action.reshape(1, -1)], skip_channels=[2]
            )
            c = falcon.fusion_art.predict(data, skip_channels=[2])[0]
            expected.append(reward_centers[c])
        np.testing.assert_array_equal(rewards_out, np.array(expected))


def test_cvi_match_memo_gives_same_answer():
    X, _ = make_blobs(60, n_features=2, centers=3, random_state=0)
    model = CVIART(FuzzyART(0.8, 1e-3, 1.0), validity=CVIART.CALINSKIHARABASZ)
    model.fit(model.prepare_data(X))
    memo = {}
    for c in range(model.n_clusters):
        extra = {"index": 5, "validity": CVIART.CALINSKIHARABASZ}
        plain = model.CVI_match(None, None, c, model.params, extra, None)
        cached = model.CVI_match(None, None, c, model.params, dict(extra, memo=memo), None)
        assert plain == cached
    assert "old_VI" in memo
