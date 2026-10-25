"""Batch scoring and the other step-loop speed-ups must not change results."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from artlib import (
    FuzzyART,
    BinaryFuzzyART,
    HypersphereART,
    GaussianART,
    SimpleARTMAP,
    FusionART,
    TopoART,
    DualVigilanceART,
)
from artlib.common.BaseART import stacked_weights


def _disable_batch(model):
    """Force the per-cluster code path."""
    model._batch_category_choice = lambda i, params: None
    if hasattr(model, "_intersection_counts"):
        model._intersection_counts = lambda x: None
    return model


MODELS = {
    "fuzzy64": (lambda: FuzzyART(0.9, 1e-3, 1.0), np.float64, False),
    "fuzzy32": (lambda: FuzzyART(0.9, 1e-3, 0.7), np.float32, False),
    "hypersphere": (lambda: HypersphereART(0.9, 1e-3, 0.5, 0.8), np.float64, False),
    "gaussian": (lambda: GaussianART(0.2, 0.05 * np.ones(6), 1e-10), np.float64, False),
    "binary_fuzzy": (lambda: BinaryFuzzyART(0.8), np.float64, True),
}


@pytest.fixture(scope="module")
def data():
    X, y = make_blobs(
        n_samples=700, n_features=6, centers=40, cluster_std=0.5, random_state=1
    )
    return X, y % 4


def _prepare(name, X):
    make, dtype, binary = MODELS[name]
    X = (X > X.mean(axis=0)).astype(int) if binary else X.astype(dtype)
    return make, X


def _fit(model, X, y=None, match_tracking="MT+"):
    Xp = model.prepare_data(X)
    if y is None:
        model.fit(Xp[:600])
        return np.asarray(model.labels_), np.vstack(model.W), model.predict(Xp[600:])
    artmap = SimpleARTMAP(model)
    artmap.fit(Xp[:600], y[:600], match_tracking=match_tracking, epsilon=1e-7)
    W = np.vstack(artmap.module_a.W)
    return np.asarray(artmap.module_a.labels_), W, artmap.predict(Xp[600:])


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("match_tracking", [None, "MT+", "MT-", "MT0", "MT~"])
def test_batch_matches_per_cluster(name, match_tracking, data):
    X, y = data
    make, X = _prepare(name, X)
    yy = None if match_tracking is None else y
    fast = _fit(make(), X, yy, match_tracking)
    slow = _fit(_disable_batch(make()), X, yy, match_tracking)
    assert fast[1].shape[0] > 5  # enough clusters for the check to mean something
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["fuzzy64", "hypersphere", "gaussian"])
def test_batch_scores_match_category_choice(name, data):
    X, _ = data
    make, X = _prepare(name, X)
    model = make()
    Xp = model.prepare_data(X)
    model.fit(Xp[:300])
    for x in Xp[300:320]:
        T, caches = model._batch_category_choice(x, model.params)
        for c, w in enumerate(model.W):
            t, cache = model.category_choice(x, w, model.params)
            assert T[c] == t
            assert caches[c] == cache or (caches[c] is None and cache is None)


def test_stacked_weights_follow_changes():
    model = FuzzyART(0.9, 1e-3, 1.0)
    model.W = [np.full(4, k, dtype=float) for k in range(3)]
    M, _ = stacked_weights(model)
    np.testing.assert_array_equal(M, np.vstack(model.W))
    model.W[1] = np.full(4, 9.0)  # replaced
    model.W.append(np.full(4, 7.0))  # appended
    M, _ = stacked_weights(model)
    np.testing.assert_array_equal(M, np.vstack(model.W))
    del model.W[0]  # removed
    M, _ = stacked_weights(model)
    np.testing.assert_array_equal(M, np.vstack(model.W))
    model.W = [np.zeros(4)]  # whole list replaced
    M, _ = stacked_weights(model)
    np.testing.assert_array_equal(M, np.vstack(model.W))


def test_batch_after_merge_matches_per_cluster(data):
    X, _ = data
    fast, slow = FuzzyART(0.9, 1e-3, 1.0), _disable_batch(FuzzyART(0.9, 1e-3, 1.0))
    for model in (fast, slow):
        Xp = model.prepare_data(X)
        model.fit(Xp[:300])
        model.merge(0, 1)
        model.partial_fit(Xp[300:600])
    np.testing.assert_array_equal(np.vstack(fast.W), np.vstack(slow.W))
    np.testing.assert_array_equal(fast.labels_, slow.labels_)


def test_fusion_batch_matches_per_cluster(data):
    X, y = data

    def make():
        return FusionART(
            [FuzzyART(0.8, 1e-3, 1.0), HypersphereART(0.8, 1e-3, 0.5, 0.8)],
            gamma_values=[0.6, 0.4],
            channel_dims=[6, 3],
        )

    def run(model, batch):
        if not batch:
            for module in model.modules:
                _disable_batch(module)
        Xj = model.join_channel_data(
            [model.modules[0].prepare_data(X[:, :3]), model.modules[1].prepare_data(X[:, 3:])]
        )
        model.fit(Xj[:600])
        return (
            np.asarray(model.labels_),
            model.predict(Xj[600:]),
            model.predict(Xj[600:], skip_channels=[1]),
        )

    for a, b in zip(run(make(), True), run(make(), False)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "make",
    [
        lambda: FuzzyART(0.7, 1e-3, 1.0),
        lambda: FusionART([FuzzyART(0.7, 1e-3, 1.0)], [1.0], [4]),
        lambda: BinaryFuzzyART(0.7),
    ],
)
def test_params_restored_when_reset_raises(make):
    model = make()
    X = np.random.default_rng(0).random((20, 2))
    if isinstance(model, BinaryFuzzyART):
        X = (X > 0.5).astype(int)
    if isinstance(model, FusionART):
        Xp = model.join_channel_data([model.modules[0].prepare_data(X)])
    else:
        Xp = model.prepare_data(X)
    model.fit(Xp[:10])
    before = model._copy_params()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        for x in Xp[10:]:
            model.step_fit(x, match_reset_func=explode, match_tracking="MT+")
    assert model._copy_params() == before


def test_reset_only_called_after_vigilance_passes(data):
    X, _ = data
    model = FuzzyART(0.85, 1e-3, 1.0)
    Xp = model.prepare_data(X)
    model.fit(Xp[:300])
    calls = []

    def reset(x, w, c, params, cache):
        calls.append(cache["match_criterion_bin"])
        return c % 2 == 0

    for x in Xp[300:400]:
        model.step_fit(x, match_reset_func=reset)
    assert calls and all(calls)


@pytest.mark.parametrize("tau,phi", [(50, 3), (100, 6)])
def test_topoart_prune_relabel(tau, phi):
    X, _ = make_blobs(600, n_features=2, centers=15, random_state=2)
    model = TopoART(FuzzyART(0.85, 1e-3, 1.0), beta_lower=0.3, tau=tau, phi=phi)
    Xp = model.prepare_data(X)
    model.fit(Xp)
    labels = np.asarray(model.labels_)
    assert labels.min() >= 0 and labels.max() < model.n_clusters
    assert model.adjacency.shape == (model.n_clusters, model.n_clusters)


def test_dual_vigilance_batch_matches_per_cluster(data):
    X, _ = data
    runs = []
    for batch in (True, False):
        base = FuzzyART(0.9, 1e-3, 1.0)
        if not batch:
            _disable_batch(base)
        model = DualVigilanceART(base, 0.7)
        Xp = model.prepare_data(X)
        model.fit(Xp[:600])
        runs.append((np.asarray(model.labels_), model.predict(Xp[600:])))
    for a, b in zip(*runs):
        np.testing.assert_array_equal(a, b)
