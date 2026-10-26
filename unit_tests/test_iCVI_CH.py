import numpy as np
import pytest
from sklearn.metrics import calinski_harabasz_score

from artlib.cvi.CVIART import CVIART
from artlib.cvi.iCVIFuzzyArt import iCVIFuzzyART
from artlib.cvi.iCVIs.BaseICVI import Base_iCVI
from artlib.cvi.iCVIs.CH import CH
from artlib.elementary.FuzzyART import FuzzyART


def _fill(X, labels):
    index = CH()
    for x, label in zip(X, labels):
        index.update(index.add_sample(x, int(label)))
    return index


def _wgss(X, labels):
    return sum(
        ((X[labels == k] - X[labels == k].mean(axis=0)) ** 2).sum()
        for k in np.unique(labels)
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.random((80, 3))
    labels = rng.integers(0, 5, 80)
    return X, labels


def test_ch_is_a_base_icvi():
    assert isinstance(CH(), Base_iCVI)
    assert CH.increasing_optimal


def test_add_matches_sklearn(data):
    X, labels = data
    index = CH()
    for i, (x, label) in enumerate(zip(X, labels)):
        index.update(index.add_sample(x, int(label)))
        n_labels = len(np.unique(labels[: i + 1]))
        if 2 <= n_labels <= i:
            expected = calinski_harabasz_score(X[: i + 1], labels[: i + 1])
            assert index.criterion_value == pytest.approx(expected, rel=1e-10)


def test_switch_label_matches_sklearn(data):
    X, labels = data
    labels = labels.copy()
    index = _fill(X, labels)
    for i in range(0, len(X), 7):
        new = (labels[i] + 1) % 5
        index.update(index.switch_label(X[i], int(labels[i]), int(new)))
        labels[i] = new
    assert index.criterion_value == pytest.approx(
        calinski_harabasz_score(X, labels), rel=1e-10
    )
    assert index.WGSS == pytest.approx(_wgss(X, labels), rel=1e-10)


def test_switch_label_to_same_label_is_unchanged(data):
    X, labels = data
    index = _fill(X, labels)
    params = index.switch_label(X[0], int(labels[0]), int(labels[0]))
    assert params["criterion_value"] == index.criterion_value
    assert index.clustering_improved(params)


def test_trial_does_not_mutate(data):
    X, labels = data
    index = _fill(X, labels)
    before = (index.criterion_value, index.WGSS, index.n_samples, len(index.CD))
    index.add_sample(X[0], 99)
    index.switch_label(X[1], int(labels[1]), int((labels[1] + 2) % 5))
    index.remove_sample(X[2], int(labels[2]))
    index.merge_clusters(3, 1)
    after = (index.criterion_value, index.WGSS, index.n_samples, len(index.CD))
    assert before == after


def test_remove_matches_sklearn(data):
    X, labels = data
    index = _fill(X, labels)
    keep = np.ones(len(X), dtype=bool)
    for i in (3, 10, 20):
        index.update(index.remove_sample(X[i], int(labels[i])))
        keep[i] = False
    assert index.criterion_value == pytest.approx(
        calinski_harabasz_score(X[keep], labels[keep]), rel=1e-10
    )


def test_merge_matches_sklearn(data):
    X, labels = data
    labels = labels.copy()
    index = _fill(X, labels)
    params = index.merge_clusters(3, 1)
    trial_value = params["criterion_value"]
    index.update(params)
    labels[labels == 3] = 1
    assert 3 not in index.CD
    assert index.criterion_value == trial_value
    assert index.criterion_value == pytest.approx(
        calinski_harabasz_score(X, labels), rel=1e-10
    )
    assert index.WGSS == pytest.approx(_wgss(X, labels), rel=1e-10)


def test_switch_out_of_singleton_deletes_cluster(data):
    X, _ = data
    X = X[:10]
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    index = _fill(X, labels)
    index.update(index.switch_label(X[9], 3, 0))
    labels[9] = 0
    assert sorted(index.CD) == [0, 1, 2]
    assert index.criterion_value == pytest.approx(
        calinski_harabasz_score(X, labels), rel=1e-10
    )


@pytest.mark.parametrize("offline", [True, False])
def test_icvi_fuzzy_art_fits(offline):
    rng = np.random.default_rng(0)
    model = iCVIFuzzyART(
        rho=0.6,
        alpha=1e-3,
        beta=1.0,
        validity=iCVIFuzzyART.CALINSKIHARABASZ,
        offline=offline,
    )
    X = model.prepare_data(rng.random((40, 2)))
    model.fit(X)
    assert model.labels_.shape == (40,)
    assert len(model.W) > 0
    assert isinstance(model.iCVI, CH)


def test_cviart_accepts_keeping_current_label():
    rng = np.random.default_rng(0)
    model = CVIART(FuzzyART(0.5, 0.01, 1.0), CVIART.CALINSKIHARABASZ)
    X = rng.random((10, 4))
    model.data = X
    model.labels_ = np.array([0, 1] * 5)
    model.base_module.W = [X[0], X[1]]
    extra = {"validity": CVIART.CALINSKIHARABASZ, "index": 0}
    assert model.CVI_match(X[0], X[0], 0, model.params, extra, {})
