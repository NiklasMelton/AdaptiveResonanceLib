import numpy as np
import pytest

from artlib.elementary.ART1 import ART1
from artlib.elementary.BayesianART import BayesianART
from artlib.elementary.GaussianART import GaussianART
from artlib.elementary.HypersphereART import HypersphereART


def _train_weight(model, samples):
    model.dim_ = samples.shape[1]
    weight = model.new_weight(samples[0], model.params)
    for sample in samples[1:]:
        weight = model.update(sample, weight, model.params, cache={})
    return weight


@pytest.mark.parametrize("model_type", [GaussianART, BayesianART])
def test_gaussian_merges_equal_pooled_sample_statistics(model_type):
    prior = np.array([[0.4, 0.1], [0.1, 0.7]])
    if model_type is GaussianART:
        model = GaussianART(rho=0.5, sigma_init=np.sqrt(np.diag(prior)))
        prior = np.diag(np.diag(prior))
    else:
        model = BayesianART(rho=1.0, cov_init=prior)

    left = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 4.0]])
    right = np.array([[4.0, 0.0], [5.0, 2.0]])
    all_samples = np.vstack([left, right])
    model.W = [_train_weight(model, left), _train_weight(model, right)]
    model.weight_sample_counter_ = [len(left), len(right)]
    model.labels_ = np.array([0, 0, 0, 1, 1])

    merged_idx = model.merge(1, 0)
    weight = model.W[merged_idx]
    expected_mean = np.mean(all_samples, axis=0)
    centered = all_samples - expected_mean
    expected_cov = (centered.T @ centered + prior) / len(all_samples)

    assert merged_idx == 0
    np.testing.assert_allclose(weight[:2], expected_mean)
    if model_type is GaussianART:
        np.testing.assert_allclose(np.square(weight[2:4]), np.diag(expected_cov))
        np.testing.assert_allclose(weight[4:6], 1 / np.diag(expected_cov))
        np.testing.assert_allclose(weight[-2], np.sqrt(np.prod(np.diag(expected_cov))))
    else:
        np.testing.assert_allclose(weight[2:-1].reshape(2, 2), expected_cov)
    assert weight[-1] == len(all_samples)
    assert model.weight_sample_counter_ == [len(all_samples)]
    np.testing.assert_array_equal(model.labels_, np.zeros(len(all_samples), dtype=int))

    # A later update must retain the same pooled-statistics interpretation.
    new_sample = np.array([3.0, 5.0])
    updated = model.update(new_sample, weight, model.params, cache={})
    extended = np.vstack([all_samples, new_sample])
    centered = extended - np.mean(extended, axis=0)
    expected_cov = (centered.T @ centered + prior) / len(extended)
    np.testing.assert_allclose(updated[:2], np.mean(extended, axis=0))
    if model_type is GaussianART:
        np.testing.assert_allclose(np.square(updated[2:4]), np.diag(expected_cov))
    else:
        np.testing.assert_allclose(updated[2:-1].reshape(2, 2), expected_cov)


@pytest.mark.parametrize("model_type", [GaussianART, BayesianART])
def test_repeated_gaussian_merges_keep_one_initial_covariance(model_type):
    if model_type is GaussianART:
        model = GaussianART(rho=0.5, sigma_init=np.array([0.5, 0.8]))
        prior = np.diag(np.square(model.params["sigma_init"]))
    else:
        prior = np.array([[0.5, 0.1], [0.1, 0.8]])
        model = BayesianART(rho=1.0, cov_init=prior)
    samples = np.array([[0.0, 1.0], [2.0, 4.0], [5.0, 3.0]])
    model.dim_ = 2
    model.W = [model.new_weight(x, model.params) for x in samples]
    model.weight_sample_counter_ = [1, 1, 1]
    model.labels_ = np.arange(3)

    model.merge(0, 1)
    model.merge(0, 1)

    centered = samples - samples.mean(axis=0)
    expected_cov = (centered.T @ centered + prior) / len(samples)
    if model_type is GaussianART:
        np.testing.assert_allclose(np.square(model.W[0][2:4]), np.diag(expected_cov))
    else:
        np.testing.assert_allclose(model.W[0][2:-1].reshape(2, 2), expected_cov)


@pytest.mark.parametrize("target,source,expected_idx", [(0, 2, 0), (2, 0, 1)])
def test_art1_merge_intersects_templates_and_reindexes(target, source, expected_idx):
    model = ART1(rho=0.5, L=2.0)
    model.dim_ = 3
    model.W = [
        model.new_weight(np.array([1, 1, 0]), model.params),
        model.new_weight(np.array([0, 1, 1]), model.params),
        model.new_weight(np.array([1, 0, 1]), model.params),
    ]
    model.weight_sample_counter_ = [2, 3, 4]
    model.labels_ = np.array([0, 1, 2])

    assert model.merge(target, source) == expected_idx
    np.testing.assert_array_equal(model.W[expected_idx][3:], [1, 0, 0])
    np.testing.assert_allclose(model.W[expected_idx][:3], [1, 0, 0])
    assert model.weight_sample_counter_ == ([6, 3] if expected_idx == 0 else [3, 6])
    np.testing.assert_array_equal(
        model.labels_, [0, 1, 0] if expected_idx == 0 else [1, 0, 1]
    )


def test_art1_merge_empty_template_with_l_one():
    model = ART1(rho=0.5, L=1.0)
    model.dim_ = 2
    model.W = [
        model.new_weight(np.array([1, 0]), model.params),
        model.new_weight(np.array([0, 1]), model.params),
    ]
    model.weight_sample_counter_ = [1, 1]
    np.testing.assert_array_equal(model.W[model.merge(0, 1)], np.zeros(4))


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ([0.0, 0.0, 1.0], [4.0, 0.0, 1.0], [2.0, 0.0, 3.0]),
        ([0.0, 0.0, 3.0], [1.0, 0.0, 1.0], [0.0, 0.0, 3.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 3.0], [1.0, 0.0, 3.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]),
    ],
)
def test_hypersphere_merge_smallest_enclosing_ball(first, second, expected):
    model = HypersphereART(rho=0.5, alpha=1e-5, beta=1.0, r_hat=10.0)
    model.W = [np.array(first), np.array(second)]
    model.weight_sample_counter_ = [2, 3]
    model.labels_ = np.array([0, 1])

    merged_idx = model.merge(1, 0)

    assert merged_idx == 0
    np.testing.assert_allclose(model.W[0], expected)
    assert model.weight_sample_counter_ == [5]
    np.testing.assert_array_equal(model.labels_, [0, 0])
    for original in (first, second):
        assert np.linalg.norm(model.W[0][:-1] - original[:-1]) + original[-1] <= (
            model.W[0][-1] + 1e-12
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ART1(rho=0.5, L=2.0),
        lambda: HypersphereART(rho=0.5, alpha=1e-5, beta=1.0, r_hat=10.0),
        lambda: GaussianART(rho=0.5, sigma_init=np.ones(2)),
        lambda: BayesianART(rho=1.0, cov_init=np.eye(2)),
    ],
)
def test_elementary_merge_rejects_invalid_indices_without_changes(factory):
    model = factory()
    model.dim_ = 2
    samples = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    model.W = [model.new_weight(sample, model.params) for sample in samples]
    model.weight_sample_counter_ = [1, 1]
    model.labels_ = np.array([0, 1])
    original_weights = [weight.copy() for weight in model.W]

    for indices, error in [
        ((0, 0), ValueError),
        ((0, 2), IndexError),
        ((-1, 0), IndexError),
        ((True, 0), TypeError),
    ]:
        with pytest.raises(error):
            model.merge(*indices)

    for weight, original in zip(model.W, original_weights):
        np.testing.assert_array_equal(weight, original)
    assert model.weight_sample_counter_ == [1, 1]
    np.testing.assert_array_equal(model.labels_, [0, 1])
