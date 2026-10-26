"""Contracts shared by the native ART model extensions."""

import importlib
import subprocess
import sys

import numpy as np
import pytest


CONTINUOUS = np.array(
    [[0.1, 0.8, 0.9, 0.2], [0.2, 0.7, 0.8, 0.3], [0.9, 0.1, 0.1, 0.9],
     [0.8, 0.2, 0.2, 0.8], [0.15, 0.75, 0.85, 0.25]],
    dtype=np.float64,
)
RAW = CONTINUOUS[:, :2].copy()
BINARY = np.array(
    [[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]],
    dtype=np.int32,
)
ART1_INPUT = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int16)
TARGET = np.array([0, 0, 1, 1, 0], dtype=np.int32)


CASES = [
    ("ART1", ART1_INPUT.astype(np.float64), {"rho": 0.4, "L": 2.0}, False),
    ("ART1MAP", ART1_INPUT, {"rho": 0.4, "L": 2.0}, True),
    ("BinaryFuzzyART", BINARY.astype(np.bool_), {"rho": 0.5}, False),
    ("BinaryFuzzyARTMAP", BINARY, {"rho": 0.5}, True),
    ("FuzzyART", CONTINUOUS, {"rho": 0.5, "alpha": 1e-10, "beta": 1.0}, False),
    ("FuzzyARTMAP", CONTINUOUS,
     {"rho": 0.5, "alpha": 1e-10, "beta": 1.0}, True),
    ("GaussianARTMAP", RAW,
     {"rho": 0.5, "alpha": 1e-10, "sigma_init": np.array([0.2, 0.2])}, True),
    ("HypersphereARTMAP", RAW,
     {"rho": 0.5, "alpha": 1e-10, "beta": 1.0, "r_hat": 2.0}, True),
]


@pytest.mark.parametrize("name,X,params,supervised", CASES)
def test_native_state_roundtrip(name, X, params, supervised):
    module = importlib.import_module(f"artlib.optimized.backends.cpp.cpp{name}")
    assert hasattr(module, "fit") and hasattr(module, "predict")
    assert not hasattr(module, f"Fit{name}")
    assert not hasattr(module, f"cpp{name}")

    modes = ("MT+", "MT-", "MT0", "MT1", "MT~") if supervised else (None,)
    for mode in modes:
        kwargs = dict(params)
        if supervised:
            kwargs.update(MT=mode, epsilon=1 if name == "BinaryFuzzyARTMAP" else 1e-10)
        full = module.fit(X, TARGET, **kwargs) if supervised else module.fit(X, **kwargs)
        first = (module.fit(X[:2], TARGET[:2], **kwargs) if supervised
                 else module.fit(X[:2], **kwargs))
        restored = dict(kwargs, weights=first[1])
        if supervised:
            restored["cluster_labels"] = first[2]
            second = module.fit(X[2:], TARGET[2:], **restored)
        else:
            second = module.fit(X[2:], **restored)

        np.testing.assert_array_equal(
            np.concatenate((first[0], second[0])), full[0]
        )
        for actual, expected in zip(second[1], full[1]):
            np.testing.assert_array_equal(actual, expected)
        if supervised:
            np.testing.assert_array_equal(second[2], full[2])

        restored = dict(kwargs, weights=full[1])
        if supervised:
            restored["cluster_labels"] = full[2]
        prediction = module.predict(X, **restored)
        labels_a = prediction[0] if supervised else prediction
        assert labels_a.shape == (len(X),)


def test_binary_artmap_preserves_negative_labels():
    module = importlib.import_module(
        "artlib.optimized.backends.cpp.cppBinaryFuzzyARTMAP"
    )
    labels = np.array([-3, -3, 7, 7, -3], dtype=np.int32)
    fitted = module.fit(BINARY, labels, rho=0.5, MT="MT+", epsilon=1)
    assert -3 in fitted[2]
    _, predicted = module.predict(
        BINARY, rho=0.5, MT="", epsilon=0,
        weights=fitted[1], cluster_labels=fitted[2],
    )
    np.testing.assert_array_equal(predicted, labels)


def test_noncontiguous_input_matches_contiguous_input():
    module = importlib.import_module("artlib.optimized.backends.cpp.cppFuzzyART")
    view = np.empty((len(CONTINUOUS), CONTINUOUS.shape[1] * 2))
    view[:, ::2] = CONTINUOUS
    view[:, 1::2] = -1
    assert not view[:, ::2].flags.c_contiguous
    args = {"rho": 0.5, "alpha": 1e-10, "beta": 1.0}
    actual = module.fit(view[:, ::2], **args)
    expected = module.fit(CONTINUOUS, **args)
    np.testing.assert_array_equal(actual[0], expected[0])
    for a, b in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "code",
    [
        "m.fit(np.ones((2, 3)), rho=.5, alpha=1e-10, beta=1.)",
        "m.fit(np.ones((2, 4)), np.array([1]), rho=.5, alpha=1e-10, "
        "beta=1., MT='MT+', epsilon=1e-10)",
    ],
)
def test_malformed_input_raises_instead_of_aborting(code):
    model = "cppFuzzyARTMAP" if "np.array([1])" in code else "cppFuzzyART"
    native_path = importlib.import_module(
        f"artlib.optimized.backends.cpp.{model}"
    ).__file__
    script = (
        "import importlib.util, numpy as np, sys\n"
        "spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "try:\n"
        f"    {code}\n"
        "except ValueError:\n"
        "    print('VALUE_ERROR')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, model, native_path],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "VALUE_ERROR" in result.stdout


def test_unfitted_prediction_raises_runtime_error():
    module = importlib.import_module("artlib.optimized.backends.cpp.cppFuzzyART")
    with pytest.raises(RuntimeError, match="no clusters"):
        module.predict(CONTINUOUS, rho=0.5, alpha=1e-10, beta=1.0)


def test_empty_fit_and_invalid_loaded_state():
    module = importlib.import_module("artlib.optimized.backends.cpp.cppFuzzyART")
    params = {"rho": 0.5, "alpha": 1e-10, "beta": 1.0}
    labels, weights = module.fit(np.empty((0, 4)), **params)
    assert labels.size == 0 and weights == []
    with pytest.raises(ValueError, match="weight length"):
        module.predict(CONTINUOUS, weights=[np.ones(3)], **params)

    artmap = importlib.import_module("artlib.optimized.backends.cpp.cppFuzzyARTMAP")
    with pytest.raises(ValueError, match="size mismatch"):
        artmap.predict(
            CONTINUOUS, rho=0.5, alpha=1e-10, beta=1.0,
            MT="", epsilon=0.0, weights=[np.ones(4)],
            cluster_labels=np.array([], dtype=np.int32),
        )


@pytest.mark.parametrize("name,X,params,supervised", CASES)
def test_python_wrapper_partial_fit_matches_fit(name, X, params, supervised):
    wrapper = importlib.import_module(f"artlib.optimized.backends.cpp.{name}")
    model_type = getattr(wrapper, name)
    full = model_type(**params)
    incremental = model_type(**params)
    if supervised:
        full.fit(X, TARGET)
        incremental.partial_fit(X[:2], TARGET[:2])
        incremental.partial_fit(X[2:], TARGET[2:])
        full_state = full.module_a
        incremental_state = incremental.module_a
    else:
        full.fit(X)
        incremental.partial_fit(X[:2])
        incremental.partial_fit(X[2:])
        full_state = full
        incremental_state = incremental
    np.testing.assert_array_equal(incremental_state.labels_, full_state.labels_)
    for actual, expected in zip(incremental_state.W, full_state.W):
        np.testing.assert_array_equal(actual, expected)
    assert len(incremental_state.W) == len(full_state.W)


def test_first_partial_fit_uses_training_and_keeps_label_length():
    from artlib.optimized.backends.cpp.ART1 import ART1
    from artlib.optimized.backends.cpp.BinaryFuzzyART import BinaryFuzzyART

    for model, X in (
        (ART1(rho=0.4, L=2.0), ART1_INPUT),
        (BinaryFuzzyART(rho=0.5), BINARY),
    ):
        model.partial_fit(X[:2])
        model.partial_fit(X[2:])
        assert len(model.labels_) == len(X)
        assert len(model.W) > 0


def _threaded_predict_case(name, rng):
    """Inputs big enough for predict to split the samples across threads.

    The feature count is not a multiple of 8, so the distance kernels' fused
    tail runs too.
    """
    n, k, d = 600, 64, 301
    if name == "BinaryFuzzyARTMAP":
        Xh = rng.random((n, d)) < 0.5
        X = np.hstack([Xh, ~Xh]).astype(np.int32)
        W = (rng.random((k, 2 * d)) < 0.4).astype(np.int32)
        params = {"rho": 0.5, "MT": "MT+", "epsilon": 1}
    elif name == "FuzzyARTMAP":
        Xh = rng.random((n, d))
        X = np.hstack([Xh, 1 - Xh])
        Wh = rng.random((k, d))
        W = np.hstack([0.9 * Wh, 0.9 * (1 - Wh)])
        params = {"rho": 0.5, "alpha": 1e-3, "beta": 1.0, "MT": "", "epsilon": 0.0}
    elif name == "GaussianARTMAP":
        X = rng.random((n, d))
        sigma = 0.5 + rng.random((k, d))
        W = np.hstack([
            rng.random((k, d)), sigma, 1 / sigma**2, np.ones((k, 1)),
            rng.integers(1, 10, (k, 1)).astype(float),
        ])
        params = {"rho": 0.5, "alpha": 1e-10, "sigma_init": np.full(d, 0.5),
                  "MT": "", "epsilon": 0.0}
    else:
        X = rng.random((n, d))
        W = np.hstack([rng.random((k, d)), 0.2 * rng.random((k, 1))])
        params = {"rho": 0.5, "alpha": 1e-3, "beta": 1.0,
                  "r_hat": float(np.sqrt(d)) + 1, "MT": "", "epsilon": 0.0}
    return X, W, params


@pytest.mark.parametrize(
    "name", ["BinaryFuzzyARTMAP", "FuzzyARTMAP", "GaussianARTMAP", "HypersphereARTMAP"]
)
def test_threaded_predict_matches_single_thread(name):
    module = importlib.import_module(f"artlib.optimized.backends.cpp.cpp{name}")
    X, W, params = _threaded_predict_case(name, np.random.default_rng(0))
    k = len(W)
    # Every category twice: a duplicate ties its original and must never win.
    weights = list(np.vstack([W, W]))
    cluster_labels = np.tile(np.arange(k, dtype=np.int32) % 3, 2)
    single = module.predict(X, **params, weights=weights, cluster_labels=cluster_labels,
                            n_threads=1)
    assert single[0].max() < k
    assert len(np.unique(single[0])) > 1
    for threads in (2, 5):
        threaded = module.predict(X, **params, weights=weights,
                                  cluster_labels=cluster_labels, n_threads=threads)
        np.testing.assert_array_equal(threaded[0], single[0])
        np.testing.assert_array_equal(threaded[1], single[1])


def _threaded_fit_case(name, rng):
    """Training data where scoring one sample is big enough to split across threads.

    About 400 categories form, and the feature counts are not multiples of 8, so
    the distance kernels' fused tail runs too.
    """
    n, d = 800, 301
    centres = rng.random((400, d))
    lab = rng.integers(0, 400, n)
    Xh = np.clip(centres[lab] + 0.02 * rng.standard_normal((n, d)), 0, 1)
    y = (lab % 3).astype(np.int32)
    if name == "BinaryFuzzyARTMAP":
        Xb = np.tile(Xh > 0.5, 7)
        X = np.hstack([Xb, ~Xb]).astype(np.int32)
        params = {"rho": 0.9, "MT": "MT+", "epsilon": 1}
    elif name == "FuzzyARTMAP":
        X = np.hstack([Xh, 1 - Xh])
        params = {"rho": 0.9, "alpha": 1e-3, "beta": 1.0, "MT": "MT+", "epsilon": 1e-10}
    elif name == "GaussianARTMAP":
        X = Xh
        params = {"rho": 1e-60, "alpha": 1e-10, "sigma_init": np.full(d, 0.1),
                  "MT": "MT+", "epsilon": 1e-10}
    else:
        X = Xh
        params = {"rho": 0.9, "alpha": 1e-3, "beta": 0.5, "r_hat": float(np.sqrt(d)),
                  "MT": "MT+", "epsilon": 1e-10}
    return X, y, params


@pytest.mark.parametrize(
    "name", ["BinaryFuzzyARTMAP", "FuzzyARTMAP", "GaussianARTMAP", "HypersphereARTMAP"]
)
def test_threaded_fit_matches_single_thread(name):
    module = importlib.import_module(f"artlib.optimized.backends.cpp.cpp{name}")
    X, y, params = _threaded_fit_case(name, np.random.default_rng(0))
    labels, W, cluster_labels = module.fit(X, y, **params, n_threads=1)
    assert len(W) > 200
    cut = len(X) // 2
    _, W_half, cl_half = module.fit(X[:cut], y[:cut], **params, n_threads=1)
    partial = module.fit(X[cut:], y[cut:], **params, weights=W_half,
                         cluster_labels=cl_half, n_threads=1)
    for threads in (2, 5):
        threaded = module.fit(X, y, **params, n_threads=threads)
        np.testing.assert_array_equal(threaded[0], labels)
        np.testing.assert_array_equal(threaded[2], cluster_labels)
        assert len(threaded[1]) == len(W)
        for actual, expected in zip(threaded[1], W):
            np.testing.assert_array_equal(actual, expected)

        threaded = module.fit(X[cut:], y[cut:], **params, weights=W_half,
                              cluster_labels=cl_half, n_threads=threads)
        np.testing.assert_array_equal(threaded[0], partial[0])
        np.testing.assert_array_equal(threaded[2], partial[2])
        for actual, expected in zip(threaded[1], partial[1]):
            np.testing.assert_array_equal(actual, expected)


_DEFAULT_THREADS = (
    "import os, json; "
    "from artlib.optimized.backends.cpp import cppFuzzyARTMAP as m; "
    "{setup}"
    "print(json.dumps([m.__file__, os.cpu_count()] + list(m._default_threads())))"
)


def _default_threads(setup="", env_threads=None):
    """(machine CPUs, default fit threads, default predict threads) in a fresh
    process, optionally after ``setup`` code such as a CPU affinity change.

    The defaults are read once per process, so each case needs its own. The
    process runs in an empty directory: ``python -c`` puts its working directory
    first on sys.path, and from the repository root that would import the
    uncompiled source tree instead of the installed package.
    """
    import json
    import os
    import tempfile

    from artlib.optimized.backends.cpp import cppFuzzyARTMAP

    env = dict(os.environ)
    env.pop("ARTLIB_NUM_THREADS", None)
    if env_threads is not None:
        env["ARTLIB_NUM_THREADS"] = env_threads
    with tempfile.TemporaryDirectory() as empty:
        out = subprocess.run(
            [sys.executable, "-c", _DEFAULT_THREADS.format(setup=setup)],
            capture_output=True, text=True, env=env, cwd=empty,
        )
    assert out.returncode == 0, out.stderr
    module_file, *counts = json.loads(out.stdout)
    # The child must load the same build as this test process.
    assert os.path.samefile(module_file, cppFuzzyARTMAP.__file__)
    return tuple(counts)


def test_default_threads_use_all_cpus_to_predict_and_half_to_fit():
    import os

    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < os.cpu_count():
        pytest.skip("this process already runs with a restricted CPU affinity mask")
    cpus, fit, predict = _default_threads()
    assert predict == cpus
    assert fit == max(1, cpus // 2)


def test_env_threads_override_the_defaults():
    assert _default_threads(env_threads="3")[1:] == (3, 3)
    # Invalid values fall back to the defaults.
    assert _default_threads(env_threads="zero")[1:] == _default_threads()[1:]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs sched_setaffinity")
def test_default_threads_follow_the_cpu_affinity_mask():
    """Slurm, taskset and container cpusets restrict the affinity mask; the
    defaults must count only the CPUs this process may run on."""
    import os

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs at least two CPUs")
    for n in sorted({1, len(allowed) // 2}):
        cpus = set(allowed[:n])
        _, fit, predict = _default_threads(setup=f"os.sched_setaffinity(0, {cpus!r}); ")
        # A restricted mask is an allocation: fit and predict both use all of it.
        assert (fit, predict) == (n, n)
