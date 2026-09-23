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
    script = (
        "import numpy as np\n"
        f"import artlib.optimized.backends.cpp.{model} as m\n"
        "try:\n"
        f"    {code}\n"
        "except ValueError:\n"
        "    print('VALUE_ERROR')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
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
