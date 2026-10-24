"""Shared MNIST parity check for the optimized ARTMAP backends.

Training is sequential, so a backend bug can stay hidden until a rare event
(a near-tie in choice, a long match-tracking chain, many categories) happens
thousands of samples in. We therefore train on the full ``N_TRAIN`` samples and
compare the per-sample category assignments, which pins down the first sample
where two backends diverge.

Prediction is stateless per sample, and the training-time assignments already
exercise category choice on every training sample, so a smaller test slice is
enough there. Both sizes can be raised with environment variables for a full
run, e.g. ``ARTLIB_PARITY_TEST=10000``.
"""

import os
from functools import lru_cache
from time import perf_counter

import numpy as np
from sklearn.datasets import fetch_openml

N_TRAIN = int(os.environ.get("ARTLIB_PARITY_TRAIN", 10000))
N_TEST = int(os.environ.get("ARTLIB_PARITY_TEST", 2000))


@lru_cache(maxsize=1)
def _load_mnist_raw():
    # The pandas ARFF parser fails on MNIST with pandas 3 (sklearn 1.9).
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    return mnist["data"].astype(np.float32), mnist["target"].astype(int)


def load_mnist(binary: bool = False):
    """Return (X, y, n_train): the first N_TRAIN training rows followed by the
    first N_TEST rows of the standard 10k test split."""
    X_all, y_all = _load_mnist_raw()
    if binary:
        X_all = (X_all > 128).astype(np.int16)
    else:
        X_all = X_all / 255.0
    X = np.vstack([X_all[:N_TRAIN], X_all[60000 : 60000 + N_TEST]])
    y = np.concatenate([y_all[:N_TRAIN], y_all[60000 : 60000 + N_TEST]])
    return X, y, N_TRAIN


def _first_mismatch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    bad = np.flatnonzero(a != b)
    if bad.size == 0:
        return None
    i = int(bad[0])
    return f"{bad.size} differ, first at index {i} ({a[i]} vs {b[i]})"


def run_parity(models: dict, capsys, binary: bool = False):
    """Fit every model in ``models`` (name -> unfitted model) on the same data
    and check that all of them agree with the first one."""
    X, y, n_train = load_mnist(binary=binary)
    y_train = y[:n_train]
    errors = []

    def timed(label, fn, *args):
        t0 = perf_counter()
        out = fn(*args)
        with capsys.disabled():
            print(f"[TIMING] {label}: {perf_counter() - t0:.3f} s", flush=True)
        return out

    def check(name, msg):
        if msg is not None:
            errors.append(f"{name}: {msg}")
            with capsys.disabled():
                print(f"[MISMATCH] {name}: {msg}", flush=True)

    results = {}
    for name, model in models.items():
        x = timed(f"prepare_data ({name})", model.prepare_data, X)
        x_train, x_test = x[:n_train], x[n_train:]
        model = timed(f"fit ({name})", model.fit, x_train, y_train)
        y_a, y_b = timed(f"predict_ab ({name})", model.predict_ab, x_test)
        results[name] = dict(
            x=x,
            labels=np.asarray(model.module_a.labels_),
            map=[model.map[c] for c in range(len(model.map))],
            W=np.vstack(model.module_a.W),
            y_a=np.asarray(y_a),
            y_b=np.asarray(y_b),
        )

    ref_name, ref = next(iter(results.items()))
    for name, res in results.items():
        if name == ref_name:
            continue
        tag = f"{name} vs {ref_name}"
        if not np.allclose(res["x"], ref["x"]):
            check(tag, "prepared data differs")
        # Per-sample training assignments: the earliest signal of divergence.
        check(f"{tag} train labels", _first_mismatch(res["labels"], ref["labels"]))
        check(f"{tag} category map", _first_mismatch(res["map"], ref["map"]))
        if res["W"].shape != ref["W"].shape:
            check(f"{tag} weights", f"shape {res['W'].shape} vs {ref['W'].shape}")
        elif not np.allclose(res["W"], ref["W"]):
            rows = np.flatnonzero(~np.isclose(res["W"], ref["W"]).all(axis=1))
            check(f"{tag} weights", f"{rows.size} categories differ, first {rows[0]}")
        check(f"{tag} predict A", _first_mismatch(res["y_a"], ref["y_a"]))
        check(f"{tag} predict B", _first_mismatch(res["y_b"], ref["y_b"]))

    if errors:
        raise AssertionError(" | ".join(errors))
