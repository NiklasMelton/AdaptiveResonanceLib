import numpy as np
import pytest

torch = pytest.importorskip("torch")

from artlib.optimized.backends.torch._TorchSimpleARTMAP import (  # noqa: E402
    _resolve_float_dtype,
    _scratch,
)

HAS_MPS = torch.backends.mps.is_available()


def test_float_dtype_defaults_to_float32_only_on_mps():
    assert _resolve_float_dtype("cpu", None) == torch.float64
    assert _resolve_float_dtype("mps", None) == torch.float32
    assert _resolve_float_dtype("cpu", torch.float32) == torch.float32
    with pytest.raises(ValueError, match="float64"):
        _resolve_float_dtype("mps", torch.float64)


def test_scratch_reuses_one_buffer_while_rows_grow():
    class Owner:
        pass

    owner, like = Owner(), torch.empty((0, 3))
    first = _scratch(owner, "x", 5, like)
    assert first.shape == (5, 3)
    # Growing within capacity, and shrinking, reuse the same memory.
    for rows in (6, 10, 2, 16):
        assert _scratch(owner, "x", rows, like).data_ptr() == first.data_ptr()
    # Past capacity it reallocates once, with room to grow again.
    bigger = _scratch(owner, "x", 17, like)
    assert bigger.data_ptr() != first.data_ptr()
    assert _scratch(owner, "x", 30, like).data_ptr() == bigger.data_ptr()


@pytest.mark.skipif(not HAS_MPS, reason="needs an Apple GPU (MPS)")
@pytest.mark.parametrize("model", ["fuzzy", "binary", "hypersphere"])
def test_mps_predicts_like_cpu(model):
    from artlib.optimized.BinaryFuzzyARTMAPFactory import BinaryFuzzyARTMAPFactory
    from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory
    from artlib.optimized.HypersphereARTMAPFactory import HypersphereARTMAPFactory

    rng = np.random.default_rng(0)
    centers = rng.random((30, 16))
    lab = rng.integers(0, 30, 400)
    X = np.clip(centers[lab] + 0.03 * rng.standard_normal((400, 16)), 0, 1)
    y = lab % 4
    make = {
        "fuzzy": lambda d: FuzzyARTMAPFactory(0.8, 1e-3, 1.0, backend="torch", device=d),
        "binary": lambda d: BinaryFuzzyARTMAPFactory(0.8, backend="torch", device=d),
        "hypersphere": lambda d: HypersphereARTMAPFactory(
            0.8, 1e-3, 1.0, 4.0, backend="torch", device=d
        ),
    }[model]
    if model == "binary":
        X = (X > 0.5).astype(np.int16)

    preds = {}
    for device in ("cpu", "mps"):
        m = make(device)
        Xp = m.prepare_data(X)
        m.fit(Xp[:300], y[:300])
        preds[device] = m.predict(Xp[300:])
    np.testing.assert_array_equal(preds["mps"], preds["cpu"])
