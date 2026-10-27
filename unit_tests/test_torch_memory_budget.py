"""Memory budget for the torch backends' chunked predict."""

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from artlib.optimized.backends.torch import _TorchSimpleARTMAP as base  # noqa: E402

GiB = 1 << 30


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_takes_the_tightest_limit_up_the_hierarchy(tmp_path):
    # Slurm-style nesting: the job cgroup sets the limit, the step cgroup does not.
    proc = tmp_path / "cgroup"
    proc.write_text("0::/slurm/job_1/step_0\n")
    root = tmp_path / "fs"
    _write(root / "memory.max", "max\n")
    _write(root / "slurm/memory.max", "max\n")
    _write(root / "slurm/job_1/memory.max", str(32 * GiB))
    _write(root / "slurm/job_1/memory.current", str(10 * GiB))
    _write(root / "slurm/job_1/memory.stat", f"anon 1\ninactive_file {2 * GiB}\n")
    _write(root / "slurm/job_1/step_0/memory.max", "max\n")
    _write(root / "slurm/job_1/step_0/memory.current", str(9 * GiB))
    # Inactive page cache counts as reclaimable: 32 - (10 - 2) = 24 GiB.
    assert base._cgroup_memory_headroom(str(proc), str(root)) == 24 * GiB


def test_cgroup_v2_container_root_limit(tmp_path):
    # Inside a container with its own cgroup namespace the limit is at the root.
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n")
    root = tmp_path / "fs"
    _write(root / "memory.max", str(4 * GiB))
    _write(root / "memory.current", str(1 * GiB))
    assert base._cgroup_memory_headroom(str(proc), str(root)) == 3 * GiB


def test_cgroup_v1_memory_controller(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text(
        "12:cpu,cpuacct:/slurm/uid_1/job_2\n"
        "4:memory:/slurm/uid_1/job_2/step_batch\n"
    )
    root = tmp_path / "fs"
    mem = root / "memory"
    unlimited = str(9223372036854771712)
    _write(mem / "memory.limit_in_bytes", unlimited)
    _write(mem / "slurm/uid_1/job_2/memory.limit_in_bytes", str(16 * GiB))
    _write(mem / "slurm/uid_1/job_2/memory.usage_in_bytes", str(6 * GiB))
    _write(mem / "slurm/uid_1/job_2/step_batch/memory.limit_in_bytes", unlimited)
    assert base._cgroup_memory_headroom(str(proc), str(root)) == 10 * GiB


def test_no_cgroup_limit_means_no_headroom_cap(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice\n")
    root = tmp_path / "fs"
    _write(root / "memory.max", "max\n")
    _write(root / "user.slice/memory.max", "max\n")
    assert base._cgroup_memory_headroom(str(proc), str(root)) is None
    assert base._cgroup_memory_headroom(str(tmp_path / "missing"), str(root)) is None


def test_usage_above_the_limit_leaves_no_headroom(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n")
    root = tmp_path / "fs"
    _write(root / "memory.max", str(1 * GiB))
    _write(root / "memory.current", str(2 * GiB))
    assert base._cgroup_memory_headroom(str(proc), str(root)) == 0


def test_cpu_memory_is_capped_by_the_cgroup(monkeypatch):
    monkeypatch.setattr(base, "_system_available_memory", lambda: 150 * GiB)
    monkeypatch.setattr(base, "_cgroup_memory_headroom", lambda: 20 * GiB)
    assert base._available_memory_bytes("cpu") == 20 * GiB
    assert base._predict_budget_bytes("cpu") == 5 * GiB

    monkeypatch.setattr(base, "_cgroup_memory_headroom", lambda: None)
    assert base._available_memory_bytes("cpu") == 150 * GiB


def test_budget_has_a_floor_and_a_fallback(monkeypatch):
    monkeypatch.setattr(base, "_system_available_memory", lambda: None)
    monkeypatch.setattr(base, "_cgroup_memory_headroom", lambda: 0)
    assert base._predict_budget_bytes("cpu") == base._MIN_PREDICT_BUDGET

    # Nothing readable (macOS without psutil): same 512 MiB budget as before.
    monkeypatch.setattr(base, "_cgroup_memory_headroom", lambda: None)
    assert base._predict_budget_bytes("cpu") == 512 << 20


def test_gaussian_in_place_quadratic_matches_the_original():
    gen = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        I = torch.rand(7, 5, generator=gen, dtype=dtype)
        mu = torch.rand(11, 5, generator=gen, dtype=dtype)
        iv = torch.rand(11, 5, generator=gen, dtype=dtype) + 0.5
        diff = I.unsqueeze(1) - mu.unsqueeze(0)
        expected = (diff * diff * iv.unsqueeze(0)).sum(dim=2)
        diff.mul_(diff).mul_(iv.unsqueeze(0))
        assert torch.equal(diff.sum(dim=2), expected)


def _model(kind, D):
    from artlib import (
        BinaryFuzzyARTMAP,
        FuzzyARTMAP,
        GaussianARTMAP,
        HypersphereARTMAP,
    )

    if kind == "fuzzy":
        return FuzzyARTMAP(0.8, 1e-3, 1.0, backend="torch", device="cpu")
    if kind == "binary":
        return BinaryFuzzyARTMAP(0.8, backend="torch", device="cpu")
    if kind == "hypersphere":
        return HypersphereARTMAP(0.8, 1e-3, 1.0, 4.0, backend="torch", device="cpu")
    return GaussianARTMAP(1e-60, 1e-10, np.full(D, 0.1), backend="torch", device="cpu")


@pytest.mark.parametrize("kind", ["fuzzy", "binary", "hypersphere", "gaussian"])
def test_predict_is_unchanged_by_tiny_chunks(monkeypatch, kind):
    """A budget too small for more than one category per chunk must give the same
    predictions as one large chunk."""
    rng = np.random.default_rng(0)
    D = 6
    centers = rng.random((25, D))
    lab = rng.integers(0, 25, 300)
    X = np.clip(centers[lab] + 0.03 * rng.standard_normal((300, D)), 0, 1)
    if kind == "binary":
        X = (X > 0.5).astype(np.int16)
    y = lab % 4
    m = _model(kind, D)
    Xp = m.prepare_data(X)
    m.fit(Xp[:200], y[:200])
    assert len(m.module_a.W) > 3
    expected = m.predict(Xp[200:])
    name = {"fuzzy": "FuzzyARTMAP", "binary": "BinaryFuzzyARTMAP",
            "hypersphere": "HypersphereARTMAP", "gaussian": "GaussianARTMAP"}[kind]
    module = importlib.import_module(f"artlib.optimized.backends.torch.{name}")
    monkeypatch.setattr(module, "_predict_budget_bytes", lambda device: 1)
    np.testing.assert_array_equal(m.predict(Xp[200:]), expected)
