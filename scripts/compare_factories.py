#!/usr/bin/env python3
"""
Driver for distributed, isolated ARTMAP benchmarks (one task per Slurm array index).

This script:
  • maps each array index to exactly one (method, backend) task
  • loads MNIST (binary or float) once per task
  • times prepare/fit/predict
  • writes a compact JSON result with timings + node/SLURM metadata
  • exits non-zero on failure (so Slurm marks the task as failed)

Within-task threading is left to the model backends.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import subprocess
from typing import Dict, Tuple, Any, Optional

import numpy as np

# --- artlib imports ---
from artlib.optimized.BinaryFuzzyARTMAPFactory import BinaryFuzzyARTMAPFactory
from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory
from artlib.optimized.HypersphereARTMAPFactory import HypersphereARTMAPFactory
from artlib.optimized.GaussianARTMAPFactory import GaussianARTMAPFactory


import torch
from torchvision.datasets import MNIST
from torchvision import transforms

def _mnist_root():
    # Prefer node-local cache if available
    return os.environ.get("TORCH_HOME", os.environ.get("SLURM_TMPDIR", "/tmp/torchdata"))

def _load_mnist_numpy_binary():
    """
    Returns:
        X_train (10000, 784) int32 in {0,1}
        y_train (10000,) int
        X_test  (10000, 784) int32 in {0,1}
        y_test  (10000,) int
    """
    root = _mnist_root()
    # ToTensor -> float32 in [0,1], then we threshold > 0.5
    tfm = transforms.ToTensor()

    ds_train = MNIST(root=root, train=True, download=True, transform=tfm)
    ds_test  = MNIST(root=root, train=False, download=True, transform=tfm)

    n_train = 10_000
    n_test = len(ds_test)  # 10,000

    # Stack efficiently with torch, then convert to numpy
    Xtr = torch.stack([ds_train[i][0].view(-1) for i in range(n_train)])  # (10000, 784), float32 [0,1]
    ytr = np.array([int(ds_train[i][1]) for i in range(n_train)], dtype=int)

    Xte = torch.stack([ds_test[i][0].view(-1) for i in range(n_test)])     # (10000, 784), float32 [0,1]
    yte = np.array([int(ds_test[i][1]) for i in range(n_test)], dtype=int)

    # Binarize
    X_train = (Xtr.numpy() > 0.5).astype(np.int32)
    X_test  = (Xte.numpy() > 0.5).astype(np.int32)

    return X_train, ytr, X_test, yte


def _load_mnist_numpy():
    """
    Returns:
        X_train (10000, 784) float32 in [0,1]
        y_train (10000,) int
        X_test  (10000, 784) float32 in [0,1]
        y_test  (10000,) int
    """
    root = _mnist_root()
    tfm = transforms.ToTensor()  # yields float32 in [0,1]

    ds_train = MNIST(root=root, train=True, download=True, transform=tfm)
    ds_test  = MNIST(root=root, train=False, download=True, transform=tfm)

    n_train = 10_000
    n_test = len(ds_test)

    Xtr = torch.stack([ds_train[i][0].view(-1) for i in range(n_train)])  # (10000, 784), float32 [0,1]
    ytr = np.array([int(ds_train[i][1]) for i in range(n_train)], dtype=int)

    Xte = torch.stack([ds_test[i][0].view(-1) for i in range(n_test)])     # (10000, 784), float32 [0,1]
    yte = np.array([int(ds_test[i][1]) for i in range(n_test)], dtype=int)

    X_train = Xtr.numpy().astype(np.float32)
    X_test  = Xte.numpy().astype(np.float32)

    return X_train, ytr, X_test, yte



# ------------------------------
# timing helper
# ------------------------------
def time_call(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[TIMING] {label}: {dt:.3f} s", flush=True)
    return out, dt


# ------------------------------
# task registry
# ------------------------------
# Hyperparams from your sample
RHO = 0.8
ALPHA = 1e-10
BETA = 0.33 * np.ones((784,), dtype=np.float32)
R_HAT = 28.0
SIGMA_INIT = 0.33 * np.ones((784,), dtype=np.float32)

METHODS = ["BinaryFuzzyARTMAP", "FuzzyARTMAP", "HypersphereARTMAP", "GaussianARTMAP"]
BACKENDS = ["python", "torch", "c++"]  # task granularity is method x backend

def make_factory(method: str, backend: str):
    if method == "BinaryFuzzyARTMAP":
        return BinaryFuzzyARTMAPFactory(RHO, ALPHA, backend=backend)
    if method == "FuzzyARTMAP":
        return FuzzyARTMAPFactory(RHO, ALPHA, BETA, backend=backend)
    if method == "HypersphereARTMAP":
        return HypersphereARTMAPFactory(RHO, ALPHA, BETA, R_HAT, backend=backend)
    if method == "GaussianARTMAP":
        return GaussianARTMAPFactory(RHO, ALPHA, SIGMA_INIT, backend=backend)
    raise ValueError(f"Unknown method: {method}")

def load_data_for_method(method: str):
    if method == "BinaryFuzzyARTMAP":
        return _load_mnist_numpy_binary()
    return _load_mnist_numpy()

# Build a stable list of tasks (indexable)
TASKS = [(m, b) for m in METHODS for b in BACKENDS]  # length 12


# ------------------------------
# metadata capture for comparability
# ------------------------------
def _read_lscpu() -> Dict[str, str]:
    try:
        out = subprocess.check_output(["bash", "-lc", "LC_ALL=C lscpu"], text=True)
        md = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                md[k.strip()] = v.strip()
        return md
    except Exception:
        return {}

def _read_nvidia_smi() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"],
            text=True,
        )
        gpus = []
        for row in out.strip().splitlines():
            name, driver, mem = [c.strip() for c in row.split(",")]
            gpus.append({"name": name, "driver": driver, "memory_total": mem})
        return {"gpus": gpus}
    except Exception:
        return {}

def _slurm_env() -> Dict[str, str]:
    keys = [
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_NODELIST",
        "SLURM_SUBMIT_DIR", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_NODE"
    ]
    return {k: os.environ.get(k, "") for k in keys}

def _versions() -> Dict[str, str]:
    # local imports to avoid mandatory torch on non-torch tasks
    import numpy
    import sklearn
    ver = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
    }
    try:
        import torch
        ver["torch"] = torch.__version__
        ver["torch_cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        pass
    return ver


# ------------------------------
# main execution for one task
# ------------------------------
@dataclass
class Result:
    method: str
    backend: str
    timings: Dict[str, float]
    n_train: int
    n_test: int
    accuracy: Optional[float]  # we can compute simple accuracy for sanity
    meta: Dict[str, Any]

def run_single_task(method: str, backend: str, seed: int = 0) -> Result:
    # Reproducibility where possible
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # Don't touch intra-op threading; models handle it
    except Exception:
        pass

    # Load data for this method
    X_train, y_train, X_test, y_test = load_data_for_method(method)
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    n_train = X_train.shape[0]

    # Build factory for this backend
    factory = make_factory(method, backend)

    # Timed prepare
    x, t_prepare = time_call(f"prepare_data ({backend})", factory.prepare_data, X)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train2, y_test2 = y[:n_train], y[n_train:]

    # Timed fit
    model, t_fit = time_call(f"fit ({backend})", factory.fit, x_train, y_train2)

    # Timed predict
    y_pred, t_pred = time_call(f"predict ({backend})", model.predict, x_test)

    # Simple accuracy sanity-check
    acc = float((y_pred == y_test2).mean()) if y_pred is not None else None

    meta = {
        "versions": _versions(),
        "slurm": _slurm_env(),
        "lscpu": _read_lscpu(),
        "nvidia_smi": _read_nvidia_smi(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
    }

    return Result(
        method=method,
        backend=backend,
        timings={
            "prepare": t_prepare,
            "fit": t_fit,
            "predict": t_pred,
            "total": t_prepare + t_fit + t_pred,
        },
        n_train=int(n_train),
        n_test=int(len(y_test2)),
        accuracy=acc,
        meta=meta,
    )


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run a single ARTMAP benchmark task.")
    parser.add_argument("--task-index", type=int, default=None,
                        help="Index in the TASKS list (e.g., use SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--method", choices=METHODS, default=None,
                        help="Override: method name (ignores --task-index backend part unless --backend also set).")
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help="Override: backend name.")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Directory for JSON outputs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--require-cpu-model", type=str, default=None,
                        help="If set, fail fast unless lscpu 'Model name' matches exactly (enforce like machines).")
    args = parser.parse_args()

    # Resolve task
    if args.method and args.backend:
        method, backend = args.method, args.backend
    elif args.task_index is not None:
        try:
            method, backend = TASKS[args.task_index]
        except IndexError:
            print(f"ERROR: task-index {args.task_index} out of range [0, {len(TASKS)-1}]", file=sys.stderr)
            sys.exit(2)
    else:
        print("ERROR: provide --task-index or both --method and --backend", file=sys.stderr)
        sys.exit(2)

    # Optional enforcement of CPU model (runtime guard)
    if args.require_cpu_model:
        cpuinfo = _read_lscpu()
        model_name = cpuinfo.get("Model name", "")
        if model_name != args.require_cpu_model:
            print(f"ERROR: CPU model mismatch. Expected '{args.require_cpu_model}', got '{model_name}'", file=sys.stderr)
            sys.exit(3)

    # Run task
    try:
        res = run_single_task(method, backend, seed=args.seed)
    except Exception as e:
        print(f"FATAL: task ({method}, {backend}) failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Write output JSON (unique, deterministic name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Embed array id if present to avoid collisions
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    suffix = f"{method}-{backend}".lower()
    if array_id:
        fname = f"{suffix}-array{array_id}.json"
    else:
        fname = f"{suffix}.json"
    out_path = out_dir / fname

    with open(out_path, "w") as f:
        json.dump(asdict(res), f, indent=2)
    print(f"[OK] wrote {out_path}", flush=True)

    # Also echo summary to stdout for live logs
    print(json.dumps({
        "method": res.method,
        "backend": res.backend,
        "n_train": res.n_train,
        "n_test": res.n_test,
        "accuracy": res.accuracy,
        "timings": res.timings,
        "host": res.meta.get("hostname", "")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
