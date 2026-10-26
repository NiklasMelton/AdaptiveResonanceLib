import numpy as np
import torch
from typing import Literal, Optional, Union, Tuple

from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

import os

from torch import Tensor


def _to_device(x: Union[Tensor, "np.ndarray"], device, dtype=torch.float32) -> Tensor:
    """Copy ``x`` to ``device`` as ``dtype``.

    The copy is blocking. An asynchronous copy can read its source after it is freed
    (for example the temporary made by a dtype change), and on MPS that returns garbage
    values.

    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    raise TypeError("Expected torch.Tensor or numpy.ndarray")


# cgroup files report "no limit" as "max" (v2) or a value near 2**63 (v1).
_UNLIMITED = 1 << 60
# Used when the available memory cannot be read (no psutil and no /proc/meminfo,
# as on macOS). It gives the same 512 MiB predict budget as before.
_FALLBACK_AVAILABLE = 2 << 30
# Smallest predict budget, so chunks stay usable when memory is nearly full.
_MIN_PREDICT_BUDGET = 64 << 20


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if text == "max":
        return _UNLIMITED
    try:
        return int(text)
    except ValueError:
        return None


def _read_stat(path: str, key: str) -> int:
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _cgroup_memory_headroom(
    proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup"
) -> Optional[int]:
    """Bytes this process can still allocate under its cgroup memory limits.

    Containers (Docker, Kubernetes) and schedulers such as Slurm cap memory with
    cgroups, which /proc/meminfo and psutil do not see. A limit can be set on any
    ancestor of the process's cgroup, so every level from the process's cgroup up
    to the root is checked and the smallest headroom wins. Usage excludes inactive
    page cache, which the kernel reclaims before killing anything (as ``docker
    stats`` does). Handles cgroup v2 and v1. Returns None when no limit applies or
    the files cannot be read (for example, outside Linux).

    """
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    headroom: Optional[int] = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            base, limit_name, usage_name, cache_key = (
                root,
                "memory.max",
                "memory.current",
                "inactive_file",
            )
        elif "memory" in controllers.split(","):
            base, limit_name, usage_name, cache_key = (
                os.path.join(root, "memory"),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                "total_inactive_file",
            )
        else:
            continue
        rel = path.strip("/")
        while True:
            level = os.path.join(base, rel) if rel else base
            limit = _read_int(os.path.join(level, limit_name))
            if limit is not None and limit < _UNLIMITED:
                usage = _read_int(os.path.join(level, usage_name)) or 0
                usage -= _read_stat(os.path.join(level, "memory.stat"), cache_key)
                room = max(limit - max(usage, 0), 0)
                headroom = room if headroom is None else min(headroom, room)
            if not rel:
                break
            rel = os.path.dirname(rel)
    return headroom


def _system_available_memory() -> Optional[int]:
    """Available system memory from psutil or /proc/meminfo, else None."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _available_memory_bytes(device) -> int:
    """Memory that a computation on ``device`` can still allocate.

    CUDA: free device memory. CPU and MPS: available system memory, capped by any
    cgroup (container or scheduler) limit; MPS is also capped by the Metal
    working-set limit, since its tensors live in the same unified memory.

    """
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info(dev)
            return int(free)
        except Exception:
            pass
    limits = [_system_available_memory(), _cgroup_memory_headroom()]
    if dev.type == "mps":
        try:
            limits.append(
                int(torch.mps.recommended_max_memory())
                - int(torch.mps.driver_allocated_memory())
            )
        except Exception:
            pass
    known = [x for x in limits if x is not None]
    return max(min(known), 0) if known else _FALLBACK_AVAILABLE


def _predict_budget_bytes(device) -> int:
    """Bytes the [batch, categories, features] temporary in predict may use.

    A quarter of the memory still available on ``device`` (see
    ``_available_memory_bytes``), and at least 64 MiB. Each backend keeps a single
    temporary of that size alive at once.

    """
    return max(_available_memory_bytes(device) // 4, _MIN_PREDICT_BUDGET)


def _resolve_float_dtype(device, dtype: Optional[torch.dtype]) -> torch.dtype:
    """Pick the floating point dtype for ``device``.

    ``None`` means float64, except on MPS, which has no float64 and gets float32.

    """
    is_mps = torch.device(device).type == "mps"
    if dtype is None:
        return torch.float32 if is_mps else torch.float64
    if is_mps and dtype == torch.float64:
        raise ValueError(
            "MPS does not support float64. Pass dtype=torch.float32 or leave it unset."
        )
    return dtype


def _scratch(owner, name: str, rows: int, like: Tensor) -> Tensor:
    """Return a ``[rows, *like.shape[1:]]`` view of a reusable buffer.

    Fit makes one ``[K, D]`` temporary per sample while K grows one category at
    a time. Allocating it fresh asks the caching allocator for a new, slightly
    larger size each step, and MPS keeps every old block cached, so held memory
    grows with K^2 * D. A buffer that grows by doubling needs only a few sizes.

    """
    buffers = owner.__dict__.setdefault("_scratch_buffers", {})
    tail = tuple(like.shape[1:])
    buf = buffers.get(name)
    if (
        buf is None
        or buf.shape[0] < rows
        or tuple(buf.shape[1:]) != tail
        or buf.dtype != like.dtype
    ):
        buffers.pop(name, None)  # release the old buffer before allocating
        buf = torch.empty(
            (max(16, 2 * rows),) + tail, device=like.device, dtype=like.dtype
        )
        buffers[name] = buf
    return buf[:rows]


def _append_rows(owner, n: int, **rows: Tensor) -> None:
    """Append one row to each named tensor attribute of ``owner``.

    Each attribute becomes a view of the first ``n + 1`` rows of a buffer with
    spare capacity, so adding a category does not copy every existing row (as
    ``torch.cat`` did). Existing rows can still be edited in place through the
    view.

    """
    buffers = owner.__dict__.setdefault("_row_buffers", {})
    for name, row in rows.items():
        current = getattr(owner, name)
        buf = buffers.get(name)
        stale = (
            buf is None
            or n >= buf.shape[0]
            or buf.shape[1:] != row.shape
            or buf.dtype != row.dtype
            or (n > 0 and current.data_ptr() != buf.data_ptr())
        )
        if stale:
            new = torch.empty(
                (max(16, 2 * n),) + tuple(row.shape), device=row.device, dtype=row.dtype
            )
            if n:
                new[:n] = current[:n]
            buf = new
            buffers[name] = buf
        buf[n] = row
        setattr(owner, name, buf[: n + 1])


class _TorchSimpleARTMAP(SimpleARTMAP):
    def _synchronize_torch_results(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        cluster_labels_out: np.ndarray,
        incremental: bool = False,
    ):
        if not incremental:
            self.map: dict[int, int] = {}
            self.module_a.labels_ = np.array((), dtype=int)
            self.module_a.weight_sample_counter_ = []

        # labels
        self.module_a.labels_ = np.concatenate(
            [self.module_a.labels_, labels_a_out.astype(int)]
        )

        # sample counters
        new_counts = np.bincount(labels_a_out, minlength=len(weights_arrays))
        if len(self.module_a.weight_sample_counter_) < len(new_counts):
            self.module_a.weight_sample_counter_.extend(
                [0] * (len(new_counts) - len(self.module_a.weight_sample_counter_))
            )
        for k, c in enumerate(new_counts):
            self.module_a.weight_sample_counter_[k] += int(c)

        # weights (store as float64 numpy arrays to match artlib expectations)
        self.module_a.W = [w for w in weights_arrays]

        # A→B mapping
        for c_a, c_b in enumerate(cluster_labels_out):
            if c_a in self.map:
                assert self.map[c_a] == int(c_b), "Incremental fit changed cluster map."
            else:
                self.map[c_a] = int(c_b)

    # --- public API (matches the C++ wrapper)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.
        max_iter : int, default=1
            Number of iterations to fit the model on the same data set.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.
        verbose : bool, default=False
            non functional. Left for compatibility
        leave_progress_bar : bool, default=True
            non functional. Left for compatibility

        Returns
        -------
        self : SimpleARTMAP
            The fitted model.

        """
        SimpleARTMAP.validate_data(self, X, y)
        self._ensure_backend(X)
        assert self._backend is not None

        # artlib-style bookkeeping
        self.classes_ = unique_labels(y)
        self.labels_ = y
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        # Expect X already normalized and complement-coded,
        Xp = _to_device(X, self._backend.device, self._backend.dtype)
        la, W, cl = self._backend.partial_fit_and_export(
            Xp, y, epsilon=epsilon, match_tracking=match_tracking
        )
        self._synchronize_torch_results(la, W, cl, incremental=False)
        self.module_a.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        """Partial fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.

        Returns
        -------
        self : SimpleARTMAP
            The partially fitted model.

        """
        SimpleARTMAP.validate_data(self, X, y)
        self._ensure_backend(X)
        assert self._backend is not None

        if not hasattr(self, "labels_"):
            self.labels_ = y
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, (0, len(y)))
            self.labels_[j:] = y

        Xp = _to_device(X, self._backend.device, self._backend.dtype)
        la, W, cl = self._backend.partial_fit_and_export(
            Xp, y, epsilon=epsilon, match_tracking=match_tracking
        )
        self._synchronize_torch_results(la, W, cl, incremental=True)
        self.module_a.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Predict labels for the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        np.ndarray
            B labels for the data.

        """
        check_is_fitted(self)
        assert self._backend is not None
        # Optional clipping, mirroring C++ wrapper behavior
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)

        # Use backend for batched predict; mirror C++ by ignoring vigilance here.
        y_a, y_b = self._backend.predict_ab_prepared(
            _to_device(X, self._backend.device, self._backend.dtype)
        )
        return y_b

    def predict_ab(
        self, X: np.ndarray, clip: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels for the data, both A-side and B-side.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A labels for the data, B labels for the data.

        """
        check_is_fitted(self)
        assert self._backend is not None
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        y_a, y_b = self._backend.predict_ab_prepared(
            _to_device(X, self._backend.device, self._backend.dtype)
        )
        return y_a, y_b
