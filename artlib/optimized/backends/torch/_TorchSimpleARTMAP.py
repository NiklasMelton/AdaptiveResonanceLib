import warnings

import numpy as np
import torch
from typing import Literal, Optional, Union, Tuple

from artlib.optimized.backend_warnings import BackendPerformanceWarning
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

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


# Below this many weight entries (categories x weight length), a model trained on
# CUDA is small enough that the C++ backend is usually much faster. Training on
# CUDA costs a few hundred microseconds per sample at any size; on a V100 with 16
# CPU threads, C++ trained 40-category models 100-500x faster, while CUDA only
# started to win around a few million entries. 100,000 is a conservative point
# in between.
_SMALL_CUDA_MODEL = 100_000


def _backend_advice(device, n_weights: int) -> Optional[str]:
    """Why this torch device is a poor choice for a model this size, or None."""
    kind = torch.device(device).type
    if kind == "mps":
        return (
            "The torch backend on an Apple GPU (device='mps') was slower than the "
            "C++ backend in every benchmark. Use backend='c++'."
        )
    if kind == "cpu":
        return (
            "The torch backend on CPU was slower than, or at best about as fast as, "
            "the C++ backend in every benchmark. Use backend='c++', or "
            "device='cuda' for large models."
        )
    if kind == "cuda" and n_weights < _SMALL_CUDA_MODEL:
        return (
            f"This model is small ({n_weights:,} weight entries). Training on CUDA "
            "has a fixed cost per sample, so the C++ backend is usually much faster "
            "for models this size. Use backend='c++'."
        )
    return None


class _TorchSimpleARTMAP(SimpleARTMAP):
    def _warn_on_slow_backend(self) -> None:
        """Warn when another backend is likely much faster for this model."""
        W = self.module_a.W
        n_weights = len(W) * (len(W[0]) if len(W) else 0)
        advice = _backend_advice(self._backend.device, n_weights)
        if advice is not None:
            # stacklevel=3 points at the caller of fit / partial_fit.
            warnings.warn(advice, BackendPerformanceWarning, stacklevel=3)

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
        self._warn_on_slow_backend()
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
        self._warn_on_slow_backend()
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
