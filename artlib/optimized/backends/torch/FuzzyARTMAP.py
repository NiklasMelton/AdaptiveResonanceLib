"""Fuzzy ARTMAP :cite:`carpenter1991fuzzy`."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, cast
import numpy as np
import torch

from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.elementary.FuzzyART import FuzzyART
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


from torch import Tensor


# ------------------------------
# utilities
# ------------------------------
def _to_device(x: Union[Tensor, "np.ndarray"], device, dtype=torch.float32) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    import numpy as np

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True)
    raise TypeError("Expected torch.Tensor or numpy.ndarray")


def _complement_code(x: Tensor) -> Tensor:
    # x ∈ [0,1]^M  →  [x, 1-x]
    return torch.cat([x, 1.0 - x], dim=-1)


# -----------------
# Torch GPU backend
# -----------------
@dataclass
class _TorchFuzzyARTMAPConfig:
    input_dim: int
    alpha: float = 1e-3
    rho: float = 0.75
    beta: float = 1.0
    epsilon: float = 1e-7
    complement: bool = True
    match_tracking: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float64
    clamp_inputs: bool = True
    fallback_to_choice_on_fail: bool = True


class _TorchFuzzyARTMAP:
    """GPU-accelerated Fuzzy ARTMAP with export hooks for artlib synchronization."""

    def __init__(self, cfg: _TorchFuzzyARTMAPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.input_dim = int(cfg.input_dim)
        self.code_dim = self.input_dim * (2 if cfg.complement else 1)

        self.W: Optional[Tensor] = None  # [K, D]
        self.map_y: Optional[Tensor] = None  # [K]
        self._lower_bounds: Optional[Tensor] = None
        self._upper_bounds: Optional[Tensor] = None
        self._prep_tol: float = 1e-6

    @property
    def n_cat(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def _ensure_capacity(self):
        if self.W is None:
            self.W = torch.empty(
                (0, self.code_dim), device=self.device, dtype=self.dtype
            )
            self.map_y = torch.empty((0,), device=self.device, dtype=torch.long)

    def _prep_input(self, X: Tensor) -> Tensor:
        if self.cfg.clamp_inputs:
            X = torch.clamp(X, 0.0, 1.0)
        return _complement_code(X) if self.cfg.complement else X

    def _validate_prepared(self, X: Tensor):
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D]")
        if self.cfg.complement:
            if X.shape[1] != 2 * self.input_dim:
                raise ValueError(
                    f"With complement=True, expected D={2*self.input_dim}, "
                    f"got {X.shape[1]}"
                )
            D = self.input_dim
            a, b = X[:, :D], X[:, D:]
            if not (
                torch.all(a >= -self._prep_tol)
                and torch.all(a <= 1.0 + self._prep_tol)
                and torch.all(b >= -self._prep_tol)
                and torch.all(b <= 1.0 + self._prep_tol)
            ):
                raise ValueError("Prepared inputs must be in [0,1].")
            if not torch.allclose(b, 1.0 - a, atol=1e-5, rtol=0):
                raise ValueError(
                    "Second half must be 1 - first half (complement coding)."
                )
        else:
            if X.shape[1] != self.input_dim:
                raise ValueError(
                    f"With complement=False, expected D={self.input_dim}, "
                    f"got {X.shape[1]}"
                )
            if not (
                torch.all(X >= -self._prep_tol) and torch.all(X <= 1.0 + self._prep_tol)
            ):
                raise ValueError("Prepared inputs must be in [0,1].")

    def set_data_bounds(
        self, lower: Union[Tensor, "np.ndarray"], upper: Union[Tensor, "np.ndarray"]
    ):
        lb = _to_device(lower, self.device, self.dtype).view(-1)
        ub = _to_device(upper, self.device, self.dtype).view(-1)
        if lb.numel() != self.input_dim or ub.numel() != self.input_dim:
            raise ValueError(f"lower/upper must have length input_dim={self.input_dim}")
        if not torch.all(ub > lb):
            raise ValueError(
                "All upper bounds must be strictly greater than lower bounds."
            )
        self._lower_bounds, self._upper_bounds = lb, ub

    def prepare_data(
        self, X: Union[Tensor, "np.ndarray"], complement: Optional[bool] = None
    ) -> Tensor:
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError(
                "Call set_data_bounds(lower, upper) before prepare_data()."
            )
        X = _to_device(X, self.device, self.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected raw dimensionality {self.input_dim}, got {X.shape[1]}"
            )
        denom = self._upper_bounds - self._lower_bounds
        Xn = (X - self._lower_bounds) / (denom + 1e-12)
        Xn = torch.clamp(Xn, 0.0, 1.0)
        use_comp = self.cfg.complement if complement is None else complement
        return _complement_code(Xn) if use_comp else Xn

    # ---- core ops
    def _choice_and_match(self, I: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (T, m, I_sum, IandW_sum) for a single prepared input I."""
        if self.W is None or self.W.shape[0] == 0:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            return (
                empty,
                empty,
                torch.tensor(0.0, device=self.device, dtype=self.dtype),
                empty,
            )
        IandW = torch.minimum(I.unsqueeze(0), self.W)  # [K, D]
        IandW_sum = IandW.sum(dim=1)  # [K]
        W_sum = self.W.sum(dim=1)  # [K]
        I_sum = I.sum()  # scalar
        T = IandW_sum / (self.cfg.alpha + W_sum)
        m = IandW_sum / I_sum.clamp_min(1e-12)
        return T, m, I_sum, IandW_sum

    def _commit_new_category(self, I: Tensor, y: int):
        self._ensure_capacity()
        if self.cfg.beta < 1.0:
            w0 = torch.ones((1, self.code_dim), device=self.device, dtype=self.dtype)
            w_new = (
                self.cfg.beta * torch.minimum(I.unsqueeze(0), w0)
                + (1.0 - self.cfg.beta) * w0
            )
        else:
            w_new = I.unsqueeze(0)
        self.W = torch.cat([self.W, w_new], dim=0)
        self.map_y = torch.cat(
            [self.map_y, torch.tensor([y], device=self.device, dtype=torch.long)], dim=0
        )

    def partial_fit_and_export(
        self,
        X_prepared: Union[Tensor, "np.ndarray"],
        y: Union[Tensor, "np.ndarray"],
        epsilon: float = 1e-10,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
    ) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Incremental training on already-prepared inputs.

        Returns:
        labels_a_out (np.ndarray): per-sample chosen A-side category indices
        weights_arrays (list[np.ndarray]): per-category weights (float64)
        cluster_labels_out (np.ndarray): map from A categories to B labels

        """
        Xp = _to_device(X_prepared, self.device, self.dtype)
        y = _to_device(y, self.device, torch.long)
        self._validate_prepared(Xp)

        # training
        la: list[int] = []
        for i in range(Xp.shape[0]):
            Ii = Xp[i]
            yi = int(y[i].item())

            if self.n_cat == 0:
                self._commit_new_category(Ii, yi)
                la.append(0)
                continue
            assert self.map_y is not None and self.W is not None
            T, m, _, _ = self._choice_and_match(Ii)
            order = torch.argsort(T, descending=True, stable=True)

            rho_eff = float(self.cfg.rho)
            found = False
            chosen_idx = None

            for idx in order.tolist():
                if m[idx].item() < rho_eff:
                    continue

                if int(self.map_y[idx].item()) == yi:
                    # resonance + learn
                    wj = self.W[idx]
                    I_and_w = torch.minimum(Ii, wj)
                    beta = self.cfg.beta
                    self.W[idx] = beta * I_and_w + (1.0 - beta) * wj
                    found = True
                    chosen_idx = idx
                    break
                else:
                    if (
                        match_tracking != ""
                    ):  # mimic MT variants simply by enabling/disabling
                        rho_eff = float(m[idx].item()) + float(epsilon)

            if not found:
                self._commit_new_category(Ii, yi)
                chosen_idx = self.n_cat - 1

            la.append(int(cast(int, chosen_idx)))

        # export numpy payloads for wrapper synchronization
        assert self.W is not None and self.map_y is not None
        W_np = [
            self.W[k].detach().to("cpu").numpy().astype(np.float64, copy=True)
            for k in range(self.n_cat)
        ]
        cl_np = self.map_y.detach().to("cpu").numpy().astype(int, copy=True)
        la_np = np.asarray(la, dtype=int)
        return la_np, W_np, cl_np

    @torch.no_grad()
    def predict_ab_prepared(
        self, X_prepared: Union[Tensor, "np.ndarray"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xp = _to_device(X_prepared, self.device, self.dtype)
        self._validate_prepared(Xp)
        if self.n_cat == 0:
            raise RuntimeError("Model has no categories. Train first.")

        W = cast(Tensor, self.W)
        W_sum = W.sum(dim=1).unsqueeze(0)  # [1, K]
        N = Xp.shape[0]
        block = 2048
        a_idx = []
        b_lab = []

        for s in range(0, N, block):
            e = min(N, s + block)
            I = Xp[s:e]  # [B, D]
            IandW = torch.minimum(I.unsqueeze(1), W.unsqueeze(0))  # [B, K, D]
            IandW_sum = IandW.sum(dim=2)  # [B, K]
            T = IandW_sum / (self.cfg.alpha + W_sum)  # [B, K]
            idx = torch.argmax(T, dim=1)  # [B]
            a_idx.append(idx.to("cpu"))
            b_lab.append(cast(Tensor, self.map_y)[idx].to("cpu"))

        y_a = torch.cat(a_idx, dim=0).numpy().astype(int)
        y_b = torch.cat(b_lab, dim=0).numpy().astype(int)
        return y_a, y_b


class FuzzyARTMAP(SimpleARTMAP):
    """FuzzyARTMAP for Classification. optimized with torch.

    This module implements FuzzyARTMAP

    FuzzyARTMAP is a non-modular classification model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    torch for efficient execution. This class acts as a wrapper for the underlying torch
    functions and to provide compatibility with the artlib style and usage.
    Functionally, FuzzyARTMAP behaves as a special case of
    :class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` instantiated with
    :class:`~artlib.elementary.FuzzyART.FuzzyART`.

    """

    def __init__(
        self,
        rho: float,
        alpha: float,
        beta: float,
        input_dim: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
        complement: bool = True,
    ):
        """Initialize the Fuzzy ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning rate.

        """
        module_a = FuzzyART(rho=rho, alpha=alpha, beta=beta)
        super().__init__(module_a)

        # torch back-end
        self._device = device
        self._dtype = dtype
        self._complement = complement
        self._backend: Optional[_TorchFuzzyARTMAP] = None
        self._declared_input_dim = input_dim  # raw dimensionality (pre-complement)

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

    # --- helpers
    def _ensure_backend(self, X: np.ndarray):
        if self._backend is not None:
            return
        d_raw = X.shape[1]
        # Infer raw input dimension if X is already prepared
        if self._complement and d_raw % 2 == 0:
            inferred_raw = d_raw // 2
        else:
            inferred_raw = d_raw
        cfg = _TorchFuzzyARTMAPConfig(
            input_dim=inferred_raw,
            alpha=self.module_a.params["alpha"],
            rho=self.module_a.params["rho"],
            beta=self.module_a.params["beta"],
            complement=self._complement,
            device=self._device,
            dtype=self._dtype,
        )
        self._backend = _TorchFuzzyARTMAP(cfg)

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
