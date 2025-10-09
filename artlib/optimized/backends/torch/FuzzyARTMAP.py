"""Fuzzy ARTMAP :cite:`carpenter1991fuzzy`."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, cast
import numpy as np
import torch
from torch import Tensor

from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.backends.torch._TorchSimpleARTMAP import _TorchSimpleARTMAP


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
    match_tracking: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float64
    clamp_inputs: bool = True
    fallback_to_choice_on_fail: bool = True


class _TorchFuzzyARTMAP:
    """Torch accelerated Fuzzy ARTMAP with export hooks for artlib synchronization."""

    def __init__(self, cfg: _TorchFuzzyARTMAPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.input_dim = int(cfg.input_dim)
        self.code_dim = self.input_dim * 2

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

    def _validate_prepared(self, X: Tensor):
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D]")
        if X.shape[1] != 2 * self.input_dim:
            raise ValueError(
                f"With complement=True, expected D={2 * self.input_dim}, "
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
            raise ValueError("Second half must be 1 - first half (complement coding).")

    def _free_mem_bytes(self) -> int:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                with torch.cuda.device(self.device):
                    free_b, _ = torch.cuda.mem_get_info()
                return int(free_b)
            except Exception:
                pass
        # CPU path
        try:
            import psutil  # type: ignore

            return int(psutil.virtual_memory().available)
        except Exception:
            # Linux fallback
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
        # Last-resort conservative default
        return 512 * 1024 * 1024

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

    def prepare_data(self, X: Union[Tensor, "np.ndarray"]) -> Tensor:
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
        return _complement_code(Xn)

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
        K, D = W.shape
        N = Xp.shape[0]
        elem_size = torch.tensor([], dtype=self.dtype).element_size()  # bytes/elem

        # Precompute and keep small tensors
        W_sum = W.sum(dim=1)  # [K]
        alpha = float(self.cfg.alpha)

        budget_bytes = max(int(0.25 * self._free_mem_bytes()), 512 * 1024 * 1024)

        def k_chunk_for(B_chunk: int) -> int:
            # max K_chunk so that B_chunk*K_chunk*D*elem_size <= budget
            denom = max(1, B_chunk * D * elem_size)
            return max(1, min(K, budget_bytes // denom))

        # We still chunk batch to keep CPU/GPU caches happy.
        B_block = 1024  # start point; real cap comes from k_chunk_for()
        a_idx_parts: list[Tensor] = []
        b_lab_parts: list[Tensor] = []

        for b0 in range(0, N, B_block):
            b1 = min(N, b0 + B_block)
            I = Xp[b0:b1]  # [B_cur, D]
            B_cur = I.shape[0]

            # Running best T and argmax over K for this batch block
            best_T = torch.full((B_cur,), -float("inf"), device=I.device, dtype=I.dtype)
            best_idx = torch.zeros((B_cur,), device="cpu", dtype=torch.long)

            K_block = k_chunk_for(B_cur)
            for k0 in range(0, K, K_block):
                k1 = min(K, k0 + K_block)
                Wc = W[k0:k1]  # [Kc, D]
                # [B_cur, Kc, D] -> sum over D -> [B_cur, Kc]
                IandW_sum = torch.minimum(I.unsqueeze(1), Wc.unsqueeze(0)).sum(dim=2)
                Tc = IandW_sum / (alpha + W_sum[k0:k1].unsqueeze(0))

                # best within this K-chunk
                Tc_max, Tc_arg = Tc.max(dim=1)  # [B_cur]
                better = Tc_max > best_T
                if better.any():
                    best_T[better] = Tc_max[better]
                    # store *global* K indices (offset by k0) on CPU to save device RAM
                    best_idx[better] = (k0 + Tc_arg[better]).to(torch.long).to("cpu")

            a_idx_parts.append(best_idx)
            # Map to B labels
            idx_dev = best_idx.to(self.device)
            b_lab_parts.append(cast(Tensor, self.map_y)[idx_dev].to("cpu"))

        y_a = torch.cat(a_idx_parts, dim=0).numpy().astype(int)
        y_b = torch.cat(b_lab_parts, dim=0).numpy().astype(int)
        return y_a, y_b


class FuzzyARTMAP(_TorchSimpleARTMAP):
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
        input_dim: Optional[int]
            number of features
        device: str
            "cuda" or "cpu". Only applied when backend=torch. Defaults to "cpu".

        """
        module_a = FuzzyART(rho=rho, alpha=alpha, beta=beta)
        super().__init__(module_a)

        # torch back-end
        self._device = device
        self._dtype = dtype
        self._backend: Optional[_TorchFuzzyARTMAP] = None
        self._declared_input_dim = input_dim  # raw dimensionality (pre-complement)

    # --- helpers
    def _ensure_backend(self, X: np.ndarray):
        if self._backend is not None:
            return
        d_raw = X.shape[1]
        # Infer raw input dimension
        inferred_raw = d_raw // 2
        cfg = _TorchFuzzyARTMAPConfig(
            input_dim=inferred_raw,
            alpha=self.module_a.params["alpha"],
            rho=self.module_a.params["rho"],
            beta=self.module_a.params["beta"],
            device=self._device,
            dtype=self._dtype,
        )
        self._backend = _TorchFuzzyARTMAP(cfg)
