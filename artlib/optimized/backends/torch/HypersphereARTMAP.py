"""Hypersphere ARTMAP (Torch-accelerated backend)"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, cast
import numpy as np
import torch

from artlib.elementary.HypersphereART import HypersphereART

from torch import Tensor

from artlib.optimized.backends.torch._TorchSimpleARTMAP import (
    _TorchSimpleARTMAP,
    _append_rows,
    _predict_budget_bytes,
    _resolve_float_dtype,
    _scratch,
    _to_device,
)


# ---------
# utilities
# ---------


def _euclidean(X: Tensor, C: Tensor) -> Tensor:
    """Pairwise distances sqrt(sum((x - c)^2)), shape [len(X), len(C)].

    Sums (x - c)^2 directly, like the C++ backend. The matmul form
    x^2 + c^2 - 2 c.x is faster on paper but cancels badly when x is close to c.

    """
    return torch.cdist(X, C, compute_mode="donot_use_mm_for_euclid_dist")


# -----------------
# Torch GPU backend
# -----------------
@dataclass
class _TorchHypersphereARTMAPConfig:
    input_dim: int
    alpha: float = 1e-3
    rho: float = 0.75
    beta: float = 1.0
    r_hat: float = 1.0
    device: str = "cuda"
    dtype: Optional[torch.dtype] = None  # None: float64, or float32 on MPS
    clamp_inputs: bool = False  # optional; Hypersphere doesn’t require [0,1]


class _TorchHypersphereARTMAP:
    """GPU-accelerated Hypersphere ARTMAP with export hooks for artlib
    synchronization."""

    def __init__(self, cfg: _TorchHypersphereARTMAPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = _resolve_float_dtype(self.device, cfg.dtype)

        self.input_dim = int(cfg.input_dim)
        # Each weight is [centroid (D), radius (1)]
        self.weight_dim = self.input_dim + 1

        self.W: Optional[Tensor] = None  # [K, D+1] (centroid..., radius)
        self.map_y: Optional[Tensor] = None  # [K]
        self._map_list: list[int] = []  # map_y as Python ints, avoids device syncs
        self._prep_tol: float = 1e-6

    @property
    def n_cat(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def _ensure_capacity(self):
        if self.W is None:
            self.W = torch.empty(
                (0, self.weight_dim), device=self.device, dtype=self.dtype
            )
            self.map_y = torch.empty((0,), device=self.device, dtype=torch.long)

    def _prep_input(self, X: Tensor) -> Tensor:
        return torch.clamp(X, 0.0, 1.0) if self.cfg.clamp_inputs else X

    def _validate_prepared(self, X: Tensor):
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D]")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected raw dimensionality {self.input_dim}, got {X.shape[1]}"
            )

    # ---- core ops (Hypersphere choice/match)
    def _choice_and_match(self, I: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (T, m, I_radius, max_radius) for a single prepared input I."""
        if self.W is None or self.W.shape[0] == 0:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            z = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return empty, empty, z, empty

        W = cast(Tensor, self.W)

        r_hat = float(self.cfg.r_hat)
        alpha = float(self.cfg.alpha)

        centroids = W[:, : self.input_dim]  # [K, D]
        radii = W[:, self.input_dim]  # [K]

        # sqrt(sum((c - x)^2)) in a reused [K, D] buffer; cdist allocates a new
        # temporary each call, which grows held memory on MPS as K grows.
        sq = torch.sub(
            centroids, I.unsqueeze(0), out=_scratch(self, "diff", W.shape[0], centroids)
        )
        I_radius = sq.mul_(sq).sum(dim=1).sqrt_()  # [K]

        max_radius = torch.maximum(radii, I_radius)  # [K]
        T = (r_hat - max_radius) / (r_hat - radii + alpha)
        m = 1.0 - (max_radius / r_hat)

        return T, m, I_radius, max_radius

    def _commit_new_category(self, I: Tensor, y: int):
        """Start a new hypersphere at sample I with radius 0."""
        self._ensure_capacity()
        w_new = torch.cat(
            [I.view(-1), torch.zeros((1,), device=self.device, dtype=self.dtype)]
        )
        _append_rows(
            self,
            self.n_cat,
            W=w_new,
            map_y=torch.tensor(y, device=self.device, dtype=torch.long),
        )
        self._map_list.append(int(y))

    def partial_fit_and_export(
        self,
        X_prepared: Union[Tensor, "np.ndarray"],
        y: Union[Tensor, "np.ndarray"],
        epsilon: float = 1e-10,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
    ) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        Xp = _to_device(X_prepared, self.device, self.dtype)
        y = _to_device(y, self.device, torch.long)
        Xp = self._prep_input(Xp)
        self._validate_prepared(Xp)

        la: list[int] = []
        y_list = y.tolist()
        for i in range(Xp.shape[0]):
            Ii = Xp[i]
            yi = int(y_list[i])

            if self.n_cat == 0:
                self._commit_new_category(Ii, yi)
                la.append(0)
                continue

            assert self.map_y is not None and self.W is not None
            T, m, I_radius_all, max_radius_all = self._choice_and_match(Ii)
            order = torch.argsort(T, descending=True, stable=True)
            # One transfer per sample instead of one .item() per candidate.
            m_list = m.tolist()

            rho_eff = float(self.cfg.rho)
            found = False
            chosen_idx: Optional[int] = None

            for idx in order.tolist():
                if m_list[idx] < rho_eff:
                    continue

                if self._map_list[idx] == yi:
                    # resonance + learn (Hypersphere update)
                    wj = self.W[idx]
                    centroid = wj[: self.input_dim]
                    radius = wj[self.input_dim]

                    i_radius = I_radius_all[idx]
                    max_radius = max_radius_all[idx]
                    beta = float(self.cfg.beta)
                    alpha = float(self.cfg.alpha)

                    # radius update
                    radius_new = radius + (beta / 2.0) * (max_radius - radius)

                    # centroid update
                    # term = 1 - min(radius, i_radius) / (i_radius + alpha).
                    # With alpha == 0 and the sample on the centroid the ratio is
                    # 0/0; use its limit as alpha -> 0, which is 0 (as in C++).
                    denom = i_radius + alpha
                    ratio = torch.where(
                        denom > 0,
                        torch.minimum(radius, i_radius) / denom,
                        torch.zeros_like(denom),
                    )
                    term = 1.0 - ratio
                    centroid_new = centroid + (beta / 2.0) * (Ii - centroid) * term

                    self.W[idx, : self.input_dim] = centroid_new
                    self.W[idx, self.input_dim] = radius_new

                    found = True
                    chosen_idx = idx
                    break
                else:
                    if match_tracking != "":
                        rho_eff = float(m_list[idx]) + float(epsilon)

            if not found:
                self._commit_new_category(Ii, yi)
                chosen_idx = self.n_cat - 1

            la.append(int(cast(int, chosen_idx)))

        assert self.W is not None and self.map_y is not None
        W_cpu = self.W.detach().to("cpu").numpy().astype(np.float64, copy=True)
        W_np = [W_cpu[k] for k in range(self.n_cat)]
        cl_np = self.map_y.detach().to("cpu").numpy().astype(int, copy=True)
        la_np = np.asarray(la, dtype=int)
        return la_np, W_np, cl_np

    @torch.no_grad()
    def predict_ab_prepared(
        self, X_prepared: Union[Tensor, "np.ndarray"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xp = _to_device(X_prepared, self.device, self.dtype)
        Xp = self._prep_input(Xp)
        self._validate_prepared(Xp)
        if self.n_cat == 0:
            raise RuntimeError("Model has no categories. Train first.")

        W = cast(Tensor, self.W)
        K, WD = W.shape
        D = self.input_dim
        assert WD == D + 1
        N = Xp.shape[0]
        elem_size = torch.tensor([], dtype=self.dtype).element_size()  # bytes/elem

        # Precompute small tensors
        r_hat = float(self.cfg.r_hat)
        alpha = float(self.cfg.alpha)

        budget_bytes = _predict_budget_bytes(self.device)

        def k_chunk_for(B_chunk: int) -> int:
            # Approximate memory for (B*K*D) intermediates
            denom = max(1, B_chunk * D * elem_size)
            return max(1, min(K, budget_bytes // denom))

        B_block = 1024
        a_idx_parts: list[Tensor] = []
        b_lab_parts: list[Tensor] = []

        for b0 in range(0, N, B_block):
            b1 = min(N, b0 + B_block)
            I = Xp[b0:b1]  # [B_cur, D]
            B_cur = I.shape[0]

            best_T = torch.full((B_cur,), -float("inf"), device=I.device, dtype=I.dtype)
            best_idx = torch.zeros((B_cur,), device="cpu", dtype=torch.long)

            K_block = k_chunk_for(B_cur)
            for k0 in range(0, K, K_block):
                k1 = min(K, k0 + K_block)
                Wc = W[k0:k1]
                centroids = Wc[:, :D].contiguous()
                radii_c = Wc[:, D].contiguous()

                I_radius = _euclidean(I, centroids)  # [B_cur, Kc]

                max_radius = torch.maximum(I_radius, radii_c.unsqueeze(0))
                Tc = (r_hat - max_radius) / (r_hat - radii_c.unsqueeze(0) + alpha)

                Tc_max, Tc_arg = Tc.max(dim=1)  # [B_cur]
                better = Tc_max > best_T
                if better.any():
                    best_T[better] = Tc_max[better]
                    best_idx[better] = (k0 + Tc_arg[better]).to(torch.long).to("cpu")

            a_idx_parts.append(best_idx)
            idx_dev = best_idx.to(self.device)
            b_lab_parts.append(cast(Tensor, self.map_y)[idx_dev].to("cpu"))

        y_a = torch.cat(a_idx_parts, dim=0).numpy().astype(int)
        y_b = torch.cat(b_lab_parts, dim=0).numpy().astype(int)
        return y_a, y_b


class HypersphereARTMAP(_TorchSimpleARTMAP):
    """HypersphereARTMAP for Classification. optimized with torch.

    This module implements HypersphereARTMAP

    HypersphereARTMAP is a non-modular classification model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    torch for efficient execution. This class acts as a wrapper for the underlying torch
    functions and to provide compatibility with the artlib style and usage.
    Functionally, HypersphereARTMAP behaves as a special case of
    :class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` instantiated with
    :class:`~artlib.elementary.HypersphereART.HypersphereART`.

    """

    def __init__(
        self,
        rho: float,
        alpha: float,
        beta: float,
        r_hat: float,
        input_dim: Optional[int] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        clamp_inputs: bool = False,
    ):
        """
        Parameters
        ----------
        rho : float
            Vigilance parameter (used with match criterion m).
        alpha : float
            Choice stabilizer (>0 to avoid division by zero).
        beta : float
            Learning rate.
        r_hat : float
            Maximum permissible category radius.
        input_dim : Optional[int]
            Raw input dimensionality (if known a priori).
        device : str
            Torch device string (e.g., 'cuda', 'cpu', 'mps').
        dtype : Optional[torch.dtype]
            Floating point dtype. Defaults to float64, or float32 on MPS, which has
            no float64.
        """
        module_a = HypersphereART(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)
        super().__init__(module_a)

        self._device = device
        self._dtype = dtype
        self._clamp = clamp_inputs
        self._backend: Optional[_TorchHypersphereARTMAP] = None
        self._declared_input_dim = input_dim  # raw dimensionality D

    def _ensure_backend(self, X: np.ndarray):
        if self._backend is not None:
            return
        d_raw = X.shape[1]
        cfg = _TorchHypersphereARTMAPConfig(
            input_dim=d_raw
            if self._declared_input_dim is None
            else int(self._declared_input_dim),
            alpha=self.module_a.params["alpha"],
            rho=self.module_a.params["rho"],
            beta=self.module_a.params["beta"],
            r_hat=self.module_a.params["r_hat"],
            device=self._device,
            dtype=self._dtype,
            clamp_inputs=self._clamp,
        )
        self._backend = _TorchHypersphereARTMAP(cfg)
