"""Gaussian ARTMAP :cite:`williamson1996gaussian` (Torch backend)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, cast
import numpy as np
import torch
from torch import Tensor

# artlib elementary module (numpy-side) for param compatibility
from artlib.elementary.GaussianART import (
    GaussianART,
)  # uses params: rho, sigma_init, alpha
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


# -----------------
# Torch GPU backend
# -----------------
@dataclass
class _TorchGaussianARTMAPConfig:
    input_dim: int
    rho: float
    alpha: float = 1e-10
    # sigma_init may be a scalar or length-D vector; handled in __init__
    sigma_init: Union["np.ndarray", Tensor, float] = 0.1
    epsilon: float = 1e-7
    match_tracking: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float64
    clamp_inputs: bool = False  # kept for parity; unused
    fallback_to_choice_on_fail: bool = True  # parity; unused


class _TorchGaussianARTMAP:
    """Torch-accelerated Gaussian ARTMAP with export hooks for artlib synchronization.

    Category weight layout (numpy-compatible on export):     [ mean(D), sigma(D),
    inv_sigma(D), sqrt_det_sigma(1), n(1) ]  -> length 3D + 2

    """

    def __init__(self, cfg: _TorchGaussianARTMAPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.input_dim = int(cfg.input_dim)

        # Internal per-category state
        self.mean: Optional[Tensor] = None  # [K, D]
        self.sigma: Optional[Tensor] = None  # [K, D]    (stddev; positive)
        self.inv_var: Optional[Tensor] = None  # [K, D]    (1 / sigma^2)
        self.sqrt_det: Optional[Tensor] = None  # [K]       (prod(sigma))
        self.counts: Optional[Tensor] = None  # [K]       (n; float)
        self.map_y: Optional[Tensor] = None  # [K]       (labels; long)

        # Preprocessed sigma_init (vector length D)
        if isinstance(cfg.sigma_init, (float, int)):
            si = torch.full(
                (self.input_dim,),
                float(cfg.sigma_init),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            si = _to_device(cfg.sigma_init, self.device, self.dtype).view(-1)
            if si.numel() == 1:
                si = si.repeat(self.input_dim)
        if si.numel() != self.input_dim:
            raise ValueError("sigma_init must be scalar or length equal to input_dim.")
        if not torch.all(si > 0):
            raise ValueError("All sigma_init entries must be strictly positive.")
        self._sigma_init = si

    @property
    def n_cat(self) -> int:
        return 0 if self.mean is None else int(self.mean.shape[0])

    # --- housekeeping
    def _ensure_capacity(self):
        if self.mean is None:
            self.mean = torch.empty(
                (0, self.input_dim), device=self.device, dtype=self.dtype
            )
            self.sigma = torch.empty(
                (0, self.input_dim), device=self.device, dtype=self.dtype
            )
            self.inv_var = torch.empty(
                (0, self.input_dim), device=self.device, dtype=self.dtype
            )
            self.sqrt_det = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.counts = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.map_y = torch.empty((0,), device=self.device, dtype=torch.long)

    def _validate_prepared(self, X: Tensor):
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D]")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected dimensionality D={self.input_dim}, got {X.shape[1]}"
            )

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
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
        return 512 * 1024 * 1024

    # --- data prep (Gaussian ART uses raw inputs; no complement coding)
    def prepare_data(self, X: Union[Tensor, "np.ndarray"]) -> Tensor:
        X = _to_device(X, self.device, self.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        return X

    # ---- math helpers
    def _gaussian_terms_for(self, I: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute exp(-0.5 * (x-μ)^T Σ^{-1} (x-μ)) and priors p(cj) for all categories.

        Returns:
            exp_term: [K]  (likelihood numerator without constants and det term)
            p_cj:     [K]  (counts / sum_counts; zero-safe)

        """
        assert (
            self.mean is not None
            and self.inv_var is not None
            and self.counts is not None
        )
        diff = I.unsqueeze(0) - self.mean  # [K, D]
        quad = (diff * diff * self.inv_var).sum(dim=1)  # [K]
        exp_term = torch.exp(-0.5 * quad)  # [K]
        sum_counts = self.counts.sum()
        if sum_counts.item() <= 0:
            p_cj = torch.full_like(self.counts, 1.0 / max(1, self.n_cat))
        else:
            p_cj = self.counts / sum_counts
        return exp_term, p_cj

    def _choice_and_match(self, I: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (T, m) over all categories for single input I.

        T = p(x|cj) * p(cj) with p(x|cj) ∝ exp_term / (alpha + sqrt_det). m = exp_term
        (match criterion).

        """
        if self.n_cat == 0:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            return empty, empty
        assert self.sqrt_det is not None
        exp_term, p_cj = self._gaussian_terms_for(I)  # [K], [K]
        denom = self.cfg.alpha + self.sqrt_det  # [K]
        p_x_given_c = exp_term / denom.clamp_min(1e-30)
        T = p_x_given_c * p_cj
        m = exp_term
        return T, m

    # ---- category ops
    def _commit_new_category(self, I: Tensor, y: int):
        """Initialize new Gaussian with mean=I, sigma=sigma_init, n=1."""
        self._ensure_capacity()
        si = self._sigma_init
        inv_var_init = 1.0 / (si * si)
        sqrt_det_init = si.prod()

        self.mean = torch.cat([self.mean, I.unsqueeze(0)], dim=0)
        self.sigma = torch.cat([self.sigma, si.unsqueeze(0)], dim=0)
        self.inv_var = torch.cat([self.inv_var, inv_var_init.unsqueeze(0)], dim=0)
        self.sqrt_det = torch.cat([self.sqrt_det, sqrt_det_init.view(1)], dim=0)
        self.counts = torch.cat(
            [self.counts, torch.tensor([1.0], device=self.device, dtype=self.dtype)],
            dim=0,
        )
        self.map_y = torch.cat(
            [self.map_y, torch.tensor([y], device=self.device, dtype=torch.long)], dim=0
        )

    def _learn_in_category(self, j: int, I: Tensor):
        """Update mean, sigma, inv_var, sqrt_det, counts; replicate numpy reference."""
        assert (
            self.mean is not None
            and self.sigma is not None
            and self.inv_var is not None
        )
        assert self.sqrt_det is not None and self.counts is not None

        m_j = self.mean[j]  # [D]
        s_j = self.sigma[j]  # [D]
        n_j = self.counts[j]  # scalar

        n_new = n_j + 1.0
        # mean_new = (1 - 1/n_new) * mean + (1/n_new) * I
        w1 = 1.0 - (1.0 / n_new)
        w2 = 1.0 / n_new
        mean_new = w1 * m_j + w2 * I

        # sigma_new = sqrt( (1 - 1/n_new)*sigma^2 + (1/n_new)*(mean_new - I)^2 )
        sigma2_old = s_j * s_j
        delta = mean_new - I
        sigma2_new = w1 * sigma2_old + w2 * (delta * delta)
        sigma_new = torch.sqrt(sigma2_new.clamp_min(1e-30))

        inv_var_new = 1.0 / (sigma_new * sigma_new)
        sqrt_det_new = sigma_new.prod()

        # write back
        self.mean[j] = mean_new
        self.sigma[j] = sigma_new
        self.inv_var[j] = inv_var_new
        self.sqrt_det[j] = sqrt_det_new
        self.counts[j] = n_new

    # ---- training/export API
    def partial_fit_and_export(
        self,
        X_prepared: Union[Tensor, "np.ndarray"],
        y: Union[Tensor, "np.ndarray"],
        epsilon: float = 1e-10,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
    ) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Incremental training on already-prepared inputs (raw, not complement-coded).

        Returns:
            labels_a_out (np.ndarray): chosen category indices per-sample
            weights_arrays (list[np.ndarray]): per-category weights (float64)
            cluster_labels_out (np.ndarray): map from categories to class labels

        """
        Xp = _to_device(X_prepared, self.device, self.dtype)
        y = _to_device(y, self.device, torch.long)
        self._validate_prepared(Xp)

        la: list[int] = []
        for i in range(Xp.shape[0]):
            Ii = Xp[i]
            yi = int(y[i].item())

            if self.n_cat == 0:
                self._commit_new_category(Ii, yi)
                la.append(0)
                continue

            T, m = self._choice_and_match(Ii)
            order = torch.argsort(T, descending=True, stable=True)

            rho_eff = float(self.cfg.rho)
            found = False
            chosen_idx: Optional[int] = None

            for idx in order.tolist():
                if m[idx].item() < rho_eff:
                    continue
                if int(cast(Tensor, self.map_y)[idx].item()) == yi:
                    self._learn_in_category(idx, Ii)
                    found = True
                    chosen_idx = idx
                    break
                else:
                    if match_tracking != "":
                        rho_eff = float(m[idx].item()) + float(epsilon)

            if not found:
                self._commit_new_category(Ii, yi)
                chosen_idx = self.n_cat - 1

            la.append(int(cast(int, chosen_idx)))

        # export numpy payloads for wrapper synchronization
        assert (
            self.mean is not None
            and self.sigma is not None
            and self.inv_var is not None
        )
        assert (
            self.sqrt_det is not None
            and self.counts is not None
            and self.map_y is not None
        )
        K, D = self.mean.shape

        weights_np: list[np.ndarray] = []
        for k in range(K):
            mk = self.mean[k].detach().to("cpu").numpy()
            sk = self.sigma[k].detach().to("cpu").numpy()
            iv = self.inv_var[k].detach().to("cpu").numpy()
            sd = self.sqrt_det[k].detach().to("cpu").numpy().reshape(1)
            nk = self.counts[k].detach().to("cpu").numpy().reshape(1)
            wk = np.concatenate([mk, sk, iv, sd, nk]).astype(np.float64, copy=False)
            weights_np.append(wk)

        cl_np = self.map_y.detach().to("cpu").numpy().astype(int, copy=True)
        la_np = np.asarray(la, dtype=int)
        return la_np, weights_np, cl_np

    @torch.no_grad()
    def predict_ab_prepared(
        self, X_prepared: Union[Tensor, "np.ndarray"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (a_idx, b_labels) for prepared inputs."""
        Xp = _to_device(X_prepared, self.device, self.dtype)
        self._validate_prepared(Xp)
        if self.n_cat == 0:
            raise RuntimeError("Model has no categories. Train first.")

        assert self.mean is not None and self.inv_var is not None
        assert (
            self.sqrt_det is not None
            and self.counts is not None
            and self.map_y is not None
        )

        K, D = self.mean.shape
        N = Xp.shape[0]
        elem_size = torch.tensor([], dtype=self.dtype).element_size()

        # precompute constants
        priors = (self.counts / self.counts.sum().clamp_min(1e-30)).to(
            self.dtype
        )  # [K]
        budget_bytes = max(int(0.25 * self._free_mem_bytes()), 512 * 1024 * 1024)

        def k_chunk_for(B_chunk: int) -> int:
            # we will allocate an intermediate [B_chunk, K_chunk, D]
            denom = max(1, B_chunk * self.input_dim * elem_size)
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

                mu = self.mean[k0:k1]  # [Kc, D]
                iv = self.inv_var[k0:k1]  # [Kc, D]
                sd = self.sqrt_det[k0:k1]  # [Kc]
                pr = priors[k0:k1]  # [Kc]

                # [B, Kc, D]
                diff = I.unsqueeze(1) - mu.unsqueeze(0)
                quad = (diff * diff * iv.unsqueeze(0)).sum(dim=2)  # [B, Kc]
                exp_term = torch.exp(-0.5 * quad)  # [B, Kc]
                denom = (self.cfg.alpha + sd).unsqueeze(0).clamp_min(1e-30)  # [1, Kc]
                T_chunk = (exp_term / denom) * pr.unsqueeze(0)  # [B, Kc]

                Tc_max, Tc_arg = T_chunk.max(dim=1)
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


class GaussianARTMAP(_TorchSimpleARTMAP):
    """GaussianARTMAP for Classification. optimized with torch.

    This module implements GaussianARTMAP

    GaussianARTMAP is a non-modular classification model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    torch for efficient execution. This class acts as a wrapper for the underlying torch
    functions and to provide compatibility with the artlib style and usage.
    Functionally, GaussianARTMAP behaves as a special case of
    :class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` instantiated with
    :class:`~artlib.elementary.GaussianART.GaussianART`.

    """

    def __init__(
        self,
        rho: float,
        sigma_init: Union[np.ndarray, float],
        alpha: float = 1e-10,
        input_dim: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize the Gaussian ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        sigma_init : np.ndarray or float
            Initial diagonal stddev(s), scalar or length-D.
        alpha : float
            Small constant for numerical stability and choice denominator.
        input_dim : Optional[int]
            Raw input dimensionality (if known a priori).
        device : str
            Torch device string (e.g., 'cuda', 'cpu', 'mps').
        dtype : torch.dtype
            Torch dtype; default float64 for numerical stability.

        """
        module_a = GaussianART(rho=rho, sigma_init=np.asarray(sigma_init), alpha=alpha)
        super().__init__(module_a)

        self._device = device
        self._dtype = dtype
        self._backend: Optional[_TorchGaussianARTMAP] = None
        self._declared_input_dim = input_dim  # raw dimensionality

    # --- helpers
    def _ensure_backend(self, X: np.ndarray):
        if self._backend is not None:
            return
        d_raw = X.shape[1]
        inferred_raw = d_raw  # GaussianART uses raw dimension (no complement code)
        cfg = _TorchGaussianARTMAPConfig(
            input_dim=inferred_raw,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            sigma_init=self.module_a.params["sigma_init"],
            device=self._device,
            dtype=self._dtype,
        )
        self._backend = _TorchGaussianARTMAP(cfg)
