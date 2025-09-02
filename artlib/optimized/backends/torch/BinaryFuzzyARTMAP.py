"""Binary Fuzzy ARTMAP (Torch-accelerated backend)"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, cast
import numpy as np
import torch
from torch import Tensor

from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART  # type: ignore
from artlib.optimized.backends.torch._TorchSimpleARTMAP import _TorchSimpleARTMAP


# ------------------------------
# utilities
# ------------------------------
def _to_device(x: Union[Tensor, "np.ndarray"], device, dtype=torch.bool) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    import numpy as np

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True)
    raise TypeError("Expected torch.Tensor or numpy.ndarray")


def _complement_code_any(x: Tensor) -> Tensor:
    """Complement-code that works for float or bool tensors."""
    if x.dtype == torch.bool:
        return torch.cat([x, ~x], dim=-1)
    return torch.cat([x, 1.0 - x], dim=-1)


# -----------------
# Torch GPU backend
# -----------------
@dataclass
class _TorchBinaryFuzzyARTMAPConfig:
    input_dim: int  # raw (pre-complement) dimension
    alpha: float = 1e-3  # choice
    rho: float = 0.75  # vigilance
    epsilon: float = 1e-7  # MT epsilon
    match_tracking: bool = True
    device: str = "cuda"
    # Binary model stores weights as bool
    dtype: torch.dtype = torch.bool
    clamp_inputs: bool = True  # used only if prepare_data normalizes
    fallback_to_choice_on_fail: bool = True


class _TorchBinaryFuzzyARTMAP:
    """Torch-accelerated Binary Fuzzy ARTMAP with export hooks for artlib sync."""

    def __init__(self, cfg: _TorchBinaryFuzzyARTMAPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.input_dim = int(cfg.input_dim)  # original D
        self.code_dim = self.input_dim * 2  # complement-coded

        self.W: Optional[Tensor] = None  # [K, D*2], bool
        self.map_y: Optional[Tensor] = None  # [K], long
        self._lower_bounds: Optional[Tensor] = None
        self._upper_bounds: Optional[Tensor] = None
        self._prep_tol: float = 1e-6

    @property
    def n_cat(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def _ensure_capacity(self):
        if self.W is None:
            self.W = torch.empty(
                (0, self.code_dim), device=self.device, dtype=torch.bool
            )
            self.map_y = torch.empty((0,), device=self.device, dtype=torch.long)

    # ---------- preparing / validation ----------
    def set_data_bounds(
        self, lower: Union[Tensor, "np.ndarray"], upper: Union[Tensor, "np.ndarray"]
    ):
        # Bounds stored. only used if you call prepare_data on continuous inputs
        lb = _to_device(lower, self.device, torch.float64).view(-1)
        ub = _to_device(upper, self.device, torch.float64).view(-1)
        if lb.numel() != self.input_dim or ub.numel() != self.input_dim:
            raise ValueError(f"lower/upper must have length input_dim={self.input_dim}")
        if not torch.all(ub > lb):
            raise ValueError(
                "All upper bounds must be strictly greater than lower bounds."
            )
        self._lower_bounds, self._upper_bounds = lb, ub

    def prepare_data(self, X: Union[Tensor, "np.ndarray"]) -> Tensor:
        """Optional helper: normalize -> complement-code -> binarize -> bool.
        If your inputs are already binary and complement-coded, skip this and
        pass them straight to partial_fit_and_export/predict_*.
        """
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError(
                "Call set_data_bounds(lower, upper) before prepare_data()."
            )

        Xf = _to_device(X, self.device, torch.float64)
        if Xf.ndim == 1:
            Xf = Xf.unsqueeze(0)
        if Xf.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected raw dimensionality {self.input_dim}, got {Xf.shape[1]}"
            )
        denom = (self._upper_bounds - self._lower_bounds).clamp_min(1e-12)
        Xn = ((Xf - self._lower_bounds) / denom).clamp(0.0, 1.0)
        # Complement-code then binarize to {0,1}, then cast to bool
        Xcc = _complement_code_any(Xn)
        Xbin = Xcc >= 0.5
        return Xbin.to(self.dtype)

    def _validate_prepared(self, X: Tensor):
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D]")
        if X.shape[1] != 2 * self.input_dim:
            msg = (
                f"With complement coding, expected D={2 * self.input_dim}, "
                f"got {X.shape[1]}"
            )
            raise ValueError(msg)
        if X.dtype not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("BinaryFuzzyARTMAP requires integer/bool prepared inputs.")
        # Coerce a,b to bool and verify complement relation b == ~a
        D = self.input_dim
        a = X[:, :D] != 0
        b = X[:, D:] != 0
        if not torch.all(b == ~a):
            raise ValueError(
                "Prepared inputs must be complement-coded binary: X = [x, ~x]."
            )

    # ---------- system utils ----------
    def _free_mem_bytes(self) -> int:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                with torch.cuda.device(self.device):
                    free_b, _ = torch.cuda.mem_get_info()
                return int(free_b)
            except Exception:
                pass
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

    # ---------- core ops ----------
    def _choice_and_match(self, I: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """For a single prepared binary input I (bool), return:

        T: [K] choice values
        m: [K] match values
        w_sum: [K] |w| (as float64)

        """
        if self.W is None or self.W.shape[0] == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.float64)
            return empty, empty, empty

        # bool AND then count
        IandW = I.unsqueeze(0) & self.W  # [K, D]
        w1 = IandW.sum(dim=1, dtype=torch.int32)  # [K] int
        w_sum = self.W.sum(dim=1, dtype=torch.int32)  # [K] int

        # Convert to float for ratios
        w1f = w1.to(torch.float64)
        wsumf = w_sum.to(torch.float64)

        # Choice: w1 / (alpha + |w|)
        T = w1f / (float(self.cfg.alpha) + wsumf)
        # Match: w1 / dim_original
        m = w1f / float(self.input_dim)
        return T, m, wsumf

    def _commit_new_category(self, I: Tensor, y: int):
        self._ensure_capacity()
        w_new = I.unsqueeze(0)  # β = 1.0 -> new = input
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
        """Incremental training on already-prepared, binary, complement-coded inputs.

        Returns:
          labels_a_out: (N,) chosen A-side category indices (int)
          weights_arrays: list of K arrays, each weight as uint8
          cluster_labels_out: (K,) map from A categories to B labels (int)

        """
        # Coerce to bool on device
        Xp = _to_device(X_prepared, self.device, torch.bool)
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

            assert self.map_y is not None and self.W is not None
            T, m, _ = self._choice_and_match(Ii)
            order = torch.argsort(T, descending=True, stable=True)

            rho_eff = float(self.cfg.rho)
            found = False
            chosen_idx: Optional[int] = None

            for idx in order.tolist():
                if m[idx].item() < rho_eff:
                    continue

                if int(self.map_y[idx].item()) == yi:
                    # resonance + learn (β=1 => w := I & w)
                    self.W[idx] = Ii & self.W[idx]
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
        assert self.W is not None and self.map_y is not None
        W_np = [
            self.W[k].detach().to("cpu").numpy().astype(np.uint8, copy=True)
            for k in range(self.n_cat)
        ]
        cl_np = self.map_y.detach().to("cpu").numpy().astype(int, copy=True)
        la_np = np.asarray(la, dtype=int)
        return la_np, W_np, cl_np

    @torch.no_grad()
    def predict_ab_prepared(
        self, X_prepared: Union[Tensor, "np.ndarray"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xp = _to_device(X_prepared, self.device, torch.bool)
        self._validate_prepared(Xp)
        if self.n_cat == 0:
            raise RuntimeError("Model has no categories. Train first.")

        W = cast(Tensor, self.W)  # [K, D], bool
        K, D = W.shape
        N = Xp.shape[0]

        # Precompute |w| counts
        W_sum = W.sum(dim=1, dtype=torch.int32).to(torch.float64)  # [K]
        alpha = float(self.cfg.alpha)

        # Memory-aware chunking like the float version
        elem_size = torch.tensor([], dtype=torch.float64).element_size()  # 8 bytes
        budget_bytes = max(int(0.25 * self._free_mem_bytes()), 512 * 1024 * 1024)

        def k_chunk_for(B_chunk: int) -> int:
            # Approx upper bound: bool -> we accumulate into int32/float64 internally,
            # but use float64 size for safety.
            denom = max(1, B_chunk * D * elem_size)
            return max(1, min(K, budget_bytes // denom))

        B_block = 1024
        a_idx_parts: list[Tensor] = []
        b_lab_parts: list[Tensor] = []

        for b0 in range(0, N, B_block):
            b1 = min(N, b0 + B_block)
            I = Xp[b0:b1]  # [B_cur, D], bool
            B_cur = I.shape[0]

            best_T = torch.full(
                (B_cur,), -float("inf"), device=I.device, dtype=torch.float64
            )
            best_idx = torch.zeros((B_cur,), device="cpu", dtype=torch.long)

            K_block = k_chunk_for(B_cur)
            for k0 in range(0, K, K_block):
                k1 = min(K, k0 + K_block)
                Wc = W[k0:k1]  # [Kc, D], bool

                # [B_cur, Kc, D] -> AND -> sum over D -> counts -> float64
                IandW_cnt = (
                    (I.unsqueeze(1) & Wc.unsqueeze(0))
                    .sum(dim=2, dtype=torch.int32)
                    .to(torch.float64)
                )
                Tc = IandW_cnt / (alpha + W_sum[k0:k1].unsqueeze(0))  # [B_cur, Kc]

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


# -----------------
# Public wrapper
# -----------------
class BinaryFuzzyARTMAP(_TorchSimpleARTMAP):
    """BinaryFuzzyARTMAP for Classification. optimized with torch.

    This module implements BinaryFuzzyARTMAP

    BinaryFuzzyARTMAP is a non-modular classification model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    torch for efficient execution. This class acts as a wrapper for the underlying torch
    functions and to provide compatibility with the artlib style and usage.
    Functionally, BinaryFuzzyARTMAP behaves as a special case of
    :class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` instantiated with
    :class:`~artlib.elementary.BinaryFuzzyART.BinaryFuzzyART`.

    """

    def __init__(
        self,
        rho: float,
        alpha: float,
        input_dim: Optional[int] = None,  # raw (pre-complement) D
        device: str = "cuda",
    ):
        """Initialize the Binary Fuzzy ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        input_dim: Optional[int]
            number of features
        device: str
            "cuda" or "cpu". Only applied when backend=torch. Defaults to "cpu".

        """
        module_a = BinaryFuzzyART(rho=rho, alpha=alpha)
        super().__init__(module_a)

        self._device = device
        self._backend: Optional[_TorchBinaryFuzzyARTMAP] = None
        self._declared_input_dim = input_dim  # raw D before complement

    def _ensure_backend(self, X: np.ndarray):
        """Initialize backend using prepared X to infer raw dimension when needed."""
        if self._backend is not None:
            return
        d_prepped = X.shape[1]  # should be 2*D
        inferred_raw = d_prepped // 2
        cfg = _TorchBinaryFuzzyARTMAPConfig(
            input_dim=inferred_raw,
            alpha=self.module_a.params["alpha"],
            rho=self.module_a.params["rho"],
            device=self._device,
        )
        self._backend = _TorchBinaryFuzzyARTMAP(cfg)
