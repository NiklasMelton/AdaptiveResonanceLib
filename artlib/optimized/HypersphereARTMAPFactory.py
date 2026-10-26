"""Factory for generating optimized HypersphereARTMAP models using various backends."""
import warnings
from typing import Optional


class HypersphereARTMAPFactory:
    """Factory for generating optimized HypersphereARTMAP models using various
    backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        r_hat: float,
        *,
        input_dim: Optional[int] = None,
        backend: str = "c++",
        device: str = "cpu",
    ):
        """Initialize the Hyperpshere ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning‑rate parameter.
        r_hat : float
            Global upper bound on cluster radius (must be > 0).
        backend: str
            "c++" (default), "torch", or "python". C++ is the fastest choice except
            for large models on an NVIDIA GPU; see "Choosing a backend" in the README.
        device: str
            "cuda", "cpu" or "mps". Only applied when backend="torch". Defaults to
            "cpu". The torch backend on "cpu" or "mps" is slower than the C++ backend
            and raises a BackendPerformanceWarning.

        """
        b = backend.lower()

        if b == "torch":
            try:
                # just to check availability
                import torch  # noqa
            except ImportError:
                warnings.warn(
                    "Backend 'torch' was requested but PyTorch is not installed. "
                    "Falling back to 'c++' backend.",
                    RuntimeWarning,
                )
                b = "cpp"

        if b == "torch":
            from artlib.optimized.backends.torch.HypersphereARTMAP import (
                HypersphereARTMAP as TorchHA,
            )

            return TorchHA(
                rho=rho,
                alpha=alpha,
                beta=beta,
                r_hat=r_hat,
                input_dim=input_dim,
                device=device,
            )

        elif b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.HypersphereARTMAP import (
                HypersphereARTMAP as CppHA,
            )

            return CppHA(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)

        elif b == "python":
            from artlib import SimpleARTMAP, HypersphereART

            return SimpleARTMAP(
                HypersphereART(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)
            )

        else:
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'c++'.",
                RuntimeWarning,
            )
            from artlib.optimized.backends.cpp.HypersphereARTMAP import (
                HypersphereARTMAP as CppHA,
            )

            return CppHA(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)
