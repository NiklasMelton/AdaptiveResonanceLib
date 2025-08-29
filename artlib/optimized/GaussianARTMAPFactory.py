"""Factory for generating optimized GaussianARTMAP models using various backends."""
import warnings
from typing import Optional
import numpy as np


class GaussianARTMAPFactory:
    """Factory for generating optimized GaussianARTMAP models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        sigma_init: np.ndarray,
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
        sigma_init : np.ndarray
            Initial diagonal standard deviations (length = n_features).
        backend: str
            torch, c++, or python. Defaults to c++
        device: str
            "cuda" or "cpu". Only applied when backend=torch. Defaults to "cpu".

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
            from artlib.optimized.backends.torch.GaussianARTMAP import (
                GaussianARTMAP as TorchGA,
            )

            return TorchGA(
                rho=rho,
                alpha=alpha,
                sigma_init=sigma_init,
                input_dim=input_dim,
                device=device,
            )

        elif b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.GaussianARTMAP import (
                GaussianARTMAP as CppGA,
            )

            return CppGA(rho=rho, alpha=alpha, sigma_init=sigma_init)

        elif b == "python":
            from artlib import SimpleARTMAP, GaussianART

            module_a = GaussianART(rho=rho, alpha=alpha, sigma_init=sigma_init)
            return SimpleARTMAP(module_a)

        else:
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'c++'.",
                RuntimeWarning,
            )
            from artlib.optimized.backends.cpp.GaussianARTMAP import (
                GaussianARTMAP as CppGA,
            )

            return CppGA(rho=rho, alpha=alpha, sigma_init=sigma_init)
