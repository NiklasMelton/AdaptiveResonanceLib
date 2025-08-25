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
            torch, c++, or python. Defaults to c++
        device: str
            "cuda" or "cpu". Only applied when backend=torch. Defaults to "cpu".

        """
        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not yet implemented for HypersphereARTMAP."
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
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
