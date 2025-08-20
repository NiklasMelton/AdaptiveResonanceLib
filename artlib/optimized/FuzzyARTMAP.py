"""Factory for generating optimized FuzzyARTMAP models using various backends."""
import warnings
from typing import Optional


class FuzzyARTMAP:
    """Factory for generating optimized FuzzyARTMAP models using various backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
        beta: float,
        *,
        input_dim: Optional[int] = None,
        backend: str = "c++",
        device: str = "cpu",
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
            from artlib.optimized.backends.torch.FuzzyARTMAP import (
                FuzzyARTMAP as TorchFA,
            )

            return TorchFA(
                rho=rho, alpha=alpha, beta=beta, input_dim=input_dim, device=device
            )

        elif b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.FuzzyARTMAP import FuzzyARTMAP as CppFA

            return CppFA(rho=rho, alpha=alpha, beta=beta)

        elif b == "python":
            from artlib import SimpleARTMAP, FuzzyART

            return SimpleARTMAP(FuzzyART(rho=rho, alpha=alpha, beta=beta))

        else:
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'c++'.",
                RuntimeWarning,
            )
            from artlib.optimized.backends.cpp.FuzzyARTMAP import FuzzyARTMAP as CppFA

            return CppFA(rho=rho, alpha=alpha, beta=beta)
