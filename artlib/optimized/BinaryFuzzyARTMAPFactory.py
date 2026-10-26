"""Factory for generating optimized BinaryFuzzyARTMAP models using various backends."""
import warnings
from typing import Optional


class BinaryFuzzyARTMAPFactory:
    """Factory for generating optimized BinaryFuzzyARTMAP models using various
    backends."""

    def __new__(
        cls,
        rho: float,
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
        input_dim: Optional[int]
            number of features
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
            from artlib.optimized.backends.torch.BinaryFuzzyARTMAP import (
                BinaryFuzzyARTMAP as TorchBFA,
            )

            return TorchBFA(rho=rho, alpha=1e-10, input_dim=input_dim, device=device)

        elif b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import (
                BinaryFuzzyARTMAP as CppBFA,
            )

            return CppBFA(rho=rho)

        elif b == "python":
            from artlib import SimpleARTMAP, BinaryFuzzyART

            return SimpleARTMAP(BinaryFuzzyART(rho=rho))

        else:
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'c++'.",
                RuntimeWarning,
            )
            from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import (
                BinaryFuzzyARTMAP as CppBFA,
            )

            return CppBFA(rho=rho)
