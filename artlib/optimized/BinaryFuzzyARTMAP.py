"""Factory for generating optimized BinaryFuzzyARTMAP models using various backends."""
import warnings
from typing import Optional


class BinaryFuzzyARTMAP:
    """Factory for generating optimized BinaryFuzzyARTMAP models using various
    backends."""

    def __new__(
        cls,
        rho: float,
        alpha: float,
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
        input_dim: Optional[int]
            number of features
        backend: str
            torch, c++, or python. Defaults to c++
        device: str
            "cuda" or "cpu". Only applied when backend=torch. Defaults to "cpu".

        """
        b = backend.lower()

        if b == "torch":
            warnings.warn(
                "Backend 'torch' is not yet implemented for BinaryFuzzyARTMAP."
                "Falling back to 'c++' backend.",
                RuntimeWarning,
            )
            b = "cpp"

        if b in ("c++", "cpp"):
            from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import (
                BinaryFuzzyARTMAP as CppBFA,
            )

            return CppBFA(rho=rho, alpha=alpha)

        elif b == "python":
            from artlib import SimpleARTMAP, BinaryFuzzyART

            return SimpleARTMAP(BinaryFuzzyART(rho=rho, alpha=alpha))

        else:
            warnings.warn(
                f"Unknown backend '{backend}', defaulting to 'c++'.",
                RuntimeWarning,
            )
            from artlib.optimized.backends.cpp.BinaryFuzzyARTMAP import (
                BinaryFuzzyARTMAP as CppBFA,
            )

            return CppBFA(rho=rho, alpha=alpha)
