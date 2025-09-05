"""This module implements ARTMAP models with hybridized backends."""
from artlib.optimized.backends.hybrid._HybridSimpleARTMAP import _HybridSimpleARTMAP
from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory
from artlib.optimized.BinaryFuzzyARTMAPFactory import BinaryFuzzyARTMAPFactory
from artlib.optimized.GaussianARTMAPFactory import GaussianARTMAPFactory
from artlib.optimized.HypersphereARTMAPFactory import HypersphereARTMAPFactory

from artlib.elementary.FuzzyART import FuzzyART
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from artlib.elementary.HypersphereART import HypersphereART
from artlib.elementary.GaussianART import GaussianART

import numpy as np


class FuzzyARTMAP(_HybridSimpleARTMAP):
    """Hybrid backend Fuzzy ARTMAP."""

    def __init__(self, rho: float, alpha: float, beta: float):
        """Initialize the Fuzzy ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning rate.

        """
        module_a = FuzzyART(rho=rho, alpha=alpha, beta=beta)
        super().__init__(module_a)

    def _create_backend(self, backend: str, device: str):
        params = dict(self.module_a.params)
        return FuzzyARTMAPFactory(**params, backend=backend, device=device)


class BinaryFuzzyARTMAP(_HybridSimpleARTMAP):
    """Hybrid backend Binary Fuzzy ARTMAP."""

    def __init__(self, rho: float, alpha: float):
        """Initialize the Binary Fuzzy ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.

        """
        module_a = BinaryFuzzyART(rho=rho, alpha=alpha)
        super().__init__(module_a)

    def _create_backend(self, backend: str, device: str):
        params = dict(self.module_a.params)
        return BinaryFuzzyARTMAPFactory(**params, backend=backend, device=device)


class GaussianARTMAP(_HybridSimpleARTMAP):
    """Hybrid backend Gaussian ARTMAP."""

    def __init__(self, rho: float, sigma_init: np.ndarray, alpha: float = 1e-10):
        """Initialize the Gaussian ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        sigma_init : np.ndarray
            Initial diagonal standard deviations (length = n_features).
        alpha : float, default=1e-10
            Small constant to avoid division by zero in likelihood term.

        """
        module_a = GaussianART(rho=rho, sigma_init=sigma_init, alpha=alpha)
        super().__init__(module_a)

    def _create_backend(self, backend: str, device: str):
        params = dict(self.module_a.params)
        return GaussianARTMAPFactory(**params, backend=backend, device=device)


class HypersphereARTMAP(_HybridSimpleARTMAP):
    """Hybrid backend Hypersphere ARTMAP."""

    def __init__(self, rho: float, alpha: float, beta: float, r_hat: float):
        """
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
        """
        module_a = HypersphereART(rho=rho, alpha=alpha, beta=beta, r_hat=r_hat)
        super().__init__(module_a)

    def _create_backend(self, backend: str, device: str):
        params = dict(self.module_a.params)
        return HypersphereARTMAPFactory(**params, backend=backend, device=device)
