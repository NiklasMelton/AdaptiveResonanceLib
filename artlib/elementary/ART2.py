"""
Carpenter, G. A., & Grossberg, S. (1987b).
ART 2: self-organization of stable category recognition codes for analog input patterns.
Appl. Opt., 26, 4919–4930. doi:10.1364/AO.26.004919.

Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991b).
ART 2-A: An adaptive resonance algorithm for rapid category learning and recognition.
Neural Networks, 4, 493 – 504. doi:10.1016/0893-6080(91) 90045-7.
"""

"""
==================================================================
DISCLAIMER: DO NOT USE ART2!!!
IT DOES NOT WORK
It is provided for completeness only.
Stephan Grossberg himself has said ART2 does not work.
==================================================================
"""

import numpy as np
from typing import Optional
from warnings import warn
from artlib.common.BaseART import BaseART


class ART2A(BaseART):
    warn("Do Not Use ART2. It does not work. This module is provided for completeness only")
    # implementation of ART 2-A
    def __init__(self, rho: float, alpha: float, beta: float):
        """
        Parameters:
        - rho: vigilance parameter
        - alpha: choice parameter
        - beta: learning rate

        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1. >= params["rho"] >= 0.
        assert 1. >= params["alpha"] >= 0.
        assert 1. >= params["beta"] >= 0.
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)
        
    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            assert self.params["alpha"] <= 1 / np.sqrt(self.dim_)
        else:
            assert X.shape[1] == self.dim_
        
    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        activation = float(np.dot(i, w))
        cache = {"activation": activation}
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        """
        get the match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion, cache used for later processing

        """
        if cache is None:
            raise ValueError("No cache provided")
        # TODO: make this more efficient
        M = cache["activation"]
        M_u = params["alpha"]*np.sum(i)
        # suppress if uncommitted activation is higher
        if M < M_u:
            return -1., cache
        else:
            return M, cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        """
        get the binary match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion binary, cache used for later processing

        """
        M, cache = self.match_criterion(i, w, params, cache)
        return M >= params["rho"], cache

    def update(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> np.ndarray:
        """
        get the updated cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            updated cluster weight, cache used for later processing

        """
        return params["beta"]*i + (1-params["beta"])*w

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        return i
