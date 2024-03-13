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
from common.BaseART import BaseART
from common.utils import normalize

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class ART2A(BaseART):
    warn("Do Not Use ART2. It does not work. This module is provided for completeness only")
    # implementation of ART 2-A
    def __init__(self, rho: float, alpha: float, beta: float):
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1. >= params["rho"] >= 0.
        assert 1. >= params["alpha"] >= 0.
        assert 1. >= params["beta"] >= 0.
        
    def check_dimensions(self, X: np.ndarray):
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            assert self.params["alpha"] <= 1 / np.sqrt(self.dim_)
        else:
            assert X.shape[1] == self.dim_
        
    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        activation = float(np.dot(i, w))
        cache = {"activation": activation}
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
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
        M, cache = self.match_criterion(i, w, params, cache)
        return M >= params["rho"], cache

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        return params["beta"]*i + (1-params["beta"])*w

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return i
