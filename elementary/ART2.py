"""
Carpenter, G. A., & Grossberg, S. (1987b).
ART 2: self-organization of stable category recognition codes for analog input patterns.
Appl. Opt., 26, 4919–4930. doi:10.1364/AO.26.004919.

Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991b).
ART 2-A: An adaptive resonance algorithm for rapid category learning and recognition.
Neural Networks, 4, 493 – 504. doi:10.1016/0893-6080(91) 90045-7.
"""

import numpy as np
from typing import Optional
from common.BaseART import BaseART
from common.utils import normalize

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class ART2A(BaseART):
    # implementation of ART 2-A

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1. >= params["rho"] >= 0.
        assert 1. >= params["alpha"] >= 0.
        assert 1. >= params["beta"] >= 0.
        
    def check_dimensions(self, X: np.ndarray):
        if not self.dim_:
            self.dim_ = X.shape[1]
            assert self.params["alpha"] <= 1 / np.sqrt(self.dim_)
        else:
            assert X.shape[1] == self.dim_
        
    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        activation = float(np.dot(i, w))
        cache = {"activation": activation}
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        if cache is None:
            raise ValueError("No cache provided")
        return cache["activation"]

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        # TODO: make this more efficient
        return self.match_criterion(i, w, params, cache) > params["alpha"]*np.sum(i)

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        return params["beta"]*i + (1-params["beta"])*w

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return i
