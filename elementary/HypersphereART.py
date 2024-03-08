import numpy as np
from typing import Optional
from common.BaseART import BaseART
from common.utils import l2norm2

class HypersphereART(BaseART):
    # implementation of HypersphereART

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert "r_hat" in params
        assert 1.0 >= params["rho"] >= 0.
        assert params["alpha"] >= 0.
        assert 1.0 >= params["beta"] >= 0.

    @staticmethod
    def category_distance(i: np.ndarray, centroid: np.ndarray, radius: float, params) -> float:
        return np.sqrt(l2norm2(i-centroid))


    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        centroid = w[:-1]
        radius = w[-1]

        i_radius = self.category_distance(i, centroid, radius, params)
        max_radius = max(radius, i_radius)

        cache = {
            "max_radius": max_radius,
            "i_radius": i_radius,
        }
        return (params["r_hat"] - max_radius)/(params["r_hat"] - radius + params["alpha"]), cache


    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]

        return 1 - (max(radius, max_radius)/params["r_hat"])


    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.match_criterion(i, w, params=params, cache=cache) >= params["rho"]


    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        centroid = w[:-1]
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]
        i_radius = cache["i_radius"]

        radius_new = radius + (params["beta"]/2)*(max_radius-radius)
        centroid_new = centroid + (params["beta"]/2)*(i-centroid)*(1-(min(radius, i_radius)/i_radius))

        return np.concatenate([centroid_new, [radius_new]])


    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return np.concatenate([i, [0.]])






