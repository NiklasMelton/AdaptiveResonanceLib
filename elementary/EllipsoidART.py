import numpy as np
from typing import Optional
from common.BaseART import BaseART
from common.utils import l2norm2

class EllipsoidART(BaseART):
    # implementation of EllipsoidART

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert "mu" in params
        assert "r_hat" in params
        assert 1.0 >= params["rho"] >= 0.
        assert 1.0 >= params["alpha"] >= 0.
        assert 1.0 >= params["beta"] >= 0.
        assert 1.0 >= params["mu"] >= 0.

    @staticmethod
    def category_distance(centroid: np.ndarray, major_axis: np.ndarray, params):
        ic_dist = (1 - centroid)

        if major_axis.any():
            return (1. / params["mu"]) * np.sqrt(
                l2norm2(ic_dist) - (1 - params["mu"] * params["mu"]) * (np.matmul(major_axis, ic_dist) ** 2)
            )
        else:
            return l2norm2(ic_dist)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        centroid = w[:self.dim_]
        major_axis = w[self.dim_:-1]
        radius = w[-1]

        dist = self.category_distance(centroid, major_axis, params)

        cache = {
            "dist": dist
        }
        return (params["r_hat"] - radius - max(radius, dist)) / (params["r_hat"] - 2*radius + params["alpha"]), cache


    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        dist = cache["dist"]

        return 1 - (radius + max(radius, dist))/params["r_hat"]

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.match_criterion(i, w, params=params, cache=cache) >= params["rho"]

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        centroid = w[:self.dim_]
        major_axis = w[self.dim_:-1]
        radius = w[-1]

        if cache is None:
            raise ValueError("No cache provided")
        dist = cache["dist"]

        radius_new = radius + (params["beta"]/2)*(max(radius, dist) - radius)
        centroid_new = centroid + (params["beta"]/2)*(i-centroid)*(1-(min(radius, dist)/dist))
        if not radius == 0.:
            major_axis_new = (i-centroid_new)/l2norm2((i-centroid_new))
        else:
            major_axis_new = major_axis

        return np.concatenate([centroid_new, major_axis_new, [radius_new]])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return np.concatenate([i, np.zeros_like(i), [0.]])






