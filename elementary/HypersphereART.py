"""
Anagnostopoulos, G. C., & Georgiopulos, M. (2000).
Hypersphere ART and ARTMAP for unsupervised and supervised, incremental learning.
In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
(pp. 59–64). volume 6. doi:10.1109/IJCNN.2000.859373.
"""
import numpy as np
from typing import Optional, Iterable
from matplotlib.axes import Axes
from common.BaseART import BaseART
from common.utils import l2norm2

class HypersphereART(BaseART):
    # implementation of HypersphereART
    def __init__(self, rho: float, alpha: float, beta: float, r_hat: float):
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
            "r_hat": r_hat,
        }
        super().__init__(params)

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


    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]

        return 1 - (max(radius, max_radius)/params["r_hat"]), cache


    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        M, cache = self.match_criterion(i, w, params=params, cache=cache)
        return M >= params["rho"], cache


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


    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        from matplotlib.patches import Circle

        for w, col in zip(self.W, colors):
            centroid = (w[0], w[1])
            radius = w[-1]
            circ = Circle(
                centroid,
                radius,
                linewidth=linewidth,
                edgecolor=col,
                facecolor='none'
            )
            ax.add_patch(circ)






