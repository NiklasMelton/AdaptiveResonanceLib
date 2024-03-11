"""
Anagnostopoulos, G. C., & Georgiopoulos, M. (2001a).
Ellipsoid ART and ARTMAP for incremental clustering and classification.
In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
(pp. 1221–1226). volume 2. doi:10.1109/IJCNN.2001.939535.

Anagnostopoulos, G. C., & Georgiopoulos, M. (2001b).
Ellipsoid ART and ARTMAP for incremental unsupervised and supervised learning.
In Aerospace/Defense Sensing, Simulation, and Controls (pp. 293– 304).
International Society for Optics and Photonics. doi:10.1117/12.421180.
"""
import numpy as np
from typing import Optional, Iterable
from matplotlib.axes import Axes
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
        assert 1.0 >= params["mu"] > 0.

    @staticmethod
    def category_distance(i: np.ndarray, centroid: np.ndarray, major_axis: np.ndarray, params):
        ic_dist = (i - centroid)

        if major_axis.any():
            return (1. / params["mu"]) * np.sqrt(
                l2norm2(ic_dist) - (1 - params["mu"] * params["mu"]) * (np.matmul(major_axis, ic_dist) ** 2)
            )
        else:
            return np.sqrt(l2norm2(ic_dist))

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        centroid = w[:self.dim_]
        major_axis = w[self.dim_:-1]
        radius = w[-1]

        dist = self.category_distance(i, centroid, major_axis, params)

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
            major_axis_new = (i-centroid_new)/np.sqrt(l2norm2((i-centroid_new)))
        else:
            major_axis_new = major_axis

        return np.concatenate([centroid_new, major_axis_new, [radius_new]])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return np.concatenate([i, np.zeros_like(i), [0.]])

    def get_2d_ellipsoids(self) -> list[tuple]:
        ellipsoids = []
        for w in self.W:
            centroid = w[:2]
            major_axis = w[self.dim_:-1]
            radius = w[-1]

            angle = np.arctan2(major_axis[1], major_axis[0])
            height = radius*2
            width = self.params["mu"]*height

            ellipsoids.append((centroid, width, height, angle))

        return ellipsoids

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        from matplotlib.patches import Ellipse

        ellipsoids = self.get_2d_ellipsoids()
        for (centroid, width, height, angle), col in zip(ellipsoids, colors):
            print(centroid, width, height, angle)
            ellip = Ellipse(
                centroid,
                width,
                height,
                angle,
                linewidth=linewidth,
                edgecolor=col,
                facecolor='none'
            )
            ax.add_patch(ellip)






