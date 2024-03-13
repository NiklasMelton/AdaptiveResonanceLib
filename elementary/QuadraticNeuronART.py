"""
Su, M.-C., & Liu, T.-K. (2001).
Application of neural networks using quadratic junctions in cluster analysis.
Neurocomputing, 37, 165 – 175. doi:10.1016/S0925-2312(00)00343-X.

Su, M.-C., & Liu, Y.-C. (2005).
A new approach to clustering data with arbitrary shapes.
Pattern Recognition, 38, 1887 – 1901. doi:10.1016/j.patcog.2005.04.010.
"""

import numpy as np
from typing import Optional, Iterable
from matplotlib.axes import Axes
from common.BaseART import BaseART
from common.utils import normalize, l2norm2, plot_weight_matrix_as_ellipse

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class QuadraticNeuronART(BaseART):
    # implementation of QuadraticNeuronART

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "s_init" in params
        assert "lr_b" in params
        assert "lr_w" in params
        assert "lr_s" in params
        assert 1.0 >= params["rho"] >= 0.

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        dim2 = self.dim_ * self.dim_
        w_ = w[:dim2].reshape((self.dim_, self.dim_))
        b = w[dim2:-1]
        s = w[-1]
        z = np.matmul(w_, i)
        l2norm2_z_b = l2norm2(z-b)
        activation = np.exp(-s*s*l2norm2_z_b)

        cache = {
            "activation": activation,
            "l2norm2_z_b": l2norm2_z_b,
            "w": w_,
            "b": b,
            "s": s,
            "z": z
        }
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        if cache is None:
            raise ValueError("No cache provided")
        return cache["activation"], cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        M, cache = self.match_criterion(i, w, params, cache)
        return M >= params["rho"], cache

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        s = cache["s"]
        w_ = cache["w"]
        b = cache["b"]
        z = cache["z"]
        T = cache["activation"]
        l2norm2_z_b = cache["l2norm2_z_b"]

        sst2 = 2*s*s*T

        b_new = b + params["lr_b"]*(sst2*(z-b))
        w_new = w_ + params["lr_w"]*(-sst2*((z-b).reshape((-1, 1))*i.reshape((1, -1))))
        s_new = s + params["lr_s"]*(-2*s*T*l2norm2_z_b)

        return np.concatenate([w_new.flatten(), b_new, [s_new]])


    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        w_new = np.identity(self.dim_)
        return np.concatenate([w_new.flatten(), i, [params["s_init"]]])

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        # kinda works
        from matplotlib.patches import Rectangle
        for w, col in zip(self.W, colors):
            dim2 = self.dim_ * self.dim_
            w_ = w[:dim2].reshape((self.dim_, self.dim_))
            b = w[dim2:-1]
            s = w[-1]
            plot_weight_matrix_as_ellipse(ax, s, w_, b, col)

