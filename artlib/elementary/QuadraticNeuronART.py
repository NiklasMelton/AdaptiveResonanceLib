"""Quadratic Neuron ART :cite:`su2001application`, :cite:`su2005new`."""
# Su, M.-C., & Liu, T.-K. (2001).
# Application of neural networks using quadratic junctions in cluster analysis.
# Neurocomputing, 37, 165 – 175. doi:10.1016/S0925-2312(00)00343-X.

# Su, M.-C., & Liu, Y.-C. (2005).
# A new approach to clustering data with arbitrary shapes.
# Pattern Recognition, 38, 1887 – 1901. doi:10.1016/j.patcog.2005.04.010.

import numpy as np
from typing import Optional, Iterable, List, Tuple, Dict
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import l2norm2
from artlib.common.visualization import plot_weight_matrix_as_ellipse


class QuadraticNeuronART(BaseART):
    """Quadratic Neuron ART for Clustering.

    This module implements Quadratic Neuron ART as first published in:
    :cite:`su2001application`, :cite:`su2005new`.

    .. # Su, M.-C., & Liu, T.-K. (2001).
    .. # Application of neural networks using quadratic junctions in cluster analysis.
    .. # Neurocomputing, 37, 165 – 175. doi:10.1016/S0925-2312(00)00343-X.

    .. # Su, M.-C., & Liu, Y.-C. (2005).
    .. # A new approach to clustering data with arbitrary shapes.
    .. # Pattern Recognition, 38, 1887 – 1901. doi:10.1016/j.patcog.2005.04.010.

    Quadratic Neuron ART clusters data in Hyper-ellipsoid by utilizing a quadratic
    neural network for activation and resonance.

    """

    def __init__(
        self, rho: float, s_init: float, lr_b: float, lr_w: float, lr_s: float
    ):
        """Initialize the Quadratic Neuron ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        s_init : float
            Initial quadratic term.
        lr_b : float
            Learning rate for cluster mean (bias).
        lr_w : float
            Learning rate for cluster weights.
        lr_s : float
            Learning rate for the quadratic term.

        """
        params = {
            "rho": rho,
            "s_init": s_init,
            "lr_b": lr_b,
            "lr_w": lr_w,
            "lr_s": lr_s,
        }
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert "s_init" in params
        assert "lr_b" in params
        assert "lr_w" in params
        assert "lr_s" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert 1.0 >= params["lr_b"] > 0.0
        assert 1.0 >= params["lr_w"] >= 0.0
        assert 1.0 >= params["lr_s"] >= 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["s_init"], float)
        assert isinstance(params["lr_b"], float)
        assert isinstance(params["lr_w"], float)
        assert isinstance(params["lr_s"], float)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        """Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        float
            Cluster activation.
        dict, optional
            Cache used for later processing.

        """
        dim2 = self.dim_ * self.dim_
        w_ = w[:dim2].reshape((self.dim_, self.dim_))
        b = w[dim2:-1]
        s = w[-1]
        z = np.matmul(w_, i)
        l2norm2_z_b = l2norm2(z - b)
        activation = np.exp(-s * s * l2norm2_z_b)

        cache = {
            "activation": activation,
            "l2norm2_z_b": l2norm2_z_b,
            "w": w_,
            "b": b,
            "s": s,
            "z": z,
        }
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values from previous calculations.

        Returns
        -------
        float
            Cluster match criterion.
        dict
            Cache used for later processing.

        """
        if cache is None:
            raise ValueError("No cache provided")
        return cache["activation"], cache

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Get the updated cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values from previous calculations.

        Returns
        -------
        np.ndarray
            Updated cluster weight, cache used for later processing.

        """
        assert cache is not None
        s = cache["s"]
        w_ = cache["w"]
        b = cache["b"]
        z = cache["z"]
        T = cache["activation"]
        l2norm2_z_b = cache["l2norm2_z_b"]

        sst2 = 2 * s * s * T

        b_new = b + params["lr_b"] * (sst2 * (z - b))
        w_new = w_ + params["lr_w"] * (
            -sst2 * ((z - b).reshape((-1, 1)) * i.reshape((1, -1)))
        )
        s_new = s + params["lr_s"] * (-2 * s * T * l2norm2_z_b)

        return np.concatenate([w_new.flatten(), b_new, [s_new]])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        np.ndarray
            New cluster weight.

        """
        w_new = np.identity(self.dim_)
        return np.concatenate([w_new.flatten(), i, [params["s_init"]]])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        dim2 = self.dim_ * self.dim_
        return [w[dim2:-1] for w in self.W]

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """Visualize the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        # kinda works
        for w, col in zip(self.W, colors):
            dim2 = self.dim_ * self.dim_
            w_ = w[:dim2].reshape((self.dim_, self.dim_))
            b = w[dim2:-1]
            s = w[-1]
            plot_weight_matrix_as_ellipse(ax, s, w_, b, col)
