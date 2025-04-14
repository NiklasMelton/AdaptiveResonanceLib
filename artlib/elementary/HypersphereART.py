"""Hyperpshere ART :cite:`anagnostopoulos2000hypersphere`."""
# Anagnostopoulos, G. C., & Georgiopulos, M. (2000).
# Hypersphere ART and ARTMAP for unsupervised and supervised, incremental learning.
# In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
# (pp. 59–64). volume 6. doi:10.1109/IJCNN.2000.859373.

import numpy as np
from typing import Optional, Iterable, List
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import l2norm2


class HypersphereART(BaseART):
    """Hypersphere ART for Clustering.

    This module implements Ellipsoid ART as first published in:
    :cite:`anagnostopoulos2000hypersphere`.

    .. # Anagnostopoulos, G. C., & Georgiopulos, M. (2000).
    .. # Hypersphere ART and ARTMAP for unsupervised and supervised, incremental
    .. # learning.
    .. # In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
    .. # (pp. 59–64). volume 6. doi:10.1109/IJCNN.2000.859373.

    Hyperpshere ART clusters data in Hyper-spheres similar to k-means with a dynamic k.

    """

    def __init__(self, rho: float, alpha: float, beta: float, r_hat: float):
        """Initialize the Hypersphere ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning rate.
        r_hat : float
            Maximum possible category radius.

        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
            "r_hat": r_hat,
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
        assert "alpha" in params
        assert "beta" in params
        assert "r_hat" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert params["alpha"] >= 0.0
        assert 1.0 >= params["beta"] >= 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["r_hat"], float)

    @staticmethod
    def category_distance(
        i: np.ndarray, centroid: np.ndarray, radius: float, params
    ) -> float:
        """Compute the category distance between a data sample and a centroid.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        centroid : np.ndarray
            Cluster centroid.
        radius : float
            Cluster radius.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        float
            Category distance.

        """
        return np.sqrt(l2norm2(i - centroid))

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
        centroid = w[:-1]
        radius = w[-1]

        i_radius = self.category_distance(i, centroid, radius, params)
        max_radius = max(radius, i_radius)

        cache = {
            "max_radius": max_radius,
            "i_radius": i_radius,
        }
        return (params["r_hat"] - max_radius) / (
            params["r_hat"] - radius + params["alpha"]
        ), cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> tuple[float, Optional[dict]]:
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
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]

        return 1 - (max(radius, max_radius) / params["r_hat"]), cache

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
            Updated cluster weight.

        """
        centroid = w[:-1]
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]
        i_radius = cache["i_radius"]

        radius_new = radius + (params["beta"] / 2) * (max_radius - radius)
        centroid_new = centroid + (params["beta"] / 2) * (i - centroid) * (
            1 - (min(radius, i_radius) / (i_radius + params["alpha"]))
        )

        return np.concatenate([centroid_new, [radius_new]])

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
        return np.concatenate([i, [0.0]])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return [w[:-1] for w in self.W]

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
        from matplotlib.patches import Circle

        for w, col in zip(self.W, colors):
            centroid = (w[0], w[1])
            radius = w[-1]
            circ = Circle(
                centroid,
                radius,
                linewidth=linewidth,
                edgecolor=col,
                facecolor="none",
            )
            ax.add_patch(circ)
