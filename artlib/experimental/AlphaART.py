import numpy as np
from matplotlib.axes import Axes
from typing import Optional, Iterable, List, Tuple, Dict, Callable
from warnings import warn
from pyalphashape import (AlphaShape, plot_polygon_edges)
from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable
import operator
from copy import deepcopy

class AlphaART(BaseART):
    """
    Alpha-Shape ART for Clustering
    """

    def __init__(self, rho: float, alpha: float):
        """
        Initializes the AlphaART object.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            alpha shape parameter.
        """
        params = {"rho": rho, "alpha": alpha}
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """
        Validates clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert params["rho"] >= 0.0
        assert isinstance(params["rho"], float)

        assert "alpha" in params
        assert params["alpha"] >= 0.0
        assert isinstance(params["alpha"], float)


    def category_choice(
        self, i: np.ndarray, w: AlphaShape, params: dict
    ) -> tuple[float, Optional[dict]]:
        """
        Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : AlphaShape
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
        dim = i.size
        activation = 1.0-(w.distance_to_surface(i)/np.sqrt(dim))

        cache = {"internal": activation >= 1.0}

        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: AlphaShape,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : AlphaShape
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values cached from previous calculations.

        Returns
        -------
        float
            Cluster match criterion.
        dict
            Cache used for later processing.

        """
        assert cache is not None and "internal" in cache
        if cache["internal"]:
            M = 1.0
        else:
            dim = i.size
            dist = np.max(np.linalg.norm(w.perimeter_points - i, axis=1))
            M = 1.0 - (dist/np.sqrt(dim))

        return M, cache

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
        op: Callable = operator.ge,
    ) -> Tuple[bool, Dict]:
        """Get the binary match criterion of the cluster.

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
        tuple
            Binary match criterion and cache used for later processing.

        """
        M, cache = self.match_criterion(i, w, params=params, cache=cache)
        M_bin = op(M, params["rho"])
        new_w = None

        if cache is None:
            cache = dict()
        cache["valid_shape"] = True
        if M_bin:
            if "internal" in cache and cache["internal"] == True:
                new_w = w
            else:
                new_w = deepcopy(w)
                new_w.add_points(i.reshape(1,-1))
                # if new_w.is_empty or not new_w.contains_point(i):
                if new_w.is_empty:
                    M_bin = False
                    cache["valid_shape"] = False

        cache["new_w"] = new_w
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin
        return M_bin, cache

    def update(
        self,
        i: np.ndarray,
        w: AlphaShape,
        params: dict,
        cache: Optional[dict] = None,
    ) -> AlphaShape:
        """
        Get the updated cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : AlphaShape
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values cached from previous calculations.

        Returns
        -------
        AlphaShape
            Updated cluster weight.

        """
        assert cache is not None and "new_w" in cache
        return cache["new_w"]

    def new_weight(self, i: np.ndarray, params: dict) -> AlphaShape:
        """
        Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        AlphaShape
            New cluster weight.

        """
        new_w = AlphaShape(i.reshape((1, -1)), alpha=params["alpha"])
        return new_w

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        centers = []
        for w in self.W:
            centers.append(w.centroid)
        return centers

    def plot_cluster_bounds(
        self, ax: Axes, colors: Iterable, linewidth: int = 1
    ):
        """
        Visualize the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        for c, w in zip(colors, self.W):
            edges = [(e1, e2) for e1, e2 in w.perimeter_edges]

            plot_polygon_edges(
                edges, ax, line_width=linewidth, line_color=c
            )

    def visualize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ax: Optional[Axes] = None,
        marker_size: int = 10,
        linewidth: int = 1,
        colors: Optional[IndexableOrKeyable] = None,
    ):
        """Visualize the clustering of the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray
            Sample labels.
        ax : matplotlib.axes.Axes, optional
            Figure axes.
        marker_size : int, default=10
            Size used for data points.
        linewidth : int, default=1
            Width of boundary line.
        colors : IndexableOrKeyable, optional
            Colors to use for each cluster.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            if X.shape[1] > 2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))


        for k, col in enumerate(colors):
            cluster_data = y == k
            if X.shape[1] > 2:
                ax.scatter(
                    X[cluster_data, 0],
                    X[cluster_data, 1],
                    X[cluster_data, 2],
                    color=col,
                    marker=".",
                    s=marker_size,
                )
            else:
                ax.scatter(
                    X[cluster_data, 0],
                    X[cluster_data, 1],
                    color=col,
                    marker=".",
                    s=marker_size,
                )

        try:
            self.plot_cluster_bounds(ax, colors, linewidth)
        except NotImplementedError:
            warn(f"{self.__class__.__name__} does not support plotting cluster bounds.")



