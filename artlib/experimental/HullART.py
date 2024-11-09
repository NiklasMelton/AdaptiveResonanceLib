import numpy as np
from matplotlib.axes import Axes
from copy import deepcopy
from typing import Optional, Iterable, List, Tuple, Dict

from artlib.experimental.alphashape import AlphaShape, equalateral_simplex_volume
from artlib.common.BaseART import BaseART


def plot_polygon_edges(
    edges: np.ndarray,
    ax: Axes,
    line_color: str = "b",
    line_width: float = 1.0,
):
    """
    Plots a convex polygon given its vertices using Matplotlib.

    Parameters
    ----------
    vertices : np.ndarray
        A list of edges representing a polygon.
    ax : matplotlib.axes.Axes
        A matplotlib Axes object to plot on.
    line_color : str, optional
        The color of the polygon lines, by default 'b'.
    line_width : float, optional
        The width of the polygon lines, by default 1.0.

    """
    for p1, p2 in edges:
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        ax.plot(
            x,
            y,
            linestyle="-",
            color=line_color,
            linewidth=line_width,
        )



class HullART(BaseART):
    """
    Hull ART for Clustering
    """

    def __init__(self, rho: float, alpha: float, alpha_hull: float, min_lambda: float, max_lambda: float):
        """
        Initializes the HullART object.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        alpha_hull : float
            alpha shape parameter.
        lambda : float
            minimum volume.
        """
        params = {"rho": rho, "alpha": alpha, "alpha_hull": alpha_hull, "min_lambda": min_lambda, "max_lambda": max_lambda}
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

        assert "alpha_hull" in params
        assert params["alpha_hull"] >= 0.0
        assert isinstance(params["alpha_hull"], float)

        assert "min_lambda" in params
        assert params["min_lambda"] >= 0.0
        assert isinstance(params["min_lambda"], float)

        assert "max_lambda" in params
        assert params["max_lambda"] >= 0.0
        assert isinstance(params["max_lambda"], float)

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

        new_w = deepcopy(w)
        new_w.add_points(i.reshape((1, -1)))
        if new_w.is_empty:
            activation = np.nan
            new_vol = 0
        else:
            min_vol = equalateral_simplex_volume(len(i), params["min_lambda"])
            new_vol = 1. - max(new_w.volume, min_vol)
            activation = new_vol / (1. - max(w.volume, min_vol) + params["alpha"])

        cache = {"new_w": new_w, "new_vol": new_vol, "activation": activation}

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
        assert cache is not None
        M = float(cache["new_vol"])
        cache["match_criterion"] = M

        return M, cache

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
        GeneralHull
            New cluster weight.

        """
        new_w = AlphaShape(i.reshape((1, -1)), alpha=params["alpha_hull"], max_perimeter_length=params["max_lambda"])
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
            edges = [(e1[:2], e2[:2]) for e1, e2 in w.perimeter_edges]

            plot_polygon_edges(
                edges, ax, line_width=linewidth, line_color=c
            )

    # def post_fit(self, X: np.ndarray):
    #     can_merge = lambda A, B: A.overlaps_with(B)
    #     merges = merge_objects(self.W, can_merge)
    #     new_W = []
    #     new_labels = np.zeros_like(self.labels_)
    #     for i, items in enumerate(merges):
    #         points = np.vstack([self.W[item].perimeter_points for item in items])
    #         new_W.append(AlphaShape(points, self.W[items[0]].alpha))
    #         for item in items:
    #             new_labels[self.labels_ == item] = i
    #
    #     self.W = new_W
    #     self.labels_ = new_labels


