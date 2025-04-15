import numpy as np
from matplotlib.axes import Axes
from typing import Optional, Iterable, List, Tuple, Dict
from warnings import warn
from artlib.experimental.AlphaShape import (AlphaShape, plot_polygon_edges)
from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable


class HullART(BaseART):
    """
    Hull ART for Clustering
    """

    def __init__(self, rho: float, alpha: float):
        """
        Initializes the HullART object.

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

        cache = None

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
        dim = i.size
        dist = np.max(np.linalg.norm(w.perimeter_points - i, axis=1))
        M = 1.0 - (dist/np.sqrt(dim))

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
        w.add_points(i.reshape(1,-1))
        return w

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


