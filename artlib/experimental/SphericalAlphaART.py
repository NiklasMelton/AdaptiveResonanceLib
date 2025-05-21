import numpy as np
from matplotlib.axes import Axes
from typing import Optional, Iterable, Tuple, Dict
from warnings import warn
from pyalphashape import (
    SphericalAlphaShape,
    plot_spherical_alpha_shape,
    latlon_to_unit_vectors
)
from artlib.common.utils import IndexableOrKeyable
from artlib.experimental.AlphaART import AlphaART

class SphericalAlphaART(AlphaART):
    """
    Spherical Alpha-Shape ART for Clustering on Spherical Surfaces
    """
    data_format = "latlon"
    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset in lat-lon form.

        Returns
        -------
        np.ndarray
            Normalized data.

        """
        return X

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            The dataset in lat-lon form.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        return X


    def check_dimensions(self, X: np.ndarray):
        """Check the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The dataset in lat-lon form.

        """
        assert X.shape[1] == 2

    def validate_data(self, X: np.ndarray):
        """Validates the data prior to clustering.

        Parameters:
        - X: The dataset in lat-lon form.

        """
        # Check latitude and longitude ranges
        lat_valid = np.all((-90.0 <= X[:, 0]) & (X[:, 0] <= 90.0))
        lon_valid = np.all((-180.0 <= X[:, 1]) & (X[:, 1] <= 180.0))
        assert lat_valid and lon_valid, ("Data must be latitude and longitude in "
                                         "decimal form")

    def category_choice(
        self, i: np.ndarray, w: SphericalAlphaShape, params: dict
    ) -> tuple[float, Optional[dict]]:
        """
        Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample in lat-lon form.
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
        activation = 1.0-(w.distance_to_surface(i)/np.pi)

        cache = {"internal": activation >= 1.0}

        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: SphericalAlphaShape,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample in lat-lon form.
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
            pt = latlon_to_unit_vectors(i[None, :])[0]
            dot_products = np.clip(w.perimeter_points @ pt, -1.0, 1.0)
            angles = np.arccos(dot_products)
            dist = np.max(angles)
            M = 1.0 - (dist/np.pi)

        return M, cache

    def new_weight(self, i: np.ndarray, params: dict) -> SphericalAlphaShape:
        """
        Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample in lat-lon form.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        AlphaShape
            New cluster weight.

        """
        new_w = SphericalAlphaShape(i.reshape((1, -1)), alpha=params["alpha"])
        return new_w


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
            plot_spherical_alpha_shape(
                w, ax=ax, line_width=linewidth, line_color=c, fill=False
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
            The dataset in lat-lon form.
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
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))

        X_3d = latlon_to_unit_vectors(X)
        for k, col in enumerate(colors):
            cluster_data = y == k
            ax.scatter(
                X_3d[cluster_data, 0],
                X_3d[cluster_data, 1],
                X_3d[cluster_data, 2],
                color=col,
                marker=".",
                s=marker_size,
            )

        try:
            self.plot_cluster_bounds(ax, colors, linewidth)
        except NotImplementedError:
            warn(f"{self.__class__.__name__} does not support plotting cluster bounds.")



