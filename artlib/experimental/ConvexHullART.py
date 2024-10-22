import numpy as np
from matplotlib.axes import Axes
from copy import deepcopy
from typing import Optional, Iterable, List, Tuple, Union, Dict
from scipy.spatial import ConvexHull

from artlib.common.BaseART import BaseART
from artlib.experimental.merging import merge_objects


def plot_convex_polygon(
    vertices: np.ndarray,
    ax: Axes,
    line_color: str = "b",
    line_width: float = 1.0,
):
    """
    Plots a convex polygon given its vertices using Matplotlib.

    Parameters
    ----------
    vertices : np.ndarray
        A list of vertices representing a convex polygon.
    ax : matplotlib.axes.Axes
        A matplotlib Axes object to plot on.
    line_color : str, optional
        The color of the polygon lines, by default 'b'.
    line_width : float, optional
        The width of the polygon lines, by default 1.0.

    """
    vertices = np.array(vertices)
    # Close the polygon by appending the first vertex at the end
    vertices = np.vstack([vertices, vertices[0]])

    ax.plot(
        vertices[:, 0],
        vertices[:, 1],
        linestyle="-",
        color=line_color,
        linewidth=line_width,
    )


def volume_of_simplex(vertices: np.ndarray) -> float:
    """
    Calculates the n-dimensional volume of a simplex defined by its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        An (n+1) x n array representing the coordinates of the simplex vertices.

    Returns
    -------
    float
        Volume of the simplex.

    """
    vertices = np.asarray(vertices)
    # Subtract the first vertex from all vertices to form a matrix
    matrix = vertices[1:] - vertices[0]
    # Calculate the absolute value of the determinant divided by factorial(n) for volume
    return np.abs(np.linalg.det(matrix)) / np.math.factorial(len(vertices) - 1)


class PseudoConvexHull:
    def __init__(self, points: np.ndarray):
        """
        Initializes a PseudoConvexHull object.

        Parameters
        ----------
        points : np.ndarray
            An array of points representing the convex hull.

        """
        self.points = points

    @property
    def vertices(self):
        return np.array([i for i in range(self.points.shape[0])])

    def add_points(self, points):
        self.points = np.vstack([self.points, points])

    @property
    def area(self):
        if self.points.shape[0] == 1:
            return 0
        else:
            return 2*np.linalg.norm(self.points[0,:]-self.points[1,:],ord=2)


HullTypes = Union[ConvexHull, PseudoConvexHull]


def centroid_of_convex_hull(hull: HullTypes):
    """
    Finds the centroid of the volume of a convex hull in n-dimensional space.

    Parameters
    ----------
    hull : HullTypes
        A ConvexHull or PseudoConvexHull object.

    Returns
    -------
    np.ndarray
        Centroid coordinates.

    """
    hull_vertices = hull.points[hull.vertices]

    centroid = np.zeros(hull_vertices.shape[1])
    total_volume = 0

    if isinstance(hull, PseudoConvexHull):
        return np.mean(hull.points, axis=0)

    for simplex in hull.simplices:
        simplex_vertices = hull_vertices[simplex]
        simplex_centroid = np.mean(simplex_vertices, axis=0)
        simplex_volume = volume_of_simplex(simplex_vertices)

        centroid += simplex_centroid * simplex_volume
        total_volume += simplex_volume

    centroid /= total_volume
    return centroid


class ConvexHullART(BaseART):
    """
    ConvexHull ART for Clustering
    """

    def __init__(self, rho: float, alpha: float):
        """
        Initializes the ConvexHullART object.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.

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
        self, i: np.ndarray, w: HullTypes, params: dict
    ) -> tuple[float, Optional[dict]]:
        """
        Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : HullTypes
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

        if isinstance(w, PseudoConvexHull):

            if w.points.shape[0] == 1:
                new_w = deepcopy(w)
                new_w.add_points(i.reshape((1, -1)))
            else:
                new_points = np.vstack(
                    [w.points[w.vertices, :], i.reshape((1, -1))]
                )
                new_w = ConvexHull(new_points, incremental=True)
        else:
            new_w = ConvexHull(w.points[w.vertices, :], incremental=True)
            new_w.add_points(i.reshape((1, -1)))

        a_max = float(2*len(i))
        new_area = a_max - new_w.area
        activation = new_area / (a_max-w.area + params["alpha"])

        cache = {"new_w": new_w, "new_area": new_area}

        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: HullTypes,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : HullTypes
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
        M = float(cache["new_area"]) / float(2*len(i))
        cache["match_criterion"] = M

        return M, cache

    def update(
        self,
        i: np.ndarray,
        w: HullTypes,
        params: dict,
        cache: Optional[dict] = None,
    ) -> HullTypes:
        """
        Get the updated cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : HullTypes
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values cached from previous calculations.

        Returns
        -------
        HullTypes
            Updated cluster weight.

        """
        return cache["new_w"]

    def new_weight(self, i: np.ndarray, params: dict) -> HullTypes:
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
        HullTypes
            New cluster weight.

        """
        new_w = PseudoConvexHull(i.reshape((1, -1)))
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
            centers.append(centroid_of_convex_hull(w))
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
            if isinstance(w, ConvexHull):
                vertices = w.points[w.vertices, :2]
            else:
                vertices = w.points[:, :2]
            plot_convex_polygon(
                vertices, ax, line_width=linewidth, line_color=c
            )
