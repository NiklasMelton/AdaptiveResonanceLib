import numpy as np
from matplotlib.axes import Axes
from copy import deepcopy
from typing import Optional, Iterable, List, Tuple, Union, Dict
from scipy.spatial import ConvexHull
from shapely import Polygon
from alphashape import alphashape

from artlib.common.BaseART import BaseART
from artlib.experimental.merging import merge_objects


def plot_polygon(
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

def centroid_of_convex_hull(hull: Union[PseudoConvexHull, ConvexHull]):
    """
    Finds the centroid of the volume of a convex hull in n-dimensional space.

    Parameters
    ----------
    hull : Union[PseudoConvexHull, ConvexHull]
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

class GeneralHull:
    def __init__(self, points: np.ndarray, alpha: float = 0.0):
        self.dim = points.shape[1]
        self.alpha = alpha
        if points.shape[0] <= 2:
            self.hull = PseudoConvexHull(points)
        elif points.shape[0] == 3 or alpha == 0.0:
            self.hull = ConvexHull(points, incremental=True)
        else:
            self.hull = alphashape(points, alpha=self.alpha)

    def add_points(self, points: np.ndarray):
        if isinstance(self.hull, PseudoConvexHull):
            if self.hull.points.shape[0] == 1:
                self.hull.add_points(points.reshape((-1, self.dim)))
            else:
                new_points = np.vstack(
                    [
                        self.hull.points[self.hull.vertices, :],
                        points.reshape((-1, self.dim))
                    ]
                )
                self.hull = ConvexHull(new_points, incremental=True)
        elif isinstance(self.hull, ConvexHull) and self.alpha == 0.0:
            self.hull.add_points(i.reshape((-1, self.dim)))
        else:
            if isinstance(self.hull, ConvexHull):
                new_points = np.vstack(
                    [
                        self.hull.points[self.hull.vertices, :],
                        points.reshape((-1, self.dim))
                    ]
                )
                self.hull = alphashape(new_points, alpha=self.alpha)
            else:
                new_points = np.vstack(
                    [
                        np.asarray(self.hull.exterior.coords),
                        points.reshape((-1, self.dim))
                    ]
                )
                self.hull = alphashape(new_points, alpha=self.alpha)

    @property
    def area(self):
        if isinstance(self.hull, (PseudoConvexHull, ConvexHull)) or self.dim > 2:
            return self.hull.area
        else:
            return self.hull.length

    @property
    def centroid(self):
        if isinstance(self.hull, (PseudoConvexHull, ConvexHull)):
            return centroid_of_convex_hull(self.hull)
        else:
            return self.hull.centroid

    @property
    def is_empty(self):
        if isinstance(self.hull, (PseudoConvexHull, ConvexHull)):
            return False
        else:
            return self.hull.is_empty

    @property
    def vertices(self):
        if isinstance(self.hull, ConvexHull):
            return self.hull.points[self.hull.vertices, :]
        elif isinstance(self.hull, Polygon):
            return np.asarray(self.hull.exterior.coords)
        else:
            return self.hull.points

    def deepcopy(self):
        if isinstance(self.hull, Polygon):
            points = np.asarray(self.hull.exterior.coords)
            return GeneralHull(points, alpha=float(self.alpha))
        elif isinstance(self.hull, ConvexHull):
            points = self.hull.points[self.hull.vertices, :]
            return GeneralHull(points, alpha=float(self.alpha))
        else:
            return deepcopy(self)


class HullART(BaseART):
    """
    Hull ART for Clustering
    """

    def __init__(self, rho: float, alpha: float, alpha_hat: float):
        """
        Initializes the HullART object.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        alpha_hat : float
            alpha shape parameter.

        """
        params = {"rho": rho, "alpha": alpha, "alpha_hat": alpha_hat}
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
        assert "alpha_hat" in params
        assert params["alpha_hat"] >= 0.0
        assert isinstance(params["alpha_hat"], float)

    def category_choice(
        self, i: np.ndarray, w: GeneralHull, params: dict
    ) -> tuple[float, Optional[dict]]:
        """
        Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : GeneralHull
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

        new_w = w.deepcopy()
        new_w.add_points(i.reshape((1,-1)))
        if new_w.is_empty:
            raise RuntimeError(
                f"alpha_hat={params['alpha_hat']} results in invalid geometry"
            )

        a_max = float(2*len(i))
        new_area = a_max - new_w.area
        activation = new_area / (a_max-w.area + params["alpha"])

        cache = {"new_w": new_w, "new_area": new_area, "activation": activation}

        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: GeneralHull,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : GeneralHull
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
        w: GeneralHull,
        params: dict,
        cache: Optional[dict] = None,
    ) -> GeneralHull:
        """
        Get the updated cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : GeneralHull
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values cached from previous calculations.

        Returns
        -------
        GeneralHull
            Updated cluster weight.

        """
        return cache["new_w"]

    def new_weight(self, i: np.ndarray, params: dict) -> GeneralHull:
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
        new_w = GeneralHull(i.reshape((1, -1)), alpha=params["alpha_hat"])
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
            vertices = w.vertices[:,:2]

            plot_polygon(
                vertices, ax, line_width=linewidth, line_color=c
            )
