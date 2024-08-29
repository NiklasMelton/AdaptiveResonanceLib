import numpy as np
from matplotlib.axes import Axes
from copy import deepcopy
from typing import Optional, Iterable, List, Union
from scipy.spatial import ConvexHull

from artlib.common.BaseART import BaseART
from artlib.experimental.merging import merge_objects


def plot_convex_polygon(vertices: np.ndarray, ax: Axes, line_color: str = 'b', line_width: float = 1.0):
    """
    Plots a convex polygon given its vertices using Matplotlib.

    Parameters:
    - vertices: A list of vertices representing a convex polygon.
    - ax: A matplotlib Axes object to plot on. If None, creates a new figure and axes.
    - line_color: The color of the polygon lines.
    - line_width: The width of the polygon lines.
    """
    vertices = np.array(vertices)
    # Close the polygon by appending the first vertex at the end
    vertices = np.vstack([vertices, vertices[0]])

    ax.plot(vertices[:, 0], vertices[:, 1], linestyle='-', color=line_color, linewidth=line_width)


def volume_of_simplex(vertices):
    """
    Calculates the n-dimensional volume of a simplex defined by its vertices.

    Parameters:
    - vertices: An (n+1) x n array representing the coordinates of the simplex vertices.

    Returns:
    - Volume of the simplex.
    """
    vertices = np.asarray(vertices)
    # Subtract the first vertex from all vertices to form a matrix
    matrix = vertices[1:] - vertices[0]
    # Calculate the absolute value of the determinant divided by factorial(n) for volume
    return np.abs(np.linalg.det(matrix)) / np.math.factorial(len(vertices) - 1)


def minimum_distance(a1, a2):
    def point_to_point_distance(P, Q):
        """Calculate the Euclidean distance between two points P and Q."""
        return np.linalg.norm(P - Q)

    def point_to_line_segment_distance(P, Q1, Q2):
        """Calculate the minimum distance between point P and line segment Q1-Q2."""
        line_vec = Q2 - Q1
        point_vec = P - Q1
        line_len = np.dot(line_vec, line_vec)

        if line_len == 0:  # Q1 and Q2 are the same point
            return np.linalg.norm(P - Q1)

        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = Q1 + t * line_vec
        return np.linalg.norm(P - projection)

    def line_segment_to_line_segment_distance(A1, A2, B1, B2):
        """Calculate the minimum distance between two line segments A1-A2 and B1-B2."""
        distances = [
            point_to_line_segment_distance(A1, B1, B2),
            point_to_line_segment_distance(A2, B1, B2),
            point_to_line_segment_distance(B1, A1, A2),
            point_to_line_segment_distance(B2, A1, A2),
        ]
        return min(distances)

    # Determine the cases and compute the distance
    if a1.shape[0] == 1 and a2.shape[0] == 1:
        # Case 1: Point to point
        return point_to_point_distance(a1[0], a2[0])

    elif a1.shape[0] == 1 and a2.shape[0] == 2:
        # Case 2: Point to line segment
        return point_to_line_segment_distance(a1[0], a2[0], a2[1])

    elif a1.shape[0] == 2 and a2.shape[0] == 1:
        # Case 3: Line segment to point
        return point_to_line_segment_distance(a2[0], a1[0], a1[1])

    elif a1.shape[0] == 2 and a2.shape[0] == 2:
        # Case 4: Line segment to line segment
        return line_segment_to_line_segment_distance(a1[0], a1[1], a2[0], a2[1])
    else:
        raise RuntimeError("INVALID DISTANCE CALCULATION")



class PseudoConvexHull:
    def __init__(self, points: np.ndarray):
        self.points = points

    @property
    def vertices(self):
        return np.array([i for i in range(self.points.shape[0])])

    def add_points(self, points):
        self.points = np.vstack([self.points, points])


HullTypes = Union[ConvexHull, PseudoConvexHull]


def centroid_of_convex_hull(hull: HullTypes):
    """
    Finds the centroid of the volume of a convex hull in n-dimensional space.

    Parameters:
    - vertices: An array of shape (m, n), where m is the number of vertices and n is the dimension.

    Returns:
    - Centroid coordinates as a numpy array of length n.
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
    def __init__(self, rho: float, merge_rho: float):
        """
        Parameters:
        - rho: vigilance parameter

        """
        params = {
            "rho": rho,
            "merge_rho": merge_rho
        }
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert params["rho"] >= 0.
        assert isinstance(params["rho"], float)

    def category_choice(self, i: np.ndarray, w: HullTypes, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        if isinstance(w, PseudoConvexHull):
            d = minimum_distance(i.reshape((1,-1)), w.points)
            activation = 1. - d**len(i)
            if w.points.shape[0] == 1:
                new_w = deepcopy(w)
                new_w.add_points(i.reshape((1,-1)))
            else:
                new_points = np.vstack([w.points[w.vertices,:], i.reshape((1,-1))])
                new_w = ConvexHull(new_points, incremental=True)
        else:
            new_w = ConvexHull(w.points[w.vertices,:], incremental=True)
            new_w.add_points(i.reshape((1,-1)))
            activation = w.area / new_w.area

        cache = {"new_w": new_w, "activation": activation}
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: HullTypes, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        """
        get the match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion, cache used for later processing

        """
        return cache["activation"], cache


    def update(self, i: np.ndarray, w: HullTypes, params: dict, cache: Optional[dict] = None) -> HullTypes:
        """
        get the updated cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            updated cluster weight, cache used for later processing

        """
        return cache["new_w"]

    def new_weight(self, i: np.ndarray, params: dict) -> HullTypes:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        new_w = PseudoConvexHull(i.reshape((1,-1)))
        return new_w

    def merge_clusters(self):
        def can_merge(w1, w2):
            combined_points = np.vstack([w1.points[w1.vertices,:], w2.points[w2.vertices,:]])

            if isinstance(w1, PseudoConvexHull) and isinstance(w2, PseudoConvexHull):
                d = minimum_distance(w1.points, w2.points)
                activation = 1.0 - d**w1.points.shape[1]
            else:
                new_w = ConvexHull(combined_points)
                if isinstance(w1, ConvexHull):
                    a1 = w1.area / new_w.area
                else:
                    a1 = np.nan
                if isinstance(w2, ConvexHull):
                    a2 = w2.area / new_w.area
                else:
                    a2 = np.nan
                activation = np.max([a1,a2])

            if activation > self.params["merge_rho"]:
                return True
            else:
                return False

        merges = merge_objects(self.W, can_merge)

        new_W = []
        new_sample_counter = []
        new_labels = np.copy(self.labels_)
        for i in range(len(merges)):
            new_labels[np.isin(self.labels_, merges[i])] = i
            new_sample_counter.append(sum(self.weight_sample_counter_[j] for j in merges[i]))
            if len(merges[i]) > 1:
                new_points = np.vstack([self.W[j].points for j in merges[i]])
                if new_points.shape[0] > 2:
                    new_W.append(ConvexHull(new_points, incremental=True))
                else:
                    new_W.append(PseudoConvexHull(new_points))
            else:
                new_W.append(self.W[merges[i][0]])
        self.W = new_W
        self.weight_sample_counter_ = new_sample_counter
        self.labels_ = new_labels

    def post_fit(self, X: np.ndarray):
        """
        function called after fit. Useful for cluster pruning

        Parameters:
        - X: data set

        """
        self.merge_clusters()




    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        centers = []
        for w in self.W:
            centers.append(centroid_of_convex_hull(w))
        return centers

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        for c, w in zip(colors, self.W):
            if isinstance(w, ConvexHull):
                vertices = w.points[w.vertices,:2]
            else:
                vertices = w.points[:, :2]
            plot_convex_polygon(vertices, ax, line_width=linewidth, line_color=c)
