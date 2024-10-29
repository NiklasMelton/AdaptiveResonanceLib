import itertools
import logging
from scipy.spatial import Delaunay
import numpy as np
from typing import Union, Tuple, List
from collections import defaultdict

def circumcenter(points: np.ndarray) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
    """
    num_rows, num_columns = points.shape
    A = np.bmat([[2 * np.dot(points, points.T),
                  np.ones((num_rows, 1))],
                 [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1),
                   np.ones((1))))
    return np.linalg.solve(A, b)[:-1]



def circumradius(points: np.ndarray) -> float:
    """
    Calculte the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))



def alphasimplices(points: np.ndarray) -> np.ndarray:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points), simplex_points
        except np.linalg.LinAlgError:
            logging.warn('Singular matrix. Likely caused by all points '
                         'lying in an N-1 space.')

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


def get_perimeter_simplices(perimeter_edges, simplices, n):

    perimeter_edges_set = set(tuple(sorted(edge)) for edge in perimeter_edges)

    # This will store the (n-1)-simplices composed of perimeter edges
    perimeter_simplices = []

    for simplex in simplices:
        # Generate all (n-1)-dimensional subsimplices (faces)
        for subsimplex in itertools.combinations(simplex, n - 1):
            # Check if all edges of this subsimplex are in the perimeter
            subsimplex_edges = itertools.combinations(
                subsimplex,
                2
            )  # get pairs of vertices
            if all(tuple(sorted(edge)) in perimeter_edges_set for edge in
                   subsimplex_edges):
                perimeter_simplices.append(subsimplex)

    return perimeter_simplices


def compute_surface_area(points, perimeter_edges, simplices):
    """
    Compute the surface area (or perimeter in 2D) of the polytope formed by the perimeter edges.

    Args:
      points (np.ndarray): The points of the polytope.
      perimeter_edges (set): The set of perimeter edges.
      simplices (set): the set of simplices

    Returns:
      float: The total surface "area" (or perimeter in 2D, hyper-volume in higher dimensions).
    """
    # Handle the 2D case (perimeter)
    if points.shape[-1] == 2:
        total_perimeter = 0.0
        for edge in perimeter_edges:
            p1, p2 = edge
            total_perimeter += np.linalg.norm(points[p1] - points[p2])
        return total_perimeter

    # Handle the 3D and higher-dimensional cases
    perimeter_simplices = get_perimeter_simplices(
        perimeter_edges,
        simplices,
        points.shape[-1]
    )
    total_area = 0.
    for perimeter_simplex in perimeter_simplices:
        simplex_points = np.array([points[p,:] for p in perimeter_simplex])
        total_area += volume_of_simplex(simplex_points)

    return total_area

def build_linked_list(edges):
    # Step 1: Create adjacency list
    edges = [tuple(sorted(edge)) for edge in edges]
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Step 2: Initialize path and visited set for edges
    visited_edges = set()
    path = []

    def dfs(node, start, path):
        # Add the node to the current path
        path.append(node)

        # If we form a cycle that returns to the start and covers all edges
        if len(path) > 1 and node == start and len(visited_edges) == len(edges):
            return path

        # Explore neighbors
        for neighbor in graph[node]:
            edge = tuple(sorted((node, neighbor)))
            # Only continue if the edge hasn't been visited
            if edge not in visited_edges:
                visited_edges.add(edge)
                result = dfs(neighbor, start, path)
                if result:
                    return result  # Stop if a valid cycle is found
                # Backtrack
                visited_edges.remove(edge)

        # If no valid path found, backtrack by removing the node from the path
        path.pop()
        return None

    # Start DFS from any node in the graph
    start_node = edges[0][0]
    cycle = dfs(start_node, start_node, path)

    if cycle:
        return cycle[:-1]
    else:
        raise RuntimeError("Cannot build a minimal cycle with the given edges")


class AlphaShape:

    def __init__(self, points: np.ndarray, alpha: float = 0.):
        """
        Compute the alpha shape (concave hull) of a set of points.  If the number
        of points in the input is three or less, the convex hull is returned to the
        user.  For two points, the convex hull collapses to a `LineString`; for one
        point, a `Point`.

        Args:

          points np.ndarray: an iterable container of points
          alpha (float): alpha value

        Returns:
          np.ndarray
        """
        assert points.shape[-1] > 1
        # Create a set to hold unique edges of simplices that pass the radius
        # filtering
        edges = set()
        perimeter_edges = set()
        simplices = set()
        self.perimeter_points = []
        self.alpha = alpha
        self.centroid = 0
        self.volume = 0
        self.surface_area = 0

        if len(points) < 3:
            # handle lines and points separately
            self.perimeter_points = points
            self.centroid = np.mean(points, axis=0)
            self.volume = 0
            if len(points) < 2:
                self.surface_area = 0
                self.perimeter_edges = []
            else:
                self.surface_area = 2*np.linalg.norm(points[0,:]-points[1,:], ord=2)
                self.perimeter_edges = [(points[0,:], points[1,:])]

        else:
            # Whenever a simplex is found that passes the radius filter, its edges
            # will be inspected to see if they already exist in the `edges` set.  If an
            # edge does not already exist there, it will be added to both the `edges`
            # set and the `permimeter_edges` set.  If it does already exist there, it
            # will be removed from the `perimeter_edges` set if found there.  This is
            # taking advantage of the property of perimeter edges that each edge can
            # only exist once.
            if alpha > 0:
                radius_filter = 1.0 / alpha
            else:
                radius_filter = np.inf
            alpha_simplices = list(alphasimplices(points))
            alpha_simplices.sort(key=lambda x: x[1])
            visited_points = set()
            for point_indices, circumradius, simplex_coords in alpha_simplices:
                # Radius filter
                if circumradius < radius_filter or not all(p in visited_points for p in point_indices):
                    for edge in itertools.combinations(
                            point_indices, r=points.shape[-1]):
                        edge = tuple(sorted(edge))

                        if all([tuple(sorted(e)) not in edges for e in
                                      itertools.combinations(
                                edge, r=len(edge))]):
                            edges.add(edge)
                            perimeter_edges.add(edge)
                        else:
                            perimeter_edges -= set(
                                tuple(sorted(e))
                                for e in itertools.combinations(edge, r=len(edge))
                            )
                    visited_points.update(tuple(point_indices))
                    simplices.add(tuple(point_indices))
                    simplex_centroid = np.mean(simplex_coords, axis=0)
                    simplex_volume = volume_of_simplex(simplex_coords)
                    self.volume += simplex_volume
                    self.centroid += simplex_centroid * simplex_volume

            self.perimeter_edges = [
                (points[p1,:], points[p2,:]) for p1, p2 in perimeter_edges
            ]
            if len(edges) > 0:
                self.centroid /= self.volume

                perimeter_indices = set(p for e in perimeter_edges for p in e)
                self.perimeter_points = np.vstack(
                    [points[p, :] for p in perimeter_indices]
                )
                self.surface_area = compute_surface_area(points, perimeter_edges, simplices)


    def add_points(self, points: np.ndarray):
        """
        Adds a new point to the alpha shape and updates the perimeter points, volume,
        surface area, and centroid accordingly.

        Args:
            point (np.ndarray): The new point to add.
        """
        # Add the new point to the existing points
        new_points = np.vstack([self.perimeter_points, points])

        # Recompute the alpha shape with the updated points
        self.__init__(new_points, alpha = float(self.alpha))


    @property
    def is_empty(self):
        return len(self.perimeter_points) == 0

    @property
    def vertices(self):
        return self.perimeter_points

    def contains_point(self, point: np.ndarray):
        if len(self.perimeter_points) < 3:
            return True
        delaunay = Delaunay(self.perimeter_points)
        return delaunay.find_simplex(point) >= 0

    @property
    def simplices(self) -> List[np.ndarray]:
        """
        Generate the Delaunay simplices for the perimeter points of this AlphaShape.

        Returns:
            List[np.ndarray]: List of simplices, where each simplex is an array of points.
        """
        delaunay = Delaunay(self.perimeter_points)
        return [self.perimeter_points[simplex] for simplex in delaunay.simplices]

    @staticmethod
    def simplex_intersects(simplex_a: np.ndarray, simplex_b: np.ndarray) -> bool:
        """
        Check if two simplices intersect by testing if any vertex of one simplex
        is within the other.

        Args:
            simplex_a (np.ndarray): A simplex represented by an array of points.
            simplex_b (np.ndarray): Another simplex represented by an array of points.

        Returns:
            bool: True if the simplices intersect, False otherwise.
        """
        delaunay_a = Delaunay(simplex_a)
        delaunay_b = Delaunay(simplex_b)

        # Check if any vertex of simplex_a is inside simplex_b
        if any(delaunay_b.find_simplex(vertex) >= 0 for vertex in simplex_a):
            return True

        # Check if any vertex of simplex_b is inside simplex_a
        if any(delaunay_a.find_simplex(vertex) >= 0 for vertex in simplex_b):
            return True

        return False


    def overlaps_with(self, other: 'AlphaShape') -> bool:
        """
        Check if this AlphaShape overlaps with another AlphaShape by examining simplex intersections.

        Args:
            other (AlphaShape): Another AlphaShape object.

        Returns:
            bool: True if the shapes overlap, False otherwise.
        """
        # Get simplices for both AlphaShapes
        simplices_self = self.simplices
        simplices_other = other.simplices

        # Check if any simplex in self intersects with any simplex in other
        for simplex_a in simplices_self:
            for simplex_b in simplices_other:
                if self.simplex_intersects(simplex_a, simplex_b):
                    return True
        return False