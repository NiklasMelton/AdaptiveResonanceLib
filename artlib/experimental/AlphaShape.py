import itertools
import logging
from scipy.spatial import Delaunay
import numpy as np
import math
from typing import Tuple, Set, List, Literal
from itertools import combinations
from matplotlib.axes import Axes


class GraphClosureTracker:
    def __init__(self, num_nodes):
        # Initialize each node to be its own parent (self-loop)
        self.parent = list(range(num_nodes))
        self.rank = [0] * num_nodes  # Rank array to optimize union operation
        self.num_nodes = num_nodes
        self.components = {i: {i} for i in range(num_nodes)}  # Initial components

    def _ensure_capacity(self, node: int):
        if node >= self.num_nodes:
            # extend parent / rank arrays and component dict
            for i in range(self.num_nodes, node + 1):
                self.parent.append(i)
                self.rank.append(0)
                self.components[i] = {i}
            self.num_nodes = node + 1

    # modify the public methods to call it
    def find(self, node):
        self._ensure_capacity(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        # Union by rank and update components
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            # Attach smaller rank tree under the larger rank tree
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                # Update the components by merging sets
                self.components[root1].update(self.components[root2])
                del self.components[root2]
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
                self.components[root2].update(self.components[root1])
                del self.components[root1]
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
                self.components[root1].update(self.components[root2])
                del self.components[root2]

    def add_edge(self, node1, node2):
        # Add an edge by connecting two nodes
        self.union(node1, node2)

    def add_fully_connected_subgraph(self, nodes):
        # Connect each pair of nodes to form a fully connected subgraph
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.union(nodes[i], nodes[j])

    def subgraph_is_already_connected(self, nodes):
        # Check if all nodes in the given list are connected

        if not nodes:
            return True  # Empty list is trivially connected
        # Find the root of the first node
        root = self.find(nodes[0])
        # Check if all other nodes share this root
        return all(self.find(node) == root for node in nodes)

    def is_connected(self, node1, node2):
        # Check if two nodes are in the same component
        return self.find(node1) == self.find(node2)

    def __iter__(self):
        # Make the class iterable over connected components
        return iter(self.components.values())

    def __getitem__(self, index):
        # Make the class indexable over connected components
        return list(self.components.values())[index]

    def __len__(self):
        # Return the number of connected components
        return len(self.components)




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
    tri = Delaunay(coords, qhull_options="Qz")

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


def get_perimeter_simplices(perimeter_edges: Set[Tuple], simplices: Set[Tuple], n: int):

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


def compute_surface_area(points: np.ndarray, perimeter_edges: Set[Tuple], simplices: Set[Tuple]):
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


def equalateral_simplex_volume(n: int, s: float):
    numerator = s ** n
    denominator = math.factorial(n)
    sqrt_term = math.sqrt((n + 1) / (2 ** n))
    volume = (numerator / denominator) * sqrt_term
    return volume


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
        if len(p1) > 2:
            z = [p1[2], p2[2]]
            ax.plot(
                x,
                y,
                z,
                linestyle="-",
                color=line_color,
                linewidth=line_width,
            )
        else:
            ax.plot(
                x,
                y,
                linestyle="-",
                color=line_color,
                linewidth=line_width,
            )

# class AlphaShape:
#
#     def __init__(self, points: np.ndarray, alpha: float = 0.):
#         """
#         Compute the alpha shape (concave hull) of a set of points.  If the number
#         of points in the input is three or less, the convex hull is returned to the
#         user.  For two points, the convex hull collapses to a `LineString`; for one
#         point, a `Point`.
#
#         Args:
#
#           points np.ndarray: an iterable container of points
#           alpha (float): alpha value
#
#         Returns:
#           np.ndarray
#         """
#         dim = points.shape[1]
#         assert dim > 1
#         self.points = points
#         # Create a set to hold unique edges of simplices that pass the radius
#         # filtering
#         edges = set()
#         perimeter_edges = set()
#         self.simplices = set()
#         self.perimeter_points = []
#         self.alpha = alpha
#
#         self.centroid = 0
#         self.volume = 0
#         # self.surface_area = 0
#         self.GCT = GraphClosureTracker(points.shape[0])
#         visited_points = set()
#         n_points = points.shape[0]
#         if n_points < dim+1:
#             # handle lines and points separately
#             self.perimeter_points = points
#             self.centroid = np.mean(points, axis=0)
#             self.volume = 0
#             if n_points < 2:
#                 # self.surface_area = 0
#                 self.perimeter_edges = []
#                 visited_points = {0}
#             else:
#                 # self.surface_area = 2*np.linalg.norm(points[0,:]-points[1,:], ord=2)
#                 edge_indices = list(combinations(range(n_points), r=dim))
#                 self.perimeter_edges = list(
#                     tuple(points[i,:] for i in es)
#                     for ps in edge_indices
#                     for es in itertools.combinations(ps, r=2)
#                 )
#                 visited_points = set(range(n_points))
#                 for ps in edge_indices:
#                     for a,b in itertools.combinations(ps, r=2):
#                         self.GCT.add_edge(a,b)
#
#         else:
#             # Whenever a simplex is found that passes the radius filter, its edges
#             # will be inspected to see if they already exist in the `edges` set.  If an
#             # edge does not already exist there, it will be added to both the `edges`
#             # set and the `permimeter_edges` set.  If it does already exist there, it
#             # will be removed from the `perimeter_edges` set if found there.  This is
#             # taking advantage of the property of perimeter edges that each edge can
#             # only exist once.
#             if alpha > 0:
#                 radius_filter = 1.0 / alpha
#             else:
#                 radius_filter = np.inf
#             alpha_simplices = list(alphasimplices(points))
#             alpha_simplices.sort(key=lambda x: x[1])
#
#             added_simplex_coords = []
#
#             for point_indices, circumradius, simplex_coords in alpha_simplices:
#                 # Radius filter
#                 check_points = point_indices.tolist()
#                 if (
#                         circumradius < radius_filter or
#                         not self.GCT.subgraph_is_already_connected(check_points)
#                 ):
#                     self.GCT.add_fully_connected_subgraph(point_indices.tolist())
#                     for edge in itertools.combinations(
#                             point_indices, r=points.shape[-1]):
#                         edge = tuple(sorted(edge))
#
#                         if all([tuple(sorted(e)) not in edges for e in
#                                       itertools.combinations(
#                                 edge, r=len(edge))]):
#                             edges.add(edge)
#                             perimeter_edges.add(edge)
#                         else:
#                             perimeter_edges -= set(
#                                 tuple(sorted(e))
#                                 for e in itertools.combinations(edge, r=len(edge))
#                             )
#                     visited_points.update(tuple(point_indices))
#                     self.simplices.add(tuple(point_indices))
#                     simplex_centroid = np.mean(simplex_coords, axis=0)
#                     simplex_volume = volume_of_simplex(simplex_coords)
#                     added_simplex_coords.append(
#                         [
#                             simplex_coords, simplex_volume, simplex_centroid
#                         ]
#                     )
#                     self.volume += simplex_volume
#                     self.centroid += simplex_centroid * simplex_volume
#
#             self.perimeter_edges = list(
#                 tuple(points[i,:] for i in es) for fs in perimeter_edges for es in
#                 itertools.combinations(fs, r=2)
#             )
#             if len(edges) > 0:
#                 self.centroid /= self.volume
#
#                 perimeter_indices = set(p for e in perimeter_edges for p in e)
#                 self.perimeter_points = np.vstack(
#                     [points[p, :] for p in perimeter_indices]
#                 )
#                 # self.surface_area = compute_surface_area(points, perimeter_edges, self.simplices)
#
#         self.is_empty = False
#         if len(self.perimeter_points) == 0 or points.shape[0] != len(visited_points):
#             self.is_empty = True
#
#
#     @property
#     def max_edge_length(self):
#         if self.perimeter_edges:
#             edge_lengths = [
#                 np.linalg.norm(a-b,ord=2)
#                 for ps in self.perimeter_edges
#                 for a,b in itertools.combinations(ps, r=2)
#             ]
#             if edge_lengths:
#                 return max(edge_lengths)
#         return 0
#
#
#
#     def add_points(self, points: np.ndarray):
#         """
#         Adds a new point to the alpha shape and updates the perimeter points, volume,
#         surface area, and centroid accordingly.
#
#         Args:
#             point (np.ndarray): The new point to add.
#         """
#         # Add the new point to the existing points
#         new_points = np.vstack([self.perimeter_points, points])
#
#         # Recompute the alpha shape with the updated points
#         self.__init__(new_points, alpha=float(self.alpha))
#
#
#     @property
#     def vertices(self):
#         return self.perimeter_points
#
#     def contains_point(self, point: np.ndarray) -> bool:
#         if len(self.points) < len(point) + 1:
#             return False
#
#         # Iterate over each simplex defined by indices in self.simplices
#         for simplex in self.simplices:
#             # Get the vertices of the simplex from perimeter points
#             vertices = self.points[list(simplex)]
#
#             # Solve for barycentric coordinates
#             try:
#                 # Create an augmented matrix for the linear system (for homogeneous coordinates)
#                 matrix = np.vstack([vertices.T, np.ones(len(vertices))])
#                 point_h = np.append(point, 1)  # Homogeneous coordinate for the point
#                 barycentric_coords = np.linalg.solve(matrix, point_h)
#
#                 # Check if all barycentric coordinates are non-negative (point lies inside simplex)
#                 if np.all(barycentric_coords >= 0):
#                     return True
#             except np.linalg.LinAlgError:
#                 # In case vertices are degenerate and can't form a valid simplex, skip it
#                 continue
#
#         # If no simplex contains the point, return False
#         return False


class AlphaShape:
    """
    Batch α‑shape (concave hull) in arbitrary dimension.
    """

    # --------------------------------------------------------------------- #
    #  construction (unchanged from your paste, but wrapped in a method so  #
    #  subclasses can re‑use it)                                            #
    # --------------------------------------------------------------------- #
    def __init__(self,
                 points: np.ndarray,
                 alpha: float = 0.,
                 max_perimeter_length: float = np.inf,
                 connectivity: Literal["strict", "relaxed"] = "strict"
    ):
        self._dim = points.shape[1]
        if self._dim < 2:
            raise ValueError("dimension must be ≥ 2")

        self.alpha = float(alpha)
        if connectivity not in {"strict", "relaxed"}:
            raise ValueError("connectivity must be 'strict' or 'relaxed'")
        self.connectivity = connectivity

        self.max_perimeter_length = float(max_perimeter_length)
        self.points = np.asarray(points, dtype=float)

        self.simplices: Set[Tuple[int, ...]] = set()
        self.perimeter_edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self.perimeter_points: np.ndarray | None = None
        self.centroid = np.zeros(self._dim)
        self.volume = 0.0
        self.GCT = GraphClosureTracker(len(points))

        # build once
        self._build_batch()

    # ------------------------------------------------------------------ #
    #  public helpers (unchanged)                                        #
    # ------------------------------------------------------------------ #

    @property
    def vertices(self):
        return self.perimeter_points
    @property
    def max_edge_length(self) -> float:
        if not self.perimeter_edges:
            return 0.0
        return max(np.linalg.norm(a - b) for a, b in self.perimeter_edges)

    def contains_point(self, pt: np.ndarray) -> bool:
        if len(self.simplices) == 0:
            return False
        for s in self.simplices:
            verts = self.points[list(s)]
            try:
                A = np.vstack([verts.T, np.ones(len(verts))])
                bary = np.linalg.solve(A, np.append(pt, 1.0))
                if np.all(bary >= 0):
                    return True
            except np.linalg.LinAlgError:
                continue
        return False

    def add_points(self, new_pts: np.ndarray):
        """
        *Batch* version – simply rebuilds everything.
        Sub‑class overrides this for incremental behaviour.
        """
        pts = np.vstack([self.points, new_pts])
        self.__init__(pts, alpha=self.alpha,
                      max_perimeter_length=self.max_perimeter_length)


    # ---------- lazy accessor for boundary (d‑1)-faces ------------------ #
    def _get_boundary_faces(self) -> Set[Tuple[int, ...]]:
        """
        Return the set of boundary faces (index tuples of length d) and cache
        it in `self._boundary_faces` so the computation is done only once.

        For `iAlphaShape` this simply forwards the attribute it already keeps.
        For the batch version we reconstruct the set from `self.simplices`
        using the usual “flip once ↔ boundary” rule.
        """
        if hasattr(self, "_boundary_faces"):
            return self._boundary_faces

        dim = self._dim
        faces: Set[Tuple[int, ...]] = set()
        for s in self.simplices:
            for f in itertools.combinations(s, dim):
                f = tuple(sorted(f))
                if f in faces:
                    faces.remove(f)
                else:
                    faces.add(f)
        # cache
        self._boundary_faces = faces
        return faces


    def distance_to_surface(self,
                            point: np.ndarray,
                            tol: float = 1e-9) -> float:
        """
        Euclidean distance from `point` to the α‑shape surface.
        Returns 0 if the point lies inside or on the surface.

        Works for any ambient dimension d ≥ 2.
        """
        p = np.asarray(point, dtype=float)
        if p.shape[-1] != self._dim:
            raise ValueError("point dimensionality mismatch")

        # 1. inside / on‑surface test
        if self.contains_point(p):
            return 0.0

        # 2. gather boundary faces and vertices
        faces = self._get_boundary_faces()
        if not faces:
            # degenerate case (e.g. only 1–2 input points)
            # fall back to nearest perimeter vertex
            return np.min(np.linalg.norm(self.perimeter_points - p, axis=1))

        dists = []

        for f in faces:
            verts = self.points[list(f)]  # shape (d, d)
            base = verts[0]
            A = verts[1:] - base  # (d‑1, d)

            # orthogonal projection of p onto the face’s affine span
            # Solve A x = (p - base)  →  least‑squares because A is tall
            x_hat, *_ = np.linalg.lstsq(A.T, (p - base), rcond=None)
            proj = base + A.T @ x_hat

            # barycentric coordinates to test if proj is inside the simplex
            # coords = [1 - sum(x_hat), *x_hat]
            bary = np.concatenate(([1.0 - x_hat.sum()], x_hat))
            if np.all(bary >= -tol):  # inside (or on) the face
                dists.append(np.linalg.norm(p - proj))
            else:
                # outside → distance to nearest vertex of this face
                dists.extend(np.linalg.norm(verts - p, axis=1))

        return float(min(dists))


    # ------------------------------------------------------------------ #
    #  internal: one‑shot batch builder (exact logic you pasted,         #
    #  only lightly reorganised so subclasses can call it)               #
    # ------------------------------------------------------------------ #
    def _build_batch(self):
        dim, pts = self._dim, self.points
        n = len(pts)
        if n < dim + 1:
            self.perimeter_points = pts
            self.centroid = pts.mean(axis=0)
            return

        r_filter = np.inf if self.alpha <= 0 else 1.0 / self.alpha
        tri = Delaunay(pts, qhull_options="Qz")

        # ---------- 1.  main sweep ---------------------------------------
        simplices = []
        for s in tri.simplices:
            r = circumradius(pts[s])
            simplices.append((tuple(s), r))

        simplices.sort(key=lambda t: t[1])  # radius ascending
        kept = []
        uf = GraphClosureTracker(n)  # temp tracker

        for simp, r in simplices:
            root_set = {uf.find(v) for v in simp}
            keep = (r <= r_filter) or \
                   (self.connectivity == "relaxed" and len(root_set) > 1)
            if not keep:
                continue
            uf.add_fully_connected_subgraph(list(simp))
            kept.append(simp)

        # ---------- 2.  strict‑mode pruning ------------------------------
        if self.connectivity == "strict":
            comp_sizes = {root: len(nodes) for root, nodes in uf.components.items()}
            main_root = max(comp_sizes, key=comp_sizes.get)
            main_verts = uf.components[main_root]
            kept = [s for s in kept if set(s) <= main_verts]

        # ---------- 3.  rebuild perimeter from *kept* simplices ----------
        self.simplices = set(kept)
        self.GCT = GraphClosureTracker(n)  # final tracker
        edges, perim_idx = set(), set()
        self.volume = 0.0
        self.centroid = np.zeros(dim)

        for s in self.simplices:
            self.GCT.add_fully_connected_subgraph(list(s))
            vol = volume_of_simplex(pts[list(s)])
            self.centroid = (self.centroid * self.volume +
                             vol * pts[list(s)].mean(axis=0)) / (self.volume + vol)
            self.volume += vol

            for f in itertools.combinations(s, dim):  # (d-1)-faces
                f = tuple(sorted(f))
                if f in edges:
                    edges.remove(f)
                else:
                    edges.add(f)
                    perim_idx.update(f)

        # ---------- 4.  store perimeter ----------------------------------
        self.perimeter_points = pts[list(sorted(perim_idx))]
        self.perimeter_edges = [(pts[i], pts[j]) for f in edges
                                for i, j in itertools.combinations(f, 2)]

    @property
    def is_empty(self):
        return len(self.perimeter_points) == 0
