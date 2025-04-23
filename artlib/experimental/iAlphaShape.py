import itertools
import numpy as np
from typing import Tuple, Set
import heapq
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import cdist
from artlib.experimental.AlphaShape import AlphaShape

def max_volume_seed(points: np.ndarray) -> np.ndarray:
    """
    Return indices of a (d+1)-simplex with near‑maximal volume using a
    greedy farthest‑point heuristic.
    """
    pts = points
    d   = pts.shape[1]
    N   = len(pts)

    # 1. start with the global centroid
    centroid = pts.mean(axis=0, keepdims=True)
    first    = np.argmax(np.linalg.norm(pts - centroid, axis=1))
    seed     = [first]

    # 2. farthest‑point iterations
    while len(seed) < d + 1:
        dist2 = np.min(cdist(pts, pts[seed])**2, axis=1)
        seed.append(int(np.argmax(dist2)))

    return np.array(seed, dtype=int)
def presort_for_alpha(points: np.ndarray,
                      alpha: float,
                      seed_layer: int = 1) -> np.ndarray:
    """
    Return a reordered copy of `points` that is well‑suited for incremental
    α‑shape construction.

    Parameters
    ----------
    points      : (N,d) array
    alpha       : α used for the shape (needed for the distance window)
    seed_layer  : how many hull layers to peel off before incremental insert

    Returns
    -------
    points_sorted : (N,d) array
    """
    pts = points.copy()
    d   = pts.shape[1]
    r_filter = np.inf if alpha <= 0 else 1.0 / alpha

    seed_idx = []
    remaining = np.arange(len(pts))

    # -------- Stage 1 : convex‑hull peeling --------------------------
    for _ in range(seed_layer):
        if len(remaining) < d + 1:
            break
        hull = ConvexHull(pts[remaining])
        layer = remaining[hull.vertices]
        seed_idx.extend(layer.tolist())
        remaining = np.setdiff1d(remaining, layer, assume_unique=True)

    if len(seed_idx) < d + 1:
        # fall back to random if hull failed (degenerate cloud)
        extra = remaining[: (d + 1 - len(seed_idx))]
        seed_idx.extend(extra.tolist())
        remaining = np.setdiff1d(remaining, extra, assume_unique=True)

    # -------- Stage 2 : sort by distance to hull ---------------------
    if len(remaining):
        hull_pts = pts[seed_idx]
        tree = KDTree(hull_pts)
        dists, _ = tree.query(pts[remaining])
        order = np.argsort(dists)          # closest‑to‑hull first
        remaining = remaining[order]

    # -------- concatenate & return -----------------------------------
    return pts[np.concatenate([seed_idx, remaining])]


def face_is_visible(all_points: np.ndarray,
                    face_idx: Tuple[int, ...],
                    new_pt: np.ndarray,
                    interior_ref: np.ndarray,
                    eps: float = 1e-12) -> bool:
    """
    Return *True* iff `new_pt` lies on the opposite side of the hyper‑plane
    that supports the boundary face `face_idx`, compared to an interior
    reference point.

    Parameters
    ----------
    all_points   : (N,d) array holding every vertex so far.
    face_idx     : tuple of length (d) – indices of the (d‑1)-simplex that
                   forms the boundary face.
    new_pt       : (d,) array – the point being inserted.
    interior_ref : (d,) array – any point guaranteed to be inside the current
                   α‑shape (the running centroid is perfect).
    eps          : numerical tolerance.

    Works for *any* ambient dimension d ≥ 2.
    """
    verts = all_points[list(face_idx)]               # shape (d, d)
    base  = verts[0]
    A     = verts[1:] - base                         # (d‑1, d) matrix

    # Find a unit normal n to the face: null‑space of A
    # (small d ⇒ SVD is cheap)
    _, _, vh = np.linalg.svd(A, full_matrices=True)
    n = vh[-1]                                       # last right‑singular vec
    n /= np.linalg.norm(n)

    # signed distances of reference and new point to the hyper‑plane
    c          = np.dot(n, base)                     # offset
    sign_int   = np.dot(n, interior_ref) - c
    sign_new   = np.dot(n, new_pt)       - c

    # If either is numerically on the plane, treat as 0
    if abs(sign_int) < eps or abs(sign_new) < eps:
        return False  # degenerate – treat as not visible

    # Visible iff they are on opposite sides
    return sign_int * sign_new < 0


# ------------------------------------------------------------------
#  For a point p, return the *minimal* circumradius it can achieve
#  with any visible boundary face of the current α‑shape.
# ------------------------------------------------------------------
def _best_possible_radius(all_points: np.ndarray,
                          boundary_faces: Set[Tuple[int, ...]],
                          p: np.ndarray,
                          centroid: np.ndarray,
                          r_filter: float) -> float:
    best_r = np.inf
    for face in boundary_faces:
        # visibility test
        if not face_is_visible(all_points, face, p, centroid):
            continue
        # build candidate simplex = face ∪ {p}
        cand_pts = np.vstack([all_points[list(face)], p])
        try:
            r = circumradius(cand_pts)
        except np.linalg.LinAlgError:
            continue
        if r < best_r:
            best_r = r
            # early exit: once r ≤ r_filter we know the point WILL pass
            if best_r <= r_filter:
                break
    return best_r


# ------------------------------------------------------------------
#  helper: signed distance of a point to the hyper‑plane of a face
# ------------------------------------------------------------------
def _signed_plane_dist(pts: np.ndarray,
                       face_idx: Tuple[int, ...],
                       q: np.ndarray) -> float:
    """
    Signed distance of point q to the hyper‑plane spanned by face_idx.
    pts        : (N,d)  all vertices
    face_idx   : tuple  indices of the (d) vertices of the face
    q          : (d,)   query point
    Returns positive if q is on the same side as the face normal.
    """
    verts = pts[list(face_idx)]
    base  = verts[0]
    A     = verts[1:] - base          # (d-1, d)
    # normal = last right‑singular vector
    _, _, vh = np.linalg.svd(A, full_matrices=True)
    n = vh[-1]
    n /= np.linalg.norm(n)
    return np.dot(n, q - base)

# ---------------------------------------------------------------
# helper: outward normal + visibility  (works for any d ≥ 2)
# ---------------------------------------------------------------
def face_is_visible_strict(all_pts: np.ndarray,
                           face: Tuple[int, ...],
                           interior_v: int,
                           new_pt: np.ndarray,
                           eps: float = 1e-12) -> bool:
    """
    Return True iff `new_pt` lies strictly outside the oriented face.

    face       : tuple of d indices (the (d‑1)-simplex)
    interior_v : index of a vertex known to be inside the polytope
                 and belonging to some simplex that owns `face`.
    """
    verts = all_pts[list(face)]
    base  = verts[0]
    # build (d-1) edge vectors
    E = verts[1:] - base                       # shape (d-1, d)
    # Gram‑Schmidt cross‑product via SVD for any dimension
    _, _, vh = np.linalg.svd(E, full_matrices=True)
    n = vh[-1]                                 # unit normal (up to sign)

    # orient outward: n·(interior - base) must be negative
    if np.dot(n, all_pts[interior_v] - base) > 0:
        n = -n

    return np.dot(n, new_pt - base) > eps      # strictly outside



# --------------------------------------------------------------------------- #
#                               iAlphaShape                                   #
# --------------------------------------------------------------------------- #
class iAlphaShape(AlphaShape):
    """
    Incremental α‑shape (expansion‑only).  Inherits every public helper from
    `AlphaShape`; overrides only the constructor and `add_points()`.
    """

    # ------------------------ constructor --------------------------------- #
    def __init__(self,
                 points: np.ndarray,
                 alpha: float = 0.,
                 max_perimeter_length: float = np.inf):

        # let the parent do a *batch* build first
        super().__init__(points, alpha, max_perimeter_length)

        # extra state needed for fast incremental updates
        self._r_filter = np.inf if alpha <= 0 else 1.0 / alpha
        self._boundary_faces: Set[Tuple[int, ...]] = set()
        self._edges_idx: Set[Tuple[int, int]] = set()

        # translate the already‑built perimeter edges into boundary faces
        self._init_boundary_faces_from_edges()

    # -------------------- incremental insertion -------------------------- #

    # ------------------ private: insert ONE point ------------------------ #
    def _insert_single(self, p: np.ndarray):
        if self.contains_point(p):
            return

        idx_new = len(self.points)
        self.points = np.vstack([self.points, p])

        # -------- map each boundary face to *one* interior vertex ----
        face2inside: dict[Tuple[int, ...], int] = {}
        for s in self.simplices:
            for f in itertools.combinations(s, self._dim):
                f = tuple(sorted(f))
                if f in self._boundary_faces:
                    interior_v = next(v for v in s if v not in f)
                    face2inside[f] = interior_v

        # -------- try to connect new point to visible faces ----------
        new_simplices = []
        for face, interior_v in face2inside.items():
            if not face_is_visible_strict(self.points, face,
                                          interior_v, p):
                continue

            cand_idx = tuple(sorted(face + (idx_new,)))
            cand_pts = self.points[list(cand_idx)]
            try:
                r = circumradius(cand_pts)
            except np.linalg.LinAlgError:
                continue
            if r > self._r_filter:
                continue

            new_simplices.append(cand_idx)

            # geometry updates
            vol = volume_of_simplex(cand_pts)
            self.centroid = (self.centroid * self.volume +
                             vol * cand_pts.mean(axis=0)) / (self.volume + vol)
            self.volume += vol
            self.GCT.add_fully_connected_subgraph(cand_idx)

        self.simplices.update(new_simplices)

        # -------- rebuild boundary faces/edges -----------------------
        self._boundary_faces.clear()
        for s in self.simplices:
            for f in itertools.combinations(s, self._dim):
                f = tuple(sorted(f))
                if f in self._boundary_faces:
                    self._boundary_faces.remove(f)
                else:
                    self._boundary_faces.add(f)

        self._edges_idx = {tuple(sorted(e))
                           for f in self._boundary_faces
                           for e in itertools.combinations(f, 2)}

        self.perimeter_edges = [(self.points[i], self.points[j])
                                for i, j in self._edges_idx]
        perim_idx = {v for f in self._boundary_faces for v in f}
        self.perimeter_points = self.points[list(sorted(perim_idx))]

    # ------------------- public: add_points ---------------------------- #
    # ------------------------------------------------------------------
    #  Predictive ordering by *minimal circumradius* (Option 4)
    # ------------------------------------------------------------------
    def add_points(self, new_pts: np.ndarray):
        """
        Insert one or more points.  They are processed in ascending order of
        their *best achievable circumradius* with any visible boundary face,
        i.e. the point most likely to satisfy the α‑criterion is inserted first.
        """
        arr = np.asarray(new_pts, float)
        if arr.ndim == 1:
            arr = arr[None, :]

        n = len(arr)
        if n == 0:
            return
        if n == 1:
            self._insert_single(arr[0])
            return

        # ---------- build predictive min‑heap -------------------------
        heap = []
        for i, pt in enumerate(arr):
            r_best = _best_possible_radius(self.points,
                                           self._boundary_faces,
                                           pt,
                                           self.centroid,
                                           self._r_filter)
            heapq.heappush(heap, (r_best, i))

        inserted = [False] * n

        # ---------- greedy loop --------------------------------------
        while heap:
            r_pred, i = heapq.heappop(heap)
            if inserted[i]:
                continue  # stale entry

            # point may have become interior; skip if so
            if self.contains_point(arr[i]):
                inserted[i] = True
                continue

            # insert the chosen point
            self._insert_single(arr[i])
            inserted[i] = True

            # re‑evaluate predictions ONLY for points whose cached
            # radius was ≤ current filter (they were “hopeful” before)
            new_heap = []
            while heap:
                r_old, j = heapq.heappop(heap)
                if inserted[j]:
                    continue
                if r_old <= self._r_filter * 1.05:  # heuristic window
                    r_new = _best_possible_radius(self.points,
                                                  self._boundary_faces,
                                                  arr[j],
                                                  self.centroid,
                                                  self._r_filter)
                    heapq.heappush(new_heap, (r_new, j))
                else:
                    heapq.heappush(new_heap, (r_old, j))
            heap = new_heap

    # --------------------- helpers --------------------------------------- #
    def _init_boundary_faces_from_edges(self):
        """
        Parent class only stores perimeter edges.  In 2‑D an edge *is* a face,
        but in higher‑D we need to recover (d‑1)‑faces.  A cheap way is to
        regenerate the boundary faces from `self.simplices`.
        """
        dim = self._dim
        for s in self.simplices:
            for face in itertools.combinations(s, dim):
                face = tuple(sorted(face))
                if face in self._boundary_faces:
                    self._boundary_faces.remove(face)
                else:
                    self._boundary_faces.add(face)

        self._edges_idx = {tuple(sorted(e))
                           for f in self._boundary_faces
                           for e in itertools.combinations(f, 2)}

