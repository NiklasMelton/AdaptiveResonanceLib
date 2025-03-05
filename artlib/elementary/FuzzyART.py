"""Fuzzy ART :cite:`carpenter1991fuzzy`."""
# Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
# Fuzzy ART: Fast stable learning and categorization of analog patterns by an
# adaptive resonance system.
# Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.

import numpy as np
from numba import njit
from typing import Optional, Iterable, List, Tuple, Dict
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import (
    normalize,
    complement_code,
    l1norm,
    fuzzy_and,
    de_complement_code,
)


def get_bounding_box(
    w: np.ndarray, n: Optional[int] = None
) -> tuple[list[int], list[int]]:
    """Extract the bounding boxes from a FuzzyART weight.

    Parameters
    ----------
    w : np.ndarray
        A fuzzy ART weight.
    n : int, optional
        Dimensions of the bounding box.

    Returns
    -------
    tuple
        A tuple containing the reference point and lengths of each edge.

    """
    n_ = int(len(w) / 2)
    if n is None:
        n = n_
    assert n <= n_, f"Requested bbox dimension {n_} exceed data dimension {n_}"

    ref_point = []
    widths = []

    for i in range(n):
        a_min = w[i]
        a_max = 1 - w[i + n]

        ref_point.append(a_min)
        widths.append(a_max - a_min)

    return ref_point, widths


@njit
def _category_choice_numba(i: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Compute the category choice (activation) using Numba optimization.

    Parameters
    ----------
    i : np.ndarray
        Data sample.
    w : np.ndarray
        Cluster weight or information.
    alpha : float
        Choice parameter for the algorithm.

    Returns
    -------
    float
        Computed cluster activation value.

    """
    return l1norm(fuzzy_and(i, w)) / (alpha + l1norm(w))


@njit
def _match_criterion_numba(i: np.ndarray, w: np.ndarray, dim_original: float) -> float:
    """Compute the match criterion using Numba optimization.

    Parameters
    ----------
    i : np.ndarray
        Data sample.
    w : np.ndarray
        Cluster weight or information.
    dim_original : float
        Original number of features before complement coding.

    Returns
    -------
    float
        Computed match criterion.

    """
    return l1norm(fuzzy_and(i, w)) / dim_original


@njit
def _update_numba(i: np.ndarray, w: np.ndarray, b: Optional[float]) -> np.ndarray:
    """Compute the updated cluster weight using Numba optimization.

    Parameters
    ----------
    i : np.ndarray
        Data sample.
    w : np.ndarray
        Cluster weight or information.
    b : float, optional
        Learning rate parameter (beta). If None, only the fuzzy AND operation is
        applied.

    Returns
    -------
    np.ndarray
        Updated cluster weight.

    """
    if b is None:
        return fuzzy_and(i, w)
    return b * fuzzy_and(i, w) + (1 - b) * w


class FuzzyART(BaseART):
    """Fuzzy ART for Clustering.

    This module implements Fuzzy ART as first published in:
    :cite:`carpenter1991fuzzy`.

    .. # Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
    .. # Fuzzy ART: Fast stable learning and categorization of analog patterns by an
    .. # adaptive resonance system.
    .. # Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.

    Fuzzy ART is a hyper-box based clustering method that is exceptionally fast and
    explainable.

    """

    def __init__(self, rho: float, alpha: float, beta: float):
        """Initialize the Fuzzy ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.
        beta : float
            Learning rate.

        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }
        super().__init__(params)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        Returns
        -------
        np.ndarray
            Normalized and complement coded data.

        """
        normalized, self.d_max_, self.d_min_ = normalize(X, self.d_max_, self.d_min_)
        cc_data = complement_code(normalized)
        return cc_data

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to its state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        out = de_complement_code(X)
        return super(FuzzyART, self).restore_data(out)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert params["alpha"] >= 0.0
        assert 1.0 >= params["beta"] > 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)

    def check_dimensions(self, X: np.ndarray):
        """Check that the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            self.dim_original = int(self.dim_ // 2)
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        """
        assert X.shape[1] % 2 == 0, "Data has not been complement coded"
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"
        assert np.all(
            abs(np.sum(X, axis=1) - float(X.shape[1] / 2)) <= 0.01
        ), "Data has not been complement coded"
        self.check_dimensions(X)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        """Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
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
        return _category_choice_numba(i, w, params["alpha"]), None

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion of the cluster.

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
        float
            Cluster match criterion.
        dict
            Cache used for later processing.

        """
        return _match_criterion_numba(i, w, self.dim_original), cache

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Get the updated cluster weight.

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
        np.ndarray
            Updated cluster weight.

        """
        return _update_numba(i, w, params.get("beta", None))

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        np.ndarray
            New cluster weight.

        """
        return i

    def get_bounding_boxes(
        self, n: Optional[int] = None
    ) -> List[tuple[list[int], list[int]]]:
        """Get the bounding boxes for each cluster.

        Parameters
        ----------
        n : int, optional
            Dimensions of the bounding box.

        Returns
        -------
        list
            List of bounding boxes.

        """
        return list(map(lambda w: get_bounding_box(w, n=n), self.W))

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return [self.restore_data(w.reshape((1, -1))).reshape((-1,)) for w in self.W]

    def shrink_clusters(self, shrink_ratio: float = 0.1):
        """Shrink the clusters by adjusting the bounding box.

        Parameters
        ----------
        shrink_ratio : float, optional
            The ratio by which to shrink the clusters, by default 0.1.

        Returns
        -------
        FuzzyART
            Self after shrinking the clusters.

        """
        new_W = []
        dim = len(self.W[0]) // 2
        for w in self.W:
            new_w = np.copy(w)
            widths = (1 - w[dim:]) - w[:dim]
            new_w[:dim] += widths * shrink_ratio
            new_w[dim:] += widths * shrink_ratio
            new_W.append(new_w)
        self.W = new_W
        return self

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """Visualize the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        from matplotlib.patches import Rectangle

        bboxes = self.get_bounding_boxes(n=2)
        for bbox, col in zip(bboxes, colors):
            rect = Rectangle(
                (bbox[0][0], bbox[0][1]),
                bbox[1][0],
                bbox[1][1],
                linewidth=linewidth,
                edgecolor=col,
                facecolor="none",
            )
            ax.add_patch(rect)
