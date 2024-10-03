"""
Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system.
Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.
"""
import numpy as np
from typing import Optional, Iterable, List
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import normalize, compliment_code, l1norm, fuzzy_and, de_compliment_code


def get_bounding_box(w: np.ndarray, n: Optional[int] = None) -> tuple[list[int], list[int]]:
    """
    extract the bounding boxes from a FuzzyART weight

    Parameters:
    - w: a fuzzy ART weight
    - n: dimensions of the bounding box

    Returns:
        reference_point, lengths of each edge
    """
    n_ = int(len(w) / 2)
    if n is None:
        n = n_
    assert n <= n_, f"Requested bbox dimension {n_} exceed data dimension {n_}"

    ref_point = []
    widths = []

    for i in range(n):
        a_min = w[i]
        a_max = 1-w[i+n]

        ref_point.append(a_min)
        widths.append(a_max-a_min)

    return ref_point, widths


class FuzzyART(BaseART):
    # implementation of FuzzyART
    def __init__(self, rho: float, alpha: float, beta: float):
        """
        Parameters:
        - rho: vigilance parameter
        - alpha: choice parameter
        - beta: learning rate

        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }
        super().__init__(params)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            normalized and compliment coded data
        """
        normalized, self.d_max_, self.d_min_ = normalize(X, self.d_max_, self.d_min_)
        cc_data = compliment_code(normalized)
        return cc_data

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            restored data
        """
        out = de_compliment_code(X)
        return super(FuzzyART, self).restore_data(out)

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1.0 >= params["rho"] >= 0.
        assert params["alpha"] >= 0.
        assert 1.0 >= params["beta"] > 0.
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            self.dim_original = int(self.dim_//2)
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        assert X.shape[1] % 2 == 0, "Data has not been compliment coded"
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"
        assert np.all(abs(np.sum(X, axis=1) - float(X.shape[1]/2)) <= 0.01), "Data has not been compliment coded"
        self.check_dimensions(X)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        return l1norm(fuzzy_and(i, w)) / (params["alpha"] + l1norm(w)), None

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
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
        return l1norm(fuzzy_and(i, w)) / self.dim_original, cache


    def update(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> np.ndarray:
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
        b = params["beta"]
        if b is None:
            return fuzzy_and(i, w)
        return b * fuzzy_and(i, w) + (1 - b) * w

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        return i

    def get_bounding_boxes(self, n: Optional[int] = None):
        return list(map(lambda w: get_bounding_box(w, n=n), self.W))

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        centers = []
        for w in self.W:
            ref_points, widths = get_bounding_box(w,None)
            centers.append(np.array(ref_points)+0.5*np.array(widths))
        return centers

    def shrink_clusters(self, shrink_ratio: float = 0.1):
        new_W = []
        dim = len(self.W[0])//2
        for w in self.W:
            new_w = np.copy(w)
            widths = (1-w[dim:]) - w[:dim]
            new_w[:dim] += widths*shrink_ratio
            new_w[dim:] += widths*shrink_ratio
            new_W.append(new_w)
        self.W = new_W
        return self


    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

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
                facecolor='none'
            )
            ax.add_patch(rect)
