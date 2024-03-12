"""
Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system.
Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.
"""
import numpy as np
from typing import Optional, Iterable
from matplotlib.axes import Axes
from common.BaseART import BaseART
from common.utils import normalize, compliment_code, l1norm, fuzzy_and

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    cc_data = compliment_code(normalized)
    return cc_data


def get_bounding_box(w: np.ndarray, n: Optional[int] = None) -> tuple[list[int], list[int]]:
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

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "beta" in params
        assert 1.0 >= params["rho"] >= 0.
        assert params["alpha"] >= 0.
        assert 1.0 >= params["beta"] > 0.

    def check_dimensions(self, X: np.ndarray):
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            self.dim_original = int(self.dim_//2)
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        assert X.shape[1] % 2 == 0, "Data has not been compliment coded"
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"
        assert np.all(abs(np.sum(X, axis=1) - float(X.shape[1]/2)) <= 0.01), "Data has not been compliment coded"
        self.check_dimensions(X)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        return l1norm(fuzzy_and(i, w)) / (params["alpha"] + l1norm(w)), None

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        return l1norm(fuzzy_and(i, w)) / self.dim_original, cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        M, cache = self.match_criterion(i, w, params)
        return M >= params["rho"], cache

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        b = params["beta"]
        if b is None:
            return fuzzy_and(i, w)
        return b * fuzzy_and(i, w) + (1 - b) * w

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return i

    def get_bounding_boxes(self, n: Optional[int] = None):
        return list(map(lambda w: get_bounding_box(w, n=n), self.W))

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
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
