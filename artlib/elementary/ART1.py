"""
Carpenter, G. A., & Grossberg, S. (1987a).
A massively parallel architecture for a self-organizing neural pattern recognition machine.
Computer Vision, Graphics, and Image Processing, 37, 54 – 115. doi:10. 1016/S0734-189X(87)80014-2.
"""

import numpy as np
from typing import Optional, List
from artlib.common.BaseART import BaseART
from artlib.common.utils import l1norm


class ART1(BaseART):
    # implementation of ART 1
    def __init__(self, rho: float, beta: float, L: float):
        """
        Parameters:
        - rho: vigilance parameter
        - beta: learning rate
        - L: uncommitted node bias

        """
        params = {
            "rho": rho,
            "beta": beta,
            "L": L
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
        assert "beta" in params
        assert "L" in params
        assert 1. >= params["rho"] >= 0.
        assert 1. >= params["beta"] >= 0.
        assert params["L"] >= 1.
        assert isinstance(params["rho"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["L"], float)

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        assert np.array_equal(X, X.astype(bool)), "ART1 only supports binary data"
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
        w_bu = w[:self.dim_]
        return float(np.dot(i, w_bu)), None

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
        w_td = w[self.dim_:]
        return l1norm(np.logical_and(i, w_td)) / l1norm(i), cache


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
        w_td = w[self.dim_:]

        w_td_new = np.logical_and(i, w_td)
        w_bu_new = (params["L"] / (params["L"] - 1 + l1norm(w_td_new)))*w_td_new
        return np.concatenate([w_bu_new, w_td_new])


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
        w_td_new = i
        w_bu_new = (params["L"] / (params["L"] - 1 + self.dim_))*w_td_new
        return np.concatenate([w_bu_new, w_td_new])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        return [w[self.dim_:] for w in self.W]
