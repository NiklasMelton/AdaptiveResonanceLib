import numpy as np
from typing import Optional, Iterable, Tuple
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import normalize, compliment_code, l1norm, fuzzy_and



TreeType = Tuple[Iterable, np.ndarray]

class GramART(BaseART):

    def __init__(self, rho: float):
        params ={
            "rho": rho
        }
        super().__init__(params)

    @staticmethod
    def prepare_data(x: TreeType) -> TreeType:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            normalized and compliment coded data
        """
        return x

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert isinstance(params["rho"], float)

    def category_choice(self, i: TreeType, w: dict, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        tree_intersection = sum([w[s]["value"] for s in i])
        return tree_intersection/len(w), {"tree_intersection": tree_intersection}

    def match_criterion(self, i: TreeType, w: dict, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
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
        return cache["tree_intersection"]/len(i[0]), cache

    def match_criterion_bin(self, i: TreeType, w: dict, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        """
        get the binary match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion binary, cache used for later processing

        """
        M, cache = self.match_criterion(i, w, params)
        return M >= params["rho"], cache

    def update(self, i: TreeType, w: dict, params: dict, cache: Optional[dict] = None) -> np.ndarray:
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


