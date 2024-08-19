import numpy as np
from typing import Optional
from artlib import BaseART
from artlib.common.utils import normalize, compliment_code, l1norm, fuzzy_and

def rectify(x: float):
    return np.maximum(0, x)

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class DistributedART(BaseART):

    # implementation of DistributedART
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

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        raise NotImplementedError

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        raise NotImplementedError

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
        t = l1norm(rectify(fuzzy_and(i, 1-w[0,:]) - w[2,:])) / (params["alpha"]-l1norm(rectify(w[0,:]-w[3,:])))
        return t, None

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
        return

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
