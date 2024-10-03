"""
Anagnostopoulos, G. C., & Georgiopulos, M. (2000).
Hypersphere ART and ARTMAP for unsupervised and supervised, incremental learning.
In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
(pp. 59–64). volume 6. doi:10.1109/IJCNN.2000.859373.
"""
import numpy as np
from typing import Optional, Iterable, List
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import l2norm2

class HypersphereART(BaseART):
    # implementation of HypersphereART
    def __init__(self, rho: float, alpha: float, beta: float, r_hat: float):
        """
        Parameters:
        - rho: vigilance parameter
        - alpha: choice parameter
        - beta: learning rate
        - r_hat: maximum possible category radius

        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
            "r_hat": r_hat,
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
        assert "alpha" in params
        assert "beta" in params
        assert "r_hat" in params
        assert 1.0 >= params["rho"] >= 0.
        assert params["alpha"] >= 0.
        assert 1.0 >= params["beta"] >= 0.
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["r_hat"], float)

    @staticmethod
    def category_distance(i: np.ndarray, centroid: np.ndarray, radius: float, params) -> float:
        return np.sqrt(l2norm2(i-centroid))


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
        centroid = w[:-1]
        radius = w[-1]

        i_radius = self.category_distance(i, centroid, radius, params)
        max_radius = max(radius, i_radius)

        cache = {
            "max_radius": max_radius,
            "i_radius": i_radius,
        }
        return (params["r_hat"] - max_radius)/(params["r_hat"] - radius + params["alpha"]), cache


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
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]

        return 1 - (max(radius, max_radius)/params["r_hat"]), cache



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
        centroid = w[:-1]
        radius = w[-1]
        if cache is None:
            raise ValueError("No cache provided")
        max_radius = cache["max_radius"]
        i_radius = cache["i_radius"]

        radius_new = radius + (params["beta"]/2)*(max_radius-radius)
        centroid_new = centroid + (params["beta"]/2)*(i-centroid)*(1-(min(radius, i_radius)/i_radius))

        return np.concatenate([centroid_new, [radius_new]])


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
        return np.concatenate([i, [0.]])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        return [w[:-1] for w in self.W]


    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        from matplotlib.patches import Circle

        for w, col in zip(self.W, colors):
            centroid = (w[0], w[1])
            radius = w[-1]
            circ = Circle(
                centroid,
                radius,
                linewidth=linewidth,
                edgecolor=col,
                facecolor='none'
            )
            ax.add_patch(circ)






