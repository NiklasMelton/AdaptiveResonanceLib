"""
Williamson, J. R. (1996).
Gaussian ARTMAP: A Neural Network for Fast Incremental Learning of Noisy Multidimensional Maps.
Neural Networks, 9, 881 – 897. doi:10.1016/0893-6080(95)00115-8.
"""

import numpy as np
from decimal import Decimal
from typing import Optional, Iterable, List
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.visualization import plot_gaussian_contours_fading


class GaussianART(BaseART):
    # implementation of GaussianART
    def __init__(self, rho: float, sigma_init: np.ndarray, alpha: float = 1e-10):
        """
        Parameters:
        - rho: vigilance parameter
        - sigma_init: initial estimate of the diagonal std
        - alpha: used to prevent division by zero errors

        """
        params = {
            "rho": rho,
            "sigma_init": sigma_init,
            "alpha": alpha
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
        assert "sigma_init" in params
        assert "alpha" in params
        assert 1.0 >= params["rho"] >= 0.
        assert params["alpha"] > 0.
        assert isinstance(params["rho"], float)
        assert isinstance(params["sigma_init"], np.ndarray)


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
        mean = w[:self.dim_]
        # sigma = w[self.dim_:2*self.dim]
        inv_sig = w[2*self.dim_:3*self.dim_]
        sqrt_det_sig = w[-2]
        n = w[-1]

        dist = mean-i
        exp_dist_sig_dist = np.exp(-0.5 * np.dot(dist, np.multiply(inv_sig, dist)))

        cache = {
            "exp_dist_sig_dist": exp_dist_sig_dist
        }
        # ignore the (2*pi)^d term as that is constant
        p_i_cj = exp_dist_sig_dist/(params["alpha"]+sqrt_det_sig)
        p_cj = n/np.sum(w_[-1] for w_ in self.W)

        activation = p_i_cj*p_cj

        return activation, cache


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
        if cache is None:
            raise ValueError("No cache provided")
        exp_dist_sig_dist = cache["exp_dist_sig_dist"]
        return exp_dist_sig_dist, cache



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
        mean = w[:self.dim_]
        sigma = w[self.dim_:2*self.dim_]
        n = w[-1]

        n_new = n+1
        mean_new = (1-(1/n_new))*mean + (1/n_new)*i
        sigma_new = np.sqrt((1-(1/n_new))*np.multiply(sigma, sigma) + (1/n_new)*((mean_new - i)**2))

        sigma2 = np.multiply(sigma_new, sigma_new)
        inv_sig = 1 / sigma2
        det_sig = np.sqrt(np.prod(sigma2))

        return np.concatenate([mean_new, sigma_new, inv_sig, [det_sig], [n_new]])


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
        sigma2 = np.multiply(params["sigma_init"], params["sigma_init"])
        inv_sig_init = 1 / sigma2
        det_sig_init = np.sqrt(np.prod(sigma2))
        return np.concatenate([i, params["sigma_init"], inv_sig_init, [det_sig_init], [1.]])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        return [w[:self.dim_] for w in self.W]


    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        for w, col in zip(self.W, colors):
            mean = w[:self.dim_]
            sigma = w[self.dim_:-1]
            plot_gaussian_contours_fading(ax, mean, sigma, col, linewidth=linewidth)
