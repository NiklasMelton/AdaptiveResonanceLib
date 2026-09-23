"""Gaussian ART :cite:`williamson1996gaussian`."""
# Williamson, J. R. (1996).
# Gaussian ARTMAP: A Neural Network for Fast Incremental Learning of Noisy
# Multidimensional Maps.
# Neural Networks, 9, 881 – 897. doi:10.1016/0893-6080(95)00115-8.

import numpy as np
from typing import Optional, Iterable, List, Tuple, Dict
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.visualization import plot_gaussian_contours_fading


class GaussianART(BaseART):
    """Gaussian ART for Clustering.

    This module implements Gaussian ART as first published in:
    :cite:`williamson1996gaussian`.

    .. # Williamson, J. R. (1996).
    .. # Gaussian ARTMAP: A Neural Network for Fast Incremental Learning of Noisy
    .. # Multidimensional Maps.
    .. # Neural Networks, 9, 881 – 897. doi:10.1016/0893-6080(95)00115-8.

    Guassian ART clusters data in Gaussian Distributions (Hyper-ellipsoids) and is
    similar to :class:`~artlib.elementary.BayesianART.BayesianART` but differs in that
    the hyper-ellipsoid always have their principal axes square to the coordinate
    frame. It is also faster than :class:`~artlib.elementary.BayesianART.BayesianART`.

    """

    def __init__(self, rho: float, sigma_init: np.ndarray, alpha: float = 1e-10):
        """Initialize the Gaussian ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        sigma_init : np.ndarray
            Initial estimate of the diagonal standard deviations.
        alpha : float, optional
            Small parameter to prevent division by zero errors, by default 1e-10.

        """
        params = {"rho": rho, "sigma_init": sigma_init, "alpha": alpha}
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert "sigma_init" in params
        assert "alpha" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert params["alpha"] > 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["sigma_init"], np.ndarray)

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
        mean = w[: self.dim_]
        # sigma = w[self.dim_:2*self.dim]
        inv_sig = w[2 * self.dim_ : 3 * self.dim_]
        sqrt_det_sig = w[-2]
        n = w[-1]

        dist = mean - i
        exp_dist_sig_dist = np.exp(-0.5 * np.dot(dist, np.multiply(inv_sig, dist)))

        cache = {"exp_dist_sig_dist": exp_dist_sig_dist}
        # ignore the (2*pi)^d term as that is constant
        p_i_cj = exp_dist_sig_dist / (params["alpha"] + sqrt_det_sig)
        total_samples = sum(w_[-1] for w_ in self.W)
        p_cj = n / total_samples

        activation = p_i_cj * p_cj

        return activation, cache

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
        if cache is None:
            raise ValueError("No cache provided")
        exp_dist_sig_dist = cache["exp_dist_sig_dist"]
        return exp_dist_sig_dist, cache

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
        mean = w[: self.dim_]
        sigma = w[self.dim_ : 2 * self.dim_]
        n = w[-1]

        n_new = n + 1
        delta = i - mean
        mean_new = mean + delta / n_new
        # n * sigma**2 = sample scatter + one initial variance contribution.
        sigma2 = (n * np.square(sigma) + delta * (i - mean_new)) / n_new
        sigma_new = np.sqrt(sigma2)

        inv_sig = 1 / sigma2
        det_sig = np.sqrt(np.prod(sigma2))

        return np.concatenate([mean_new, sigma_new, inv_sig, [det_sig], [n_new]])

    def merge(self, target_idx: int, source_idx: int) -> int:
        """Pool sample moments, retaining one initial variance contribution."""
        self._validate_merge_indices(target_idx, source_idx)
        target = self.W[target_idx]
        source = self.W[source_idx]
        n1, n2 = target[-1], source[-1]
        n = n1 + n2
        mean1, mean2 = target[: self.dim_], source[: self.dim_]
        delta = mean2 - mean1
        mean = mean1 + (n2 / n) * delta
        variance = (
            n1 * np.square(target[self.dim_ : 2 * self.dim_])
            + n2 * np.square(source[self.dim_ : 2 * self.dim_])
            - np.square(self.params["sigma_init"])
            + (n1 * n2 / n) * np.square(delta)
        ) / n
        sigma = np.sqrt(variance)
        new_w = np.concatenate(
            [mean, sigma, 1 / variance, [np.sqrt(np.prod(variance))], [n]]
        )
        return self._apply_merge(target_idx, source_idx, new_w)

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
        sigma2 = np.multiply(params["sigma_init"], params["sigma_init"])
        inv_sig_init = 1 / sigma2
        det_sig_init = np.sqrt(np.prod(sigma2))
        return np.concatenate(
            [i, params["sigma_init"], inv_sig_init, [det_sig_init], [1.0]]
        )

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return [w[: self.dim_] for w in self.W]

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
        for w, col in zip(self.W, colors):
            mean = w[: self.dim_]
            sigma = w[self.dim_ : -1]
            plot_gaussian_contours_fading(ax, mean, sigma, col, linewidth=linewidth)
