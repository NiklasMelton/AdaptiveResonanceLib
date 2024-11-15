"""Bayesian ART :cite:`vigdor2007bayesian`."""
# Vigdor, B., & Lerner, B. (2007).
# The Bayesian ARTMAP.
# IEEE Transactions on Neural
# Networks, 18, 1628–1644. doi:10.1109/TNN.2007.900234.

import numpy as np
from typing import Optional, Iterable, List, Callable, Literal, Tuple, Union, Dict
import operator
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.visualization import plot_gaussian_contours_covariance


class BayesianART(BaseART):
    """Bayesian ART for Clustering.

    This module implements Bayesian ART as first published in:
    :cite:`vigdor2007bayesian`.

     .. # Vigdor, B., & Lerner, B. (2007).
    .. # The Bayesian ARTMAP.
    .. # IEEE Transactions on Neural
    .. # Networks, 18, 1628–1644. doi:10.1109/TNN.2007.900234.

    Bayesian ART clusters data in Bayesian Distributions (Hyper-ellipsoids) and is
    similar to :class:`~artlib.elementary.GaussianART.GaussianART` but differs in that
    it allows arbitrary rotation of the hyper-ellipsoid.

    """

    pi2 = np.pi * 2

    def __init__(self, rho: float, cov_init: np.ndarray):
        """Initialize the Bayesian ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter in the range [0, 1].
        cov_init : np.ndarray
            Initial estimate of the covariance matrix for each cluster.

        """
        params = {
            "rho": rho,
            "cov_init": cov_init,
        }
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
        assert "cov_init" in params
        assert params["rho"] > 0
        assert isinstance(params["rho"], float)
        assert isinstance(params["cov_init"], np.ndarray)

    def check_dimensions(self, X: np.ndarray):
        """Check that the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            assert self.params["cov_init"].shape[0] == self.dim_
            assert self.params["cov_init"].shape[1] == self.dim_
        else:
            assert X.shape[1] == self.dim_

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
        cov = w[self.dim_ : -1].reshape((self.dim_, self.dim_))
        n = w[-1]
        dist = mean - i

        exp_dist_cov_dist = np.exp(
            -0.5 * np.matmul(dist.T, np.matmul(np.linalg.inv(cov), dist))
        )
        det_cov = np.linalg.det(cov)

        p_i_cj = exp_dist_cov_dist / np.sqrt((self.pi2**self.dim_) * det_cov)
        p_cj = n / np.sum(w_[-1] for w_ in self.W)

        activation = p_i_cj * p_cj

        cache = {"cov": cov, "det_cov": det_cov}

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
        # the original paper uses the det(cov_old) for match criterion
        # however, it makes logical sense to use the new_cov and results are
        # improved when doing so
        assert cache is not None
        new_w = self.update(i, w, params, cache)
        new_cov = new_w[self.dim_ : -1].reshape((self.dim_, self.dim_))
        cache["new_w"] = new_w
        # return cache["det_cov"]
        return np.linalg.det(new_cov), cache

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
        op: Callable = operator.ge,
    ) -> tuple[bool, dict]:
        """Get the binary match criterion of the cluster.

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
        op : callable, optional
            Operator for comparison, by default operator.ge.

        Returns
        -------
        bool
            Binary match criterion.
        dict
            Cache used for later processing.

        """
        M, cache = self.match_criterion(i, w, params=params, cache=cache)
        M_bin = op(
            params["rho"], M
        )  # note that this is backwards from the base ART: rho >= M
        if cache is None:
            cache = dict()
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin

        return M_bin, cache

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Adjust match tracking based on the method and epsilon value.

        Parameters
        ----------
        cache : dict
            Cache containing intermediate results, including the match criterion.
        epsilon : float
            Adjustment factor for the match criterion.
        params : dict
            Dictionary containing algorithm parameters.
        method : {"MT+", "MT-", "MT0", "MT1", "MT~"}
            Match tracking method to use.

        Returns
        -------
        bool
            True if match tracking continues, False otherwise.

        """
        assert isinstance(cache, dict)
        assert isinstance(params, dict)
        M = cache["match_criterion"]
        # we have to reverse some signs because bayesianART has an inverted
        # vigilence check
        if method == "MT+":
            self.params["rho"] = M - epsilon
            return True
        elif method == "MT-":
            self.params["rho"] = M + epsilon
            return True
        elif method == "MT0":
            self.params["rho"] = M
            return True
        elif method == "MT1":
            self.params["rho"] = -np.inf
            return False
        elif method == "MT~":
            return True
        else:
            raise ValueError(f"Invalid Match Tracking Method: {method}")

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
        assert cache is not None

        if "new_w" in cache:
            return cache["new_w"]

        mean = w[: self.dim_]
        cov = w[self.dim_ : -1].reshape((self.dim_, self.dim_))
        n = w[-1]

        n_new = n + 1
        mean_new = (1 - (1 / n_new)) * mean + (1 / n_new) * i

        i_mean_dist = i - mean_new
        i_mean_dist_2 = i_mean_dist.reshape((-1, 1)) * i_mean_dist.reshape((1, -1))

        cov_new = (n / n_new) * cov + (1 / n_new) * i_mean_dist_2

        return np.concatenate([mean_new, cov_new.flatten(), [n_new]])

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
            Updated cluster weight.

        """
        return np.concatenate([i, params["cov_init"].flatten(), [1]])

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
            cov = w[self.dim_ : -1].reshape((self.dim_, self.dim_))
            # sigma = np.sqrt(np.diag(cov))
            plot_gaussian_contours_covariance(ax, mean, cov, col, linewidth=linewidth)
