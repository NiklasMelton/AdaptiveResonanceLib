"""
Vigdor, B., & Lerner, B. (2007).
The Bayesian ARTMAP.
IEEE Transactions on Neural Networks, 18, 1628–1644. doi:10.1109/TNN.2007.900234.
"""
import numpy as np
from typing import Optional, Iterable, List, Callable, Literal, Tuple
import operator
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.visualization import plot_gaussian_contours_covariance


class BayesianART(BaseART):
    # implementation of Bayesian ART
    pi2 = np.pi * 2
    def __init__(self, rho: float, cov_init: np.ndarray):
        """
        Parameters:
        - rho: vigilance parameter
        - cov_init: initial estimate of covariance matrix

        """
        params = {
            "rho": rho,
            "cov_init": cov_init,
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
        assert "cov_init" in params
        assert params["rho"] > 0
        assert isinstance(params["rho"], float)
        assert isinstance(params["cov_init"], np.ndarray)

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            assert self.params["cov_init"].shape[0] == self.dim_
            assert self.params["cov_init"].shape[1] == self.dim_
        else:
            assert X.shape[1] == self.dim_

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
        cov = w[self.dim_:-1].reshape((self.dim_, self.dim_))
        n = w[-1]
        dist = mean - i

        exp_dist_cov_dist = np.exp(-0.5 * np.matmul(dist.T, np.matmul(np.linalg.inv(cov), dist)))
        det_cov = np.linalg.det(cov)

        p_i_cj = exp_dist_cov_dist / np.sqrt((self.pi2 ** self.dim_) * det_cov)
        p_cj = n / np.sum(w_[-1] for w_ in self.W)

        activation = p_i_cj * p_cj

        cache = {
            "cov": cov,
            "det_cov": det_cov
        }

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
        # the original paper uses the det(cov_old) for match criterion
        # however, it makes logical sense to use the new_cov and results are improved when doing so
        new_w = self.update(i, w, params, cache)
        new_cov = new_w[self.dim_:-1].reshape((self.dim_, self.dim_))
        cache["new_w"] = new_w
        # if cache is None:
        #     raise ValueError("No cache provided")
        # return cache["det_cov"]
        return np.linalg.det(new_cov), cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None, op: Callable = operator.ge) -> tuple[bool, dict]:
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
        M, cache = self.match_criterion(i, w, params=params, cache=cache)
        M_bin = op(params["rho"], M) # note that this is backwards from the base ART: rho >= M
        if cache is None:
            cache = dict()
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin

        return M_bin, cache

    def _match_tracking(self, cache: dict, epsilon: float, params: dict, method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"]) -> bool:
        M = cache["match_criterion"]
        # we have to reverse some signs becayse bayesianART has an inverted vigilence check
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
        if cache is None:
            raise ValueError("No cache provided")

        if "new_w" in cache:
            return cache["new_w"]

        mean = w[:self.dim_]
        cov = w[self.dim_:-1].reshape((self.dim_, self.dim_))
        n = w[-1]

        n_new = n+1
        mean_new = (1-(1/n_new))*mean + (1/n_new)*i

        i_mean_dist = i-mean_new
        i_mean_dist_2 = i_mean_dist.reshape((-1, 1))*i_mean_dist.reshape((1, -1))

        cov_new = (n / n_new) * cov + (1 / n_new) * i_mean_dist_2

        return np.concatenate([mean_new, cov_new.flatten(), [n_new]])

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
        return np.concatenate([i, params["cov_init"].flatten(), [1]])

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
            cov = w[self.dim_:-1].reshape((self.dim_, self.dim_))
            # sigma = np.sqrt(np.diag(cov))
            plot_gaussian_contours_covariance(ax, mean, cov, col, linewidth=linewidth)
