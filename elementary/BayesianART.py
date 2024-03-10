"""
Vigdor, B., & Lerner, B. (2007).
The Bayesian ARTMAP.
IEEE Transactions on Neural Networks, 18, 1628–1644. doi:10.1109/TNN.2007.900234.
"""
import numpy as np
from typing import Optional
from common.BaseART import BaseART
from common.utils import normalize

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class BayesianART(BaseART):
    # implementation of Bayesian ART
    pi2 = np.pi * 2

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "cov_init" in params
        assert params["rho"] > 0

    def check_dimensions(self, X: np.ndarray):
        if not self.dim_:
            self.dim_ = X.shape[1]
            assert self.params["cov_init"].shape[0] == self.dim_
            assert self.params["cov_init"].shape[1] == self.dim_
        else:
            assert X.shape[1] == self.dim_

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        mean = w[:self.dim_]
        cov = w[self.dim_:self.dim_*self.dim_].reshape((self.dim_, self.dim_))
        n = w[-1]
        dist = mean - i
        exp_dist_cov_dist = np.exp(-0.5 * np.matmul(dist.T, np.matmul((1 / cov), dist)))
        cache = {
            "exp_dist_cov_dist": exp_dist_cov_dist,
            "cov": cov
        }
        p_i_cj = exp_dist_cov_dist / np.sqrt((self.pi2 ** self.dim_) * np.linalg.det(cov))
        p_cj = n / np.sum(w_[-1] for w_ in self.W)

        return p_i_cj * p_cj, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        if cache is None:
            raise ValueError("No cache provided")
        return cache["cov"]

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.match_criterion(i, w, params=params, cache=cache) >= params["rho"]

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        if cache is None:
            raise ValueError("No cache provided")

        mean = w[:self.dim_]
        cov = cache["cov"]
        n = w[-1]

        n_new = n+1
        mean_new = (1-(1/n_new))*mean + (1/n_new)*i
        cov_new = (n/n_new)*cov + (1/n_new)*np.multiply(
            ((i-mean_new).reshape((-1, 1))*(i-mean_new).reshape((1, -1))).T,
            np.identity(self.dim_)
        )

        return np.concatenate([mean_new, cov_new.flatten(), [n_new]])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return np.concatenate[i, params["cov_init"].flatten(), [1]]
