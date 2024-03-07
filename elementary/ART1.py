import numpy as np
from typing import Optional
from elementary.BaseART import BaseART
from art_utils.utils import normalize, l1norm

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class ART1(BaseART):
    # implementation of ART 1

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "beta" in params
        assert "L" in params
        assert 1. >= params["rho"] >= 0.
        assert 1. >= params["beta"] >= 0.
        assert params["L"] >= 1.

    def validate_data(self, X: np.ndarray):
        assert ((X == 0) | (X == 1)), "ART1 only supports binary data"

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        w_bu = w[:self.dim_]
        return float(np.dot(i, w_bu)), None

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        w_td = w[self.dim_:]
        return l1norm(np.logical_and(i, w_td)) / self.dim_

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.match_criterion(i, w, params, cache) >= params["rho"]

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        w_td = w[self.dim_:]

        w_td_new = np.logical_and(i, w_td)
        w_bu_new = (params["L"] / (params["L"] - 1 + l1norm(w_td_new)))*w_td_new
        return np.concatenate([w_bu_new, w_td_new])


    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        w_td_new = i
        w_bu_new = (params["L"] / (params["L"] - 1 + self.dim_))
        return np.concatenate([w_bu_new, w_td_new])
