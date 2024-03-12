"""
Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
Intelligence Through Interaction: Towards a Unified Theory for Learning.
In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
Berlin, Heidelberg: Springer Berlin Heidelberg.
doi:10.1007/ 978-3-540-72383-7_128.
"""
import numpy as np
from typing import Optional, Union
from common.BaseART import BaseART

def get_channel_position_tuples(channel_dims: list[int]) -> list[tuple[int, int]]:
    positions = []
    start = 0
    for length in channel_dims:
        end = start + length
        positions.append((start, end))
        start = end
    return positions

class FusionART(BaseART):
    # implementation of FusionART

    def __init__(
            self,
            modules: list[BaseART],
            gamma_values: Union[list[float], np.ndarray],
            channel_dims: Union[list[int], np.ndarray]
    ):
        assert len(modules) == len(gamma_values) == len(channel_dims)
        params = {"gamma_values": gamma_values}
        super().__init__(params)
        self.modules = modules
        self.n = len(self.modules)
        self.channel_dims = channel_dims
        self._channel_indices = get_channel_position_tuples(self.channel_dims)
        self.dim_ = sum(channel_dims)

    @property
    def W(self):
        W = np.concatenate(
            self.modules[k].W
            for k in range(self.n)
        )
        return W

    @W.setter
    def W(self, new_W):
        for k in range(self.n):
            if len(new_W) > 0:
                self.modules[k].W = new_W[self._channel_indices[k][0]:self._channel_indices[k][1]]
            else:
                self.modules[k].W = []

    @staticmethod
    def validate_params(params: dict):
        assert "gamma_values" in params
        assert all([1.0 >= g >= 0.0 for g in params["gamma_values"]])
        assert sum(params["gamma_values"]) == 1.0

    def validate_data(self, X: np.ndarray):
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"

    def check_dimensions(self, X: np.ndarray):
        assert X.shape[1] == self.dim_, "Invalid data shape"

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        activations, caches = zip(
            *[
                self.modules[k].category_choice(
                    i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    w[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    self.modules[k].params
                )
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        activation = sum([a*self.params["gamma_values"][k] for k, a in enumerate(activations)])
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[list[float], dict]:
        if cache is None:
            raise ValueError("No cache provided")
        M, caches = zip(
            *[
                self.modules[k].match_criterion(
                    i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    w[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    self.modules[k].params,
                    cache[k]
                )
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return M, cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        M_bin, caches = zip(
            *[
                self.modules[k].match_criterion_bin(
                    i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    w[self._channel_indices[k][0]:self._channel_indices[k][1]],
                    self.modules[k].params,
                    cache[k]
                )
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return all(M_bin), cache

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        W = [
            self.modules[k].update(
                i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                w[self._channel_indices[k][0]:self._channel_indices[k][1]],
                self.modules[k].params,
                cache[k]
            )
            for k in range(self.n)
        ]
        return np.concatenate(W)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        W = [
            self.modules[k].new_weight(
                i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                self.modules[k].params,
            )
            for k in range(self.n)
        ]
        return np.concatenate(W)
