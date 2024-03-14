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
from artlib.common.BaseART import BaseART

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

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = self.params
        for i, module in enumerate(self.modules):
            deep_items = module.get_params().items()
            out.update((f"module_{i}" + "__" + k, val) for k, val in deep_items)
            out[f"module_{i}"] = module
        return out


    @property
    def W(self):
        W = [
            np.concatenate(
                [
                    self.modules[k].W[i]
                    for k in range(self.n)
                 ]
            )
            for i
            in range(self.modules[0].n_clusters)
        ]
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
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "gamma_values" in params
        assert all([1.0 >= g >= 0.0 for g in params["gamma_values"]])
        assert sum(params["gamma_values"]) == 1.0
        assert isinstance(params["gamma_values"], np.ndarray)


    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        self.check_dimensions(X)
        for k in range(self.n):
            X_k = X[:, self._channel_indices[k][0]:self._channel_indices[k][1]]
            self.modules[k].validate_data(X_k)

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        assert X.shape[1] == self.dim_, "Invalid data shape"

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
        if cache is None:
            raise ValueError("No cache provided")
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
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        W = [
            self.modules[k].new_weight(
                i[self._channel_indices[k][0]:self._channel_indices[k][1]],
                self.modules[k].params,
            )
            for k in range(self.n)
        ]
        return np.concatenate(W)

    def add_weight(self, new_w: np.ndarray):
        """
        add a new cluster weight

        Parameters:
        - new_w: new cluster weight to add

        """
        for k in range(self.n):
            new_w_k = new_w[self._channel_indices[k][0]:self._channel_indices[k][1]]
            self.modules[k].add_weight(new_w_k)

    def set_weight(self, idx: int, new_w: np.ndarray):
        """
        set the value of a cluster weight

        Parameters:
        - idx: index of cluster to update
        - new_w: new cluster weight

        """
        for k in range(self.n):
            new_w_k = new_w[self._channel_indices[k][0]:self._channel_indices[k][1]]
            self.modules[k].set_weight(idx, new_w_k)
