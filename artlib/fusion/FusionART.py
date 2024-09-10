"""
Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
Intelligence Through Interaction: Towards a Unified Theory for Learning.
In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
Berlin, Heidelberg: Springer Berlin Heidelberg.
doi:10.1007/ 978-3-540-72383-7_128.
"""
import numpy as np
from typing import Optional, Union, Callable, List, Literal, Tuple
from copy import deepcopy
from artlib.common.BaseART import BaseART
from sklearn.utils.validation import check_is_fitted
import operator

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
    def n_clusters(self) -> int:
        return self.modules[0].n_clusters

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

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict, skip_channels: List[int] = []) -> tuple[float, Optional[dict]]:
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
                if k not in skip_channels
                else (1., dict())
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        activation = sum([a*self.params["gamma_values"][k] for k, a in enumerate(activations)])
        return activation, cache

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None, skip_channels: List[int] = []) -> tuple[list[float], dict]:
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
                if k not in skip_channels
                else (np.inf, {"match_criterion": np.inf})
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return M, cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None, skip_channels: List[int] = [], op: Callable = operator.ge) -> tuple[bool, dict]:
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
                    cache[k],
                    op
                )
                if k not in skip_channels
                else (True, {"match_criterion": np.inf})
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return all(M_bin), cache


    def _match_tracking(self, cache: List[dict], epsilon: float, params: List[dict], method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"]) -> bool:
        keep_searching = []
        for i in range(len(cache)):
            if cache[i]["match_criterion_bin"]:
                keep_searching_i = self.modules[i]._match_tracking(cache[i], epsilon, params[i], method)
                keep_searching.append(keep_searching_i)
            else:
                keep_searching.append(True)
        return all(keep_searching)


    def _set_params(self, new_params):
        for i in range(self.n):
            self.modules[i].params = new_params[i]

    def _deep_copy_params(self):
        return {i: deepcopy(module.params) for i, module in enumerate(self.modules)}


    def partial_fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        """
        iteratively fit the model to the data

        Parameters:
        - X: data set
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """

        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ =  True

        if not hasattr(self.modules[0], 'W'):
            self.W: list[np.ndarray] = []
            self.labels_ = np.zeros((X.shape[0], ), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode='constant')
        for i, x in enumerate(X):
            c = self.step_fit(x, match_reset_func=match_reset_func, match_reset_method=match_reset_method, epsilon=epsilon)
            self.labels_[i+j] = c
        return self

    def step_pred(self, x, skip_channels: List[int] = []) -> int:
        """
        predict the label for a single sample

        Parameters:
        - x: data sample

        Returns:
            cluster label of the input sample

        """
        assert len(self.W) >= 0, "ART module is not fit."

        T, _ = zip(*[self.category_choice(x, w, params=self.params, skip_channels=skip_channels) for w in self.W])
        c_ = int(np.argmax(T))
        return c_

    def predict(self, X: np.ndarray, skip_channels: List[int] = []) -> np.ndarray:
        """
        predict labels for the data

        Parameters:
        - X: data set

        Returns:
            labels for the data

        """

        check_is_fitted(self)
        self.validate_data(X)
        self.check_dimensions(X)

        y = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c = self.step_pred(x, skip_channels=skip_channels)
            y[i] = c
        return y

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

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        centers_ = [module.get_cluster_centers() for module in self.modules]
        centers = [
            np.concatenate(
                [
                    centers_[k][i]
                    for k in range(self.n)
                ]
            )
            for i
            in range(self.n_clusters)
        ]
        return centers

    def get_channel_centers(self, channel: int):
        return self.modules[channel].get_cluster_centers()

    def predict_regression(self, X: np.ndarray, target_channels: List[int] = [-1]) -> Union[np.ndarray, List[np.ndarray]]:
        target_channels = [self.n+k if k < 0 else k for k in target_channels]
        C = self.predict(X, skip_channels=target_channels)
        centers = [self.get_channel_centers(k) for k in target_channels]
        if len(target_channels) == 1:
            return np.array([centers[0][c] for c in C])
        else:
            return [np.array([centers[k][c] for c in C]) for k in target_channels]

    def join_channel_data(self, channel_data: List[np.ndarray], skip_channels: List[int] = []) -> np.ndarray:
        skip_channels = [self.n+k if k < 0 else k for k in skip_channels]
        n_samples = channel_data[0].shape[0]

        formatted_channel_data = []
        i = 0
        for k in range(self.n):
            if k not in skip_channels:
                formatted_channel_data.append(channel_data[i])
                i += 1
            else:
                formatted_channel_data.append(0.5*np.ones((n_samples, self._channel_indices[k][1]-self._channel_indices[k][0])))

        X = np.hstack(formatted_channel_data)
        return X