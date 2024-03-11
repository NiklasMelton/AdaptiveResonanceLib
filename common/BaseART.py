import numpy as np
from typing import Optional, Callable, Iterable
from matplotlib.axes import Axes
from warnings import warn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted


class BaseART(BaseEstimator, ClusterMixin):
    # Generic implementation of Adaptive Resonance Theory (ART)
    def __init__(self, params: dict):
        self.validate_params(params)
        self.params = params

    @property
    def n_clusters(self) -> int:
        if hasattr(self, "W"):
            return len(self.W)
        else:
            return 0

    @staticmethod
    def validate_params(params: dict):
        raise NotImplementedError

    def check_dimensions(self, X: np.ndarray):
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"
        self.check_dimensions(X)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        raise NotImplementedError

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        raise NotImplementedError

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        raise NotImplementedError

    def update(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> np.ndarray:
        raise NotImplementedError

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        raise NotImplementedError

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        if len(self.W) == 0:
            self.W.append(self.new_weight(x, self.params))
            return 0
        else:
            T_values, T_cache = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.argmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m = self.match_criterion_bin(x, w, params=self.params, cache=cache)
                no_match_reset = (
                        match_reset_func is None or
                        match_reset_func(x, w, c_, params=self.params, cache=cache)
                )
                if m and no_match_reset:
                    self.W[c_] = self.update(x, w, self.params, cache=cache)
                    return c_
                else:
                    T[c_] = -1

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.W.append(w_new)
            return c_new

    def step_pred(self, x) -> int:
        assert len(self.W) >= 0, "ART module is not fit."

        T, _ = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
        c_ = int(np.argmax(T))
        return c_

    def pre_step_fit(self, X: np.ndarray):
        # this is where pruning steps can go
        pass


    def fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None, max_iter=1):
        self.validate_data(X)
        self.check_dimensions(X)

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ), dtype=int)
        for _ in range(max_iter):
            for i, x in enumerate(X):
                self.pre_step_fit(X)
                c = self.step_fit(x, match_reset_func=match_reset_func)
                self.labels_[i] = c
        return self


    def partial_fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None):
        self.validate_data(X)
        self.check_dimensions(X)

        if not hasattr(self, 'W'):
            self.W: list[np.ndarray] = []
            self.labels_ = np.zeros((X.shape[0], ), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode='constant')
        for i, x in enumerate(X):
            c = self.step_fit(x, match_reset_func=match_reset_func)
            self.labels_[i+j] = c
        return self


    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        self.validate_data(X)
        self.check_dimensions(X)

        y = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c = self.step_pred(x)
            y[i] = c
        return y

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        raise NotImplementedError

    def visualize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            ax: Optional[Axes] = None,
            marker_size: int = 10,
            linewidth: int = 1,
            colors: Optional[Iterable] = None
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm
            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))

        for k, col in enumerate(colors):
            cluster_data = y == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], color=col, marker=".", s=marker_size)

        try:
            self.plot_cluster_bounds(ax, colors, linewidth)
        except NotImplementedError:
            warn(f"{self.__name__} does not support plotting cluster bounds." )







