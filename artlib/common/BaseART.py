import numpy as np
from typing import Optional, Callable, Iterable
from collections import defaultdict
from matplotlib.axes import Axes
from warnings import warn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from artlib.common.utils import normalize


class BaseART(BaseEstimator, ClusterMixin):
    # Generic implementation of Adaptive Resonance Theory (ART)
    def __init__(self, params: dict):
        """

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        self.validate_params(params)
        self.params = params
        self.sample_counter_ = 0
        self.weight_sample_counter_: list[int] = []

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        else:
            # If the key is not in params, raise an AttributeError
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key in self.__dict__.get('params', {}):
            # If key is in params, set its value
            self.params[key] = value
        else:
            # Otherwise, proceed with normal attribute setting
            super().__setattr__(key, value)


    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        return self.params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Specific redefinition of sklearn.BaseEstimator.set_params for ART classes

        Parameters:
        - **params : Estimator parameters.

        Returns:
        - self : estimator instance
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        local_params = dict()

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = list(valid_params.keys())
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
                local_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        self.validate_params(local_params)
        return self


    @staticmethod
    def prepare_data(X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            normalized data
        """
        return normalize(X)

    @property
    def n_clusters(self) -> int:
        """
        get the current number of clusters

        Returns:
            the number of clusters
        """
        if hasattr(self, "W"):
            return len(self.W)
        else:
            return 0

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        raise NotImplementedError

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        assert np.all(X >= 0), "Data has not been normalized"
        assert np.all(X <= 1.0), "Data has not been normalized"
        self.check_dimensions(X)

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def add_weight(self, new_w: np.ndarray):
        """
        add a new cluster weight

        Parameters:
        - new_w: new cluster weight to add

        """
        self.weight_sample_counter_.append(1)
        self.W.append(new_w)

    def set_weight(self, idx: int, new_w: np.ndarray):
        """
        set the value of a cluster weight

        Parameters:
        - idx: index of cluster to update
        - new_w: new cluster weight

        """
        self.weight_sample_counter_[idx] += 1
        self.W[idx] = new_w

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        """
        fit the model to a single sample

        Parameters:
        - x: data sample
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise

        Returns:
            cluster label of the input sample

        """
        self.sample_counter_ += 1
        if len(self.W) == 0:
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return 0
        else:
            T_values, T_cache = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.argmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m, cache = self.match_criterion_bin(x, w, params=self.params, cache=cache)
                no_match_reset = (
                        match_reset_func is None or
                        match_reset_func(x, w, c_, params=self.params, cache=cache)
                )
                if m and no_match_reset:
                    self.set_weight(c_, self.update(x, w, self.params, cache=cache))
                    return c_
                else:
                    T[c_] = -1

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return c_new

    def step_pred(self, x) -> int:
        """
        predict the label for a single sample

        Parameters:
        - x: data sample

        Returns:
            cluster label of the input sample

        """
        assert len(self.W) >= 0, "ART module is not fit."

        T, _ = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
        c_ = int(np.argmax(T))
        return c_

    def pre_step_fit(self, X: np.ndarray):
        """
        undefined function called prior to each sample fit. Useful for cluster pruning

        Parameters:
        - X: data set

        """
        # this is where pruning steps can go
        pass

    def post_step_fit(self, X: np.ndarray):
        """
        undefined function called after each sample fit. Useful for cluster pruning

        Parameters:
        - X: data set

        """
        # this is where pruning steps can go
        pass


    def fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None, max_iter=1):
        """
        Fit the model to the data

        Parameters:
        - X: data set
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise
        - max_iter: number of iterations to fit the model on the same data set

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ), dtype=int)
        for _ in range(max_iter):
            for i, x in enumerate(X):
                self.pre_step_fit(X)
                c = self.step_fit(x, match_reset_func=match_reset_func)
                self.labels_[i] = c
                self.post_step_fit(X)
        return self


    def partial_fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None):
        """
        iteratively fit the model to the data

        Parameters:
        - X: data set
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise

        """

        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ =  True

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


    def predict(self, X: np.ndarray) -> np.ndarray:
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
            c = self.step_pred(x)
            y[i] = c
        return y

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
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
        """
        Visualize the clustering of the data

        Parameters:
        - X: data set
        - y: sample labels
        - ax: figure axes
        - marker_size: size used for data points
        - linewidth: width of boundary line
        - colors: colors to use for each cluster

        """
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
            warn(f"{self.__class__.__name__} does not support plotting cluster bounds.")







