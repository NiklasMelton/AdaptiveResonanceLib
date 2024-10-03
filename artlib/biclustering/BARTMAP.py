"""
Xu, R., & Wunsch II, D. C. (2011).
BARTMAP: A viable structure for biclustering.
Neural Networks, 24, 709–716. doi:10.1016/j.neunet.2011.03.020.

Xu, R., Wunsch II, D. C., & Kim, S. (2012).
Methods and systems for biclustering algorithm.
U.S. Patent 9,043,326 Filed January 28, 2012,
claiming priority to Provisional U.S. Patent Application,
January 28, 2011, issued May 26, 2015.
"""

import numpy as np
from typing import Optional
from collections import defaultdict
from matplotlib.colors import Colormap
from artlib.common.BaseART import BaseART
from sklearn.base import BaseEstimator, BiclusterMixin
from scipy.stats import pearsonr

class BARTMAP(BaseEstimator, BiclusterMixin):
    rows_: np.ndarray #bool
    columns_: np.ndarray #bool

    def __init__(self, module_a: BaseART, module_b: BaseART, eta: float):
        """

        Parameters:
        - module_a: a-side ART module
        - module_b: b-side ART module
        - eta: minimum correlation

        """
        params: dict = {"eta": eta}
        self.validate_params(params)
        self.params = params
        self.module_a = module_a
        self.module_b = module_b

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
        out = self.params

        deep_a_items = self.module_a.get_params().items()
        out.update(("module_a" + "__" + k, val) for k, val in deep_a_items)
        out["module_a"] = self.module_a

        deep_b_items = self.module_b.get_params().items()
        out.update(("module_b" + "__" + k, val) for k, val in deep_b_items)
        out["module_b"] = self.module_b
        return out

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
    def validate_params(params):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "eta" in params
        assert isinstance(params["eta"], float)

    @property
    def column_labels_(self):
        return self.module_b.labels_

    @property
    def row_labels_(self):
        return self.module_a.labels_

    @property
    def n_row_clusters(self):
        return self.module_a.n_clusters

    @property
    def n_column_clusters(self):
        return self.module_b.n_clusters

    def _get_x_cb(self, x: np.ndarray, c_b: int):
        """
        get the components of a vector belonging to a b-side cluster

        Parameters:
        - x: a sample vector
        - c_b: b-side cluster label

        Returns:
            x filtered to features belonging to the b-side cluster c_b

        """
        b_components = self.module_b.labels_ == c_b
        return x[b_components]

    @staticmethod
    def _pearsonr(a: np.ndarray, b: np.ndarray):
        """
        get the correlation between two vectors

        Parameters:
        - a: some vector
        - b: some vector

        Returns:
            Pearson correlation

        """
        r, _ = pearsonr(a, b)
        return r

    def _average_pearson_corr(self, X: np.ndarray, k: int, c_b: int) -> float:
        """
        get the average correlation between for a sample for all features in cluster b

        Parameters:
        - X: data set A
        - k: sample index
        - c_b: b-side cluster to check

        """
        X_a = X[self.column_labels_ == c_b, :]
        if len(X_a) == 0:
            raise ValueError("HERE")
        X_k_cb = self._get_x_cb(X[k,:], c_b)
        mean_r = np.mean(
            [
                self._pearsonr(X_k_cb, self._get_x_cb(x_a_l, c_b))
                for x_a_l in X_a
            ]
        )

        return float(mean_r)

    def validate_data(self, X_a: np.ndarray, X_b: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set A
        - y: data set B

        """
        self.module_a.validate_data(X_a)
        self.module_b.validate_data(X_b)

    def match_criterion_bin(self, X: np.ndarray, k: int, c_b: int, params: dict) -> bool:
        """
        get the binary match criterion of the cluster

        Parameters:
        - X: data set
        - k: sample index
        - c_b: b-side cluster to check
        - params: dict containing parameters for the algorithm

        Returns:
            cluster match criterion binary

        """
        M = self._average_pearson_corr(X, k, c_b)
        return M >= self.params["eta"]

    def match_reset_func(
            self,
            i: np.ndarray,
            w: np.ndarray,
            cluster_a,
            params: dict,
            extra: dict,
            cache: Optional[dict] = None
    ) -> bool:
        """
        Permits external factors to influence cluster creation.

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - cluster_a: a-side cluster label
        - params: dict containing parameters for the algorithm
        - extra: additional parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            true if match is permitted

        """
        k = extra["k"]
        for cluster_b in range(len(self.module_b.W)):
            if self.match_criterion_bin(self.X, k, cluster_b, params):
                return True
        return False

    def step_fit(self, X: np.ndarray, k: int) -> int:
        """
        fit the model to a single sample

        Parameters:
        - X: data set
        - k: sample index

        Returns:
            cluster label of the input sample

        """
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"k": k}, cache=cache
        )
        c_a = self.module_a.step_fit(X[k, :], match_reset_func=match_reset_func)
        return c_a

    def fit(self, X: np.ndarray, max_iter=1):
        """
        Fit the model to the data

        Parameters:
        - X: data set
        - max_iter: number of iterations to fit the model on the same data set

        """
        # Check that X and y have correct shape
        self.X = X

        n = X.shape[0]
        X_a = self.module_b.prepare_data(X)
        X_b = self.module_b.prepare_data(X.T)
        self.validate_data(X_a, X_b)


        self.module_b = self.module_b.fit(X_b, max_iter=max_iter)

        # init module A
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        for _ in range(max_iter):
            for k in range(n):
                self.module_a.pre_step_fit(X)
                c_a = self.step_fit(X_a, k)
                self.module_a.labels_[k] = c_a
                self.module_a.post_step_fit(X)

        self.rows_ = np.vstack(
            [
                self.row_labels_ == label
                for label in range(self.module_a.n_clusters)
                for _ in range(self.module_b.n_clusters)
            ]
        )
        self.columns_ = np.vstack(
            [
                self.column_labels_ == label
                for _ in range(self.module_a.n_clusters)
                for label in range(self.module_b.n_clusters)
            ]
        )
        return self


    def visualize(
            self,
            cmap: Optional[Colormap] = None
    ):
        """
        Visualize the clustering of the data

        Parameters:
        - cmap: some colormap

        """
        import matplotlib.pyplot as plt

        if cmap is None:
            from matplotlib.pyplot import cm
            cmap=plt.cm.Blues

        plt.matshow(
            np.outer(np.sort(self.row_labels_) + 1, np.sort(self.column_labels_) + 1),
            cmap=cmap,
        )
