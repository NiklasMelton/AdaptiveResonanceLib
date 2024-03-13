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
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from common.BaseART import BaseART
from sklearn.base import BaseEstimator, BiclusterMixin
from scipy.stats import pearsonr

class BARTMAP(BaseEstimator, BiclusterMixin):
    rows_: np.ndarray #bool
    columns_: np.ndarray #bool

    def __init__(self, module_a: BaseART, module_b: BaseART, params: dict):
        self.validate_params(params)
        self.params = params
        self.module_a = module_a
        self.module_b = module_b

    @staticmethod
    def validate_params(params):
        assert "eta" in params

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

    def _get_x_cb(self, X: np.ndarray, c_b: int):
        b_components = self.module_b.labels_ == c_b
        return X[b_components]

    @staticmethod
    def _pearsonr(a: np.ndarray, b: np.ndarray):
        r, _ = pearsonr(a, b)
        return r

    def _average_pearson_corr(self, X: np.ndarray, k: int, c_b: int) -> float:
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
        self.module_a.validate_data(X_a)
        self.module_b.validate_data(X_b)

    def match_criterion_bin(self, X: np.ndarray, k: int, c_b: int, params: dict) -> bool:
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
        k = extra["k"]
        for cluster_b in range(len(self.module_b.W)):
            if self.match_criterion_bin(self.X, k, cluster_b, params):
                return True
        return False

    def step_fit(self, X: np.ndarray, k: int) -> int:
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"k": k}, cache=cache
        )
        c_a = self.module_a.step_fit(X[k, :], match_reset_func=match_reset_func)
        return c_a

    def fit(self, X: np.ndarray, max_iter=1):
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
                print(k, self.module_a.n_clusters)
                self.module_a.pre_step_fit(X)
                c_a = self.step_fit(X_a, k)
                self.module_a.labels_[k] = c_a

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
        import matplotlib.pyplot as plt

        if cmap is None:
            from matplotlib.pyplot import cm
            cmap=plt.cm.Blues

        plt.matshow(
            np.outer(np.sort(self.row_labels_) + 1, np.sort(self.column_labels_) + 1),
            cmap=cmap,
        )
