import numpy as np
from typing import Optional
from elementary.BaseART import BaseART
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

    def _get_x_cb(self, X: np.ndarray, c_b: int):
        b_components = self.module_b.labels_ == c_b
        return X[b_components]

    @staticmethod
    def _pearsonr(a: np.ndarray, b: np.ndarray):
        r, _ = pearsonr(a, b)
        return r

    def _average_pearson_corr(self, X: np.ndarray, k: int, c_a: int, c_b: int) -> float:
        X_a = X[self.column_labels_ == c_a, :]
        X_k_cb = self._get_x_cb(X[k,:], c_b)
        mean_r = np.mean(
            [
                self._pearsonr(X_k_cb, self._get_x_cb(x_a_l, c_b))
                for x_a_l in X_a
            ]
        )

        return mean_r

    def match_criterion_bin(self, X: np.ndarray, k: int, c_a: int, c_b: int, params: dict) -> bool:
        return self._average_pearson_corr(X, k, c_a, c_b) >= params["eta"]

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
            if self.match_criterion_bin(self.X, k, cluster_a, cluster_b, params):
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
        self.validate_data(X)
        self.X = X

        n = X.shape[0]
        self.module_b = self.module_b.fit(X.T, max_iter=max_iter)

        # init module A
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],))

        for _ in range(max_iter):
            for k in range(n):
                c_a = self.step_fit(X, k)
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



