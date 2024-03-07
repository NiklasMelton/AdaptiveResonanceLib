import numpy as np
from typing import Optional, Callable
from elementary.BaseART import BaseART
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.utils.validation import check_is_fitted
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

    def match_criterion_bin(self, X: np.ndarray, k: int, c_a: int, c_b: int) -> bool:
        return self._average_pearson_corr(X, k, c_a, c_b) >= self.params["eta"]


    def fit(self, X: np.ndarray, max_iter: int = 1):
        n = X.shape[0]
        self.module_b = self.module_b.fit(X.T, max_iter=max_iter)

        num_row_clusters = 0
        for k in range(n):
            if k == 0:
                self.row_labels_[0] = 0
                num_row_clusters = 1
            else:
                for c_a in range(num_row_clusters):





