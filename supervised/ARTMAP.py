import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from elementary.BaseART import BaseART
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels


class BaseARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):
    map: dict[int, int]

    def map_a2b(self, y_a: np.ndarray) -> np.ndarray:
        u, inv = np.unique(y_a, return_inverse=True)
        return np.array([self.map[x] for x in u])[inv].reshape(y_a.shape)

    def validate_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SimpleARTMAP(BaseARTMAP):

    def match_reset_func(
            self,
            i: np.ndarray,
            w: np.ndarray,
            cluster_a,
            params: dict,
            extra: dict,
            cache: Optional[dict] = None
    ) -> bool:
        cluster_b = extra["cluster_b"]
        if cluster_a in self.map and self.map[cluster_a] != cluster_b:
            return False
        return True

    def __init__(self, module_a: BaseART):
        self.module_a = module_a
        self.map: dict[int, int] = dict()


    def validate_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(X, y)
        X = self.module_a.validate_data(X)
        return X, y

    def step_fit(self, x: np.ndarray, c_b: int) -> int:
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"cluster_b": c_b}, cache=cache
        )
        c_a = self.module_a.step_fit(x, match_reset_func=match_reset_func)
        if c_a not in self.map:
            self.map[c_a] = c_b
        else:
            assert self.map[c_a] == c_b
        return c_a

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        # Check that X and y have correct shape
        X, y = self.validate_data(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.labels_ = y
        # init module A
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],))

        for _ in range(max_iter):
            for i, (x, c_b) in enumerate(zip(X, y)):
                c_a = self.step_fit(x, c_b)
                self.module_a.labels_[i] = c_a
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X, y = self.validate_data(X, y)
        if not hasattr(self, 'labels_'):
            self.labels_ = y
            self.module_a.W = []
            self.module_a.labels_ = np.zeros((X.shape[0],))
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode='constant')
            self.labels_[j:] = y
            self.module_a.labels_ = np.pad(self.module_a.labels_, [(0, X.shape[0])], mode='constant')
        for i, (x, c_b) in enumerate(zip(X, y)):
            c_a = self.step_fit(x, c_b)
            self.module_a.labels_[i+j] = c_a
        return self

    @property
    def labels_a(self):
        return self.module_a.labels_

    @property
    def labels_b(self):
        return self.labels_

    @property
    def labels_ab(self):
        return {"A": self.labels_a, "B": self.labels_}

    def step_pred(self, x: np.ndarray) -> tuple[int, int]:
        c_a = self.module_a.step_pred(x)
        c_b = self.map[c_a]
        return c_a, c_b


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        y_a = np.zeros((X.shape[0],))
        y_b = np.zeros((X.shape[0],))
        for i, x in enumerate(X):
            c_a, c_b = self.step_pred(x)
            y_a[i] = c_a
            y_b[i] = c_b
        return y_a, y_b



class ARTMAP(BaseARTMAP):
    def __init__(self, module_a: BaseART, module_b: BaseART):
        self.module_b = module_b
        self.simpleARTMAP = SimpleARTMAP(module_a)

    @property
    def module_a(self):
        return self.simpleARTMAP.module_a

    @property
    def map(self):
        return self.simpleARTMAP.map

    @property
    def labels_(self):
        return self.simpleARTMAP.labels_

    @property
    def labels_a(self):
        return self.module_a.labels_

    @property
    def labels_b(self):
        return self.labels_

    @property
    def labels_ab(self):
        return {"A": self.labels_a, "B": self.labels_}

    def validate_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(X, y)
        self.module_a.validate_data(X)
        self.module_b.validate_data(y)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        # Check that X and y have correct shape
        self.validate_data(X, y)

        self.module_b.fit(y, max_iter=max_iter)

        y_c = self.module_b.labels_

        self.simpleARTMAP.fit(X, y_c, max_iter=max_iter)

        return self


    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        self.validate_data(X, y)
        self.module_b.partial_fit(y)
        self.simpleARTMAP.partial_fit(X, self.labels_b)
        return self


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        return self.simpleARTMAP.predict(X)
