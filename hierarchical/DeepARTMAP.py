import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from elementary.BaseART import BaseART
from typing import Optional, cast, Union
from supervised.ARTMAP import SimpleARTMAP, ARTMAP, BaseARTMAP

class DeepARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):

    def __init__(self, modules: list[BaseART]):
        assert len(modules) >= 1, "Must provide at least one ART module"
        self.modules = modules
        self.layers: list[BaseARTMAP]
        self.is_supervised: Optional[bool] = None

    @property
    def labels_(self):
        return self.layers[0].labels_

    @property
    def n_modules(self):
        return len(self.modules)

    @property
    def n_layers(self):
        return len(self.layers)

    def validate_data(
            self,
            X: list[np.ndarray],
            y: Optional[np.ndarray] = None
    ) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        assert len(X) == self.n_modules, \
            f"Must provide {self.n_modules} input matrices for {self.n_modules} ART modules"
        if y is not None:
            n = len(y)
        else:
            n = X[0].shape[0]
        assert all(x.shape[0] == n for x in X), "Inconsistent sample number in input matrices"
        return X, y


    def fit(self, X: list[np.ndarray], y: Optional[np.ndarray] = None, max_iter=1):
        X, y = self.validate_data(X, y)
        if y is not None:
            self.is_supervised = True
            self.layers = [SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)]
            self.layers[0] = self.layers[0].fit(X[0], y, max_iter=max_iter)
        else:
            self.is_supervised = False
            assert self.n_modules >= 2, "Must provide at least two ART modules when providing cluster labels"
            self.layers = cast(list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]) + \
                           cast(list[BaseARTMAP], [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)])
            self.layers[0] = self.layers[0].fit(X[1], X[0], max_iter=max_iter)

        for art_i in range(1, self.n_layers):
            y_i = self.layers[art_i-1].labels_
            self.layers[art_i] = self.layers[art_i].fit(X[art_i], y_i, max_iter=max_iter)

        return self


    def partial_fit(self, X: list[np.ndarray], y: Optional[np.ndarray] = None):
        X, y = self.validate_data(X, y)
        if y is not None:
            if len(self.layers) == 0:
                self.is_supervised = True
                self.layers = [SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)]
            assert self.is_supervised, "Labels were previously provided. Must continue to provide labels for partial fit."
            self.layers[0] = self.layers[0].partial_fit(X[0], y)
            x_i = 1
        else:
            if len(self.layers) == 0:
                self.is_supervised = False
                assert self.n_modules >= 2, "Must provide at least two ART modules when providing cluster labels"
                self.layers = cast(list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]) + \
                               cast(list[BaseARTMAP], [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)])
            assert not self.is_supervised, "Labels were not previously provided. Do not provide labels to continue partial fit."
            self.layers[0] = self.layers[0].partial_fit(X[1], X[0])
            x_i = 2

        n_samples = X[0].shape[0]

        for art_i in range(1, self.n_modules):
            y_i = self.layers[art_i-1].labels_a[-n_samples:]
            self.layers[art_i] = self.layers[art_i].partial_fit(X[x_i], y_i)
            x_i += 1
        return self


    def predict(self, X: Union[np.ndarray, list[np.ndarray]]) -> list[np.ndarray]:
        if isinstance(X, list):
            x = X[-1]
        else:
            x = X
        pred_a, pred_b = self.layers[-1].predict(x)
        pred = [pred_a, pred_b]
        for layer in self.layers[:-1][::-1]:
            pred.append(layer.map_a2b(pred[-1]))

        return pred[::-1]





