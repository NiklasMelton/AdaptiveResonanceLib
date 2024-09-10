"""
Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
ARTMAP: Supervised real-time learning and classification of nonstationary data by a self-organizing neural network.
Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from typing import Optional, cast, Union, Literal, Tuple
from collections import defaultdict
from artlib.common.BaseART import BaseART
from artlib.common.BaseARTMAP import BaseARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.supervised.ARTMAP import ARTMAP

class DeepARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):

    def __init__(self, modules: list[BaseART]):
        """

        Parameters:
        - modules: list of ART modules

        """
        assert len(modules) >= 1, "Must provide at least one ART module"
        self.modules = modules
        self.layers: list[BaseARTMAP]
        self.is_supervised: Optional[bool] = None

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = dict()
        for i, module in enumerate(self.modules):
            out[f"module_{i}"] = module
            if deep:
                deep_items = module.get_params().items()
                out.update((f"module_{i}" + "__" + k, val) for k, val in deep_items)
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Specific redefinition of sklearn.BaseEstimator.set_params for ARTMAP classes

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
        return self

    @property
    def labels_(self):
        return self.layers[0].labels_

    @property
    def labels_deep_(self):
        return np.concatenate(
            [
                layer.labels_.reshape((-1, 1))
                for layer in self.layers
            ]+[
                self.layers[-1].labels_a.reshape((-1, 1))
            ],
            axis=1
        )

    @property
    def n_modules(self):
        return len(self.modules)

    @property
    def n_layers(self):
        return len(self.layers)

    def map_deep(self, level: int, y_a: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        map a label from one arbitrary level to the highest (B) level

        Parameters:
        - level: level the label is from
        - y_a: the cluster label(s)

        Returns:
            cluster label(s) at highest level

        """
        if level < 0:
            level += len(self.layers)
        y_b = self.layers[level].map_a2b(y_a)
        if level > 0:
            return self.map_deep(level-1, y_b)
        else:
            return y_b


    def validate_data(
            self,
            X: list[np.ndarray],
            y: Optional[np.ndarray] = None
    ):
        """
        validates the data prior to clustering

        Parameters:
        - X: list of deep data sets
        - y: optional labels for data

        """
        assert len(X) == self.n_modules, \
            f"Must provide {self.n_modules} input matrices for {self.n_modules} ART modules"
        if y is not None:
            n = len(y)
        else:
            n = X[0].shape[0]
        assert all(x.shape[0] == n for x in X), "Inconsistent sample number in input matrices"

    def prepare_data(self,  X: list[np.ndarray], y: Optional[np.ndarray] = None) ->Tuple[list[np.ndarray], Optional[np.ndarray]]:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            prepared data
        """
        return [self.modules[i].prepare_data(X[i]) for i in range(self.n_modules)], y

    def restore_data(self,  X: list[np.ndarray], y: Optional[np.ndarray] = None) ->Tuple[list[np.ndarray], Optional[np.ndarray]]:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            prepared data
        """
        return [self.modules[i].restore_data(X[i]) for i in range(self.n_modules)], y


    def fit(self, X: list[np.ndarray], y: Optional[np.ndarray] = None, max_iter=1, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        """
        Fit the model to the data

        Parameters:
        - X: list of deep datasets
        - y: optional labels
        - max_iter: number of iterations to fit the model on the same data set
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """
        self.validate_data(X, y)
        if y is not None:
            self.is_supervised = True
            self.layers = [SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)]
            self.layers[0] = self.layers[0].fit(X[0], y, max_iter=max_iter, match_reset_method=match_reset_method, epsilon=epsilon)
        else:
            self.is_supervised = False
            assert self.n_modules >= 2, "Must provide at least two ART modules when providing cluster labels"
            self.layers = cast(list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]) + \
                           cast(list[BaseARTMAP], [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)])
            self.layers[0] = self.layers[0].fit(X[1], X[0], max_iter=max_iter, match_reset_method=match_reset_method, epsilon=epsilon)

        for art_i in range(1, self.n_layers):
            y_i = self.layers[art_i-1].labels_a
            self.layers[art_i] = self.layers[art_i].fit(X[art_i], y_i, max_iter=max_iter, match_reset_method=match_reset_method, epsilon=epsilon)

        return self


    def partial_fit(self, X: list[np.ndarray], y: Optional[np.ndarray] = None, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        """
        Partial fit the model to the data

        Parameters:
        - X: list of deep datasets
        - y: optional labels
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """
        self.validate_data(X, y)
        if y is not None:
            if len(self.layers) == 0:
                self.is_supervised = True
                self.layers = [SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)]
            assert self.is_supervised, "Labels were previously provided. Must continue to provide labels for partial fit."
            self.layers[0] = self.layers[0].partial_fit(X[0], y, match_reset_method=match_reset_method, epsilon=epsilon)
            x_i = 1
        else:
            if len(self.layers) == 0:
                self.is_supervised = False
                assert self.n_modules >= 2, "Must provide at least two ART modules when providing cluster labels"
                self.layers = cast(list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]) + \
                               cast(list[BaseARTMAP], [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)])
            assert not self.is_supervised, "Labels were not previously provided. Do not provide labels to continue partial fit."
            self.layers[0] = self.layers[0].partial_fit(X[1], X[0], match_reset_method=match_reset_method, epsilon=epsilon)
            x_i = 2

        n_samples = X[0].shape[0]

        for art_i in range(1, self.n_modules):
            y_i = self.layers[art_i-1].labels_a[-n_samples:]
            self.layers[art_i] = self.layers[art_i].partial_fit(X[x_i], y_i, match_reset_method=match_reset_method, epsilon=epsilon)
            x_i += 1
        return self


    def predict(self, X: Union[np.ndarray, list[np.ndarray]]) -> list[np.ndarray]:
        """
        predict labels for the data

        Parameters:
        - X: list of deep data sets

        Returns:
            B labels for the data

        """
        if isinstance(X, list):
            x = X[-1]
        else:
            x = X
        pred_a, pred_b = self.layers[-1].predict_ab(x)
        pred = [pred_a, pred_b]
        for layer in self.layers[:-1][::-1]:
            pred.append(layer.map_a2b(pred[-1]))

        return pred[::-1]





