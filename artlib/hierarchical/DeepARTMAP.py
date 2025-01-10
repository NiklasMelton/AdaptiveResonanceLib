"""Deep ARTMAP :cite:`carpenter1991artmap`."""
# Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
# ARTMAP: Supervised real-time learning and classification of nonstationary data by a
# self-organizing neural network.
# Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from typing import Optional, cast, Union, Literal, Tuple
from collections import defaultdict
from artlib.common.BaseART import BaseART
from artlib.common.BaseARTMAP import BaseARTMAP
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.supervised.ARTMAP import ARTMAP


class DeepARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):
    """DeepARTMAP for Hierachical Supervised and Unsupervised Learning.

    This module implements DeepARTMAP, a generalization of the
    :class:`~artlib.supervised.ARTMAP.ARTMAP` class :cite:`carpenter1991artmap` that
    allows an arbitrary number of data channels to be divisively clustered. DeepARTMAP
    support both supervised and unsupervised modes. If only two ART modules are
    provided, DeepARTMAP reverts to standard :class:`~artlib.supervised.ARTMAP.ARTMAP`
    where the first module is the A-module and the second module is the B-module.
    DeepARTMAP does not currently have a direct citation and is an original creation
    of this library.

    .. # Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
    .. # ARTMAP: Supervised real-time learning and classification of nonstationary data
    .. # by a self-organizing neural network.
    .. # Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.

    """

    def __init__(self, modules: list[BaseART]):
        """Initialize the DeepARTMAP model.

        Parameters
        ----------
        modules : list of BaseART
            A list of instantiated ART modules to use as layers in the DeepARTMAP model.

        Raises
        ------
        AssertionError
            If no ART modules are provided.

        """
        assert len(modules) >= 1, "Must provide at least one ART module"
        self.modules = modules
        self.layers: list[BaseARTMAP] = []
        self.is_supervised: Optional[bool] = None

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional, default=True
            If True, will return the parameters for this class and contained subobjects
            that are estimators.

        Returns
        -------
        dict
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

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DeepARTMAP
            The estimator instance.

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
    def labels_(self) -> np.ndarray:
        """Get the labels from the first layer.

        Returns
        -------
        np.ndarray
            The labels from the first ART layer.

        """
        return self.layers[0].labels_

    @property
    def labels_deep_(self) -> np.ndarray:
        """Get the deep labels from all layers.

        Returns
        -------
        np.ndarray
            Deep labels from all ART layers concatenated together.

        """
        return np.concatenate(
            [layer.labels_.reshape((-1, 1)) for layer in self.layers]
            + [self.layers[-1].labels_a.reshape((-1, 1))],
            axis=1,
        )

    @property
    def n_modules(self) -> int:
        """Get the number of ART modules.

        Returns
        -------
        int
            The number of ART modules.

        """
        return len(self.modules)

    @property
    def n_layers(self) -> int:
        """Get the number of layers.

        Returns
        -------
        int
            The number of layers in DeepARTMAP.

        """
        return len(self.layers)

    def map_deep(
        self, level: int, y_a: Union[np.ndarray, int]
    ) -> Union[np.ndarray, int]:
        """Map a label from one arbitrary level to the highest (B) level.

        Parameters
        ----------
        level : int
            The level from which the label is taken.
        y_a : np.ndarray or int
            The cluster label(s) at the input level.

        Returns
        -------
        np.ndarray or int
            The cluster label(s) at the highest level (B).

        """
        if level < 0:
            level += len(self.layers)
        y_b = self.layers[level].map_a2b(y_a)
        if level > 0:
            return self.map_deep(level - 1, y_b)
        else:
            return y_b

    def validate_data(self, X: list[np.ndarray], y: Optional[np.ndarray] = None):
        """Validate the data before clustering.

        Parameters
        ----------
        X : list of np.ndarray
            The input data sets for each module.
        y : np.ndarray, optional
            The corresponding labels, by default None.

        Raises
        ------
        AssertionError
            If the input data is inconsistent or does not match the expected format.

        """
        assert len(X) == self.n_modules, (
            f"Must provide {self.n_modules} input matrices for "
            f"{self.n_modules} ART modules"
        )
        if y is not None:
            n = len(y)
        else:
            n = X[0].shape[0]
        assert all(
            x.shape[0] == n for x in X
        ), "Inconsistent sample number in input matrices"

    def prepare_data(
        self, X: Union[np.ndarray, list[np.ndarray]], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[list[np.ndarray], Optional[np.ndarray]]]:
        """Prepare the data for clustering.

        Parameters
        ----------
        X : list of np.ndarray
            The input data set for each module.
        y : np.ndarray, optional
            The corresponding labels, by default None.

        Returns
        -------
        tuple of (list of np.ndarray, np.ndarray)
            The prepared data set and labels (if any).

        """
        return [self.modules[i].prepare_data(X[i]) for i in range(self.n_modules)], y

    def restore_data(
        self, X: Union[np.ndarray, list[np.ndarray]], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[list[np.ndarray], Optional[np.ndarray]]]:
        """Restore the data to its original state before preparation.

        Parameters
        ----------
        X : list of np.ndarray
            The input data set for each module.
        y : np.ndarray, optional
            The corresponding labels, by default None.

        Returns
        -------
        tuple of (list of np.ndarray, np.ndarray)
            The restored data set and labels (if any).

        """
        return [self.modules[i].restore_data(X[i]) for i in range(self.n_modules)], y

    def fit(
        self,
        X: list[np.ndarray],
        y: Optional[np.ndarray] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Fit the DeepARTMAP model to the data.

        Parameters
        ----------
        X : list of np.ndarray
            The input data sets for each module.
        y : np.ndarray, optional
            The corresponding labels for supervised learning, by default None.
        max_iter : int, optional
            The number of iterations to fit the model, by default 1.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            The method to reset vigilance if a mismatch occurs, by default "MT+".
        epsilon : float, optional
            A small adjustment factor for match tracking, by default 0.0.

        Returns
        -------
        DeepARTMAP
            The fitted DeepARTMAP model.

        """
        self.validate_data(X, y)
        if y is not None:
            self.is_supervised = True
            self.layers = [SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)]
            self.layers[0] = self.layers[0].fit(
                X[0],
                y,
                max_iter=max_iter,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )
        else:
            self.is_supervised = False
            assert (
                self.n_modules >= 2
            ), "Must provide at least two ART modules when providing cluster labels"
            self.layers = cast(
                list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]
            ) + cast(
                list[BaseARTMAP],
                [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)],
            )
            self.layers[0] = self.layers[0].fit(
                X[1],
                X[0],
                max_iter=max_iter,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )

        for art_i in range(1, self.n_layers):
            y_i = self.layers[art_i - 1].labels_a
            self.layers[art_i] = self.layers[art_i].fit(
                X[art_i],
                y_i,
                max_iter=max_iter,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )

        return self

    def partial_fit(
        self,
        X: list[np.ndarray],
        y: Optional[np.ndarray] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Partially fit the DeepARTMAP model to the data.

        Parameters
        ----------
        X : list of np.ndarray
            The input data sets for each module.
        y : np.ndarray, optional
            The corresponding labels for supervised learning, by default None.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            The method to reset vigilance if a mismatch occurs, by default "MT+".
        epsilon : float, optional
            A small adjustment factor for match tracking, by default 0.0.

        Returns
        -------
        DeepARTMAP
            The partially fitted DeepARTMAP model.

        """
        self.validate_data(X, y)
        if y is not None:
            if len(self.layers) == 0:
                self.is_supervised = True
                self.layers = [
                    SimpleARTMAP(self.modules[i]) for i in range(self.n_modules)
                ]
            assert self.is_supervised, (
                "Labels were previously provided. "
                "Must continue to provide labels for partial fit."
            )
            self.layers[0] = self.layers[0].partial_fit(
                X[0], y, match_tracking=match_tracking, epsilon=epsilon
            )
            x_i = 1
        else:
            if len(self.layers) == 0:
                self.is_supervised = False
                assert (
                    self.n_modules >= 2
                ), "Must provide at least two ART modules when providing cluster labels"
                self.layers = cast(
                    list[BaseARTMAP], [ARTMAP(self.modules[1], self.modules[0])]
                ) + cast(
                    list[BaseARTMAP],
                    [SimpleARTMAP(self.modules[i]) for i in range(2, self.n_modules)],
                )
            assert not self.is_supervised, (
                "Labels were not previously provided. "
                "Do not provide labels to continue partial fit."
            )

            self.layers[0] = self.layers[0].partial_fit(
                X[1],
                X[0],
                match_tracking=match_tracking,
                epsilon=epsilon,
            )
            x_i = 2

        n_samples = X[0].shape[0]
        for art_i in range(1, self.n_layers):
            y_i = self.layers[art_i - 1].labels_a[-n_samples:]
            self.layers[art_i] = self.layers[art_i].partial_fit(
                X[x_i],
                y_i,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )
            x_i += 1
        return self

    def predict(
        self, X: Union[np.ndarray, list[np.ndarray]], clip: bool = False
    ) -> list[np.ndarray]:
        """Predict the labels for the input data.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            The input data set for prediction.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        list of np.ndarray
            The predicted labels for each layer.

        """
        if isinstance(X, list):
            x = X[-1]
        else:
            x = X
        pred_a, pred_b = self.layers[-1].predict_ab(x, clip=clip)
        pred = [pred_a, pred_b]
        for layer in self.layers[:-1][::-1]:
            pred.append(layer.map_a2b(pred[-1]))

        return pred[::-1]
