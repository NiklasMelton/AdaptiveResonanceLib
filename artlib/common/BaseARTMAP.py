"""Base class for all ARTMAP objects."""
import numpy as np
from typing import Union, Optional, Literal
from collections import defaultdict
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from artlib.common.utils import IndexableOrKeyable


class BaseARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):
    """Generic implementation of Adaptive Resonance Theory MAP (ARTMAP)"""

    def __init__(self):
        """Instantiate the BaseARTMAP object."""
        self.map: dict[int, int] = dict()

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Specific redefinition of `sklearn.BaseEstimator.set_params` for ARTMAP classes.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        local_params = dict(valid_params)

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

    def map_a2b(self, y_a: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """Map an a-side label to a b-side label.

        Parameters
        ----------
        y_a : Union[np.ndarray, int]
            Side A label(s).

        Returns
        -------
        Union[np.ndarray, int]
            Side B cluster label(s).

        """
        if isinstance(y_a, int):
            return self.map[y_a]
        u, inv = np.unique(y_a, return_inverse=True)
        return np.array([self.map[x] for x in u], dtype=int)[inv].reshape(y_a.shape)

    def validate_data(self, X: np.ndarray, y: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset A.
        y : np.ndarray
            Dataset B.

        """
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Dataset A.
        y : np.ndarray
            Dataset B.
        max_iter : int, optional
            Number of iterations to fit the model on the same dataset.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting match criterion.
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 1e-10.

        """
        raise NotImplementedError

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        """Partial fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Dataset A.
        y : np.ndarray
            Dataset B.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting match criterion.
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 1e-10.

        """
        raise NotImplementedError

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Predict labels for the data.

        Parameters
        ----------
        X : np.ndarray
            Dataset A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        np.ndarray
            B-side labels for the data.

        """
        raise NotImplementedError

    def predict_ab(
        self, X: np.ndarray, clip: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels for the data, both A-side and B-side.

        Parameters
        ----------
        X : np.ndarray
            Dataset A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A-side labels for the data, B-side labels for the data.

        """
        raise NotImplementedError

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Visualize the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        raise NotImplementedError

    def visualize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ax: Optional[Axes] = None,
        marker_size: int = 10,
        linewidth: int = 1,
        colors: Optional[IndexableOrKeyable] = None,
    ):
        """Visualize the clustering of the data.

        Parameters
        ----------
        X : np.ndarray
            Dataset.
        y : np.ndarray
            Sample labels.
        ax : matplotlib.axes.Axes, optional
            Figure axes, by default None.
        marker_size : int, optional
            Size used for data points, by default 10.
        linewidth : int, optional
            Width of boundary line, by default 1.
        colors : IndexableOrKeyable, optional
            Colors to use for each cluster, by default None.

        """
        raise NotImplementedError
