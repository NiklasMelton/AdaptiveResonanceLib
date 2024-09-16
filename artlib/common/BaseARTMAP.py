import numpy as np
from typing import Union, Optional, Iterable, Literal
from collections import defaultdict
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin

class BaseARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):

    def __init__(self):
        self.map: dict[int, int] = dict()

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

    def map_a2b(self, y_a: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        map an a-side label to a b-side label

        Parameters:
        - y_a: side a label(s)

        Returns:
            side B cluster label(s)

        """
        if isinstance(y_a, int):
            return self.map[y_a]
        u, inv = np.unique(y_a, return_inverse=True)
        return np.array([self.map[x] for x in u], dtype=int)[inv].reshape(y_a.shape)

    def validate_data(self, X: np.ndarray, y: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set A
        - y: data set B

        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 1e-10):
        """
        Fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B
        - max_iter: number of iterations to fit the model on the same data set

        """
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, y: np.ndarray, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 1e-10):
        """
        Partial fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B

        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict labels for the data

        Parameters:
        - X: data set A

        Returns:
            B labels for the data

        """
        raise NotImplementedError

    def predict_ab(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        predict labels for the data, both A-side and B-side

        Parameters:
        - X: data set A

        Returns:
            A labels for the data, B labels for the data

        """
        raise NotImplementedError

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
        raise NotImplementedError