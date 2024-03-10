import numpy as np
from typing import Union, Optional, Iterable
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin

class BaseARTMAP(BaseEstimator, ClassifierMixin, ClusterMixin):
    map: dict[int, int]

    def map_a2b(self, y_a: Union[np.ndarray, int]) -> np.ndarray:
        if isinstance(y_a, int):
            return self.map[y_a]
        u, inv = np.unique(y_a, return_inverse=True)
        return np.array([self.map[x] for x in u], dtype=int)[inv].reshape(y_a.shape)

    def validate_data(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        raise NotImplementedError