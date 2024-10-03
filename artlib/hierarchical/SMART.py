"""
Bartfai, G. (1994).
Hierarchical clustering with ART neural networks.
In Proc. IEEE International Conference on Neural Networks (ICNN)
(pp. 940–944). volume 2. doi:10.1109/ICNN.1994.374307.
"""

import numpy as np
from typing import Union, Type, Optional, Iterable, Literal
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.hierarchical.DeepARTMAP import DeepARTMAP

class SMART(DeepARTMAP):

    def __init__(self, base_ART_class: Type, rho_values: Union[list[float], np.ndarray], base_params: dict, **kwargs):
        """

        Parameters:
        - base_ART: some ART class
        - rho_values: rho parameters for each sub-module
        - base_params: base param dict for each sub-module

        """

        assert all(np.diff(rho_values) > 0), "rho_values must be monotonically increasing"
        self.rho_values = rho_values

        layer_params = [dict(base_params, **{"rho": rho}) for rho in self.rho_values]
        layers = [base_ART_class(**params, **kwargs) for params in layer_params]
        for layer in layers:
            assert isinstance(layer, BaseART), "Only elementary ART-like objects are supported"
        super().__init__(layers)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            prepared data
        """
        X_, _ = super(SMART, self).prepare_data([X]*self.n_modules)
        return X_[0]

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            restored data
        """
        X_, _ = super(SMART, self).restore_data([X] * self.n_modules)
        return X_[0]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, max_iter=1, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        """
        Fit the model to the data

        Parameters:
        - X: data set A
        - y: not used
        - max_iter: number of iterations to fit the model on the same data set
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """
        X_list = [X]*self.n_modules
        return super().fit(X_list, max_iter=max_iter, match_reset_method=match_reset_method, epsilon=epsilon)

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        X_list = [X] * self.n_modules
        return self.partial_fit(X_list, match_reset_method=match_reset_method, epsilon=epsilon)

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        for j in range(len(self.modules)):
            layer_colors = []
            for k in range(self.modules[j].n_clusters):
                if j == 0:
                    layer_colors.append(colors[k])
                else:
                    layer_colors.append(colors[self.map_deep(j - 1, k)])
            self.modules[j].plot_cluster_bounds(ax, layer_colors, linewidth)


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
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm
            colors = cm.rainbow(np.linspace(0, 1, self.modules[0].n_clusters))

        for k, col in enumerate(colors):
            cluster_data = y == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], color=col, marker=".", s=marker_size)

        self.plot_cluster_bounds(ax, colors, linewidth)