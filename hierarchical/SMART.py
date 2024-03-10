import numpy as np
from typing import Union, Type, Optional, Iterable
from matplotlib.axes import Axes
from common.BaseART import BaseART
from hierarchical.DeepARTMAP import DeepARTMAP

class SMART(DeepARTMAP):

    def __init__(self, base_ART_class: Type, rho_values: Union[list[float], np.ndarray], base_params: dict, **kwargs):

        assert all(np.diff(rho_values) > 0), "rho_values must be monotonically increasing"
        self.rho_values = rho_values

        layer_params = [dict(base_params, **{"rho": rho}) for rho in self.rho_values]
        layers = [base_ART_class(params, **kwargs) for params in layer_params]
        for layer in layers:
            assert isinstance(layer, BaseART), "Only elementary ART-like objects are supported"
        super().__init__(layers)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, max_iter=1):
        X_list = [X]*self.n_modules
        return super().fit(X_list, max_iter=max_iter)

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_list = [X] * self.n_modules
        return self.partial_fit(X_list)

    def visualize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            ax: Optional[Axes] = None,
            marker_size: int = 10,
            linewidth: int = 1,
            colors: Optional[Iterable] = None
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm
            colors = cm.rainbow(np.linspace(0, 1, self.modules[0].n_clusters))

        for k, col in enumerate(colors):
            cluster_data = y == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], color=col, marker=".", s=marker_size)

        for j in range(len(self.modules)):
            layer_colors = []
            for k in range(self.modules[j].n_clusters):
                if j == 0:
                    layer_colors.append(colors[k])
                else:
                    layer_colors.append(colors[self.map_deep(j - 1, k)])
            self.modules[j].plot_bounding_boxes(ax, layer_colors, linewidth)
