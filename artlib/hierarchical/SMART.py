"""SMART :cite:`bartfai1994hierarchical`."""
# Bartfai, G. (1994).
# Hierarchical clustering with ART neural networks.
# In Proc. IEEE International Conference on Neural Networks (ICNN)
# (pp. 940–944). volume 2.
# doi:10.1109/ICNN.1994.374307.

import numpy as np
from typing import Union, Type, Optional, Literal, Tuple
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable
from artlib.hierarchical.DeepARTMAP import DeepARTMAP


class SMART(DeepARTMAP):
    """SMART for Hierachical Clustering.

    This module implements SMART as first published in: :cite:`bartfai1994hierarchical`

    .. # Bartfai, G. (1994).
    .. # Hierarchical clustering with ART neural networks.
    .. # In Proc. IEEE International Conference on Neural Networks (ICNN)
    .. # (pp. 940–944). volume 2.
    .. # doi:10.1109/ICNN.1994.374307.

    SMART accepts an uninstantiated :class:`~artlib.common.BaseART.BaseART` class and
    hierarchically clusters data in a divisive fashion by using a set of vigilance
    values that monotonically increase in their restrictiveness. SMART is a special
    case of :class:`~artlib.hierarchical.DeepARTMAP.DeepARTMAP`, which forms the
    backbone of this class, where all channels receive the same data.

    """

    def __init__(
        self,
        base_ART_class: Type,
        rho_values: Union[list[float], np.ndarray],
        base_params: dict,
        **kwargs
    ):
        """Initialize the SMART model.

        Parameters
        ----------
        base_ART_class : Type
            Some ART class to instantiate the layers.
        rho_values : list of float or np.ndarray
            The vigilance parameter values for each layer, must be monotonically
            increasing for most ART modules.
        base_params : dict
            Parameters for the base ART module, used to instantiate each layer.
        **kwargs :
            Additional keyword arguments for ART module initialization.

        """
        if base_ART_class.__name__ != "BayesianART":
            assert all(
                np.diff(rho_values) > 0
            ), "rho_values must be monotonically increasing"
        else:
            assert all(
                np.diff(rho_values) < 0
            ), "rho_values must be monotonically decreasing for BayesianART"
        self.rho_values = rho_values

        layer_params = [dict(base_params, **{"rho": rho}) for rho in self.rho_values]
        modules = [base_ART_class(**params, **kwargs) for params in layer_params]
        for module in modules:
            assert isinstance(
                module, BaseART
            ), "Only elementary ART-like objects are supported"
        super().__init__(modules)

    def prepare_data(
        self, X: Union[np.ndarray, list[np.ndarray]], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[list[np.ndarray], Optional[np.ndarray]]]:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset to prepare.

        Returns
        -------
        np.ndarray
            Prepared data.

        """
        X_, _ = super(SMART, self).prepare_data([X] * self.n_modules)
        return X_[0]

    def restore_data(
        self, X: Union[np.ndarray, list[np.ndarray]], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[list[np.ndarray], Optional[np.ndarray]]]:
        """Restore data to its original form before preparation.

        Parameters
        ----------
        X : np.ndarray
            The dataset to restore.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        X_, _ = super(SMART, self).restore_data([X] * self.n_modules)
        return X_[0]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Fit the SMART model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset to fit the model on.
        y : np.ndarray, optional
            Not used, present for compatibility.
        max_iter : int, optional
            The number of iterations to run the model on the data.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            The match reset method to use when adjusting vigilance.
        epsilon : float, optional
            A small value to adjust vigilance during match tracking.

        Returns
        -------
        SMART
            Fitted SMART model.

        """
        X_list = [X] * self.n_modules
        return super().fit(
            X_list,
            max_iter=max_iter,
            match_tracking=match_tracking,
            epsilon=epsilon,
        )

    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Partial fit the SMART model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset to partially fit the model on.
        y : np.ndarray, optional
            Not used, present for compatibility.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            The match reset method to use when adjusting vigilance.
        epsilon : float, optional
            A small value to adjust vigilance during match tracking.

        Returns
        -------
        SMART
            Partially fitted SMART model.

        """
        X_list = [X] * self.n_modules
        return super(SMART, self).partial_fit(
            X_list, match_tracking=match_tracking, epsilon=epsilon
        )

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Visualize the cluster boundaries.

        Parameters
        ----------
        ax : Axes
            The matplotlib axes on which to plot the cluster boundaries.
        colors : IndexableOrKeyable
            The colors to use for each cluster.
        linewidth : int, optional
            The width of the boundary lines.

        Returns
        -------
        None

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
        colors: Optional[IndexableOrKeyable] = None,
    ):
        """Visualize the clustering of the data with cluster boundaries.

        Parameters
        ----------
        X : np.ndarray
            The dataset to visualize.
        y : np.ndarray
            The cluster labels for the data points.
        ax : Axes, optional
            The matplotlib axes on which to plot the visualization.
        marker_size : int, optional
            The size of the data points in the plot.
        linewidth : int, optional
            The width of the cluster boundary lines.
        colors : IndexableOrKeyable, optional
            The colors to use for each cluster.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, self.modules[0].n_clusters))

        for k, col in enumerate(colors):
            cluster_data = y == k
            plt.scatter(
                X[cluster_data, 0],
                X[cluster_data, 1],
                color=col,
                marker=".",
                s=marker_size,
            )

        self.plot_cluster_bounds(ax, colors, linewidth)
