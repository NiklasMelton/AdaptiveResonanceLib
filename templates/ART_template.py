"""Template for custom ART Modules.

This template contains the minimum methods necessary to define for a custom ART
module. Additional functions may be defined as well depending on how customized the
behavior of the model is. See :class:`~artlib.common.BaseART.BaseART` for the
complete set of pre-defined methods.

"""
import numpy as np
from matplotlib.axes import Axes
from typing import Optional, List, Tuple, Dict

from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable


class MyART(BaseART):
    """Generic Template for custom ART modules."""

    def __init__(self, rho: float):
        """Initializes the ART object.

        Parameters
        ----------
        rho : float
            Vigilance parameter.

        """
        # add additional parameters as needed
        params = {"rho": rho}
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """Validates clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert params["rho"] >= 0.0
        assert isinstance(params["rho"], float)
        # check additional parameters if you add them

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: Dict
    ) -> Tuple[float, Optional[Dict]]:
        """Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        tuple
            Cluster activation and cache used for later processing.

        """
        # implement your custom logic here
        raise NotImplementedError

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values from previous calculations.

        Returns
        -------
        tuple
            Cluster match criterion and cache used for later processing.

        """
        # implement your custom logic here
        raise NotImplementedError

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
    ) -> np.ndarray:
        """Get the updated cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Dictionary containing parameters for the algorithm.
        cache : dict, optional
            Cache containing values from previous calculations.

        Returns
        -------
        np.ndarray
            Updated cluster weight.

        """
        # implement your custom logic here
        raise NotImplementedError

    def new_weight(self, i: np.ndarray, params: Dict) -> np.ndarray:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        np.ndarray
            Updated cluster weight.

        """
        # implement your custom logic here
        raise NotImplementedError

    # ==================================================================================
    # These functions are not strictly necessary but should be defined if possible
    # ==================================================================================
    def get_cluster_centers(self) -> List[np.ndarray]:
        """Undefined function for getting centers of each cluster. Used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        # implement your custom logic here
        raise NotImplementedError

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Undefined function for visualizing the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, default=1
            Width of boundary line.

        """
        # implement your custom logic here
        raise NotImplementedError
