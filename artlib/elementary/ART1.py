"""ART1.

.. # Carpenter, G. A., & Grossberg, S. (1987a).
.. # A massively parallel architecture for a self-organizing neural pattern
.. # recognition machine.
.. # Computer Vision, Graphics, and Image
.. # Processing, 37, 54 – 115. doi:10. 1016/S0734-189X(87)80014-2.

.. bibliography::
   :filter: citation_key == "carpenter1987massively"
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict
from artlib.common.BaseART import BaseART
from artlib.common.utils import l1norm


class ART1(BaseART):
    """ART1 for Clustering.

    This module implements ART1 as first published in Carpenter, G. A., & Grossberg, S.
    (1987a). A massively parallel architecture for a self-organizing neural pattern
    recognition machine. Computer Vision, Graphics, and Image Processing, 37, 54 – 115.
    doi:10. 1016/S0734-189X(87)80014-2. ART1 is intended for binary data clustering
    only.

    """

    def __init__(self, rho: float, beta: float, L: float):
        """Initialize the ART1 model.

        Parameters
        ----------
        rho : float
            Vigilance parameter in the range [0, 1].
        beta : float
            Learning parameter in the range [0, 1]. A value of 1 is recommended for fast
            learning.
        L : float
            Uncommitted node bias, a value greater than or equal to 1.

        """
        params = {"rho": rho, "beta": beta, "L": L}
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert "beta" in params
        assert "L" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert 1.0 >= params["beta"] >= 0.0
        assert params["L"] >= 1.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["L"], float)

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        assert np.array_equal(X, X.astype(bool)), "ART1 only supports binary data"
        self.check_dimensions(X)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
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
        float
            Cluster activation.
        dict, optional
            Cache used for later processing.

        """
        w_bu = w[: self.dim_]
        return float(np.dot(i, w_bu)), None

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[Union[float, List[float]], Optional[Dict]]:
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
        float
            Cluster match criterion.
        dict
            Cache used for later processing.

        """
        w_td = w[self.dim_ :]
        return l1norm(np.logical_and(i, w_td)) / l1norm(i), cache

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
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
        w_td = w[self.dim_ :]

        w_td_new = np.logical_and(i, w_td)
        w_bu_new = (params["L"] / (params["L"] - 1 + l1norm(w_td_new))) * w_td_new
        return np.concatenate([w_bu_new, w_td_new])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
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
        w_td_new = i
        w_bu_new = (params["L"] / (params["L"] - 1 + self.dim_)) * w_td_new
        return np.concatenate([w_bu_new, w_td_new])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return [w[self.dim_ :] for w in self.W]
