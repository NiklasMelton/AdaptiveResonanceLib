"""ART2 :cite:`carpenter1987art`, :cite:`carpenter1991art`.

::

    ==================================================================
    DISCLAIMER: DO NOT USE ART2!!!
    IT DOES NOT WORK
    It is provided for completeness only.
    Stephan Grossberg himself has said ART2 does not work.
    ==================================================================

"""

# Carpenter, G. A., & Grossberg, S. (1987b).
# ART 2: self-organization of stable category recognition codes for analog input
# patterns.
# Appl. Opt., 26, 4919–4930. doi:10.1364/AO.26.004919.

# Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991b).
# ART 2-A: An adaptive resonance algorithm for rapid category learning and
# recognition.
# Neural Networks, 4, 493 – 504. doi:10.1016/0893-6080(91) 90045-7.

import numpy as np
from typing import Optional, List
from warnings import warn
from artlib.common.BaseART import BaseART


class ART2A(BaseART):
    """ART2-A for Clustering.

    This module implements ART2-A as first published in:
    :cite:`carpenter1987art`, :cite:`carpenter1991art`


    .. # Carpenter, G. A., & Grossberg, S. (1987b).
    .. # ART 2: self-organization of stable category recognition codes for analog input
    .. # patterns.
    .. # Appl. Opt., 26, 4919–4930. doi:10.1364/AO.26.004919.

    .. # Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991b).
    .. # ART 2-A: An adaptive resonance algorithm for rapid category learning and
    .. # recognition.
    .. # Neural Networks, 4, 493 – 504. doi:10.1016/0893-6080(91) 90045-7.


    ART2-A is similar to :class:`~artlib.elementary.ART1.ART1` but designed for analog
    data. This method is implemented for historical purposes and is not recommended
    for use.

    """

    def __init__(self, rho: float, alpha: float, beta: float):
        """Initialize the ART2-A model.

        Parameters
        ----------
        rho : float
            Vigilance parameter in the range [0, 1].
        alpha : float
            Choice parameter, recommended value is 1e-7.
        beta : float
            Learning parameter in the range [0, 1]. A value of 1 is recommended for
            fast learning.

        """
        warn("Do Not Use ART2. It does not work. Module provided for completeness only")

        params = {
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }
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
        assert "alpha" in params
        assert "beta" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert 1.0 >= params["alpha"] >= 0.0
        assert 1.0 >= params["beta"] >= 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)

    def check_dimensions(self, X: np.ndarray):
        """Check that the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
            assert self.params["alpha"] <= 1 / np.sqrt(self.dim_)
        else:
            assert X.shape[1] == self.dim_

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
        activation = float(np.dot(i, w))
        cache = {"activation": activation}
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> tuple[float, Optional[dict]]:
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
        if cache is None:
            raise ValueError("No cache provided")
        # TODO: make this more efficient
        M = cache["activation"]
        M_u = params["alpha"] * np.sum(i)
        # suppress if uncommitted activation is higher
        if M < M_u:
            return -1.0, cache
        else:
            return M, cache

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
        return params["beta"] * i + (1 - params["beta"]) * w

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
        return i

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return self.W
