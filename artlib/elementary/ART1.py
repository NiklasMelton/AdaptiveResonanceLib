"""ART1 :cite:`carpenter1987massively`."""

# Carpenter, G. A., & Grossberg, S. (1987a).
# A massively parallel architecture for a self-organizing neural pattern
# recognition machine.
# Computer Vision, Graphics, and Image
# Processing, 37, 54 – 115. doi:10. 1016/S0734-189X(87)80014-2.

import numpy as np
from typing import Optional, List, Tuple, Dict
from artlib.common.BaseART import BaseART
from numba import njit


@njit
def match_criterion_numba(i, w_td, dim_):
    """Compute the match criterion for ART1 clustering.

    This function calculates the proportion of active features in the input `i`
    that match the corresponding features in the top-down weight vector `w_td`.

    Parameters
    ----------
    i : np.ndarray (int32 or uint8)
        Binary or integer input vector representing the data sample.
    w_td : np.ndarray (uint8)
        Binary top-down weight vector of the cluster.
    dim_ : int
        The number of features (dimensions) in the input vector.

    Returns
    -------
    float
        The match criterion value, computed as the ratio of matching features
        to the total number of features.

    """
    count = 0
    for j in range(dim_):
        if (i[j] != 0) & (
            w_td[j] != 0
        ):  # Ensure binary logic while allowing integer input
            count += 1
    return count / dim_


@njit
def category_choice_numba(i, w_bu):
    """Compute the category choice activation for ART1.

    This function calculates the category choice value, which represents the
    strength of association between an input `i` and a bottom-up weight vector `w_bu`.

    Parameters
    ----------
    i : np.ndarray (int32 or uint8)
        Binary or integer input vector representing the data sample.
    w_bu : np.ndarray (float32)
        Bottom-up weight vector of the cluster.

    Returns
    -------
    float
        The activation value, computed as the sum of element-wise multiplications.

    """
    activation = 0.0
    for j in range(len(i)):
        activation += (
            i[j] * w_bu[j]
        )  # Supports integer input and floating-point weights
    return activation


@njit
def update_numba(i, w, L, dim_):
    """Optimized update function for ART1 using Numba.

    This function updates the cluster weight vector based on the input `i`,
    using ART1 learning rules.

    Parameters
    ----------
    i : np.ndarray (int32 or uint8)
        Binary or integer input vector.
    w : np.ndarray (float32)
        Cluster weight vector, containing both bottom-up and top-down weights.
    L : float
        Uncommitted node bias parameter.
    dim_ : int
        Feature dimension.

    Returns
    -------
    np.ndarray
        Updated cluster weight vector.

    """
    # Extract the top-down weights (last dim_ elements)
    w_td_new = np.empty(dim_, dtype=np.uint8)
    count = 0

    # Compute the new top-down weight using bitwise AND (but allowing integer input)
    for j in range(dim_):
        w_td_new[j] = (i[j] != 0) & (w[j + dim_] != 0)  # Ensures binary logic
        if w_td_new[j]:  # Count nonzero elements
            count += 1

    # Compute the new bottom-up weights
    scaling_factor = L / (L - 1 + count)
    w_bu_new = np.empty(dim_, dtype=np.float32)

    for j in range(dim_):
        w_bu_new[j] = scaling_factor * w_td_new[j]  # Multiplication remains float32

    # Concatenate updated weights
    updated_weights = np.empty(2 * dim_, dtype=np.float32)
    updated_weights[:dim_] = w_bu_new
    updated_weights[dim_:] = w_td_new

    return updated_weights


class ART1(BaseART):
    """ART1 for Binary Clustering.

    This module implements ART1 as first published in: :cite:`carpenter1987massively`.


    .. # Carpenter, G. A., & Grossberg, S. (1987a).
    .. # A massively parallel architecture for a self-organizing neural pattern
    .. # recognition machine.
    .. # Computer Vision, Graphics, and Image
    .. # Processing, 37, 54 – 115. doi:10. 1016/S0734-189X(87)80014-2.

    ART1 is exclusively for clustering binary data.

    """

    def __init__(self, rho: float, L: float):
        """Initialize the ART1 model.

        Parameters
        ----------
        rho : float
            Vigilance parameter in the range [0, 1].
        L : float
            Uncommitted node bias, a value greater than or equal to 1.

        """
        params = {"rho": rho, "L": L}
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
        assert "L" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert params["L"] >= 1.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["L"], float)

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        assert X.dtype == np.bool_ or np.issubdtype(
            X.dtype, np.integer
        ), "ART1 only supports binary data"
        assert ((X == 0) | (X == 1)).all(), "ART1 only supports binary data"
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
        # # In practice, numba seems to be slower for this function despite what
        # # profiling would indicate
        # return category_choice_numba(i, w_bu), None

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
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
        float
            Cluster match criterion.
        dict
            Cache used for later processing.

        """
        w_td = w[self.dim_ :]
        # return np.count_nonzero(i & w_td) / self.dim_, cache
        return match_criterion_numba(i, w_td, self.dim_), cache

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
        # w_td = w[self.dim_ :]
        # w_td_new = i & w_td
        # w_bu_new = (
        #     params["L"] / (params["L"] - 1 + np.count_nonzero(w_td_new))
        # ) * w_td_new
        #
        # return np.concatenate([w_bu_new, w_td_new])
        return update_numba(i, w, params["L"], self.dim_)

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
        scaling_factor_ = params["L"] / (params["L"] - 1 + self.dim_)
        w_bu_new = scaling_factor_ * i
        return np.concatenate([w_bu_new, w_td_new])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return [w[self.dim_ :] for w in self.W]
