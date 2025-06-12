"""Fuzzy ART :cite:`carpenter1991fuzzy`."""
# Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
# Fuzzy ART: Fast stable learning and categorization of analog patterns by an
# adaptive resonance system.
# Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.

from artlib.elementary.FuzzyART import FuzzyART
from typing import Optional, Callable, Literal, Tuple, Dict
import warnings
import numpy as np
from numba import njit


@njit
def _category_choice_binary(
    i: np.ndarray, w: np.ndarray, alpha: float, pre_MT: bool, rho_w1: int
) -> Tuple[float, int]:
    """Optimized category choice for binary data using count_nonzero."""
    w1 = np.count_nonzero(i & w)
    if not pre_MT or w1 >= rho_w1:
        return w1 / (alpha + np.count_nonzero(w)), w1
    else:
        return np.nan, w1


@njit
def _match_criterion_binary(i: np.ndarray, w: np.ndarray, dim_original: float) -> float:
    """Optimized match criterion for binary data using count_nonzero."""
    return np.count_nonzero(i & w) / dim_original


@njit
def _update_binary(i: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Optimized update rule for binary data."""
    return i & w  # Using bitwise AND for binary updates


class BinaryFuzzyART(FuzzyART):
    """Fuzzy ART optimized for binary input data."""

    def __init__(self, rho: float, alpha: float):
        """Initialize the Binary Fuzzy ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.

        """
        super().__init__(rho, alpha, beta=1.0)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        Returns
        -------
        np.ndarray
            Normalized and complement coded data.

        """
        cc_data = super().prepare_data(X)
        return cc_data.astype(np.uint8)  # TODO: convert to bool

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
        assert 1.0 >= params["rho"] >= 0.0
        assert params["alpha"] >= 0.0
        assert isinstance(params["rho"], float)
        assert isinstance(params["alpha"], float)

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset.

        """
        assert X.shape[1] % 2 == 0, "Data has not been complement coded"
        assert X.dtype == np.bool_ or np.issubdtype(
            X.dtype, np.integer
        ), "Binary Fuzzy ART only supports binary data"
        assert ((X == 0) | (X == 1)).all(), "Binary Fuzzy ART only supports binary data"
        assert np.all(
            abs(np.sum(X, axis=1) - float(X.shape[1] / 2)) <= 0.01
        ), "Data has not been complement coded"
        self.check_dimensions(X)

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        """Get the activation of the cluster using optimized binary operations."""
        pre_MT = params["MT"] not in [None, "MT-"]
        T, w1 = _category_choice_binary(i, w, params["alpha"], pre_MT, params["rho_w1"])
        return T, {"w1": w1}

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion using optimized binary operations."""
        if cache is None:
            warnings.warn(
                "Cache is None during Match Criterion. This will reduce performance"
            )
            w1 = np.count_nonzero(i & w)
            cache = {"w1": w1}
        return cache["w1"] / self.dim_original, cache

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Get the updated cluster weight using optimized binary operations."""
        return _update_binary(i, w)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """Generate a new cluster weight."""
        return i

    def step_pred(self, x) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        self.params["MT"] = None
        return super().step_pred(x)

    def step_fit(
        self,
        x: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ) -> int:
        """Fit the model to a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        self.sample_counter_ += 1
        base_params = self._deep_copy_params()
        self.params["MT"] = match_tracking
        self.params["rho_w1"] = int(self.params["rho"] * self.dim_original)
        mt_operator = self._match_tracking_operator(match_tracking)
        if len(self.W) == 0:
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return 0
        else:
            if match_tracking in ["MT~"] and match_reset_func is not None:
                T_values, T_cache = zip(
                    *[
                        self.category_choice(x, w, params=self.params)
                        if match_reset_func(x, w, c_, params=self.params, cache=None)
                        else (np.nan, None)
                        for c_, w in enumerate(self.W)
                    ]
                )
            else:
                T_values, T_cache = zip(
                    *[self.category_choice(x, w, params=self.params) for w in self.W]
                )
            T = np.array(T_values)
            while any(~np.isnan(T)):
                c_ = int(np.nanargmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m, cache = self.match_criterion_bin(
                    x, w, params=self.params, cache=cache, op=mt_operator
                )
                if match_tracking in ["MT~"] and match_reset_func is not None:
                    no_match_reset = True
                else:
                    no_match_reset = match_reset_func is None or match_reset_func(
                        x, w, c_, params=self.params, cache=cache
                    )
                if m and no_match_reset:
                    self.set_weight(c_, self.update(x, w, self.params, cache=cache))
                    self._set_params(base_params)
                    return c_
                else:
                    T[c_] = np.nan
                    if m and not no_match_reset:
                        keep_searching = self._match_tracking(
                            cache, epsilon, self.params, match_tracking
                        )
                        if not keep_searching:
                            T[:] = np.nan
                        else:
                            self.params["rho_w1"] = int(
                                self.params["rho"] * self.dim_original
                            )

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            self._set_params(base_params)
            return c_new
