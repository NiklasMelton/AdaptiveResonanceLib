"""Fuzzy ART :cite:`carpenter1991fuzzy`."""
# Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991c).
# Fuzzy ART: Fast stable learning and categorization of analog patterns by an
# adaptive resonance system.
# Neural Networks, 4, 759 – 771. doi:10.1016/0893-6080(91)90056-B.

from artlib.elementary.FuzzyART import FuzzyART
from artlib.common.utils import fracsort, fracargmax
from typing import Optional, Callable, Literal, Tuple, Dict, List, Union
import warnings
import numpy as np
from numpy.typing import NDArray
from numba import njit
import operator


@njit
def _and_popcount(i, w):
    s = 0
    for j in range(i.size):
        # bool & bool -> bool, adding a bool increments by 1 when True
        s += i[j] & w[j]
    return s


@njit
def _category_choice_binary(
    i: np.ndarray, w: np.ndarray, pre_MT: bool, rho_int: int
) -> Tuple[int, bool]:
    """Optimized category choice for binary data using count_nonzero."""
    iw_count = _and_popcount(i, w)
    if (not pre_MT) or (iw_count >= rho_int):
        return iw_count, True
    else:
        return iw_count, False


class BinaryFuzzyART(FuzzyART):
    """Fuzzy ART optimized for binary input data."""

    def __init__(self, rho: float):
        """Initialize the Binary Fuzzy ART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.

        """
        super().__init__(rho, alpha=1e-10, beta=1.0)
        self.w_count_cache: List[int] = []

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
        return cc_data.astype(np.bool)  # TODO: convert to bool

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "rho" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert isinstance(params["rho"], float)

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
    ) -> tuple[int, Optional[dict]]:
        """Get the activation of the cluster using optimized binary operations."""
        pre_MT = params["MT"] not in ["MT-"]
        iw_count, mt_status = _category_choice_binary(i, w, pre_MT, params["rho_int"])
        cache = {
            "iw_count": iw_count,  # scalar
            "mt_status": mt_status,  # bool
        }
        return iw_count, cache

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
            iw_count = _and_popcount(i, w)
            cache = {"iw_count": iw_count}
        return cache["iw_count"], cache

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
        op: Callable = operator.ge,
    ) -> Tuple[bool, Dict]:
        """Get the binary match criterion of the cluster.

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
            Binary match criterion and cache used for later processing.

        """
        M, cache = self.match_criterion(i, w, params=params, cache=cache)
        M_bin = op(M, params["rho_int"])
        if cache is None:
            cache = {"iw_count": M}
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin
        return M_bin, cache

    def set_weight(self, idx: int, new_w: np.ndarray, cache: Optional[dict] = None):
        """Set the value of a cluster weight.

        Parameters
        ----------
        idx : int
            Index of cluster to update.
        new_w : np.ndarray
            New cluster weight.
        cache : Optional[dict]
            cache of values created during training step

        """
        self.weight_sample_counter_[idx] += 1
        self.W[idx] = new_w
        if cache is None:
            wc = np.count_nonzero(new_w)
        else:
            wc = cache["iw_count"]
        self.w_count_cache[idx] = wc if wc > 0 else 1

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Get the updated cluster weight using optimized binary operations."""
        return i & w

    def add_weight(self, new_w: np.ndarray):
        """Add a new cluster weight.

        Parameters
        ----------
        new_w : np.ndarray
            New cluster weight to add.

        """
        self.weight_sample_counter_.append(1)
        wc = np.count_nonzero(new_w)
        self.w_count_cache.append(wc if wc > 0 else 1)
        self.W.append(new_w)

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
        assert len(self.W) >= 0, "ART module is not fit."
        self.params["MT"] = None
        T_num, _ = zip(
            *[self.category_choice(x, w, params=self.params) for w in self.W]
        )
        T_num = np.ascontiguousarray(T_num, dtype=np.uint32)
        T_den = np.ascontiguousarray(self.w_count_cache, dtype=np.uint32)

        c_ = int(fracargmax(T_num, T_den))
        return c_

    def _match_tracking_integer(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: int,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Perform match tracking using the specified method.

        Parameters
        ----------
        cache : dict
            Cached match criterion value.
        epsilon : float
            Small adjustment factor for match tracking.
        params : dict
            Parameters
        method : Literal["MT+", "MT-", "MT0", "MT1", "MT~"]
            Match tracking method to apply.

        Returns
        -------
        bool
            Whether to continue searching for a match.

        """
        assert isinstance(cache, dict)
        assert isinstance(params, dict)
        M = cache["match_criterion"]
        if method == "MT+":
            self.params["rho_int"] = M + epsilon
            # return True
        elif method == "MT-":
            self.params["rho_int"] = M - epsilon
            # return True
        elif method == "MT0":
            self.params["rho_int"] = M
            # return True
        elif method == "MT1":
            self.params["rho_int"] = np.inf
            # return False
        elif method == "MT~":
            pass
            # return True
        else:
            raise ValueError(f"Invalid Match Tracking Method: {method}")

        if method == "MT1" or self.params["rho_int"] > self.dim_original:
            return False
        else:
            return True

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
            Epsilon value used for adjusting match criterion. Rounded up to nearest int

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        self.sample_counter_ += 1
        base_params = self._deep_copy_params()
        self.params["MT"] = match_tracking
        self.params["rho_int"] = int(np.ceil(self.params["rho"] * self.dim_original))
        epsilon_int = int(np.ceil(epsilon))
        mt_operator = self._match_tracking_operator(match_tracking)
        if len(self.W) == 0:
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return 0
        else:
            n = len(self.W)
            T_num: NDArray[np.uint32] = np.empty(n, dtype=np.uint32)
            T_den: NDArray[np.uint32] = np.empty(n, dtype=np.uint32)
            T_idx: NDArray[np.int32] = np.empty(n, dtype=np.int32)
            k = 0
            pre_MT = self.params["MT"] not in ["MT-"]
            rho_int = self.params["rho_int"]

            if match_tracking in ["MT~"] and match_reset_func is not None:
                for c_ in range(len(self.W)):
                    w = self.W[c_]
                    t_num, mt_status = _category_choice_binary(x, w, pre_MT, rho_int)
                    if (not mt_status) or (
                        not match_reset_func(x, w, c_, params=self.params, cache=None)
                    ):
                        continue
                    T_num[k] = t_num
                    T_den[k] = self.w_count_cache[c_]
                    T_idx[k] = c_
                    k += 1
            else:
                for c_ in range(len(self.W)):
                    w = self.W[c_]
                    t_num, mt_status = _category_choice_binary(x, w, pre_MT, rho_int)
                    if not mt_status:
                        continue
                    T_num[k] = t_num
                    T_den[k] = self.w_count_cache[c_]
                    T_idx[k] = c_
                    k += 1
            if k:
                T_num = T_num[:k]
                T_den = T_den[:k]
                T_idx = T_idx[:k]
                order = fracsort(T_num, T_den)
            else:
                order = ()

            for t_ in order:
                c_ = T_idx[t_]
                w = self.W[c_]
                cache = {"iw_count": int(T_num[t_])}
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
                    if m and not no_match_reset:
                        keep_searching = self._match_tracking_integer(
                            cache, epsilon_int, self.params, match_tracking
                        )
                        if not keep_searching:
                            break

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            self._set_params(base_params)
            return c_new
