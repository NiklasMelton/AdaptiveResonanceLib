"""Base class for all ART objects."""
import numpy as np
from typing import (
    TYPE_CHECKING,
    Optional,
    Callable,
    Literal,
    List,
    Tuple,
    Union,
    Dict,
    Sequence,
)
from collections import defaultdict
from warnings import warn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from artlib.common.utils import normalize, de_normalize, IndexableOrKeyable
import operator

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class LazyCaches(Sequence):
    """Per-cluster cache dicts built on demand from arrays of scores.

    Batch scoring computes every cluster's cache values at once, but only the few
    clusters that are actually visited need a dict.

    ``match``, when given, holds each cluster's :meth:`BaseART.match_criterion`
    value, which lets the search skip clusters that fail vigilance.

    """

    __slots__ = ("_columns", "match")

    def __init__(
        self, columns: Dict[str, np.ndarray], match: Optional[np.ndarray] = None
    ):
        """Wrap per-cluster cache arrays, keyed by cache name."""
        self._columns = columns
        self.match = match

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def __getitem__(self, index):
        return {key: values[index] for key, values in self._columns.items()}


class _NoCaches(Sequence):
    """A sequence of ``n`` empty (None) caches, with an optional ``match`` array (see
    :class:`LazyCaches`)."""

    __slots__ = ("_n", "match")

    def __init__(self, n: int, match: Optional[np.ndarray] = None):
        self._n = n
        self.match = match

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index):
        if not -self._n <= index < self._n:
            raise IndexError(index)
        return None


def stacked_weights(model, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``model.W`` as one 2-D array, reusing rows that have not changed.

    The rows are cached and refreshed by identity: a weight that was replaced (as
    every artlib module does on update) is copied in again, and new weights are
    appended. A weight array edited in place is not detected.

    Returns
    -------
    tuple
        The stacked weights, and the numpy dtype ``char`` of each original weight
        (some kernels must reproduce float32 arithmetic for float32 weights).

    """
    W = model.W
    n = len(W)
    key = "_stacked_weights_cache_"
    cached = model.__dict__.get(key)
    if cached is not None and cached[0] is W and cached[2].dtype == dtype:
        _, refs, M, kinds = cached
        m = len(refs)
        width = np.size(W[0]) if n else M.shape[1]
        if m <= n and M.shape[1] == width:
            if not all(map(operator.is_, refs, W)):
                for c in [c for c in range(m) if refs[c] is not W[c]]:
                    M[c] = W[c]
                    kinds[c] = np.asarray(W[c]).dtype.char
                    refs[c] = W[c]
            if n > m:
                if n > M.shape[0]:
                    size = max(n, 2 * M.shape[0])
                    grown = np.empty((size, width), dtype=dtype)
                    grown[:m] = M[:m]
                    grown_kinds = np.empty(size, dtype="U1")
                    grown_kinds[:m] = kinds[:m]
                    M, kinds = grown, grown_kinds
                for c in range(m, n):
                    M[c] = W[c]
                    kinds[c] = np.asarray(W[c]).dtype.char
                refs.extend(W[m:n])
                model.__dict__[key] = (W, refs, M, kinds)
            return M[:n], kinds[:n]
    width = np.size(W[0]) if n else 0
    M = np.empty((n, width), dtype=dtype)
    kinds = np.empty(n, dtype="U1")
    for c in range(n):
        M[c] = W[c]
        kinds[c] = np.asarray(W[c]).dtype.char
    model.__dict__[key] = (W, list(W), M, kinds)
    return M, kinds


class BaseART(BaseEstimator, ClusterMixin):
    """Generic implementation of Adaptive Resonance Theory (ART)"""

    data_format = "default"

    def __init__(self, params: Dict):
        """
        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        self.validate_params(params)
        self.params = params
        self.sample_counter_ = 0
        self.weight_sample_counter_: List[int] = []
        self.d_min_ = None
        self.d_max_ = None
        self.is_fitted_ = False
        self.labels_ = np.zeros((0,), dtype=int)

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        else:
            # If the key is not in params, raise an AttributeError
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        if key in self.__dict__.get("params", {}):
            # If key is in params, set its value
            self.params[key] = value
        else:
            # Otherwise, proceed with normal attribute setting
            super().__setattr__(key, value)

    def __getstate__(self):
        # The stacked-weight cache is rebuilt on demand; do not pickle a copy.
        state = dict(super().__getstate__())
        state.pop("_stacked_weights_cache_", None)
        return state

    def get_params(self, deep: bool = True) -> Dict:
        """
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this class and contained subobjects
            that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return self.params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Specific redefinition of `sklearn.BaseEstimator.set_params` for ART classes.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        local_params = dict(valid_params)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = list(valid_params.keys())
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
                local_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        self.validate_params(local_params)
        return self

    def set_data_bounds(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """Manually set the data bounds for normalization.

        Parameters
        ----------
        lower_bounds : np.ndarray
            The lower bounds for each column.

        upper_bounds : np.ndarray
            The upper bounds for each column.

        """
        if self.is_fitted_:
            raise ValueError("Cannot change data limits after fit.")
        self.d_min_ = lower_bounds
        self.d_max_ = upper_bounds

    def find_data_bounds(
        self, *data_batches: list[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Automatically find the data bounds for normalization from a list of data
        batches.

        Parameters
        ----------
        *data_batches : list[np.ndarray]
            Batches of data to be presented to the model

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for data.

        """
        all_data = np.vstack(data_batches)
        lower_bounds = np.min(all_data, axis=0)
        upper_bounds = np.max(all_data, axis=0)

        return lower_bounds, upper_bounds

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        Returns
        -------
        np.ndarray
            Normalized data.

        """
        normalized, self.d_max_, self.d_min_ = normalize(X, self.d_max_, self.d_min_)
        return normalized

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        return de_normalize(X, d_max=self.d_max_, d_min=self.d_min_)

    @property
    def n_clusters(self) -> int:
        """Get the current number of clusters.

        Returns
        -------
        int
            The number of clusters.

        """
        if hasattr(self, "W"):
            return len(self.W)
        else:
            return 0

    @staticmethod
    def validate_params(params: Dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        raise NotImplementedError

    def check_dimensions(self, X: np.ndarray):
        """Check the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
        else:
            assert X.shape[1] == self.dim_

    def validate_data(self, X: np.ndarray):
        """Validates the data prior to clustering.

        Parameters:
        - X: data set

        """
        normalization_message = (
            "Data has not been normalized or was not normalized "
            "correctly. All values must fall between 0 and 1, "
            "inclusively."
        )
        if self.is_fitted_:
            normalization_message += (
                "\nThis appears to not be the first batch of "
                "data. Data boundaries must be calculated for "
                "the entire data space. Prior to fitting, use "
                "BaseART.set_data_bounds() to manually set the "
                "bounds for your data or use "
                "BaseART.find_data_bounds() to identify the "
                "bounds automatically for multiple batches."
            )
        assert np.all(X >= 0), normalization_message
        assert np.all(X <= 1.0), normalization_message
        self.check_dimensions(X)

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
        raise NotImplementedError

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
        M_bin = op(M, params["rho"])
        if cache is None:
            cache = dict()
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin
        return M_bin, cache

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
        raise NotImplementedError

    def merge(self, target_idx: int, source_idx: int) -> int:
        """Merge a source cluster into a target cluster in place.

        Parameters
        ----------
        target_idx : int
            Index of the cluster to keep.
        source_idx : int
            Index of the cluster to remove.

        Returns
        -------
        int
            Index of the target cluster after removing the source.

        """
        raise NotImplementedError

    def _validate_merge_indices(self, target_idx: int, source_idx: int):
        """Validate indices before calculating or applying a cluster merge."""
        n_clusters = len(self.W)
        for idx in (target_idx, source_idx):
            if not isinstance(idx, (int, np.integer)) or isinstance(
                idx, (bool, np.bool_)
            ):
                raise TypeError("Cluster indices must be integers")
            if not 0 <= idx < n_clusters:
                raise IndexError(f"Cluster index {idx} is out of range")
        if target_idx == source_idx:
            raise ValueError("Cannot merge a cluster with itself")

    def _apply_merge(self, target_idx: int, source_idx: int, new_w: np.ndarray) -> int:
        """Replace two categories with one and update their shared bookkeeping."""
        self.W[target_idx] = new_w
        self.weight_sample_counter_[target_idx] += self.weight_sample_counter_[
            source_idx
        ]
        del self.W[source_idx]
        del self.weight_sample_counter_[source_idx]

        self.labels_[self.labels_ == source_idx] = target_idx
        self.labels_[self.labels_ > source_idx] -= 1
        return target_idx - (source_idx < target_idx)

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
        raise NotImplementedError

    def add_weight(self, new_w: np.ndarray):
        """Add a new cluster weight.

        Parameters
        ----------
        new_w : np.ndarray
            New cluster weight to add.

        """
        self.weight_sample_counter_.append(1)
        self.W.append(new_w)

    def set_weight(self, idx: int, new_w: np.ndarray):
        """Set the value of a cluster weight.

        Parameters
        ----------
        idx : int
            Index of cluster to update.
        new_w : np.ndarray
            New cluster weight.

        """
        self.weight_sample_counter_[idx] += 1
        self.W[idx] = new_w

    def _reset_sample_counters(self):
        """Clear the sample counts left over from a previous fit."""
        self.sample_counter_ = 0
        self.weight_sample_counter_ = []

    def _post_weight_update(self, i: np.ndarray, old_w: np.ndarray, idx: int):
        """Allow a module to update auxiliary state after a sample is learned."""
        pass

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
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
            self.params["rho"] = M + epsilon
            # return True
        elif method == "MT-":
            self.params["rho"] = M - epsilon
            # return True
        elif method == "MT0":
            self.params["rho"] = M
            # return True
        elif method == "MT1":
            self.params["rho"] = np.inf
            # return False
        elif method == "MT~":
            pass
            # return True
        else:
            raise ValueError(f"Invalid Match Tracking Method: {method}")

        if method == "MT1" or self.params["rho"] > 1.0:
            return False
        else:
            return True

    @staticmethod
    def _match_tracking_operator(
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"]
    ) -> Callable:
        if method in ["MT+", "MT-", "MT1"]:
            return operator.ge
        elif method in ["MT0", "MT~"]:
            return operator.gt
        else:
            raise ValueError(f"Invalid Match Tracking Method: {method}")

    def _set_params(self, new_params):
        self.params = new_params

    def _copy_params(self) -> Dict:
        """Copy the params so match tracking can be undone after a sample.

        Match tracking only replaces values in the dict, so a shallow copy is enough.

        """
        return dict(self.params)

    def _batch_category_choice(
        self, i: np.ndarray, params: Dict
    ) -> Optional[Tuple[np.ndarray, Sequence[Optional[Dict]]]]:
        """Score every cluster in one call.

        Modules with a vectorized activation override this. It must return exactly
        what calling :meth:`category_choice` on each weight would, or None to fall
        back to those calls.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Dictionary containing parameters for the algorithm.

        Returns
        -------
        tuple or None
            Activations as a float array and a sequence of per-cluster caches.

        """
        return None

    def _category_choices(
        self, i: np.ndarray, params: Dict
    ) -> Tuple[np.ndarray, Sequence[Optional[Dict]]]:
        """Score every cluster, using the batch hook when the module has one."""
        batch = self._batch_category_choice(i, params)
        if batch is not None:
            return batch
        T_values, T_cache = zip(
            *[self.category_choice(i, w, params=params) for w in self.W]
        )
        return np.array(T_values), T_cache

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
        base_params = self._copy_params()
        try:
            return self._step_fit(x, match_reset_func, match_tracking, epsilon)
        finally:
            self._set_params(base_params)

    def _step_fit(
        self,
        x: np.ndarray,
        match_reset_func: Optional[Callable],
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
        epsilon: float,
    ) -> int:
        """Body of :meth:`step_fit`; the caller restores params afterwards."""
        mt_operator = self._match_tracking_operator(match_tracking)
        if len(self.W) == 0:
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            return 0

        T, T_cache = self._category_choices(x, self.params)
        reset_first = match_tracking in ["MT~"] and match_reset_func is not None
        if reset_first:
            # MT~ checks the reset before choice. Choice has no side effects, so
            # computing it for every cluster first gives the same result.
            assert match_reset_func is not None
            T = np.array(T, dtype=float)
            for c_, w in enumerate(self.W):
                if not match_reset_func(x, w, c_, params=self.params, cache=None):
                    T[c_] = np.nan

        # Sort candidates once:
        # primary = -T (descending T), secondary = index (ascending)
        valid = ~np.isnan(T)
        if np.any(valid):
            idx = np.arange(T.shape[0])[valid]
            T_valid = T[valid]
            order = idx[np.lexsort((idx, -T_valid))]  # lexsort uses last key as primary
        else:
            order = np.array([], dtype=int)

        match = getattr(T_cache, "match", None)
        if match is not None and (
            type(self).match_criterion_bin is BaseART.match_criterion_bin
        ):
            candidates = self._vigilant_candidates(order, match[order], mt_operator)
        else:
            candidates = iter(order)

        for c_ in candidates:
            w = self.W[c_]
            cache = T_cache[c_]
            m, cache = self.match_criterion_bin(
                x, w, params=self.params, cache=cache, op=mt_operator
            )
            if not m:
                # The reset only matters once vigilance passes, so skip the call.
                continue
            no_match_reset = (
                reset_first
                or match_reset_func is None
                or match_reset_func(x, w, c_, params=self.params, cache=cache)
            )
            if no_match_reset:
                self.set_weight(c_, self.update(x, w, self.params, cache=cache))
                self._post_weight_update(x, w, c_)
                return c_
            keep_searching = self._match_tracking(
                cache, epsilon, self.params, match_tracking
            )
            if not keep_searching:
                break

        c_new = len(self.W)
        w_new = self.new_weight(x, self.params)
        self.add_weight(w_new)
        return c_new

    def _vigilant_candidates(self, order: np.ndarray, match: np.ndarray, op: Callable):
        """Yield ``order`` but skip clusters whose match fails vigilance.

        ``match`` holds the match criterion of each cluster in ``order``. The
        vigilance is read again after each yield because match tracking can change
        it. Skipped clusters would only have failed ``match_criterion_bin``, which
        has no side effects.

        """
        pos = 0
        while pos < len(order):
            passing = np.flatnonzero(op(match[pos:], self.params["rho"]))
            if passing.size == 0:
                return
            pos += int(passing[0])
            yield order[pos]
            pos += 1

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
        assert len(self.W) > 0, "ART module is not fit."

        T, _ = self._category_choices(x, self.params)
        c_ = int(np.argmax(T))
        return c_

    def pre_step_fit(self, X: np.ndarray):
        """Undefined function called prior to each sample fit. Useful for cluster
        pruning.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        # this is where pruning steps can go
        pass

    def post_step_fit(self, X: np.ndarray):
        """Undefined function called after each sample fit. Useful for cluster pruning.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        # this is where pruning steps can go
        pass

    def post_fit(self, X: np.ndarray):
        """Undefined function called after fit. Useful for cluster pruning.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        # this is where pruning steps can go
        pass

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray, optional
            Not used. For compatibility.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
        max_iter : int, default=1
            Number of iterations to fit the model on the same dataset.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.
        verbose : bool, default=False
            If True, displays progress of the fitting process.
        leave_progress_bar : bool, default=True
            If True, leaves thge progress of the fitting process. Only used when
            verbose=True

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W: List[np.ndarray] = []
        self._reset_sample_counters()
        self.labels_ = np.zeros((X.shape[0],), dtype=int)
        for _ in range(max_iter):
            if verbose:
                try:
                    from tqdm import tqdm
                except ImportError as exc:
                    raise ImportError(
                        "The 'tqdm' package is required for progress bars. "
                        "Install it with `pip install tqdm`."
                    ) from exc

                x_iter = tqdm(
                    enumerate(X), total=int(X.shape[0]), leave=leave_progress_bar
                )
            else:
                x_iter = enumerate(X)
            for i, x in x_iter:
                self.pre_step_fit(X)
                c = self.step_fit(
                    x,
                    match_reset_func=match_reset_func,
                    match_tracking=match_tracking,
                    epsilon=epsilon,
                )
                self.labels_[i] = c
                self.post_step_fit(X)
        self.post_fit(X)
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Iteratively fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        if not hasattr(self, "W"):
            self.W = []
            self.labels_ = np.zeros((X.shape[0],), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode="constant")
        for i, x in enumerate(X):
            c = self.step_fit(
                x,
                match_reset_func=match_reset_func,
                match_tracking=match_tracking,
                epsilon=epsilon,
            )
            self.labels_[i + j] = c
        return self

    def fit_gif(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
        verbose: bool = False,
        leave_progress_bar: bool = True,
        ax: Optional["Axes"] = None,
        filename: Optional[str] = None,
        colors: Optional[IndexableOrKeyable] = None,
        n_cluster_estimate: int = 20,
        fps: int = 5,
        final_hold_secs: float = 0.0,
        **kwargs,
    ):
        """Fit the model to the data and make a gif of the process.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray, optional
            Not used. For compatibility.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
        max_iter : int, default=1
            Number of iterations to fit the model on the same dataset.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.
        verbose : bool, default=False
            If True, displays progress of the fitting process.
        leave_progress_bar : bool, default=True
            If True, leaves thge progress of the fitting process. Only used when
            verbose=True
        ax : matplotlib.axes.Axes, optional
            Figure axes.
        colors : IndexableOrKeyable, optional
            Colors to use for each cluster.
        n_cluster_estimate : int, default=20
            estimate of number of clusters. Used for coloring plot.
        fps : int, default=5
            gif frames per second
        final_hold_secs : float, default=0.0
            seconds to hold the final frame (n_final_frames=ceil(final_hold_secs * fps))
        **kwargs : dict
            see :func: `artlib.common.BaseART.visualize`

        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
        if filename is None:
            filename = f"fit_gif_{self.__class__.__name__}.gif"
        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, n_cluster_estimate))
            black = np.array([[0, 0, 0, 1]])  # RGBA for black
            colors = np.vstack((colors, black))  # Add black at the end

        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W = []
        self._reset_sample_counters()
        self.labels_ = -np.ones((X.shape[0],), dtype=int)

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, filename, dpi=80):
            for _ in range(max_iter):
                if verbose:
                    try:
                        from tqdm import tqdm
                    except ImportError as exc:
                        raise ImportError(
                            "The 'tqdm' package is required for progress bars. "
                            "Install it with `pip install tqdm`."
                        ) from exc

                    x_iter = tqdm(
                        enumerate(X), total=int(X.shape[0]), leave=leave_progress_bar
                    )
                else:
                    x_iter = enumerate(X)
                for i, x in x_iter:
                    self.pre_step_fit(X)
                    c = self.step_fit(
                        x,
                        match_reset_func=match_reset_func,
                        match_tracking=match_tracking,
                        epsilon=epsilon,
                    )
                    self.labels_[i] = c
                    self.post_step_fit(X)
                    ax.clear()
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    self.visualize(X, self.labels_, ax, colors=colors, **kwargs)
                    writer.grab_frame()
                self.post_fit(X)

                n_extra_frames = int(np.ceil(final_hold_secs * fps))
                for _ in range(n_extra_frames):
                    ax.clear()
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    self.visualize(X, self.labels_, ax, colors=colors, **kwargs)
                    writer.grab_frame()
            return self

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Predict labels for the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        np.ndarray
            Labels for the data.

        """
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.d_min_, self.d_max_)
        self.validate_data(X)
        self.check_dimensions(X)

        y = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c = self.step_pred(x)
            y[i] = c
        return y

    def shrink_clusters(self, shrink_ratio: float = 0.1):
        """Shrink the clusters by a specified ratio.

        Parameters
        ----------
        shrink_ratio : float, optional
            The ratio by which to shrink the clusters. Must be between 0 and 1.
            Default is 0.1.

        Returns
        -------
        self : object
            Returns the instance with shrunken clusters.

        """
        return self

    def plot_cluster_bounds(
        self, ax: "Axes", colors: IndexableOrKeyable, linewidth: int = 1
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
        raise NotImplementedError

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Undefined function for getting centers of each cluster. Used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        raise NotImplementedError

    def visualize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ax: Optional["Axes"] = None,
        marker_size: int = 10,
        linewidth: int = 1,
        colors: Optional[IndexableOrKeyable] = None,
    ):
        """Visualize the clustering of the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray
            Sample labels.
        ax : matplotlib.axes.Axes, optional
            Figure axes.
        marker_size : int, default=10
            Size used for data points.
        linewidth : int, default=1
            Width of boundary line.
        colors : IndexableOrKeyable, optional
            Colors to use for each cluster.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))

        for k, col in enumerate(colors):
            cluster_data = y == k
            ax.scatter(
                X[cluster_data, 0],
                X[cluster_data, 1],
                color=col,
                marker=".",
                s=marker_size,
            )

        try:
            self.plot_cluster_bounds(ax, colors, linewidth)
        except NotImplementedError:
            warn(f"{self.__class__.__name__} does not support plotting cluster bounds.")
