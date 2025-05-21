"""Base class for all ART objects."""
import numpy as np
from typing import Optional, Callable, Literal, List, Tuple, Union, Dict
from copy import deepcopy
from collections import defaultdict
from matplotlib.axes import Axes
from warnings import warn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from artlib.common.utils import normalize, de_normalize, IndexableOrKeyable
import operator


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

    def _deep_copy_params(self) -> Dict:
        return deepcopy(self.params)

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

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            self._set_params(base_params)
            return c_new

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

        T, _ = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
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
        self.labels_ = np.zeros((X.shape[0],), dtype=int)
        for _ in range(max_iter):
            if verbose:
                from tqdm import tqdm

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
        ax: Optional[Axes] = None,
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
        self.labels_ = -np.ones((X.shape[0],), dtype=int)

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, filename, dpi=80):
            for _ in range(max_iter):
                if verbose:
                    from tqdm import tqdm

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
        ax: Optional[Axes] = None,
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
            plt.scatter(
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
