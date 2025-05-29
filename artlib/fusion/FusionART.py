"""Fusion ART :cite:`tan2007intelligence`."""
# Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
# Intelligence Through Interaction: Towards a Unified Theory for Learning.
# In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
# Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
# Berlin, Heidelberg: Springer Berlin Heidelberg.
# doi:10.1007/ 978-3-540-72383-7_128.

import numpy as np
from typing import Optional, Union, Callable, List, Literal, Tuple, Dict
from copy import deepcopy
from artlib.common.BaseART import BaseART
from sklearn.utils.validation import check_is_fitted
import operator


def get_channel_position_tuples(
    channel_dims: List[int],
) -> List[Tuple[int, int]]:
    """Generate the start and end positions for each channel in the input data.

    Parameters
    ----------
    channel_dims : list of int
        A list representing the number of dimensions for each channel.

    Returns
    -------
    list of tuple of int
        A list of tuples where each tuple represents the start and end index for a
        channel.

    """
    positions = []
    start = 0
    for length in channel_dims:
        end = start + length
        positions.append((start, end))
        start = end
    return positions


class FusionART(BaseART):
    """Fusion ART for Data Fusion and Regression.

    This module implements Fusion ART as first described in:
    :cite:`tan2007intelligence`.

    .. # Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007).
    .. # Intelligence Through Interaction: Towards a Unified Theory for Learning.
    .. # In D. Liu, S. Fei, Z.-G. Hou, H. Zhang, & C. Sun (Eds.),
    .. # Advances in Neural Networks – ISNN 2007 (pp. 1094–1103).
    .. # Berlin, Heidelberg: Springer Berlin Heidelberg.
    .. # doi:10.1007/ 978-3-540-72383-7_128.

    Fusion ART accepts an arbitrary number of ART modules, each assigned a different
    data channel. The activation and match functions for all ART modules are then fused
    such that all modules must be simultaneously active and resonant in order for a
    match to occur. This provides fine-grain control when clustering multi-channel or
    multi-modal data and allows for different geometries of clusters to be used for
    each channel. Fusion ART also allows for fitting regression models and specific
    functions have been implemented to allow this.

    """

    def __init__(
        self,
        modules: List[BaseART],
        gamma_values: Union[List[float], np.ndarray],
        channel_dims: Union[List[int], np.ndarray],
    ):
        """Initialize the FusionART instance.

        Parameters
        ----------
        modules : List[BaseART]
            A list of ART modules corresponding to each data channel.
        gamma_values : Union[List[float], np.ndarray]
            The activation ratio for each channel.
        channel_dims : Union[List[int], np.ndarray]
            The number of dimensions for each channel.

        """
        assert len(modules) == len(gamma_values) == len(channel_dims)
        params = {"gamma_values": gamma_values}
        super().__init__(params)
        self.modules = modules
        self.n = len(self.modules)
        self.channel_dims = channel_dims
        self._channel_indices = get_channel_position_tuples(self.channel_dims)
        self.dim_ = sum(channel_dims)

    def get_params(self, deep: bool = True) -> Dict:
        """Get the parameters of the FusionART model.

        Parameters
        ----------
        deep : bool, optional
            If True, will return parameters for this class and the contained sub-objects
            that are estimators (default is True).

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        out = self.params
        for i, module in enumerate(self.modules):
            deep_items = module.get_params().items()
            out.update((f"module_{i}" + "__" + k, val) for k, val in deep_items)
            out[f"module_{i}"] = module
        return out

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters in the first ART module.

        Returns
        -------
        int
            The number of clusters.

        """
        return self.modules[0].n_clusters

    @property
    def W(self):
        """Get the weights of all modules as a single array.

        Returns
        -------
        np.ndarray
            Concatenated weights of all channels from the ART modules.

        """
        W = [
            np.concatenate([self.modules[k].W[i] for k in range(self.n)])
            for i in range(self.modules[0].n_clusters)
        ]
        return W

    @W.setter
    def W(self, new_W):
        """Set the weights for each module by splitting the input weights.

        Parameters
        ----------
        new_W : np.ndarray
            New concatenated weights to be set for the modules.

        """
        for k in range(self.n):
            if len(new_W) > 0:
                self.modules[k].W = new_W[
                    self._channel_indices[k][0] : self._channel_indices[k][1]
                ]
            else:
                self.modules[k].W = []

    @staticmethod
    def validate_params(params: Dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            The parameters for the FusionART model.

        """
        assert "gamma_values" in params
        assert all([1.0 >= g >= 0.0 for g in params["gamma_values"]])
        assert sum(params["gamma_values"]) == 1.0
        assert isinstance(params["gamma_values"], (np.ndarray, list))

    def validate_data(self, X: np.ndarray):
        """Validate the input data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        self.check_dimensions(X)
        for k in range(self.n):
            X_k = X[:, self._channel_indices[k][0] : self._channel_indices[k][1]]
            self.modules[k].validate_data(X_k)

    def check_dimensions(self, X: np.ndarray):
        """Ensure that the input data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        assert X.shape[1] == self.dim_, "Invalid data shape"

    def prepare_data(
        self, channel_data: List[np.ndarray], skip_channels: List[int] = []
    ) -> np.ndarray:
        """Prepare the input data by processing each channel's data through its
        respective ART module.

        Parameters
        ----------
        channel_data : list of np.ndarray
            List of arrays, one for each channel.
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        np.ndarray
            Processed and concatenated data.

        """
        skip_channels = [self.n + k if k < 0 else k for k in skip_channels]
        prepared_channel_data = [
            self.modules[i].prepare_data(channel_data[i])
            for i in range(self.n)
            if i not in skip_channels
        ]

        return self.join_channel_data(
            prepared_channel_data, skip_channels=skip_channels
        )

    def restore_data(
        self, X: np.ndarray, skip_channels: List[int] = []
    ) -> List[np.ndarray]:
        """Restore data to its original state before preparation.

        Parameters
        ----------
        X : np.ndarray
            The prepared data.
        skip_channels : list of int, optional
            Channels to be skipped (default is []).
        Returns
        -------
        np.ndarray
            Restored data for each channel.

        """
        skip_channels = [self.n + k if k < 0 else k for k in skip_channels]
        channel_data = self.split_channel_data(X, skip_channels=skip_channels)
        restored_channel_data = [
            self.modules[i].restore_data(channel_data[i])
            for i in range(self.n)
            if i not in skip_channels
        ]
        return restored_channel_data

    def category_choice(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        skip_channels: List[int] = [],
    ) -> Tuple[float, Optional[Dict]]:
        """Get the activation of the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            Cluster activation and cache for further processing.

        """
        activations, caches = zip(
            *[
                self.modules[k].category_choice(
                    i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    w[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    self.modules[k].params,
                )
                if k not in skip_channels
                else (1.0, dict())
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        activation = sum(
            [a * self.params["gamma_values"][k] for k, a in enumerate(activations)]
        )
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
        skip_channels: List[int] = [],
    ) -> Tuple[float, Optional[Dict]]:
        """Get the match criterion for the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            max match_criterion across channels and the updated cache.

        """
        if cache is None:
            raise ValueError("No cache provided")
        M, caches = zip(
            *[
                self.modules[k].match_criterion(
                    i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    w[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    self.modules[k].params,
                    cache[k],
                )
                if k not in skip_channels
                else (np.nan, {"match_criterion": np.inf})
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return np.nanmax(M), cache

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
        op: Callable = operator.ge,
        skip_channels: List[int] = [],
    ) -> Tuple[bool, Dict]:
        """Get the binary match criterion for the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).
        op : Callable, optional
            Operator for comparison (default is operator.ge).
        skip_channels : list of int, optional
            Channels to be skipped (default is []).

        Returns
        -------
        tuple
            Binary match criterion and cache for further processing.

        """
        if cache is None:
            raise ValueError("No cache provided")
        M_bin, caches = zip(
            *[
                self.modules[k].match_criterion_bin(
                    i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    w[self._channel_indices[k][0] : self._channel_indices[k][1]],
                    self.modules[k].params,
                    cache[k],
                    op,
                )
                if k not in skip_channels
                else (True, {"match_criterion": np.inf})
                for k in range(self.n)
            ]
        )
        cache = {k: cache_k for k, cache_k in enumerate(caches)}
        return all(M_bin), cache

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Perform match tracking for all channels using the specified method.

        Parameters
        ----------
        cache : list of dict
            Cached match criterion values for each channel.
        epsilon : float
            Small adjustment factor for match tracking.
        params : list of dict
            Parameters for each channel module.
        method : Literal["MT+", "MT-", "MT0", "MT1", "MT~"]
            Match tracking method to apply.

        Returns
        -------
        bool
            Whether to continue searching for a match across all channels.

        """
        keep_searching = []
        for i in range(len(cache)):
            if cache[i]["match_criterion_bin"]:
                keep_searching_i = self.modules[i]._match_tracking(
                    cache[i], epsilon, params[i], method
                )
                keep_searching.append(keep_searching_i)
            else:
                keep_searching.append(True)
        return all(keep_searching)

    def _set_params(self, new_params: Dict):
        """Set the parameters for each module in FusionART.

        Parameters
        ----------
        new_params : list of dict
            A list of parameters for each module.

        """
        for i in range(self.n):
            self.modules[i].params = new_params[i]

    def _deep_copy_params(self) -> Dict:
        """Create a deep copy of the parameters for each module.

        Returns
        -------
        dict
            A dictionary with module indices as keys and their deep-copied parameters
            as values.

        """
        return {i: deepcopy(module.params) for i, module in enumerate(self.modules)}

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
                    if not (m and no_match_reset):
                        params = {i: self.modules[i].params for i in range(len(cache))}
                        keep_searching = self._match_tracking(
                            cache, epsilon, params, match_tracking
                        )
                        if not keep_searching:
                            T[:] = np.nan

            c_new = len(self.W)
            w_new = self.new_weight(x, self.params)
            self.add_weight(w_new)
            self._set_params(base_params)
            return c_new

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
            Input dataset.
        match_reset_func : callable, optional
            Function to reset the match criteria based on external factors.
        match_tracking : Literal["MT+", "MT-", "MT0", "MT1", "MT~"], optional
            Method for resetting match criteria (default is "MT+").
        epsilon : float, optional
            Value to adjust the vigilance parameter (default is 0.0).

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        if not hasattr(self.modules[0], "W"):
            self.W: List[np.ndarray] = []
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

    def step_pred(self, x, skip_channels: List[int] = []) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Input sample.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        int
            Predicted cluster label for the input sample.

        """
        assert len(self.W) >= 0, "ART module is not fit."

        T, _ = zip(
            *[
                self.category_choice(
                    x, w, params=self.params, skip_channels=skip_channels
                )
                for w in self.W
            ]
        )
        c_ = int(np.argmax(T))
        return c_

    def predict(
        self, X: np.ndarray, clip: bool = False, skip_channels: List[int] = []
    ) -> np.ndarray:
        """Predict labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            Input dataset.
        clip : bool
            clip the input values to be between the previously seen data limits
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        np.ndarray
            Predicted labels for the input data.

        """
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.d_min_, self.d_max_)
        self.validate_data(X)
        self.check_dimensions(X)

        y = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c = self.step_pred(x, skip_channels=skip_channels)
            y[i] = c
        return y

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: Dict,
        cache: Optional[Dict] = None,
    ) -> np.ndarray:
        """Update the cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Input data sample.
        w : np.ndarray
            Cluster weight information.
        params : dict
            Parameters for the ART algorithm.
        cache : dict, optional
            Cache for previous calculations (default is None).

        Returns
        -------
        np.ndarray
            Updated cluster weight.

        """
        assert cache is not None
        W = [
            self.modules[k].update(
                i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                w[self._channel_indices[k][0] : self._channel_indices[k][1]],
                self.modules[k].params,
                cache[k],
            )
            for k in range(self.n)
        ]
        return np.concatenate(W)

    def new_weight(self, i: np.ndarray, params: Dict) -> np.ndarray:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Input data sample.
        params : dict
            Parameters for the ART algorithm.

        Returns
        -------
        np.ndarray
            New cluster weight.

        """
        W = [
            self.modules[k].new_weight(
                i[self._channel_indices[k][0] : self._channel_indices[k][1]],
                self.modules[k].params,
            )
            for k in range(self.n)
        ]
        return np.concatenate(W)

    def add_weight(self, new_w: np.ndarray):
        """Add a new cluster weight.

        Parameters:
        - new_w: new cluster weight to add

        """
        for k in range(self.n):
            new_w_k = new_w[self._channel_indices[k][0] : self._channel_indices[k][1]]
            self.modules[k].add_weight(new_w_k)

    def set_weight(self, idx: int, new_w: np.ndarray):
        """Set the value of a cluster weight.

        Parameters:
        - idx: index of cluster to update
        - new_w: new cluster weight

        """
        for k in range(self.n):
            new_w_k = new_w[self._channel_indices[k][0] : self._channel_indices[k][1]]
            self.modules[k].set_weight(idx, new_w_k)

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the center points for each cluster.

        Returns
        -------
        list of np.ndarray
            Center points of the clusters.

        """
        centers_ = [module.get_cluster_centers() for module in self.modules]
        centers = [
            np.concatenate([centers_[k][i] for k in range(self.n)])
            for i in range(self.n_clusters)
        ]
        return centers

    def get_channel_centers(self, channel: int) -> List[np.ndarray]:
        """Get the center points of clusters for a specific channel.

        Parameters
        ----------
        channel : int
            The channel index.

        Returns
        -------
        list of np.ndarray
            Cluster centers for the specified channel.

        """
        return self.modules[channel].get_cluster_centers()

    def predict_regression(
        self, X: np.ndarray, clip: bool = False, target_channels: List[int] = [-1]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predict regression values for the input data using the target channels.

        Parameters
        ----------
        X : np.ndarray
            Input dataset.
        clip : bool
            clip the input values to be between the previously seen data limits
        target_channels : list of int, optional
            List of target channels to use for regression. If negative values are used,
            they are considered as channels counting backward from the last channel.
            By default, it uses the last channel (-1).

        Returns
        -------
        Union[np.ndarray, list of np.ndarray]
            Predicted regression values. If only one target channel is used, returns a
            single np.ndarray. If multiple target channels are used, returns a list of
            np.ndarray, one for each channel.

        """
        target_channels = [self.n + k if k < 0 else k for k in target_channels]
        C = self.predict(X, clip=clip, skip_channels=target_channels)
        centers = [self.get_channel_centers(k) for k in target_channels]
        if len(target_channels) == 1:
            return np.array([centers[0][c] for c in C])
        else:
            return [np.array([centers[k][c] for c in C]) for k in target_channels]

    def join_channel_data(
        self, channel_data: List[np.ndarray], skip_channels: List[int] = []
    ) -> np.ndarray:
        """Concatenate data from different channels into a single array.

        Parameters
        ----------
        channel_data : list of np.ndarray
            Data from each channel.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        np.ndarray
            Concatenated data.

        """
        skip_channels = [self.n + k if k < 0 else k for k in skip_channels]
        n_samples = channel_data[0].shape[0]

        formatted_channel_data = []
        i = 0
        for k in range(self.n):
            if k not in skip_channels:
                formatted_channel_data.append(channel_data[i])
                i += 1
            else:
                formatted_channel_data.append(
                    0.5
                    * np.ones(
                        (
                            n_samples,
                            self._channel_indices[k][1] - self._channel_indices[k][0],
                        )
                    )
                )

        X = np.hstack(formatted_channel_data)
        return X

    def split_channel_data(
        self, joined_data: np.ndarray, skip_channels: List[int] = []
    ) -> List[np.ndarray]:
        """Split the concatenated data into its original channels.

        Parameters
        ----------
        joined_data : np.ndarray
            Concatenated data from multiple channels.
        skip_channels : list of int, optional
            Channels to skip (default is []).

        Returns
        -------
        list of np.ndarray
            Split data, one array for each channel.

        """
        skip_channels = [self.n + k if k < 0 else k for k in skip_channels]

        channel_data = []
        current_col = 0

        for k in range(self.n):
            start_idx, end_idx = self._channel_indices[k]
            channel_width = end_idx - start_idx

            if k not in skip_channels:
                # Extract the original channel data
                channel_data.append(
                    joined_data[:, current_col : current_col + channel_width]
                )
                current_col += channel_width
            else:
                # If this channel was skipped, we know it was filled with 0.5,
                # so we skip those columns
                current_col += channel_width

        return channel_data
