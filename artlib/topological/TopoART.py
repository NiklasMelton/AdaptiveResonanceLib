"""Topo ART :cite:`tscherepanow2010topoart`."""
# Tscherepanow, M. (2010).
# TopoART: A Topology Learning Hierarchical ART Network.
# In K. Diamantaras, W. Duch, & L. S. Iliadis (Eds.),
# Artificial Neural Networks – ICANN 2010 (pp. 157–167).
# Berlin, Heidelberg: Springer Berlin Heidelberg.
# doi:10.1007/978-3-642-15825-4_21.

import numpy as np
from typing import Optional, Callable, List, Literal, Tuple, Union, Dict
from matplotlib.axes import Axes
from warnings import warn
from copy import deepcopy
from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable
import operator


class TopoART(BaseART):
    """Topo ART for Topological Clustering.

    This module implements Topo ART as first published in:
    :cite:`tscherepanow2010topoart`.

    .. # Tscherepanow, M. (2010).
    .. # TopoART: A Topology Learning Hierarchical ART Network.
    .. # In K. Diamantaras, W. Duch, & L. S. Iliadis (Eds.),
    .. # Artificial Neural Networks – ICANN 2010 (pp. 157–167).
    .. # Berlin, Heidelberg: Springer Berlin Heidelberg.
    .. # doi:10.1007/978-3-642-15825-4_21.

    Topo ART clusters accepts an instantiated :class:`~artlib.common.BaseART.BaseART`
    module and generates a topological clustering by recording the first and second
    resonant cluster relationships in an adjacency matrix. Further, it updates the
    second resonant cluster with a lower learning rate than the first, providing for
    a distributed learning model.

    """

    def __init__(self, base_module: BaseART, beta_lower: float, tau: int, phi: int):
        """Initialize TopoART.

        Parameters
        ----------
        base_module : BaseART
            An instantiated ART module.
        beta_lower : float
            The learning rate for the second resonant cluster.
        tau : int
            Number of samples after which clusters are pruned.
        phi : int
            Minimum number of samples a cluster must be associated with to be kept.

        """
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} "
                f"is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module: "
                f"{base_module.base_module.__class__.__name__}"
            )
        params = dict(
            base_module.params,
            **{"beta_lower": beta_lower, "tau": tau, "phi": phi},
        )
        super().__init__(params)
        self.base_module = base_module
        self.adjacency = np.zeros([], dtype=int)
        self._permanent_mask = np.zeros([], dtype=bool)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing parameters for the algorithm.

        Raises
        ------
        AssertionError
            If the required parameters are not provided or are invalid.

        """
        assert (
            "beta" in params
        ), "TopoART is only compatible with ART modules relying on 'beta' for learning."
        assert "beta_lower" in params
        assert "tau" in params
        assert "phi" in params
        assert params["beta"] >= params["beta_lower"]
        assert params["phi"] <= params["tau"]
        assert isinstance(params["beta"], float)
        assert isinstance(params["beta_lower"], float)
        assert isinstance(params["tau"], int)
        assert isinstance(params["phi"], int)

    @property
    def W(self) -> List[np.ndarray]:
        """Get the weight matrix of the base module.

        Returns
        -------
        list[np.ndarray]
            The weight matrix of the base ART module.

        """
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        """Set the weight matrix of the base module.

        Parameters
        ----------
        new_W : list[np.ndarray]
            The new weight matrix.

        """
        self.base_module.W = new_W

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        self.base_module.validate_data(X)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        Returns
        -------
        np.ndarray
            Prepared (normalized) data.

        """
        return self.base_module.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to the state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        return self.base_module.restore_data(X)

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
            Parameters for the algorithm.

        Returns
        -------
        tuple[float, Optional[dict]]
            Cluster activation and cache used for later processing.

        """
        return self.base_module.category_choice(i, w, params)

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
            Parameters for the algorithm.
        cache : dict, optional
            Values cached from previous calculations.

        Returns
        -------
        tuple[float, dict]
            Cluster match criterion and cache used for later processing.

        """
        return self.base_module.match_criterion(i, w, params, cache)

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
        op: Callable = operator.ge,
    ) -> tuple[bool, dict]:
        """Get the binary match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Parameters for the algorithm.
        cache : dict, optional
            Values cached from previous calculations.
        op : Callable, default=operator.ge
            Comparison operator to use for the binary match criterion.

        Returns
        -------
        tuple[bool, dict]
            Binary match criterion and cache used for later processing.

        """
        return self.base_module.match_criterion_bin(i, w, params, cache, op)

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Update the cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        params : dict
            Parameters for the algorithm.
        cache : dict, optional
            Values cached from previous calculations.

        Returns
        -------
        np.ndarray
            Updated cluster weight.

        """
        assert cache is not None
        if cache.get("resonant_c", -1) >= 0:
            self.adjacency[cache["resonant_c"], cache["current_c"]] += 1
        return self.base_module.update(i, w, params, cache)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """Generate a new cluster weight.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        params : dict
            Parameters for the algorithm.

        Returns
        -------
        np.ndarray
            Newly generated cluster weight.

        """
        return self.base_module.new_weight(i, params)

    def add_weight(self, new_w: np.ndarray):
        """Add a new cluster weight.

        Parameters
        ----------
        new_w : np.ndarray
            New cluster weight to add.

        """
        if len(self.W) == 0:
            self.adjacency = np.zeros((1, 1))
        else:
            self.adjacency = np.pad(self.adjacency, ((0, 1), (0, 1)), "constant")
        self._permanent_mask = np.pad(self._permanent_mask, (0, 1), "constant")
        self.weight_sample_counter_.append(1)
        self.W.append(new_w)

    def prune(self, X: np.ndarray):
        """Prune clusters based on the number of associated samples.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        a = (
            np.array(self.weight_sample_counter_).reshape(
                -1,
            )
            >= self.phi
        )
        b = self._permanent_mask
        print(a.shape, b.shape)

        self._permanent_mask += (
            np.array(self.weight_sample_counter_).reshape(
                -1,
            )
            >= self.phi
        )
        perm_labels = np.where(self._permanent_mask)[0]

        self.W = [w for w, pm in zip(self.W, self._permanent_mask) if pm]
        self.weight_sample_counter_ = [
            self.weight_sample_counter_[i] for i in perm_labels
        ]
        self.adjacency = self.adjacency[perm_labels][:, perm_labels]
        self._permanent_mask = self._permanent_mask[perm_labels]

        label_map = {
            label: np.where(perm_labels == label)[0][0]
            for label in np.unique(self.labels_)
            if label in perm_labels
        }

        for i, x in enumerate(X):
            if self.labels_[i] in label_map:
                self.labels_[i] = label_map[self.labels_[i]]
            elif len(self.W) > 0:
                # this is a more flexible approach than that described in the paper
                self.labels_[i] = self.step_pred(x)
            else:
                self.labels_[i] = -1

    def post_step_fit(self, X: np.ndarray):
        """Perform post-fit operations, such as cluster pruning, after fitting each
        sample.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.

        """
        if self.sample_counter_ > 0 and self.sample_counter_ % self.tau == 0:
            self.prune(X)

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Adjust the vigilance parameter based on match tracking methods.

        Parameters
        ----------
        cache : dict
            Cached values from previous calculations.
        epsilon : float
            Adjustment factor for the vigilance parameter.
        params : dict
            Parameters for the algorithm.
        method : Literal["MT+", "MT-", "MT0", "MT1", "MT~"]
            Method to use for match tracking.

        Returns
        -------
        bool
            True if the match tracking continues, False otherwise.

        """
        assert isinstance(cache, dict)
        assert isinstance(params, dict)
        M = cache["match_criterion"]
        if method == "MT+":
            self.base_module.params["rho"] = M + epsilon
            return True
        elif method == "MT-":
            self.base_module.params["rho"] = M - epsilon
            return True
        elif method == "MT0":
            self.base_module.params["rho"] = M
            return True
        elif method == "MT1":
            self.base_module.params["rho"] = np.inf
            return False
        elif method == "MT~":
            return True
        else:
            raise ValueError(f"Invalid Match Tracking Method: {method}")

    def _set_params(self, new_params):
        """Set new parameters for the base module.

        Parameters
        ----------
        new_params : dict
            New parameters to set.

        """
        self.base_module.params = new_params

    def _deep_copy_params(self) -> dict:
        """Create a deep copy of the parameters.

        Returns
        -------
        dict
            Deep copy of the parameters.

        """
        return deepcopy(self.base_module.params)

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
        match_reset_func : Callable, optional
            Function to reset the match based on custom criteria.
        match_tracking : Literal["MT+", "MT-", "MT0", "MT1", "MT~"], default="MT+"
            Method to reset the match.
        epsilon : float, default=0.0
            Adjustment factor for vigilance.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        base_params = self._deep_copy_params()
        mt_operator = self._match_tracking_operator(match_tracking)
        self.sample_counter_ += 1
        resonant_c: int = -1

        if len(self.W) == 0:
            new_w = self.new_weight(x, self.params)
            self.add_weight(new_w)
            self.adjacency = np.zeros((1, 1), dtype=int)
            self._permanent_mask = np.zeros((1,), dtype=bool)
            return 0
        else:
            T_values, T_cache = zip(
                *[
                    self.category_choice(x, w, params=self.base_module.params)
                    for w in self.W
                ]
            )
            T = np.array(T_values)
            while any(~np.isnan(T)):
                c_ = int(np.nanargmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m, cache = self.match_criterion_bin(
                    x,
                    w,
                    params=self.base_module.params,
                    cache=cache,
                    op=mt_operator,
                )
                no_match_reset = match_reset_func is None or match_reset_func(
                    x, w, c_, params=self.base_module.params, cache=cache
                )
                if m and no_match_reset:
                    if resonant_c < 0:
                        params = self.base_module.params
                    else:
                        params = dict(
                            self.base_module.params,
                            **{"beta": self.params["beta_lower"]},
                        )
                    # TODO: make compatible with DualVigilanceART
                    new_w = self.update(
                        x,
                        w,
                        params=params,
                        cache=dict(
                            (cache if cache else {}),
                            **{"resonant_c": resonant_c, "current_c": c_},
                        ),
                    )
                    self.set_weight(c_, new_w)
                    if resonant_c < 0:
                        resonant_c = c_
                        T[c_] = np.nan
                    else:
                        self._set_params(base_params)
                        return resonant_c
                else:
                    T[c_] = np.nan
                    if not no_match_reset:
                        keep_searching = self._match_tracking(
                            cache, epsilon, self.params, match_tracking
                        )
                        if not keep_searching:
                            T[:] = np.nan

            self._set_params(base_params)
            if resonant_c < 0:
                c_new = len(self.W)
                w_new = self.new_weight(x, self.params)
                self.add_weight(w_new)
                return c_new

            return resonant_c

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of each cluster.

        Returns
        -------
        List[np.ndarray]
            Cluster centroids.

        """
        return self.base_module.get_cluster_centers()

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Visualize the boundaries of each cluster.

        Parameters
        ----------
        ax : Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, default=1
            Width of boundary lines.

        """
        try:
            self.base_module.plot_cluster_bounds(
                ax=ax, colors=colors, linewidth=linewidth
            )
        except NotImplementedError:
            warn(
                f"{self.base_module.__class__.__name__} "
                f"does not support plotting cluster bounds."
            )
