"""
Tscherepanow, M. (2010).
TopoART: A Topology Learning Hierarchical ART Network.
In K. Diamantaras, W. Duch, & L. S. Iliadis (Eds.),
Artificial Neural Networks – ICANN 2010 (pp. 157–167).
Berlin, Heidelberg: Springer Berlin Heidelberg.
doi:10.1007/978-3-642-15825-4_21.

"""

import numpy as np
from typing import Optional, Callable, Iterable
from matplotlib.axes import Axes
from warnings import warn
from artlib.common.BaseART import BaseART


class TopoART(BaseART):

    def __init__(self, base_module: BaseART, betta_lower: float, tau: int, phi: int):
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module {base_module.base_module.__class__.__name__}"
            )
        params = dict(base_module.params, **{"beta_lower": betta_lower, "tau": tau, "phi": phi})
        super().__init__(params)
        self.base_module = base_module
        self.adjacency = np.zeros([], dtype=int)
        self._permanent_mask = np.zeros([], dtype=bool)

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "beta" in params, "TopoART is only compatible with ART modules relying on 'beta' for learning."
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
    def W(self):
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        self.base_module.W = new_W

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        self.base_module.validate_data(X)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        return self.base_module.category_choice(i, w, params)

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        """
        get the match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion, cache used for later processing

        """
        return self.base_module.match_criterion(i, w, params, cache)

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        """
        get the binary match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion binary, cache used for later processing

        """
        return self.base_module.match_criterion_bin(i, w, params, cache)

    def update(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> np.ndarray:
        """
        get the updated cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            updated cluster weight, cache used for later processing

        """
        if cache.get("resonant_c", -1) >= 0:
            self.adjacency[cache["resonant_c"], cache["current_c"]] += 1
        return self.base_module.update(i, w, params, cache)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """

        return self.base_module.new_weight(i, params)


    def add_weight(self, new_w: np.ndarray):
        """
        add a new cluster weight

        Parameters:
        - new_w: new cluster weight to add

        """
        if len(self.W) == 0:
            self.adjacency = np.zeros((1, 1))
        else:
            self.adjacency = np.pad(self.adjacency, ((0, 1), (0, 1)), "constant")
        self._permanent_mask = np.pad(self._permanent_mask, (0, 1), "constant")
        self.weight_sample_counter_.append(1)
        self.W.append(new_w)


    def prune(self, X: np.ndarray):
        self._permanent_mask += (np.array(self.weight_sample_counter_) >= self.phi)
        perm_labels = np.where(self._permanent_mask)[0]

        self.W = [w for w, pm in zip(self.W, self._permanent_mask) if pm]
        self.weight_sample_counter_ = [self.weight_sample_counter_[i] for i in perm_labels]
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
        """
        Function called after each sample fit. Used for cluster pruning

        Parameters:
        - X: data set

        """
        if self.sample_counter_ > 0 and self.sample_counter_ % self.tau == 0:
            self.prune(X)

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        """
        fit the model to a single sample

        Parameters:
        - x: data sample
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise

        Returns:
            cluster label of the input sample

        """
        self.sample_counter_ += 1
        resonant_c: int = -1

        if len(self.W) == 0:
            new_w = self.new_weight(x, self.params)
            self.add_weight(new_w)
            self.adjacency = np.zeros((1, 1), dtype=int)
            self._permanent_mask = np.zeros((1, ), dtype=bool)
            return 0
        else:
            T_values, T_cache = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.argmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m, cache = self.match_criterion_bin(x, w, params=self.params, cache=cache)
                no_match_reset = (
                        match_reset_func is None or
                        match_reset_func(x, w, c_, params=self.params, cache=cache)
                )
                if m and no_match_reset:
                    if resonant_c < 0:
                        params = self.params
                    else:
                        params = dict(self.params, **{"beta": self.params["beta_lower"]})
                    #TODO: make compatible with DualVigilanceART
                    new_w = self.update(
                        x,
                        w,
                        params=params,
                        cache=dict((cache if cache else {}), **{"resonant_c": resonant_c, "current_c": c_})
                    )
                    self.set_weight(c_, new_w)
                    if resonant_c < 0:
                        resonant_c = c_
                        T[c_] = -1
                    else:
                        return resonant_c
                else:
                    T[c_] = -1

            if resonant_c < 0:
                c_new = len(self.W)
                w_new = self.new_weight(x, self.params)
                self.add_weight(w_new)
                return c_new

            return resonant_c

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        try:
            self.base_module.plot_cluster_bounds(ax=ax, colors=colors, linewidth=linewidth)
        except NotImplementedError:
            warn(f"{self.base_module.__class__.__name__} does not support plotting cluster bounds.")