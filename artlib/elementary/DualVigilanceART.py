"""
Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
Dual vigilance fuzzy adaptive resonance theory.
Neural Networks, 109, 1–5. doi:10.1016/j.neunet.2018.09.015.
"""
import numpy as np
from typing import Optional, Callable, Iterable, List, Literal
from warnings import warn
from copy import deepcopy
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART


class DualVigilanceART(BaseART):
    # implementation of Dual Vigilance ART

    def __init__(self, base_module: BaseART, rho_lower_bound: float):
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module {base_module.base_module.__class__.__name__}"
            )
        assert "rho" in base_module.params, \
            "Dual Vigilance ART is only compatible with ART modules relying on 'rho' for vigilance."

        params = {"rho_lower_bound": rho_lower_bound}
        assert base_module.params["rho"] > params["rho_lower_bound"] >= 0
        super().__init__(params)
        self.base_module = base_module
        self.rho_lower_bound = rho_lower_bound
        self.map: dict[int, int] = dict()

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            base modules prepare_data
        """
        return self.base_module.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            restored data
        """
        return self.base_module.restore_data(X)

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = {
            "rho_lower_bound": self.params["rho_lower_bound"],
            "base_module": self.base_module
        }
        if deep:
            deep_items = self.base_module.get_params().items()
            out.update(("base_module" + "__" + k, val) for k, val in deep_items)

        return out

    @property
    def n_clusters(self) -> int:
        """
        get the current number of clusters

        Returns:
            the number of clusters
        """
        return len(set(c for c in self.map.values()))

    @property
    def dim_(self):
        return self.base_module.dim_

    @dim_.setter
    def dim_(self, new_dim):
        self.base_module.dim_ = new_dim

    @property
    def labels_(self):
        return self.base_module.labels_

    @labels_.setter
    def labels_(self, new_labels: np.ndarray):
        self.base_module.labels_ = new_labels

    @property
    def W(self):
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        self.base_module.W = new_W

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        self.base_module.check_dimensions(X)

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        self.base_module.validate_data(X)
        self.check_dimensions(X)

    def validate_params(self, params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """

        assert "rho_lower_bound" in params, \
            "Dual Vigilance ART requires a lower bound 'rho' value"
        assert params["rho_lower_bound"] >= 0
        assert isinstance(params["rho_lower_bound"], float)

    def _match_tracking(self, cache: dict, epsilon: float, params: dict, method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"]) -> bool:
        M = cache["match_criterion"]
        if method == "MT+":
            self.base_module.params["rho"] = M+epsilon
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
        self.base_module.params = new_params

    def _deep_copy_params(self) -> dict:
        return deepcopy(self.base_module.params)

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None,match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0) -> int:
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
        base_params = self._deep_copy_params()
        mt_operator = self._match_tracking_operator(match_reset_method)
        self.sample_counter_ += 1
        if len(self.base_module.W) == 0:
            new_w = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.add_weight(new_w)
            self.map[0] = 0
            return 0
        else:
            T_values, T_cache = zip(
                *[
                    self.base_module.category_choice(x, w, params=self.base_module.params)
                    for w in self.base_module.W
                ]
            )
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.nanargmax(T))
                w = self.base_module.W[c_]
                cache = T_cache[c_]
                m1, cache = self.base_module.match_criterion_bin(x, w, params=self.base_module.params, cache=cache, op=mt_operator)
                no_match_reset = (
                    match_reset_func is None or
                    match_reset_func(x, w, self.map[c_], params=self.base_module.params, cache=cache)
                )

                if no_match_reset:
                    if m1:
                        new_w = self.base_module.update(x, w, self.base_module.params, cache=cache)
                        self.base_module.set_weight(c_, new_w)
                        self._set_params(base_params)
                        return self.map[c_]
                    else:
                        lb_params = dict(self.base_module.params, **{"rho": self.rho_lower_bound})
                        m2, _ = self.base_module.match_criterion_bin(x, w, params=lb_params, cache=cache, op=mt_operator)
                        if m2:
                            c_new = len(self.base_module.W)
                            w_new = self.base_module.new_weight(x, self.base_module.params)
                            self.base_module.add_weight(w_new)
                            self.map[c_new] = self.map[c_]
                            self._set_params(base_params)
                            return self.map[c_new]
                else:
                    keep_searching = self._match_tracking(cache, epsilon, self.params, match_reset_method)
                    if not keep_searching:
                        T[:] = np.nan
                T[c_] = np.nan

            c_new = len(self.base_module.W)
            w_new = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.add_weight(w_new)
            self.map[c_new] = max(self.map.values()) + 1
            self._set_params(base_params)
            return self.map[c_new]

    def step_pred(self, x) -> int:
        """
        predict the label for a single sample

        Parameters:
        - x: data sample

        Returns:
            cluster label of the input sample

        """
        assert len(self.base_module.W) >= 0, "ART module is not fit."
        T, _ = zip(
            *[
                self.base_module.category_choice(x, w, params=self.base_module.params)
                for w in self.base_module.W
            ]
        )
        c_ = int(np.argmax(T))
        return self.map[c_]

    def get_cluster_centers(self) -> List[np.ndarray]:
        """
        function for getting centers of each cluster. Used for regression
        Returns:
            cluster centroid
        """
        return self.base_module.get_cluster_centers()

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        colors_base = []
        for k_a in range(self.base_module.n_clusters):
            colors_base.append(colors[self.map[k_a]])

        try:
            self.base_module.plot_cluster_bounds(ax=ax, colors=colors_base, linewidth=linewidth)
        except NotImplementedError:
            warn(f"{self.base_module.__class__.__name__} does not support plotting cluster bounds.")
