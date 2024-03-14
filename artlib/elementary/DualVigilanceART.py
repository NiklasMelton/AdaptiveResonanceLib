"""
Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
Dual vigilance fuzzy adaptive resonance theory.
Neural Networks, 109, 1–5. doi:10.1016/j.neunet.2018.09.015.
"""
import numpy as np
from typing import Optional, Callable, Iterable
from warnings import warn
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART


class DualVigilanceART(BaseART):
    # implementation of Dual Vigilance ART

    def __init__(self, base_module: BaseART, lower_bound: float):
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module {base_module.base_module.__class__.__name__}"
            )
        params = dict(base_module.params, **{"rho_lower_bound": lower_bound})
        super().__init__(params)
        self.base_module = base_module
        self.lower_bound = lower_bound
        self.map: dict[int, int] = dict()

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = self.params
        if deep:
            deep_items = self.base_module.get_params().items()
            out.update(("base_module" + "__" + k, val) for k, val in deep_items)
            out["base_module"] = self.base_module
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

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params, \
            "Dual Vigilance ART is only compatible with ART modules relying on 'rho' for vigilance."
        assert "rho_lower_bound" in params, \
            "Dual Vigilance ART requires a lower bound 'rho' value"
        assert params["rho"] > params["rho_lower_bound"] >= 0
        assert isinstance(params["rho"], float)
        assert isinstance(params["rho_lower_bound"], float)

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
                c_ = int(np.argmax(T))
                w = self.base_module.W[c_]
                cache = T_cache[c_]
                m1, cache = self.base_module.match_criterion_bin(x, w, params=self.base_module.params, cache=cache)
                no_match_reset = (
                    match_reset_func is None or
                    match_reset_func(x, w, self.map[c_], params=self.base_module.params, cache=cache)
                )

                if no_match_reset:
                    if m1:
                        new_w = self.base_module.update(x, w, self.params, cache=cache)
                        self.base_module.set_weight(c_, new_w)
                        return self.map[c_]
                    else:
                        lb_params = dict(self.base_module.params, **{"rho": self.lower_bound})
                        m2, _ = self.base_module.match_criterion_bin(x, w, params=lb_params, cache=cache)
                        if m2:
                            c_new = len(self.base_module.W)
                            w_new = self.base_module.new_weight(x, self.base_module.params)
                            self.base_module.add_weight(w_new)
                            self.map[c_new] = self.map[c_]
                            return self.map[c_new]
                T[c_] = -1

            c_new = len(self.base_module.W)
            w_new = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.add_weight(w_new)
            self.map[c_new] = max(self.map.values()) + 1
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

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

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
