"""
Brito da Silva LE, Rayapati N, Wunsch DC.
iCVI-ARTMAP: Using Incremental Cluster Validity Indices and Adaptive Resonance Theory Reset Mechanism to Accelerate
Validation and Achieve Multiprototype Unsupervised Representations.
IEEE Trans Neural Netw Learn Syst. 2023 Dec;
34(12):9757-9770. doi: 10.1109/TNNLS.2022.3160381.
"""
import numpy as np
from typing import Optional, Callable, Iterable
from warnings import warn
from matplotlib.axes import Axes
from cvi import CVI
from artlib.common.BaseART import BaseART
from copy import deepcopy


class iCVIART(BaseART):
    # implementation of iCVI ART

    def __init__(self, base_module: BaseART, iCVI: CVI):
        """

        Parameters:
        - base_module: base ART module
        - iCVI: the icvi to optimize for clustering

        """
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module {base_module.base_module.__class__.__name__}"
            )
        super().__init__(base_module.params)
        self.map = dict()
        self.base_module = base_module
        self.iCVI = iCVI

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = {"iCVI": self.iCVI}
        if deep:
            deep_items = self.base_module.get_params().items()
            out.update(("base_module" + "__" + k, val) for k, val in deep_items)
            out["base_module"] = self.base_module
        return out

    def _identify_icvi_cluster(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        """
        Find the best cluster using iCVIs

        Parameters:
        - x: data sample
        - match_reset_func:
            a callable accepting the data sample, a cluster weight, cluster_label, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise

        Returns:
            cluster label of the input sample

        """

        icvi_values = []
        icvi_copies = []
        for k in range(self.n_clusters+1):
            if k < self.n_clusters and match_reset_func is not None:
                no_match_reset = match_reset_func(x, self.W[k], k, self.params, None)
            else:
                no_match_reset = True
            if no_match_reset:
                local_icvi = deepcopy(self.iCVI)
                icvi_val = local_icvi.get_cvi(x, k)
                print(icvi_val)
            else:
                local_icvi = None
                icvi_val = -1
            icvi_values.append(icvi_val)
            icvi_copies.append(local_icvi)

        c_ = int(np.argmin(icvi_values))
        print("--------")
        self.iCVI = icvi_copies[c_]
        return c_


    def match_reset_func(
            self,
            i: np.ndarray,
            w: np.ndarray,
            cluster_a,
            params: dict,
            extra: dict,
            cache: Optional[dict] = None
    ) -> bool:
        """
        Permits external factors to influence cluster creation.

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - cluster_a: a-side cluster label
        - params: dict containing parameters for the algorithm
        - extra: additional parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            true if match is permitted

        """
        cluster_b = extra["cluster_b"]
        if cluster_a in self.map and self.map[cluster_a] != cluster_b:
            return False
        return True


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
        pass

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        """
        fit the model to a single sample

        Parameters:
        - x: data sample
        - match_reset_func:
            a callable accepting the data sample, a cluster weight, cluster_label, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise

        Returns:
            cluster label of the input sample

        """
        c_b = self._identify_icvi_cluster(x, match_reset_func)
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"cluster_b": c_b}, cache=cache
        )
        c_a = self.base_module.step_fit(x, match_reset_func=match_reset_func)
        if c_a not in self.map:
            self.map[c_a] = c_b
        else:
            assert self.map[c_a] == c_b
        return c_a


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
