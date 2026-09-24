"""Dual Vigilance ART :cite:`da2019dual`."""
# Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
# Dual vigilance fuzzy adaptive resonance theory.
# Neural Networks, 109, 1–5. doi:10.1016/j.neunet.2018.09.015.
import numpy as np
from typing import Optional, Callable, List, Literal, Union, Dict
from warnings import warn
from copy import deepcopy
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import IndexableOrKeyable


class DualVigilanceART(BaseART):
    """Dual Vigilance ART for Clustering.

    This module implements Dual Vigilance ART as first published in: :cite:`da2019dual`.

    .. # Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
    .. # Dual vigilance fuzzy adaptive resonance theory.
    .. # Neural Networks, 109, 1–5. doi:10.1016/j.neunet.2018.09.015.

    Dual Vigilance ART allows a :class:`~artlib.common.BaseART.BaseART` module to
    cluster with both an upper and lower vigilance value. The upper-vigilance value
    allows the :class:`~artlib.common.BaseART.BaseART` module to cluster normally,
    however, data is simultaneously clustered using the lower vigilance level to
    combine multiple base ART categories into a single abstracted category. This
    permits clusters to be combined to form arbitrary shapes. For example if the
    :class:`~artlib.common.BaseART.BaseART` module is
    :class:`~artlib.elementary.FuzzyART.FuzzyART`, a Dual Vigilance Fuzzy ART
    clustering result would look  like a series of hyper-boxes forming an arbitrary
    geometry.

    """

    def __init__(self, base_module: BaseART, rho_lower_bound: float):
        """Initialize the Dual Vigilance ART model.

        Parameters
        ----------
        base_module : BaseART
            The instantiated ART module that will serve as the base for dual vigilance.
        rho_lower_bound : float
            The lower vigilance value that will "merge" the base_module clusters.

        """
        assert isinstance(base_module, BaseART)
        if hasattr(base_module, "base_module"):
            warn(
                f"{base_module.__class__.__name__} "
                f"is an abstraction of the BaseART class. "
                f"This module will only make use of the base_module: "
                f"{base_module.base_module.__class__.__name__}"
            )
        assert "rho" in base_module.params, (
            "Dual Vigilance ART is only compatible with ART modules "
            "relying on 'rho' for vigilance."
        )

        params = {"rho_lower_bound": rho_lower_bound}
        assert base_module.params["rho"] > params["rho_lower_bound"] >= 0
        self.base_module = base_module
        super().__init__(params)
        self.rho_lower_bound = rho_lower_bound
        self.map: dict[int, int] = dict()
        self._prototype_labels_: list[int] = []

    @property
    def prototype_labels_(self) -> np.ndarray:
        """Base prototype selected for each sample in ``labels_``."""
        return np.asarray(self._prototype_labels_, dtype=int)

    def _reset_fit_state(self):
        """Discard the category mapping and sample history for a fresh fit."""
        self.map = {}
        self._prototype_labels_ = []
        self.base_module.weight_sample_counter_ = []
        self.sample_counter_ = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter: int = 1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit and retain the final pass's base prototype assignments."""
        self._reset_fit_state()
        super().fit(
            X,
            y=y,
            match_reset_func=match_reset_func,
            max_iter=max_iter,
            match_tracking=match_tracking,
            epsilon=epsilon,
            verbose=verbose,
            leave_progress_bar=leave_progress_bar,
        )
        if len(self.labels_):
            self._prototype_labels_ = self._prototype_labels_[-len(self.labels_) :]
        else:
            self._prototype_labels_ = []
        return self

    def fit_gif(self, X: np.ndarray, *args, **kwargs):
        """Fit with animation while retaining final prototype assignments."""
        self._reset_fit_state()
        super().fit_gif(X, *args, **kwargs)
        if len(self.labels_):
            self._prototype_labels_ = self._prototype_labels_[-len(self.labels_) :]
        else:
            self._prototype_labels_ = []
        return self

    def _validate_abstract_cluster(self, idx: int):
        if not isinstance(idx, (int, np.integer)) or isinstance(idx, (bool, np.bool_)):
            raise TypeError("Cluster indices must be integers")
        if idx not in self.map.values():
            raise IndexError(f"Abstract cluster index {idx} is out of range")

    def merge(self, target_idx: int, source_idx: int) -> int:
        """Merge two abstract clusters without changing base prototypes.

        Both indices refer to abstract cluster labels in ``map``. Return the
        target's index after removing and renumbering the source cluster.

        """
        self._validate_abstract_cluster(target_idx)
        self._validate_abstract_cluster(source_idx)
        if target_idx == source_idx:
            raise ValueError("Cannot merge a cluster with itself")

        def remap(label: int) -> int:
            label = target_idx if label == source_idx else label
            return int(label - (label > source_idx))

        new_map = {proto: remap(label) for proto, label in self.map.items()}
        new_labels = np.array([remap(label) for label in self.labels_], dtype=int)
        self.map = new_map
        self.labels_ = new_labels
        return remap(target_idx)

    def move_prototype(
        self,
        source_cluster_idx: int,
        source_prototype_idx: int,
        target_cluster_idx: int,
    ) -> int:
        """Assign a base prototype to another existing abstract cluster.

        ``source_prototype_idx`` indexes ``base_module.W``. The source and
        target cluster indices are abstract labels in ``map``. Return the
        target's index after any renumbering.

        Stored sample labels are recomputed from their recorded base prototype
        assignments. If the source becomes empty, later cluster IDs shift down.

        """
        self._validate_abstract_cluster(source_cluster_idx)
        self._validate_abstract_cluster(target_cluster_idx)
        if not isinstance(source_prototype_idx, (int, np.integer)) or isinstance(
            source_prototype_idx, (bool, np.bool_)
        ):
            raise TypeError("Prototype index must be an integer")
        if not 0 <= source_prototype_idx < self.base_module.n_clusters:
            raise IndexError("Source prototype index is out of range")
        if self.map.get(source_prototype_idx) != source_cluster_idx:
            raise ValueError("Source prototype is not in the source cluster")
        if source_cluster_idx == target_cluster_idx:
            raise ValueError("Source and target clusters must differ")
        history = self.__dict__.get("_prototype_labels_")
        if history is None or len(history) != len(self.labels_):
            raise RuntimeError(
                "Base prototype assignments are unavailable for stored samples"
            )

        new_map = dict(self.map)
        new_map[source_prototype_idx] = int(target_cluster_idx)
        if source_cluster_idx not in new_map.values():
            new_map = {
                proto: int(label - (label > source_cluster_idx))
                for proto, label in new_map.items()
            }
        new_labels = np.array([new_map[proto] for proto in history], dtype=int)
        self.map = new_map
        self.labels_ = new_labels
        return new_map[source_prototype_idx]

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        Returns
        -------
        np.ndarray
            Prepared data from the base module.

        """
        return self.base_module.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to its state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        Returns
        -------
        np.ndarray
            Restored data from the base module.

        """
        return self.base_module.restore_data(X)

    def get_params(self, deep: bool = True) -> dict:
        """Get the parameters of the estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, return the parameters for this class and contained subobjects that
            are estimators, by default True.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        out = {
            "rho_lower_bound": self.params["rho_lower_bound"],
            "base_module": self.base_module,
        }
        if deep:
            deep_items = self.base_module.get_params().items()
            out.update(("base_module" + "__" + k, val) for k, val in deep_items)

        return out

    @property
    def n_clusters(self) -> int:
        """Get the current number of clusters.

        Returns
        -------
        int
            The number of clusters.

        """
        return len(set(c for c in self.map.values()))

    @property
    def dim_(self):
        """Get the dimensionality of the data from the base module.

        Returns
        -------
        int
            Dimensionality of the data.

        """
        return self.base_module.dim_

    @dim_.setter
    def dim_(self, new_dim):
        self.base_module.dim_ = new_dim

    @property
    def labels_(self):
        """Get the labels from the base module.

        Returns
        -------
        np.ndarray
            Labels for the data.

        """
        return self.base_module.labels_

    @labels_.setter
    def labels_(self, new_labels: np.ndarray):
        self.base_module.labels_ = new_labels

    @property
    def W(self) -> List:
        """Get the weights from the base module.

        Returns
        -------
        list of np.ndarray
            Weights of the clusters.

        """
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        self.base_module.W = new_W

    def check_dimensions(self, X: np.ndarray):
        """Check that the data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        self.base_module.check_dimensions(X)

    def validate_data(self, X: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        self.base_module.validate_data(X)
        self.check_dimensions(X)

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert (
            "rho_lower_bound" in params
        ), "Dual Vigilance ART requires a lower bound 'rho' value"
        assert params["rho_lower_bound"] >= 0
        assert isinstance(params["rho_lower_bound"], float)

    def _match_tracking(
        self,
        cache: Union[List[Dict], Dict],
        epsilon: float,
        params: Union[List[Dict], Dict],
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Adjust match tracking based on the method and epsilon value.

        Parameters
        ----------
        cache : dict
            Cache containing intermediate results, including the match criterion.
        epsilon : float
            Adjustment factor for the match criterion.
        params : dict
            Dictionary containing algorithm parameters.
        method : {"MT+", "MT-", "MT0", "MT1", "MT~"}
            Match tracking method to use.

        Returns
        -------
        bool
            True if match tracking continues, False otherwise.

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
        self.base_module.params = new_params

    def _deep_copy_params(self) -> dict:
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
        match_reset_func : callable, optional
            A callable accepting the data sample, a cluster weight, the params dict,
            and the cache dict.
            Returns True if the cluster is valid for the sample, False otherwise.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting match criterion, by default "MT+".
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 0.0.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        base_params = self._deep_copy_params()
        mt_operator = self._match_tracking_operator(match_tracking)
        self.sample_counter_ += 1
        if len(self.base_module.W) == 0:
            new_w = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.add_weight(new_w)
            self.map[0] = 0
            self._prototype_labels_.append(0)
            return 0
        else:
            T_values, T_cache = zip(
                *[
                    self.base_module.category_choice(
                        x, w, params=self.base_module.params
                    )
                    for w in self.base_module.W
                ]
            )
            T = np.array(T_values)

            # Candidates are those with positive T (and not NaN)
            valid = ~np.isnan(T)
            if np.any(valid):
                idx = np.arange(T.shape[0])[valid]
                T_valid = T[valid]
                order = idx[
                    np.lexsort((idx, -T_valid))
                ]  # primary: -T (desc), secondary: idx (asc)
            else:
                order = np.array([], dtype=int)

            for c_ in order:
                w = self.base_module.W[c_]
                cache = T_cache[c_]
                m1, cache = self.base_module.match_criterion_bin(
                    x,
                    w,
                    params=self.base_module.params,
                    cache=cache,
                    op=mt_operator,
                )
                no_match_reset = match_reset_func is None or match_reset_func(
                    x,
                    w,
                    self.map[c_],
                    params=self.base_module.params,
                    cache=cache,
                )

                if no_match_reset:
                    if m1:
                        new_w = self.base_module.update(
                            x, w, self.base_module.params, cache=cache
                        )
                        self.base_module.set_weight(c_, new_w)
                        self._set_params(base_params)
                        self._prototype_labels_.append(int(c_))
                        return self.map[c_]
                    else:
                        lb_params = dict(
                            self.base_module.params, **{"rho": self.rho_lower_bound}
                        )
                        m2, _ = self.base_module.match_criterion_bin(
                            x, w, params=lb_params, cache=cache, op=mt_operator
                        )
                        if m2:
                            c_new = len(self.base_module.W)
                            w_new = self.base_module.new_weight(
                                x, self.base_module.params
                            )
                            self.base_module.add_weight(w_new)
                            self.map[c_new] = self.map[c_]
                            self._set_params(base_params)
                            self._prototype_labels_.append(c_new)
                            return self.map[c_new]
                else:
                    keep_searching = self._match_tracking(
                        cache, epsilon, self.params, match_tracking
                    )
                    if not keep_searching:
                        break

            c_new = len(self.base_module.W)
            w_new = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.add_weight(w_new)
            self.map[c_new] = max(self.map.values()) + 1
            self._set_params(base_params)
            self._prototype_labels_.append(c_new)
            return self.map[c_new]

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
        """Get the centers of each cluster, used for regression.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return self.base_module.get_cluster_centers()

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Visualize the bounds of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        colors_base = []
        for k_a in range(self.base_module.n_clusters):
            colors_base.append(colors[self.map[k_a]])

        try:
            self.base_module.plot_cluster_bounds(
                ax=ax, colors=colors_base, linewidth=linewidth
            )
        except NotImplementedError:
            warn(
                f"{self.base_module.__class__.__name__} "
                f"does not support plotting cluster bounds."
            )
