"""CVIART."""
import numpy as np
from copy import deepcopy
import sklearn.metrics as metrics
from typing import Optional, List, Callable, Literal, Iterable
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART


class CVIART(BaseART):
    """CVI Art Classification.

    Expanded version of Art that uses Cluster Validity Indicies to help with cluster
    selection. PBM is not implemented, can be seen here.
    git.mst.edu/acil-group/CVI-Fuzzy-ART/-/blob/master/PBM_index.m?ref_type=heads

    Note, the default step_fit function in base ART evaluates the matching function
    even if the other criteria has failed. This means it could run slower then it would
    otherwise.

    """

    CALINSKIHARABASZ = 1
    DAVIESBOULDIN = 2
    SILHOUETTE = 3
    # PBM = 4

    def __init__(self, base_module: BaseART, validity: int):
        """Initialize the CVIART model.

        Parameters
        ----------
        base_module : BaseART
            Base ART module for clustering.
        validity : int
            Validity index used for cluster evaluation.

        """
        self.base_module = base_module
        params = dict(base_module.params, **{"validity": validity})
        super().__init__(params)
        print(self.params)

    def validate_params(self, params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        self.base_module.validate_params(params)
        assert "validity" in params
        assert isinstance(params["validity"], int)
        assert params["validity"] in [
            CVIART.CALINSKIHARABASZ,
            CVIART.DAVIESBOULDIN,
            CVIART.SILHOUETTE,
        ]

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            Dataset to be normalized.

        Returns
        -------
        np.ndarray
            Normalized data.

        """
        return self.base_module.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """Restore data to state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            Dataset to be restored.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        return self.base_module.restore_data(X)

    @property
    def W(self) -> List:
        """Get the base module weights.

        Returns
        -------
        list of np.ndarray
            base module weights

        """
        return self.base_module.W

    @W.setter
    def W(self, new_W):
        self.base_module.W = new_W

    @property
    def labels_(self) -> np.ndarray:
        """Get the base module labels.

        Returns
        -------
        np.ndarray
            base module labels

        """
        return self.base_module.labels_

    @labels_.setter
    def labels_(self, new_labels_):
        self.base_module.labels_ = new_labels_

    def CVI_match(self, x, w, c_, params, extra, cache):
        """Evaluate the cluster validity index (CVI) for a match.

        Parameters
        ----------
        x : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight information.
        c_ : int
            Cluster index.
        params : dict
            Parameters for the algorithm.
        extra : dict
            Extra information including index and validity type.
        cache : dict
            Cache containing values from previous calculations.

        Returns
        -------
        bool
            True if the new validity score improves the clustering, False otherwise.

        """
        if len(self.W) < 2:
            return True

        if extra["validity"] == self.CALINSKIHARABASZ:
            valid_func = metrics.calinski_harabasz_score
        elif extra["validity"] == self.DAVIESBOULDIN:
            valid_func = metrics.davies_bouldin_score
        elif extra["validity"] == self.SILHOUETTE:
            valid_func = metrics.silhouette_score
        else:
            raise ValueError(f"Invalid Validity Parameter: {extra['validity']}")

        old_VI = valid_func(self.data, self.labels_)
        new_labels = np.copy(self.labels_)
        new_labels[extra["index"]] = c_
        new_VI = valid_func(self.data, new_labels)
        if extra["validity"] != self.DAVIESBOULDIN:
            return new_VI > old_VI
        else:
            return new_VI < old_VI

    def _match_tracking(
        self,
        cache: dict,
        epsilon: float,
        params: dict,
        method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"],
    ) -> bool:
        """Adjust the vigilance parameter (rho) based on the match tracking method.

        Parameters
        ----------
        cache : dict
            Cache containing match criterion.
        epsilon : float
            Epsilon value for adjusting the vigilance parameter.
        params : dict
            Parameters for the algorithm.
        method : {"MT+", "MT-", "MT0", "MT1", "MT~"}
            Method for resetting match criterion.

        Returns
        -------
        bool
            True if further matching is required, False otherwise.

        """
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

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray, optional
            Not used. For compatibility.
        match_reset_func : callable, optional
            A callable accepting the data sample, a cluster weight, the params dict,
            and the cache dict.
            Returns True if the cluster is valid for the sample, False otherwise.
        max_iter : int, optional
            Number of iterations to fit the model on the same dataset, by default 1.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting match criterion.
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 0.0.

        """
        self.data = X
        self.base_module.validate_data(X)
        self.base_module.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0],), dtype=int)
        for _ in range(max_iter):
            for index, x in enumerate(X):
                self.pre_step_fit(X)
                if match_reset_func is None:
                    cvi_match_reset_func = (
                        lambda i, w, cluster_a, params, cache: self.CVI_match(
                            i,
                            w,
                            cluster_a,
                            params,
                            {
                                "index": index,
                                "validity": self.params["validity"],
                            },
                            cache,
                        )
                    )
                else:
                    cvi_match_reset_func = lambda i, w, cluster_a, params, cache: (
                        match_reset_func(i, w, cluster_a, params, cache)
                        and self.CVI_match(
                            i,
                            w,
                            cluster_a,
                            params,
                            {
                                "index": index,
                                "validity": self.params["validity"],
                            },
                            cache,
                        )
                    )
                c = self.base_module.step_fit(
                    x,
                    match_reset_func=cvi_match_reset_func,
                    match_tracking=match_tracking,
                    epsilon=epsilon,
                )
                self.labels_[index] = c
                self.post_step_fit(X)

    def pre_step_fit(self, X: np.ndarray):
        """Preprocessing step before fitting each sample.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        return self.base_module.pre_step_fit(X)

    def post_step_fit(self, X: np.ndarray):
        """Postprocessing step after fitting each sample.

        Parameters
        ----------
        X : np.ndarray
            The dataset.

        """
        return self.base_module.post_step_fit(X)

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
            Method for resetting match criterion.
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 0.0.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        raise NotImplementedError

    def step_pred(self, x: np.ndarray) -> int:
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
        return self.base_module.step_pred(x)

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Get the centers of the clusters.

        Returns
        -------
        list of np.ndarray
            Cluster centroids.

        """
        return self.base_module.get_cluster_centers()

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """Plot the boundaries of each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, optional
            Width of boundary line, by default 1.

        """
        return self.base_module.plot_cluster_bounds(ax, colors, linewidth)
