"""BARTMAP :cite:`xu2011bartmap`, :cite:`xu2012biclustering`."""
# Xu, R., & Wunsch II, D. C. (2011).
# BARTMAP: A viable structure for biclustering.
# Neural Networks, 24, 709–716. doi:10.1016/j.neunet.2011.03.020.
#
# Xu, R., Wunsch II, D. C., & Kim, S. (2012).
# Methods and systems for biclustering algorithm.
# U.S. Patent 9,043,326 Filed January 28, 2012,
# claiming priority to Provisional U.S. Patent Application,
# January 28, 2011, issued May 26, 2015.

import numpy as np
from typing import TYPE_CHECKING, Optional
from collections import defaultdict
from artlib.common.BaseART import BaseART
from sklearn.base import BaseEstimator, BiclusterMixin

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


class BARTMAP(BaseEstimator, BiclusterMixin):
    """BARTMAP for Biclustering.

    This class implements BARTMAP as first published in:
    :cite:`xu2011bartmap`.

    .. # Xu, R., & Wunsch II, D. C. (2011).
    .. # BARTMAP: A viable structure for biclustering.
    .. # Neural Networks, 24, 709–716. doi:10.1016/j.neunet.2011.03.020.

    BARTMAP accepts two instantiated :class:`~artlib.common.BaseART.BaseART` modules
    `module_a` and `module_b` which cluster the rows (samples) and columns (features)
    respectively. The features are clustered independently, but the samples are
    clustered by considering samples already within a row cluster as well as the
    candidate sample and enforcing a minimum correlation within the subset of
    features belonging to at least one of the feature clusters.

    """

    rows_: np.ndarray  # bool
    columns_: np.ndarray  # bool

    def __init__(self, module_a: BaseART, module_b: BaseART, eta: float):
        """Initialize the BARTMAP model.

        Parameters
        ----------
        module_a : BaseART
            The instantiated ART module used for clustering the rows (samples).
        module_b : BaseART
            The instantiated ART module used for clustering the columns (features).
        eta : float
            The minimum Pearson correlation required for row clustering.

        """
        params: dict = {"eta": eta}
        self.validate_params(params)
        self.params = params
        self.module_a = module_a
        self.module_b = module_b

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

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters for this estimator and contained subobjects
            that are estimators.

        Returns
        -------
        dict
            Dictionary of parameter names mapped to their values.

        """
        out = dict(self.params)

        deep_a_items = self.module_a.get_params().items()
        out.update(("module_a" + "__" + k, val) for k, val in deep_a_items)
        out["module_a"] = self.module_a

        deep_b_items = self.module_b.get_params().items()
        out.update(("module_b" + "__" + k, val) for k, val in deep_b_items)
        out["module_b"] = self.module_b
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Specific redefinition of `sklearn.BaseEstimator.set_params` for ART classes.

        Parameters
        ----------
        **params : dict
            Estimator parameters as keyword arguments.

        Returns
        -------
        self : object
            The estimator instance.

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        local_params = dict()

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

    @staticmethod
    def validate_params(params: dict):
        """Validate clustering parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for the algorithm.

        """
        assert "eta" in params
        assert isinstance(params["eta"], float)

    @property
    def column_labels_(self) -> np.ndarray:
        """Cluster labels for the columns.

        Returns
        -------
        column_labels_ : ndarray of shape (n_columns,)
            Array of cluster labels assigned to each column.

        """
        return self.module_b.labels_

    @property
    def row_labels_(self) -> np.ndarray:
        """Cluster labels for the rows.

        Returns
        -------
        row_labels_ : ndarray of shape (n_rows,)
            Array of cluster labels assigned to each row.

        """
        return self.module_a.labels_

    @property
    def n_row_clusters(self) -> int:
        """Number of row clusters.

        Returns
        -------
        n_row_clusters : int
            The number of clusters for the rows.

        """
        return self.module_a.n_clusters

    @property
    def n_column_clusters(self) -> int:
        """Number of column clusters.

        Returns
        -------
        n_column_clusters : int
            The number of clusters for the columns.

        """
        return self.module_b.n_clusters

    def _get_x_cb(self, x: np.ndarray, c_b: int):
        """Get the components of a vector belonging to a b-side cluster.

        Parameters
        ----------
        x : np.ndarray
            A sample vector.
        c_b : int
            The b-side cluster label.

        Returns
        -------
        np.ndarray
            The sample vector `x` filtered to include only features belonging to
            the b-side cluster `c_b`.

        """
        b_components = self.module_b.labels_ == c_b
        return x[b_components]

    @staticmethod
    def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
        """Get the Pearson correlation between two vectors.

        Parameters
        ----------
        a : np.ndarray
            A vector.
        b : np.ndarray
            Another vector.

        Returns
        -------
        float
            The Pearson correlation between the two vectors `a` and `b`, or NaN
            when it is undefined (fewer than two values, or a constant vector).

        """
        return float(BARTMAP._pearsonr_rows(np.asarray(a)[None, :], np.asarray(b))[0])

    @staticmethod
    def _pearsonr_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pearson correlation between each row of ``A`` and the vector ``b``.

        Computed the same way as :func:`scipy.stats.pearsonr`, but for all rows at
        once.

        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if b.shape[0] < 2:
            return np.full(A.shape[0], np.nan)
        A_c = A - A.mean(axis=1, keepdims=True)
        b_c = b - b.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            A_n = A_c / np.linalg.norm(A_c, axis=1, keepdims=True)
            b_n = b_c / np.linalg.norm(b_c)
            r = A_n @ b_n
        return np.clip(r, -1.0, 1.0)

    def _row_members(self, k: int, cluster_a: Optional[int]) -> np.ndarray:
        """Rows that sample ``k`` is compared with.

        These are the rows already assigned to row cluster ``cluster_a``, or every
        other row when ``cluster_a`` is None.

        """
        n = self.X.shape[0]
        if cluster_a is None:
            members = np.ones(n, dtype=bool)
        else:
            members = np.asarray(self.module_a.labels_) == cluster_a
            seen = self.__dict__.get("_rows_seen_")
            if seen is not None and len(seen) == n:
                members &= seen
        members[k] = False
        return members

    def _average_pearson_corr(
        self, X: np.ndarray, k: int, c_b: int, cluster_a: Optional[int] = None
    ) -> float:
        """Get the average Pearson correlation between a sample and the rows of a row
        cluster, over the features in column cluster ``c_b``.

        Parameters
        ----------
        X : np.ndarray
            The dataset A.
        k : int
            The sample index.
        c_b : int
            The b-side cluster to check.
        cluster_a : int, optional
            The a-side (row) cluster to compare with. If None, sample ``k`` is
            compared with every other row.

        Returns
        -------
        float
            The average Pearson correlation, or NaN if there are no rows to compare
            with or the correlation is undefined.

        """
        members = self._row_members(k, cluster_a)
        if not members.any():
            return float("nan")
        columns = self.column_labels_ == c_b
        r = self._pearsonr_rows(X[members][:, columns], X[k, columns])
        return float(np.mean(r))

    def validate_data(self, X_a: np.ndarray, X_b: np.ndarray):
        """Validate the data prior to clustering.

        Parameters
        ----------
        X_a : np.ndarray
            Dataset A, containing the samples.
        X_b : np.ndarray
            Dataset B, containing the features.

        """
        self.module_a.validate_data(X_a)
        self.module_b.validate_data(X_b)

    def match_criterion_bin(
        self,
        X: np.ndarray,
        k: int,
        c_b: int,
        params: dict,
        cluster_a: Optional[int] = None,
    ) -> bool:
        """Get the binary match criterion of the cluster.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        k : int
            The sample index.
        c_b : int
            The b-side cluster to check.
        params : dict
            Dictionary containing parameters for the algorithm.
        cluster_a : int, optional
            The a-side (row) cluster to compare with. If None, sample ``k`` is
            compared with every other row.

        Returns
        -------
        bool
            Binary value indicating whether the cluster match criterion is met.

        """
        M = self._average_pearson_corr(X, k, c_b, cluster_a)
        return bool(M >= self.params["eta"])

    def match_reset_func(
        self,
        i: np.ndarray,
        w: np.ndarray,
        cluster_a,
        params: dict,
        extra: dict,
        cache: Optional[dict] = None,
    ) -> bool:
        """Permit external factors to influence cluster creation.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight or information.
        cluster_a : int
            A-side cluster label.
        params : dict
            Dictionary containing parameters for the algorithm.
        extra : dict
            Additional parameters for the algorithm.
        cache : dict, optional
            Dictionary containing values cached from previous calculations.

        Returns
        -------
        bool
            True if the match is permitted, otherwise False.

        """
        k = extra["k"]
        if not self._row_members(k, cluster_a).any():
            # No other rows in this cluster yet, so nothing to correlate with.
            return True
        for cluster_b in range(len(self.module_b.W)):
            if self.match_criterion_bin(self.X, k, cluster_b, params, cluster_a):
                return True
        return False

    def step_fit(self, X: np.ndarray, k: int) -> int:
        """Fit the model to a single sample.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        k : int
            The sample index.

        Returns
        -------
        int
            The cluster label of the input sample.

        """
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"k": k}, cache=cache
        )
        c_a = self.module_a.step_fit(X[k, :], match_reset_func=match_reset_func)
        return c_a

    def fit(self, X: np.ndarray, max_iter=1):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset to fit the model on.
        max_iter : int
            The number of iterations to fit the model on the same dataset.

        """
        # Check that X and y have correct shape
        self.X = X

        n = X.shape[0]
        X_a = self.module_a.prepare_data(X)
        X_b = self.module_b.prepare_data(X.T)
        self.validate_data(X_a, X_b)

        self.module_b = self.module_b.fit(X_b, max_iter=max_iter)

        # init module A
        self.module_a.W = []
        self.module_a._reset_sample_counters()
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
        # Rows not yet seen have no row cluster, so they are not compared with.
        self._rows_seen_ = np.zeros(n, dtype=bool)

        for _ in range(max_iter):
            for k in range(n):
                self.module_a.pre_step_fit(X_a)
                c_a = self.step_fit(X_a, k)
                self.module_a.labels_[k] = c_a
                self._rows_seen_[k] = True
                self.module_a.post_step_fit(X_a)

        self.rows_ = np.vstack(
            [
                self.row_labels_ == label
                for label in range(self.module_a.n_clusters)
                for _ in range(self.module_b.n_clusters)
            ]
        )
        self.columns_ = np.vstack(
            [
                self.column_labels_ == label
                for _ in range(self.module_a.n_clusters)
                for label in range(self.module_b.n_clusters)
            ]
        )
        return self

    def visualize(self, cmap: Optional["Colormap"] = None):
        """Visualize the clustering of the data.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap or str
            The colormap to use for visualization.

        """
        import matplotlib.pyplot as plt

        if cmap is None:
            cmap = plt.cm.Blues

        plt.matshow(
            np.outer(np.sort(self.row_labels_) + 1, np.sort(self.column_labels_) + 1),
            cmap=cmap,
        )
