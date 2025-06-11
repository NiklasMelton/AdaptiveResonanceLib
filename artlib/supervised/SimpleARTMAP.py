"""Simple ARTMAP :cite:`gotarredona1998adaptive`."""
# Serrano-Gotarredona, T., Linares-Barranco, B., & Andreou, A. G. (1998).
# Adaptive Resonance Theory Microchips: Circuit Design Techniques.
# Norwell, MA, USA: Kluwer Academic Publishers.
import numpy as np
from typing import Optional, Literal, Dict, Union, Tuple
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.BaseARTMAP import BaseARTMAP
from artlib.common.utils import IndexableOrKeyable
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels


class SimpleARTMAP(BaseARTMAP):
    """SimpleARTMAP for Classification.

    This module implements SimpleARTMAP as first published in:
    :cite:`gotarredona1998adaptive`.

    .. # Serrano-Gotarredona, T., Linares-Barranco, B., & Andreou, A. G. (1998).
    .. # Adaptive Resonance Theory Microchips: Circuit Design Techniques.
    .. # Norwell, MA, USA: Kluwer Academic Publishers.

    SimpleARTMAP is a special case of :class:`~artlib.supervised.ARTMAP.ARTMAP`
    specifically for classification. It allows the clustering of data samples while
    enforcing a many-to-one mapping from sample clusters to labels. It accepts an
    instantiated :class:`~artlib.common.BaseART.BaseART` module and dynamically adapts
    the vigilance function to prevent resonance when the many-to-one mapping is
    violated. This enables SimpleARTMAP to identify discrete clusters belonging to
    each category label.

    """

    def __init__(self, module_a: BaseART):
        """Initialize SimpleARTMAP.

        Parameters
        ----------
        module_a : BaseART
            The instantiated ART module used for clustering the independent channel.

        """
        self.module_a = module_a
        super().__init__()

    def match_reset_func(
        self,
        i: np.ndarray,
        w: np.ndarray,
        cluster_a,
        params: dict,
        extra: dict,
        cache: Optional[dict] = None,
    ) -> bool:
        """Permits external factors to influence cluster creation.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight / info.
        cluster_a : int
            A-side cluster label.
        params : dict
            Parameters for the algorithm.
        extra : dict
            Additional parameters, including "cluster_b".
        cache : dict, optional
            Values cached from previous calculations.

        Returns
        -------
        bool
            True if the match is permitted, False otherwise.

        """
        cluster_b = extra["cluster_b"]
        if cluster_a in self.map and self.map[cluster_a] != cluster_b:
            return False
        return True

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters of the model.

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
        out = {"module_a": self.module_a}
        if deep:
            deep_items = self.module_a.get_params().items()
            out.update(("module_a" + "__" + k, val) for k, val in deep_items)
        return out

    def validate_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The validated datasets X and y.

        """
        X, y = check_X_y(X, y, dtype=None)
        self.module_a.validate_data(X)
        return X, y

    def prepare_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for clustering.

        Parameters
        ----------
        X : np.ndarray
            Data set.
        y : Optional[np.ndarray]
            Data set B. Not used in SimpleARTMAP

        Returns
        -------
        np.ndarray
            Prepared data.

        """
        return self.module_a.prepare_data(X)

    def restore_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Restore data to state prior to preparation.

        Parameters
        ----------
        X : np.ndarray
            Data set.

        Returns
        -------
        np.ndarray
            Restored data.

        """
        return self.module_a.restore_data(X)

    def step_fit(
        self,
        x: np.ndarray,
        c_b: int,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ) -> int:
        """Fit the model to a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample for side A.
        c_b : int
            Side B label.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.

        Returns
        -------
        int
            Side A cluster label.

        """
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i,
            w,
            cluster,
            params=params,
            extra={"cluster_b": c_b},
            cache=cache,
        )
        c_a = self.module_a.step_fit(
            x,
            match_reset_func=match_reset_func,
            match_tracking=match_tracking,
            epsilon=epsilon,
        )
        if c_a not in self.map:
            self.map[c_a] = c_b
        else:
            assert self.map[c_a] == c_b
        return c_a

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.
        max_iter : int, default=1
            Number of iterations to fit the model on the same data set.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.
        verbose : bool, default=False
            If True, displays a progress bar during training.
        leave_progress_bar : bool, default=True
            If True, leaves thge progress of the fitting process. Only used when
            verbose=True

        Returns
        -------
        self : SimpleARTMAP
            The fitted model.

        """
        # Check that X and y have correct shape
        SimpleARTMAP.validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.labels_ = y
        # init module A
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        for _ in range(max_iter):
            if verbose:
                from tqdm import tqdm

                x_y_iter = tqdm(
                    enumerate(zip(X, y)),
                    total=int(X.shape[0]),
                    leave=leave_progress_bar,
                )
            else:
                x_y_iter = enumerate(zip(X, y))
            for i, (x, c_b) in x_y_iter:
                self.module_a.pre_step_fit(X)
                c_a = self.step_fit(
                    x,
                    c_b,
                    match_tracking=match_tracking,
                    epsilon=epsilon,
                )
                self.module_a.labels_[i] = c_a
                self.module_a.post_step_fit(X)
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit the model to the data and return the labels. Need to define this or
        ClusterMixin could cause issues.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.
        max_iter : int, default=1
            Number of iterations to fit the model on the same data set.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.
        verbose : bool, default=False
            If True, displays a progress bar during training.
        leave_progress_bar : bool, default=True
            If True, leaves thge progress of the fitting process. Only used when
            verbose=True

        Returns
        -------
        np.ndarray
            The labels (same as y).

        """
        self.fit(X, y, max_iter, match_tracking, epsilon, verbose, leave_progress_bar)
        return self.labels_

    def fit_gif(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
        ax: Optional[Axes] = None,
        filename: Optional[str] = None,
        fps: int = 5,
        final_hold_secs: float = 0.0,
        colors: Optional[IndexableOrKeyable] = None,
        n_class_estimate: Optional[int] = None,
        max_iter: int = 1,
        **kwargs,
    ):
        """Fit the model while recording the learning process as an animated GIF.

        The routine iterates over the training data, calling
        :py:meth:`step_fit` for each sample, and captures intermediate plots by
        repeatedly invoking :py:meth:`visualize`.  All frames are written to a
        GIF file (via ``matplotlib.animation.PillowWriter``), allowing an
        intuitive, frame‑by‑frame view of how clusters form and adjust over
        time.

        Parameters
        ----------
        X : np.ndarray
            Independent‑channel samples (side A), shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target labels (side B), shape ``(n_samples,)``.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Strategy used by the ART module to reset its match criterion.
        epsilon : float, default=1e‑10
            Small positive constant added to the vigilance when
            ``match_tracking`` triggers a reset.
        verbose : bool, default=False
            If ``True``, displays a tqdm progress bar for each epoch.
        leave_progress_bar : bool, default=True
            If ``True``, leaves the progress bar visible after completion
            (only relevant when ``verbose`` is ``True``).
        ax : matplotlib.axes.Axes, optional
            Existing axes on which to draw frames.  If ``None``, a new figure
            and axes are created.
        filename : str, optional
            Output path for the GIF.  Defaults to
            ``"fit_gif_supervised_<ClassName>.gif"`` if ``None``.
        fps : int, default=5
            Frames per second in the resulting GIF.
        final_hold_secs : float, default=0.0
            Extra seconds to hold the final frame (duplicates the last plot
            ``ceil(final_hold_secs * fps)`` times).
        colors : array‑like, optional
            Sequence of colors to use for each class when plotting.  If
            ``None``, a rainbow colormap is generated.
        n_class_estimate : int, optional
            Expected number of distinct classes.  Only used when ``colors`` is
            ``None`` to size the autogenerated colormap.
        max_iter : int, default=1
            Number of complete passes over ``(X, y)``.
        **kwargs
            Additional keyword arguments forwarded to
            :py:meth:`visualize` (e.g., ``marker_size``, ``linewidth``).

        Returns
        -------
        self : SimpleARTMAP
            The fitted estimator (identical object, returned for chaining).

        Notes
        -----
        * Generates a GIF file as a **side‑effect**.  The estimator itself is
          updated exactly as in :py:meth:`fit`; only plotting calls and file
          I/O are added.
        * For reproducible colors across different runs, supply an explicit
          ``colors`` array rather than relying on the rainbow colormap.

        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
        from sklearn.utils.multiclass import unique_labels

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)

        if filename is None:
            filename = f"fit_gif_supervised_{self.__class__.__name__}.gif"

        # Determine colors if not provided
        if colors is None:
            from matplotlib.pyplot import cm

            class_labels = np.unique(y)
            if n_class_estimate is None:
                n_class_estimate = len(class_labels)
            colormap = cm.rainbow(np.linspace(0, 1, n_class_estimate))
            colors = colormap  # assumes class indices are 0...n-1

        # Initialize and fit
        self.validate_data(X, y)
        self.classes_ = unique_labels(y)
        self.labels_ = y
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        writer = PillowWriter(fps=fps)
        with writer.saving(ax.figure, filename, dpi=80):
            for _ in range(max_iter):
                if verbose:
                    from tqdm import tqdm

                    iterator = tqdm(
                        enumerate(zip(X, y)),
                        total=len(X),
                        leave=leave_progress_bar,
                    )
                else:
                    iterator = enumerate(zip(X, y))

                for i, (x, c_b) in iterator:
                    self.module_a.pre_step_fit(X)
                    c_a = self.step_fit(
                        x,
                        c_b,
                        match_tracking=match_tracking,
                        epsilon=epsilon,
                    )
                    self.module_a.labels_[i] = c_a
                    self.module_a.post_step_fit(X)

                    ax.clear()
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    self.visualize(X[: i + 1], y[: i + 1], ax, colors=colors, **kwargs)
                    writer.grab_frame()

            self.module_a.post_fit(X)

            n_extra_frames = int(np.ceil(final_hold_secs * fps))
            for _ in range(n_extra_frames):
                ax.clear()
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                self.visualize(X, y, ax, colors=colors, **kwargs)
                writer.grab_frame()
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        """Partial fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        y : np.ndarray
            Data set B.
        match_tracking : Literal, default="MT+"
            Method to reset the match.
        epsilon : float, default=1e-10
            Small value to adjust the vigilance.

        Returns
        -------
        self : SimpleARTMAP
            The partially fitted model.

        """
        SimpleARTMAP.validate_data(self, X, y)
        if not hasattr(self, "labels_"):
            self.labels_ = y
            self.module_a.W = []
            self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode="constant")
            self.labels_[j:] = y
            self.module_a.labels_ = np.pad(
                self.module_a.labels_, [(0, X.shape[0])], mode="constant"
            )
        for i, (x, c_b) in enumerate(zip(X, y)):
            self.module_a.pre_step_fit(X)
            c_a = self.step_fit(x, c_b, match_tracking=match_tracking, epsilon=epsilon)
            self.module_a.labels_[i + j] = c_a
            self.module_a.post_step_fit(X)
        return self

    @property
    def labels_a(self) -> np.ndarray:
        """Get labels from side A (module A).

        Returns
        -------
        np.ndarray
            Labels from module A.

        """
        return self.module_a.labels_

    @property
    def labels_b(self) -> np.ndarray:
        """Get labels from side B.

        Returns
        -------
        np.ndarray
            Labels from side B.

        """
        return self.labels_

    @property
    def labels_ab(self) -> Dict[str, np.ndarray]:
        """Get labels from both A-side and B-side.

        Returns
        -------
        dict
            A dictionary with keys "A" and "B" containing labels from sides A and B,
            respectively.

        """
        return {"A": self.labels_a, "B": self.labels_}

    @property
    def n_clusters(self) -> int:
        """Get the number of clusters in side A.

        Returns
        -------
        int
            Number of clusters.

        """
        return self.module_a.n_clusters

    @property
    def n_clusters_a(self) -> int:
        """Get the number of clusters in side A.

        Returns
        -------
        int
            Number of clusters in side A.

        """
        return self.n_clusters

    @property
    def n_clusters_b(self) -> int:
        """Get the number of clusters in side B.

        Returns
        -------
        int
            Number of clusters in side B.

        """
        return len(set(c for c in self.map.values()))

    def step_pred(self, x: np.ndarray) -> tuple[int, int]:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample for side A.

        Returns
        -------
        tuple[int, int]
            Side A cluster label, side B cluster label.

        """
        c_a = self.module_a.step_pred(x)
        c_b = self.map[c_a]
        return c_a, c_b

    def predict(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Predict labels for the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        np.ndarray
            B labels for the data.

        """
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        y_b = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c_a, c_b = self.step_pred(x)
            y_b[i] = c_b
        return y_b

    def predict_ab(
        self, X: np.ndarray, clip: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels for the data, both A-side and B-side.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A labels for the data, B labels for the data.

        """
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        y_a = np.zeros((X.shape[0],), dtype=int)
        y_b = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c_a, c_b = self.step_pred(x)
            y_a[i] = c_a
            y_b[i] = c_b
        return y_a, y_b

    def plot_cluster_bounds(
        self, ax: Axes, colors: IndexableOrKeyable, linewidth: int = 1
    ):
        """Visualize the cluster boundaries.

        Parameters
        ----------
        ax : Axes
            Figure axes.
        colors : IndexableOrKeyable
            Colors to use for each cluster.
        linewidth : int, default=1
            Width of boundary lines.

        """
        colors_a = []
        for k_a in range(self.n_clusters):
            colors_a.append(colors[self.map[k_a]])
        self.module_a.plot_cluster_bounds(ax=ax, colors=colors_a, linewidth=linewidth)

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
            Data set.
        y : np.ndarray
            Sample labels.
        ax : Optional[Axes], default=None
            Figure axes.
        marker_size : int, default=10
            Size used for data points.
        linewidth : int, default=1
            Width of boundary lines.
        colors : Optional[Iterable], default=None
            Colors to use for each cluster.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            if self.module_a.data_format in ["latlon"]:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm

            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters_b))

        for k_b, col in enumerate(colors):
            cluster_data = y == k_b
            if self.module_a.data_format == "default":
                plt.scatter(
                    X[cluster_data, 0],
                    X[cluster_data, 1],
                    color=col,
                    marker=".",
                    s=marker_size,
                )

        self.plot_cluster_bounds(ax=ax, colors=colors, linewidth=linewidth)
