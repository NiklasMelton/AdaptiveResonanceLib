"""
Serrano-Gotarredona, T., Linares-Barranco, B., & Andreou, A. G. (1998).
Adaptive Resonance Theory Microchips: Circuit Design Techniques.
Norwell, MA, USA: Kluwer Academic Publishers.
"""
import numpy as np
from typing import Optional, Iterable, Literal
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.BaseARTMAP import BaseARTMAP
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels


class SimpleARTMAP(BaseARTMAP):

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

    def __init__(self, module_a: BaseART):
        self.module_a = module_a
        super().__init__()

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.

        """
        out = {"module_a": self.module_a}
        if deep:
            deep_items = self.module_a.get_params().items()
            out.update(("module_a" + "__" + k, val) for k, val in deep_items)
        return out


    def validate_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        validates the data prior to clustering

        Parameters:
        - X: data set A
        - y: data set B

        """
        X, y = check_X_y(X, y, dtype=None)
        self.module_a.validate_data(X)
        return X, y

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            prepared data
        """
        return self.module_a.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            restored data
        """
        return self.module_a.restore_data(X)

    def step_fit(self, x: np.ndarray, c_b: int, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 1e-10) -> int:
        """
        Fit the model to a single sample

        Parameters:
        - x: data sample for side A
        - c_b: side b label
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        Returns:
            side A cluster label

        """
        match_reset_func = lambda i, w, cluster, params, cache: self.match_reset_func(
            i, w, cluster, params=params, extra={"cluster_b": c_b}, cache=cache
        )
        c_a = self.module_a.step_fit(x, match_reset_func=match_reset_func, match_reset_method=match_reset_method, epsilon=epsilon)
        if c_a not in self.map:
            self.map[c_a] = c_b
        else:
            assert self.map[c_a] == c_b
        return c_a

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 1e-10, verbose: bool = False):
        """
        Fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B
        - max_iter: number of iterations to fit the model on the same data set
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

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
                x_y_iter = tqdm(enumerate(zip(X, y)), total=int(X.shape[0]))
            else:
                x_y_iter = enumerate(zip(X, y))
            for i, (x, c_b) in x_y_iter:
                self.module_a.pre_step_fit(X)
                c_a = self.step_fit(x, c_b, match_reset_method=match_reset_method, epsilon=epsilon)
                self.module_a.labels_[i] = c_a
                self.module_a.post_step_fit(X)
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 1e-10):
        """
        Partial fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """
        SimpleARTMAP.validate_data(self, X, y)
        if not hasattr(self, 'labels_'):
            self.labels_ = y
            self.module_a.W = []
            self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)
            j = 0
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode='constant')
            self.labels_[j:] = y
            self.module_a.labels_ = np.pad(self.module_a.labels_, [(0, X.shape[0])], mode='constant')
        for i, (x, c_b) in enumerate(zip(X, y)):
            self.module_a.pre_step_fit(X)
            c_a = self.step_fit(x, c_b, match_reset_method=match_reset_method, epsilon=epsilon)
            self.module_a.labels_[i+j] = c_a
            self.module_a.post_step_fit(X)
        return self

    @property
    def labels_a(self):
        return self.module_a.labels_

    @property
    def labels_b(self):
        return self.labels_

    @property
    def labels_ab(self):
        return {"A": self.labels_a, "B": self.labels_}

    @property
    def n_clusters(self):
        return self.module_a.n_clusters

    @property
    def n_clusters_a(self):
        return self.n_clusters

    @property
    def n_clusters_b(self):
        return len(set(c for c in self.map.values()))

    def step_pred(self, x: np.ndarray) -> tuple[int, int]:
        """
        Predict the label for a single sample

        Parameters:
        - x: data sample for side A

        Returns:
            side A cluster label, side B cluster label

        """
        c_a = self.module_a.step_pred(x)
        c_b = self.map[c_a]
        return c_a, c_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict labels for the data

        Parameters:
        - X: data set A

        Returns:
            B labels for the data

        """
        check_is_fitted(self)
        y_b = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c_a, c_b = self.step_pred(x)
            y_b[i] = c_b
        return y_b

    def predict_ab(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        predict labels for the data, both A-side and B-side

        Parameters:
        - X: data set A

        Returns:
            A labels for the data, B labels for the data

        """
        check_is_fitted(self)
        y_a = np.zeros((X.shape[0],), dtype=int)
        y_b = np.zeros((X.shape[0],), dtype=int)
        for i, x in enumerate(X):
            c_a, c_b = self.step_pred(x)
            y_a[i] = c_a
            y_b[i] = c_b
        return y_a, y_b

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        """
        undefined function for visualizing the bounds of each cluster

        Parameters:
        - ax: figure axes
        - colors: colors to use for each cluster
        - linewidth: width of boundary line

        """
        colors_a = []
        for k_a in range(self.n_clusters):
            colors_a.append(colors[self.map[k_a]])
        self.module_a.plot_cluster_bounds(ax, colors_a, linewidth)

    def visualize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            ax: Optional[Axes] = None,
            marker_size: int = 10,
            linewidth: int = 1,
            colors: Optional[Iterable] = None
    ):
        """
        Visualize the clustering of the data

        Parameters:
        - X: data set
        - y: sample labels
        - ax: figure axes
        - marker_size: size used for data points
        - linewidth: width of boundary line
        - colors: colors to use for each cluster

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if colors is None:
            from matplotlib.pyplot import cm
            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters_b))

        for k_b, col in enumerate(colors):
            cluster_data = y == k_b
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], color=col, marker=".", s=marker_size)

        self.plot_cluster_bounds(ax, colors, linewidth)
