"""Binary Fuzzy ART :cite:`carpenter1991fuzzy`."""
import numpy as np
from typing import Literal
from artlib.cpp_optimized.cppBinaryFuzzyARTMAP import (
    FitBinaryFuzzyARTMAP,
    PredictBinaryFuzzyARTMAP,
)
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.elementary.BinaryFuzzyART import BinaryFuzzyART
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class BinaryFuzzyARTMAP(SimpleARTMAP):
    """BinaryFuzzyARTMAP for Classification. optimized with C++

    This module implements BinaryFuzzyARTMAP

    BinaryFuzzyARTMAP is a non-modular classification model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    c++ for efficient execution. This class acts as a wrapper for the underlying c++
    functions and to provide compatibility with the artlib style and usage.
    Functionally, BinaryFuzzyARTMAP behaves as a special case of
    :class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` instantiated with
    :class:`~artlib.elementary.BinaryFuzzyART.BinaryFuzzyART`.

    """

    def __init__(self, rho: float, alpha: float):
        """Initialize the Binary Fuzzy ARTMAP model.

        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float
            Choice parameter.

        """
        module_a = BinaryFuzzyART(rho=rho, alpha=alpha)
        super().__init__(module_a)

    def _synchronize_cpp_results(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: np.ndarray,
        cluster_labels_out: np.ndarray,
        incremental: bool = False,
    ):
        """Synchronize the python class with the output of the c++ code.

        Parameters
        ----------
        labels_a_out : np.ndarray
            A 1D numpy array containing the a-side labels from fitting
        weights_arrays : np.ndarray
            A 2D numpy array where rows are the Binary Fuzzy ART weights
        cluster_labels_out : np.ndarray
            A 1D numpy array describing the map from a-side to b-side cluster labels
        incremental: bool, default=False
            This flag is set to true when synchronizing after a partial_fit

        """
        if not incremental:
            self.map: dict[int, int] = dict()
            self.module_a.labels_ = np.array(())
            self.module_a.weight_sample_counter_ = []

        # concatenate new module_a labels to the existing labels
        self.module_a.labels_ = np.concatenate(
            [self.module_a.labels_, labels_a_out.flatten()]
        ).flatten()
        # update module_a sample counter
        new_sample_counts = list(map(int, np.bincount(labels_a_out)))

        if len(self.module_a.weight_sample_counter_) < len(new_sample_counts):
            self.module_a.weight_sample_counter_.extend(
                [0]
                * (len(new_sample_counts) - len(self.module_a.weight_sample_counter_))
            )

        for i in range(len(new_sample_counts)):
            self.module_a.weight_sample_counter_[i] += new_sample_counts[i]

        # update module_a weights
        self.module_a.W = [w for w in weights_arrays]

        # update the cluster-label map
        for c_a, c_b in enumerate(cluster_labels_out):
            if c_a in self.map:
                assert self.map[c_a] == c_b, (
                    "Incremental learning has changed "
                    "cluster mapping. Something went "
                    "wrong."
                )
            else:
                self.map[c_a] = c_b

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
            non functional. Left for compatibility
        leave_progress_bar : bool, default=True
            non functional. Left for compatibility

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

        labels_a_out, weights_arrays, cluster_labels_out = FitBinaryFuzzyARTMAP(
            X,
            y,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            MT=match_tracking,
            epsilon=epsilon,
            weights=None,
            cluster_labels=None,
        )
        self._synchronize_cpp_results(labels_a_out, weights_arrays, cluster_labels_out)
        self.module_a.is_fitted_ = True
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
            existing_W = None
            existing_cluster_labels = None
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, [(0, X.shape[0])], mode="constant")
            self.labels_[j:] = y
            existing_W = np.array(self.module_a.W)
            existing_cluster_labels = np.array(
                [self.map[c_a] for c_a in range(self.module_a.n_clusters)]
            )

        labels_a_out, weights_arrays, cluster_labels_out = FitBinaryFuzzyARTMAP(
            X,
            y,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            MT=match_tracking,
            epsilon=epsilon,
            weights=existing_W,
            cluster_labels=existing_cluster_labels,
        )
        self._synchronize_cpp_results(
            labels_a_out, weights_arrays, cluster_labels_out, incremental=True
        )
        self.module_a.is_fitted_ = True
        return self

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

        existing_W = np.array(self.module_a.W)
        existing_cluster_labels = np.array(
            [self.map[c_a] for c_a in range(self.module_a.n_clusters)]
        )

        _, y_b = PredictBinaryFuzzyARTMAP(
            X, weights=existing_W, cluster_labels=existing_cluster_labels
        )
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

        existing_W = np.array(self.module_a.W)
        existing_cluster_labels = np.array(
            [self.map[c_a] for c_a in range(self.module_a.n_clusters)]
        )

        y_a, y_b = PredictBinaryFuzzyARTMAP(
            X, weights=existing_W, cluster_labels=existing_cluster_labels
        )
        return y_a, y_b
