"""ART1 :cite:`carpenter1987massively`."""
import numpy as np
from typing import Literal, Optional, Callable
from artlib.optimized.backends.cpp.cppART1 import (
    FitART1,
    PredictART1,
)
from artlib.elementary.ART1 import ART1 as pyART1
from sklearn.utils.validation import check_is_fitted


class ART1(pyART1):
    """ART1 for Clustering. optimized with C++

    This module implements ART1

    ART1 is a non-modular clustering model which has been highly
    optimized for run-time performance. Fit and predict functions are implemented in
    c++ for efficient execution. This class acts as a wrapper for the underlying c++
    functions and to provide compatibility with the artlib style and usage.
    Functionally, ART1 behaves as a special case of
    :class:`~artlib.elementary.ART1.ART1`.

    """

    def _synchronize_cpp_results(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        incremental: bool = False,
    ):
        """Synchronize the python class with the output of the c++ code.

        Parameters
        ----------
        labels_a_out : np.ndarray
            A 1D numpy array containing the a-side labels from fitting
        weights_arrays : np.ndarray
            A 2D numpy array where rows are the Fuzzy ART weights
        incremental: bool, default=False
            This flag is set to true when synchronizing after a partial_fit

        """
        if not incremental:
            self.labels_ = np.array((), dtype=int)
            self.weight_sample_counter_ = []

        # labels
        self.labels_ = np.concatenate([self.labels_, labels_a_out.astype(int)])

        # sample counters
        new_counts = np.bincount(labels_a_out, minlength=len(weights_arrays))
        if len(self.weight_sample_counter_) < len(new_counts):
            self.weight_sample_counter_.extend(
                [0] * (len(new_counts) - len(self.weight_sample_counter_))
            )
        for k, c in enumerate(new_counts):
            self.weight_sample_counter_[k] += int(c)

        # weights (float64 arrays)
        self.W = [w for w in weights_arrays]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray, optional
            Not used. For compatibility.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
            Not used. For compatibility.
        max_iter : int, default=1
            Number of iterations to fit the model on the same dataset.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
            Not used. For compatibility.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.
            Not used. For compatibility.
        verbose : bool, default=False
            If True, displays progress of the fitting process.
            Not used. For compatibility.
        leave_progress_bar : bool, default=True
            If True, leaves thge progress of the fitting process. Only used when
            verbose=True
            Not used. For compatibility.

        """
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        self.validate_data(X_)
        self.W = []
        self.labels_ = np.zeros((X_.shape[0],), dtype=int)

        la, W = FitART1(
            X_,
            rho=self.params["rho"],
            L=self.params["L"],
            weights=None,
        )
        self._synchronize_cpp_results(la, W)
        self.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        match_reset_func: Optional[Callable] = None,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Iteratively fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        match_reset_func : callable, optional
            A callable that influences cluster creation.
            Not used. For compatibility.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, default="MT+"
            Method for resetting match criterion.
            Not used. For compatibility.
        epsilon : float, default=0.0
            Epsilon value used for adjusting match criterion.
            Not used. For compatibility.

        """
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        self.validate_data(X_)

        if not hasattr(self, "labels_"):
            self.labels_ = np.zeros((X_.shape[0],), dtype=int)
            existing_W = None
        else:
            existing_W = np.ascontiguousarray(self.W, dtype=float)

        la, W = PredictART1(
            X_,
            rho=self.params["rho"],
            L=self.params["L"],
            weights=existing_W,
        )
        self._synchronize_cpp_results(la, W, incremental=True)
        self.is_fitted_ = True
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
        X_ = np.ascontiguousarray(X, dtype=np.float64)
        if clip:
            X_ = np.clip(X_, self.d_min_, self.d_max_)
        self.validate_data(X_)

        W = np.ascontiguousarray(self.W, dtype=float)

        y_a = PredictART1(
            X_,
            rho=self.params["rho"],
            L=self.params["L"],
            weights=W,
        )
        return y_a
