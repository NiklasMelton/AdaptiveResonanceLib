"""Fuzzy ARTMAP – C++‑accelerated wrapper."""
from __future__ import annotations

import numpy as np
from typing import Literal

from artlib.cpp_optimized.cppFuzzyARTMAP import (  # ← new backend
    FitFuzzyARTMAP,
    PredictFuzzyARTMAP,
)
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from artlib.elementary.FuzzyART import FuzzyART  # ← analogue of BinaryFuzzyART
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class FuzzyARTMAP(SimpleARTMAP):
    """C++‑optimised Fuzzy ARTMAP.

    A thin wrapper around the `cppFuzzyARTMAP` free functions that keeps
    the Python‑side state (`module_a`, label maps, sample counters) in sync
    with the C++ core.

    """

    # ---------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------
    def __init__(self, rho: float, alpha: float, beta: float):
        """
        Parameters
        ----------
        rho   : float  – vigilance (0 ≤ ρ ≤ 1)
        alpha : float  – choice (α ≥ 0)
        beta  : float  – learning‑rate (0 < β ≤ 1)
        """
        module_a = FuzzyART(rho=rho, alpha=alpha, beta=beta)
        super().__init__(module_a)

    # ---------------------------------------------------------------------
    # helper – mirror C++ results back to Python objects
    # ---------------------------------------------------------------------
    def _sync_cpp(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        cluster_labels_out: np.ndarray,
        *,
        incremental: bool = False,
    ):
        if not incremental:
            self.map: dict[int, int] = {}
            self.module_a.labels_ = np.array((), dtype=int)
            self.module_a.weight_sample_counter_ = []

        # labels
        self.module_a.labels_ = np.concatenate(
            [self.module_a.labels_, labels_a_out.astype(int)]
        )

        # sample counters
        new_counts = np.bincount(labels_a_out, minlength=len(weights_arrays))
        if len(self.module_a.weight_sample_counter_) < len(new_counts):
            self.module_a.weight_sample_counter_.extend(
                [0] * (len(new_counts) - len(self.module_a.weight_sample_counter_))
            )
        for k, c in enumerate(new_counts):
            self.module_a.weight_sample_counter_[k] += int(c)

        # weights (float64 arrays)
        self.module_a.W = [w for w in weights_arrays]

        # A→B mapping
        for c_a, c_b in enumerate(cluster_labels_out):
            if c_a in self.map:
                assert self.map[c_a] == c_b, "Incremental fit changed cluster map."
            else:
                self.map[c_a] = int(c_b)

    # ---------------------------------------------------------------------
    # batch fit
    # ---------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_iter: int = 1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
    ):
        SimpleARTMAP.validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.labels_ = y
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        la, W, cl = FitFuzzyARTMAP(
            X,
            y,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            beta=self.module_a.params["beta"],
            MT=match_tracking,
            epsilon=epsilon,
            weights=None,
            cluster_labels=None,
        )
        self._sync_cpp(la, W, cl)
        self.module_a.is_fitted_ = True
        return self

    # ---------------------------------------------------------------------
    # incremental fit
    # ---------------------------------------------------------------------
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        SimpleARTMAP.validate_data(self, X, y)

        if not hasattr(self, "labels_"):
            self.labels_ = y
            existing_W = None
            existing_map = None
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, (0, len(y)))
            self.labels_[j:] = y
            existing_W = np.array(self.module_a.W, dtype=float)
            existing_map = np.array(
                [self.map[c] for c in range(self.module_a.n_clusters)]
            )

        la, W, cl = FitFuzzyARTMAP(
            X,
            y,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            beta=self.module_a.params["beta"],
            MT=match_tracking,
            epsilon=epsilon,
            weights=existing_W,
            cluster_labels=existing_map,
        )
        self._sync_cpp(la, W, cl, incremental=True)
        self.module_a.is_fitted_ = True
        return self

    # ---------------------------------------------------------------------
    # prediction helpers
    # ---------------------------------------------------------------------
    def _predict_cpp(self, X: np.ndarray):
        W = np.array(self.module_a.W, dtype=float)
        cl = np.array([self.map[c] for c in range(self.module_a.n_clusters)])
        return PredictFuzzyARTMAP(
            X,
            rho=self.module_a.params["rho"],
            alpha=self.module_a.params["alpha"],
            beta=self.module_a.params["beta"],
            MT="",
            epsilon=0.0,
            weights=W,
            cluster_labels=cl,
        )

    def predict(self, X: np.ndarray, *, clip: bool = False) -> np.ndarray:
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        _, y_b = self._predict_cpp(X)
        return y_b

    def predict_ab(self, X: np.ndarray, *, clip: bool = False):
        check_is_fitted(self)
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        y_a, y_b = self._predict_cpp(X)
        return y_a, y_b
