import numpy as np
import torch
from typing import Literal, Union, Tuple

from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from torch import Tensor


def _to_device(x: Union[Tensor, "np.ndarray"], device, dtype=torch.float32) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    import numpy as np

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True)
    raise TypeError("Expected torch.Tensor or numpy.ndarray")


class _TorchSimpleARTMAP(SimpleARTMAP):
    def _synchronize_torch_results(
        self,
        labels_a_out: np.ndarray,
        weights_arrays: list[np.ndarray],
        cluster_labels_out: np.ndarray,
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

        # weights (store as float64 numpy arrays to match artlib expectations)
        self.module_a.W = [w for w in weights_arrays]

        # A→B mapping
        for c_a, c_b in enumerate(cluster_labels_out):
            if c_a in self.map:
                assert self.map[c_a] == int(c_b), "Incremental fit changed cluster map."
            else:
                self.map[c_a] = int(c_b)

    # --- public API (matches the C++ wrapper)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1,
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
        SimpleARTMAP.validate_data(self, X, y)
        self._ensure_backend(X)
        assert self._backend is not None

        # artlib-style bookkeeping
        self.classes_ = unique_labels(y)
        self.labels_ = y
        self.module_a.W = []
        self.module_a.labels_ = np.zeros((X.shape[0],), dtype=int)

        # Expect X already normalized and complement-coded,
        Xp = _to_device(X, self._backend.device, self._backend.dtype)
        la, W, cl = self._backend.partial_fit_and_export(
            Xp, y, epsilon=epsilon, match_tracking=match_tracking
        )
        self._synchronize_torch_results(la, W, cl, incremental=False)
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
        self._ensure_backend(X)
        assert self._backend is not None

        if not hasattr(self, "labels_"):
            self.labels_ = y
        else:
            j = len(self.labels_)
            self.labels_ = np.pad(self.labels_, (0, len(y)))
            self.labels_[j:] = y

        Xp = _to_device(X, self._backend.device, self._backend.dtype)
        la, W, cl = self._backend.partial_fit_and_export(
            Xp, y, epsilon=epsilon, match_tracking=match_tracking
        )
        self._synchronize_torch_results(la, W, cl, incremental=True)
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
        assert self._backend is not None
        # Optional clipping, mirroring C++ wrapper behavior
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)

        # Use backend for batched predict; mirror C++ by ignoring vigilance here.
        y_a, y_b = self._backend.predict_ab_prepared(
            _to_device(X, self._backend.device, self._backend.dtype)
        )
        return y_b

    def predict_ab(
        self, X: np.ndarray, clip: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        assert self._backend is not None
        if clip:
            X = np.clip(X, self.module_a.d_min_, self.module_a.d_max_)
        self.module_a.validate_data(X)
        self.module_a.check_dimensions(X)
        y_a, y_b = self._backend.predict_ab_prepared(
            _to_device(X, self._backend.device, self._backend.dtype)
        )
        return y_a, y_b
