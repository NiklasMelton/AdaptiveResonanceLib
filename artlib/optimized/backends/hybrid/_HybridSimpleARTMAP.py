import numpy as np
from typing import Literal, Tuple, Dict
import copy

from artlib.supervised.SimpleARTMAP import SimpleARTMAP

# TODO: auto select device and backend based on data and model


class _HybridSimpleARTMAP(SimpleARTMAP):
    _backends: Dict[str, SimpleARTMAP]

    def _create_backend(self, backend: str, device: str):
        pass

    def sync_to_backend(self, backend: SimpleARTMAP):
        # module A
        backend.module_a = copy.deepcopy(self.module_a)

        # A→B mapping
        backend.map = copy.deepcopy(self.map)

        # classes
        if hasattr(self, "classes_"):
            backend.classes_ = copy.deepcopy(self.classes_)
        # labels
        if hasattr(self, "labels_"):
            backend.labels_ = copy.deepcopy(self.labels_)

        return backend

    def sync_from_backend(self, backend: SimpleARTMAP):
        # module A
        self.module_a = copy.deepcopy(backend.module_a)

        # A→B mapping
        self.map = copy.deepcopy(backend.map)

        # classes
        if hasattr(backend, "classes_"):
            self.classes_ = copy.deepcopy(backend.classes_)
        # labels
        if hasattr(backend, "labels_"):
            self.labels_ = copy.deepcopy(backend.labels_)

    def _ensure_backend(self, X: np.ndarray, backend: str, device: str):
        backend = backend.lower().replace("c++", "cpp")
        device = device.lower().replace("gpu", "cuda")

        bd = f"{backend}-{device}"
        if bd not in self._backends:
            self._backends[bd] = self._create_backend(backend, device)
        self._backends[bd] = self.sync_to_backend(self._backends[bd])
        self._backends[bd]._ensure_backend(X)

        return bd

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
        leave_progress_bar: bool = True,
        backend: str = "cpp",
        device: str = "cpu",
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
        backend: str, default="cpp"
            torch, c++, or python.
        device: str, default="cpu"
            "cuda" or "cpu". Only applied when backend=torch.
        Returns
        -------
        self : SimpleARTMAP
            The fitted model.

        """
        bd = self._ensure_backend(X, backend, device)
        self._backends[bd] = self._backends[bd].fit(
            X, y, max_iter, match_tracking, epsilon, verbose, leave_progress_bar
        )
        self.sync_from_backend(self._backends[bd])

        self.module_a.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        backend: str = "cpp",
        device: str = "cpu",
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
        backend: str, default="cpp"
            torch, c++, or python.
        device: str, default="cpu"
            "cuda" or "cpu". Only applied when backend=torch.

        Returns
        -------
        self : SimpleARTMAP
            The partially fitted model.

        """
        bd = self._ensure_backend(X, backend, device)
        self._backends[bd] = self._backends[bd].partial_fit(
            X, y, match_tracking, epsilon
        )
        self.sync_from_backend(self._backends[bd])

        self.module_a.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        clip: bool = False,
        backend: str = "cpp",
        device: str = "cpu",
    ) -> np.ndarray:
        """Predict labels for the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits
        backend: str, default="cpp"
            torch, c++, or python.
        device: str, default="cpu"
            "cuda" or "cpu". Only applied when backend=torch.

        Returns
        -------
        np.ndarray
            B labels for the data.

        """
        bd = self._ensure_backend(X, backend, device)
        y_b = self._backends[bd].predict(X, clip)
        return y_b

    def predict_ab(
        self,
        X: np.ndarray,
        clip: bool = False,
        backend: str = "cpp",
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels for the data, both A-side and B-side.

        Parameters
        ----------
        X : np.ndarray
            Data set A.
        clip : bool
            clip the input values to be between the previously seen data limits
        backend: str, default="cpp"
            torch, c++, or python.
        device: str, default="cpu"
            "cuda" or "cpu". Only applied when backend=torch.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A labels for the data, B labels for the data.

        """
        bd = self._ensure_backend(X, backend, device)
        y_a, y_b = self._backends[bd].predict_ab(X, clip)

        return y_a, y_b
