"""ARTMAP.

Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
ARTMAP: Supervised real-time learning and classification of nonstationary data by a
self-organizing neural network.
Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.

"""
import numpy as np
from typing import Literal, Tuple, Dict, Union, Optional
from artlib.common.BaseART import BaseART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.validation import check_is_fitted


class ARTMAP(SimpleARTMAP):
    """ARTMAP for Classification and Regression.

    This module implements ARTMAP as first published in
    Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
    ARTMAP: Supervised real-time learning and classification of nonstationary data by a
    self-organizing neural network.
    Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.

    ARTMAP joins accepts two ART modules A and B which cluster the dependent channel
    (samples) and the independent channel (labels) respectively while linking them with
    a many-to-one mapping. If your labels are integers, use SimpleARTMAP for a faster
    and more direct implementation. ARTMAP also provides the ability to fit a regression
    model to data and specific functions have been implemented to allow this. However,
    FusionART provides substantially better fit for regression problems which are not
    monotonic.

    """

    def __init__(self, module_a: BaseART, module_b: BaseART):
        """Initialize the ARTMAP model with two ART modules.

        Parameters
        ----------
        module_a : BaseART
            A-side ART module for clustering the independent channel.
        module_b : BaseART
            B-side ART module for clustering the dependent channel.

        """
        self.module_b = module_b
        super(ARTMAP, self).__init__(module_a)

    def get_params(self, deep: bool = True) -> dict:
        """Get the parameters of the ARTMAP model.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this class and contained subobjects
            that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        out = {
            "module_a": self.module_a,
            "module_b": self.module_b,
        }

        if deep:
            deep_a_items = self.module_a.get_params().items()
            out.update(("module_a" + "__" + k, val) for k, val in deep_a_items)

            deep_b_items = self.module_b.get_params().items()
            out.update(("module_b" + "__" + k, val) for k, val in deep_b_items)
        return out

    @property
    def labels_a(self) -> np.ndarray:
        """Get the labels generated by the A-side ART module.

        Returns
        -------
        np.ndarray
            Labels for the A-side data (independent channel).

        """
        return self.module_a.labels_

    @property
    def labels_b(self) -> np.ndarray:
        """Get the labels generated by the B-side ART module.

        Returns
        -------
        np.ndarray
            Labels for the B-side data (dependent channel).

        """
        return self.module_b.labels_

    @property
    def labels_ab(self) -> Dict[str, np.ndarray]:
        """Get the labels generated by both the A-side and B-side ART modules.

        Returns
        -------
        dict
            Dictionary containing both A-side and B-side labels.

        """
        return {"A": self.labels_a, "B": self.module_b.labels_}

    def validate_data(self, X: np.ndarray, y: np.ndarray):
        """Validate the input data prior to clustering.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).
        y : np.ndarray
            Data set B (dependent channel).

        """
        self.module_a.validate_data(X)
        self.module_b.validate_data(y)

    def prepare_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare data for clustering by normalizing and transforming.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).
        y : np.ndarray
            Data set B (dependent channel).

        Returns
        -------
        tuple of np.ndarray
            Normalized data for both channels.

        """
        assert y is not None
        return self.module_a.prepare_data(X), self.module_b.prepare_data(y)

    def restore_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Restore data to its original state before preparation.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).
        y : np.ndarray
            Data set B (dependent channel).

        Returns
        -------
        tuple of np.ndarray
            Restored data for both channels.

        """
        assert y is not None
        return self.module_a.restore_data(X), self.module_b.restore_data(y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter=1,
        match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
        verbose: bool = False,
    ):
        """Fit the ARTMAP model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).
        y : np.ndarray
            Data set B (dependent channel).
        max_iter : int, optional
            Number of iterations to fit the model on the same data set.
        match_reset_method : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting the vigilance parameter when match criterion fails.
        epsilon : float, optional
            Small increment to modify the vigilance parameter.
        verbose : bool, default=False
            If True, displays a progress bar during training.

        Returns
        -------
        self : ARTMAP
            Fitted ARTMAP model.

        """
        # Check that X and y have correct shape
        self.validate_data(X, y)

        self.module_b.fit(
            y,
            max_iter=max_iter,
            match_reset_method=match_reset_method,
            epsilon=epsilon,
        )

        y_c = self.module_b.labels_

        super(ARTMAP, self).fit(
            X,
            y_c,
            max_iter=max_iter,
            match_reset_method=match_reset_method,
            epsilon=epsilon,
            verbose=verbose,
        )

        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        match_reset_method: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 1e-10,
    ):
        """Partially fit the ARTMAP model to the data.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).
        y : np.ndarray
            Data set B (dependent channel).
        match_reset_method : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting the vigilance parameter when match criterion fails.
        epsilon : float, optional
            Small increment to modify the vigilance parameter.

        Returns
        -------
        self : ARTMAP
            Partially fitted ARTMAP model.

        """
        self.validate_data(X, y)
        self.module_b.partial_fit(
            y, match_reset_method=match_reset_method, epsilon=epsilon
        )
        super(ARTMAP, self).partial_fit(
            X,
            self.labels_b,
            match_reset_method=match_reset_method,
            epsilon=epsilon,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the given data.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).

        Returns
        -------
        np.ndarray
            Predicted labels for data set B (dependent channel).

        """
        check_is_fitted(self)
        return super(ARTMAP, self).predict(X)

    def predict_ab(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict both A-side and B-side labels for the given data.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).

        Returns
        -------
        tuple of np.ndarray
            A labels and B labels for the data.

        """
        check_is_fitted(self)
        return super(ARTMAP, self).predict_ab(X)

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for the given data using cluster centers.
        Note: ARTMAP is not recommended for regression.
        Use FusionART for regression tasks.

        Parameters
        ----------
        X : np.ndarray
            Data set A (independent channel).

        Returns
        -------
        np.ndarray
            Predicted values using cluster centers.
        """
        check_is_fitted(self)
        C = self.predict(X)
        centers = self.module_b.get_cluster_centers()
        return np.array([centers[c] for c in C])
