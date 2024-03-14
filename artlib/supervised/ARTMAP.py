"""
Carpenter, G. A., Grossberg, S., & Reynolds, J. H. (1991a).
ARTMAP: Supervised real-time learning and classification of nonstationary data by a self-organizing neural network.
Neural Networks, 4, 565 – 588. doi:10.1016/0893-6080(91)90012-T.
"""
import numpy as np
from artlib.common.BaseART import BaseART
from artlib.supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.validation import check_is_fitted


class ARTMAP(SimpleARTMAP):
    def __init__(self, module_a: BaseART, module_b: BaseART):
        """

        Parameters:
        - module_a: a-side ART module
        - module_b: b-side ART module

        """
        self.module_b = module_b
        super(ARTMAP, self).__init__(module_a)

    def get_params(self, deep: bool = True) -> dict:
        """

        Parameters:
        - deep: If True, will return the parameters for this class and contained subobjects that are estimators.

        Returns:
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
    def labels_a(self):
        return self.module_a.labels_

    @property
    def labels_b(self):
        return self.labels_

    @property
    def labels_ab(self):
        return {"A": self.labels_a, "B": self.labels_}

    def validate_data(self, X: np.ndarray, y: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set A
        - y: data set B

        """
        self.module_a.validate_data(X)
        self.module_b.validate_data(y)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        """
        Fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B
        - max_iter: number of iterations to fit the model on the same data set

        """
        # Check that X and y have correct shape
        self.validate_data(X, y)

        self.module_b.fit(y, max_iter=max_iter)

        y_c = self.module_b.labels_

        super(ARTMAP, self).fit(X, y_c, max_iter=max_iter)

        return self


    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Partial fit the model to the data

        Parameters:
        - X: data set A
        - y: data set B

        """
        self.validate_data(X, y)
        self.module_b.partial_fit(y)
        super(ARTMAP, self).partial_fit(X, self.labels_b)
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict labels for the data

        Parameters:
        - X: data set A

        Returns:
            B labels for the data

        """
        check_is_fitted(self)
        return super(ARTMAP, self).predict(X)

    def predict_ab(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        predict labels for the data, both A-side and B-side

        Parameters:
        - X: data set A

        Returns:
            A labels for the data, B labels for the data

        """
        check_is_fitted(self)
        return super(ARTMAP, self).predict_ab(X)
