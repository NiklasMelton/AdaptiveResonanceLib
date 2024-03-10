import numpy as np
from common.BaseART import BaseART
from common.BaseARTMAP import BaseARTMAP
from supervised.SimpleARTMAP import SimpleARTMAP
from sklearn.utils.validation import check_is_fitted, check_X_y


class ARTMAP(BaseARTMAP):
    def __init__(self, module_a: BaseART, module_b: BaseART):
        self.module_b = module_b
        self.simpleARTMAP = SimpleARTMAP(module_a)

    @property
    def module_a(self):
        return self.simpleARTMAP.module_a

    @property
    def map(self):
        return self.simpleARTMAP.map

    @property
    def labels_(self):
        return self.simpleARTMAP.labels_

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
        self.module_a.validate_data(X)
        self.module_b.validate_data(y)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter=1):
        # Check that X and y have correct shape
        self.validate_data(X, y)

        self.module_b.fit(y, max_iter=max_iter)

        y_c = self.module_b.labels_

        self.simpleARTMAP.fit(X, y_c, max_iter=max_iter)

        return self


    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        self.validate_data(X, y)
        self.module_b.partial_fit(y)
        self.simpleARTMAP.partial_fit(X, self.labels_b)
        return self


    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        return self.simpleARTMAP.predict(X)
