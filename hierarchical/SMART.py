import numpy as np

from elementary.BaseART import BaseART
from typing import Union, Type, Optional
from DeepARTMAP import DeepARTMAP

class SMART(DeepARTMAP):

    def __init__(self, base_ART_class: Type, rho_values: Union[list[float], np.ndarray], base_params: dict, **kwargs):

        assert all(np.diff(rho_values) > 0), "rho_values must be monotonically increasing"
        self.rho_values = rho_values

        layer_params = [dict(base_params, **{"rho": rho}) for rho in self.rho_values]
        layers = [base_ART_class(params, **kwargs) for params in layer_params]
        for layer in layers:
            assert isinstance(layer, BaseART), "Only elementary ART-like objects are supported"
        super().__init__(layers)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, max_iter=1):
        X_list = [X]*self.n
        return super().fit(X_list, max_iter=max_iter)

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_list = [X] * self.n
        return self.partial_fit(X_list)
