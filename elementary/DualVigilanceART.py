"""
Brito da Silva, L. E., Elnabarawy, I., & Wunsch II, D. C. (2019).
Dual vigilance fuzzy adaptive resonance theory.
Neural Networks, 109, 1–5. doi:10.1016/j.neunet.2018.09.015.
"""
import numpy as np
from typing import Optional, Callable
from common.BaseART import BaseART


class DualVigilanceART(BaseART):
    # implementation of Dual Vigilance ART

    def __init__(self, base_module: BaseART, lower_bound: float):
        params = dict(base_module.params, **{"rho_lower_bound": lower_bound})
        super().__init__(params)
        self.base_module = base_module
        self.lower_bound = lower_bound
        self.map: dict[int, int] = dict()

    @property
    def dim_(self):
        return self.base_module.dim_

    @dim_.setter
    def dim_(self, new_dim):
        self.base_module.dim_ = new_dim

    @property
    def labels_(self):
        return self.base_module.labels_

    @labels_.setter
    def labels_(self, new_labels: np.ndarray):
        self.labels_ = new_labels

    @property
    def W(self):
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        self.base_module.W = new_W

    def check_dimensions(self, X: np.ndarray):
        self.base_module.check_dimensions(X)

    def validate_data(self, X: np.ndarray):
        self.base_module.validate_data(X)
        self.check_dimensions(X)

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params, \
            "Dual Vigilance ART is only compatible with ART modules relying on 'rho' for vigilance."
        assert "rho_lower_bound" in params, \
            "Dual Vigilance ART requires a lower bound 'rho' value"
        assert params["rho"] > params["rho_lower_bound"] >= 0

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        if len(self.base_module.W) == 0:
            self.base_module.W.append(self.base_module.new_weight(x, self.base_module.params))
            return 0
        else:
            T_values, T_cache = zip(
                *[
                    self.base_module.category_choice(x, w, params=self.base_module.params)
                    for w in self.base_module.W
                ]
            )
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.argmax(T))
                w = self.base_module.W[c_]
                cache = T_cache[c_]
                m1 = self.base_module.match_criterion_bin(x, w, params=self.base_module.params, cache=cache)
                no_match_reset = (
                    match_reset_func is None or
                    match_reset_func(x, w, self.map[c_], params=self.base_module.params, cache=cache)
                )

                if no_match_reset:
                    if m1:
                        self.base_module.W[c_] = self.base_module.update(x, w, self.params, cache=cache)
                        return self.map[c_]
                    else:
                        lb_params = dict(self.base_module.params, **{"rho": self.lower_bound})
                        m2 = self.base_module.match_criterion_bin(x, w, params=lb_params, cache=cache)
                        if m2:
                            c_new = len(self.base_module.W)
                            w_new = self.base_module.new_weight(x, self.base_module.params)
                            self.base_module.W.append(w_new)
                            self.map[c_new] = self.map[c_]
                            return self.map[c_new]
                T[c_] = -1

            c_new = len(self.base_module.W)
            w_new = self.base_module.new_weight(x, self.base_module.params)
            self.base_module.W.append(w_new)
            self.map[c_new] = max(self.map.values()) + 1
            return self.map[c_new]

    def step_pred(self, x) -> int:
        assert len(self.base_module.W) >= 0, "ART module is not fit."
        T, _ = zip(
            *[
                self.base_module.category_choice(x, w, params=self.base_module.params)
                for w in self.base_module.W
            ]
        )
        c_ = int(np.argmax(T))
        return self.map[c_]
