import numpy as np
from typing import Optional, Callable
from elementary.BaseART import BaseART
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels

class TopoART(BaseART):

    def __init__(self, base_module: BaseART, betta_lower: float, tau: int, phi: int):
        params = dict(base_module.params, **{"beta_lower": betta_lower, "tau": tau, "phi": phi})
        super().__init__(params)
        self.base_module = base_module
        self.adjacency = np.zeros([], dtype=int)
        self._counter = np.zeros([], dtype=int)
        self._permanent_mask = np.zeros([], dtype=bool)

    @staticmethod
    def validate_params(params: dict):
        assert "beta" in params, "TopoART is only compatible with ART modules relying on 'beta' for learning."
        assert "beta_lower" in params
        assert "tau" in params
        assert "phi" in params
        assert params["beta"] >= params["beta_lower"]
        assert params["phi"] <= params["tau"]

    @property
    def W(self):
        return self.base_module.W

    @W.setter
    def W(self, new_W: list[np.ndarray]):
        self.base_module.W = new_W

    def validate_data(self, X: np.ndarray):
        self.base_module.validate_data(X)

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        return self.base_module.category_choice(i, w, params)

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        return self.base_module.match_criterion(i, w, params, cache)

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.base_module.match_criterion_bin(i, w, params, cache)

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        if cache.get("resonant_c", -1) >= 0:
            self.adjacency[cache["resonant_c"], cache["current_c"]] += 1
        return self.base_module.update(i, w, params, cache)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        self.adjacency = np.pad(self.adjacency, ((0, 1), (0, 1)), "constant")
        self._counter = np.pad(self._counter, (0, 1), "constant", constant_values=(1,))
        self._permanent_mask = np.pad(self._permanent_mask, (0, 1), "constant")
        return self.new_weight(i, params)


    def prune(self, X: np.ndarray):
        self._permanent_mask += (self._counter >= self.params["phi"])
        perm_labels = np.where(self._permanent_mask)[0]
        self.W = [w for w, pm in zip(self.W, self._permanent_mask) if pm]
        self._counter = self._counter[perm_labels]

        label_map = {
            label: np.where(perm_labels == label)[0][0]
            for label in np.unique(self.labels_)
            if label in perm_labels
        }

        for i, x in enumerate(X):
            if self.labels_[i] in label_map:
                self.labels_[i] = label_map[self.labels_[i]]
            else:
                T_values, T_cache = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
                T = np.array(T_values)
                new_label = np.argmax(T)
                self.labels_[i] = new_label
                self._counter[new_label] += 1

    def step_prune(self, X: np.ndarray):
        sum_counter = sum(self._counter)
        if sum_counter > 0 and sum_counter % self.params["tau"] == 0:
            self.prune(X)

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None) -> int:
        resonant_c: int = -1

        if len(self.W) == 0:
            self.W.append(self.new_weight(x, self.params))
            self.adjacency = np.zeros((1, 1), dtype=int)
            self._counter = np.ones((1, ), dtype=int)
            self._permanent_mask = np.zeros((1, ), dtype=bool)
            return 0
        else:
            T_values, T_cache = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
            T = np.array(T_values)
            while any(T > 0):
                c_ = int(np.argmax(T))
                w = self.W[c_]
                cache = T_cache[c_]
                m = self.match_criterion_bin(x, w, params=self.params, cache=cache)
                no_match_reset = (
                        match_reset_func is None or
                        match_reset_func(x, w, c_, params=self.params, cache=cache)
                )
                if m and no_match_reset:
                    if resonant_c < 0:
                        params = self.params
                    else:
                        params = dict(self.params, **{"beta": self.params["beta_lower"]})
                    #TODO: make compatible with DualVigilanceART
                    self.W[c_] = self.update(
                        x,
                        w,
                        params=params,
                        cache=dict(cache, **{"resonant_c": resonant_c, "current_c": c_})
                    )
                    if resonant_c < 0:
                        resonant_c = c_
                    else:
                        return resonant_c
                else:
                    T[c_] = -1

            if resonant_c < 0:
                c_new = len(self.W)
                w_new = self.new_weight(x, self.params)
                self.W.append(w_new)
                return c_new

            return resonant_c


    def fit(self, X: np.ndarray, match_reset_func: Optional[Callable] = None, max_iter=1):
        self.validate_data(X)
        self.check_dimensions(X)

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ))
        for _ in range(max_iter):
            for i, x in enumerate(X):
                self.step_prune(X)
                c = self.step_fit(x, match_reset_func=match_reset_func)
                self.labels_[i] = c
        return self
