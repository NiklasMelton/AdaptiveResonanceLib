import numpy as np
from typing import Optional
from common.BaseART import BaseART
from common.utils import normalize

def prepare_data(data: np.ndarray) -> np.ndarray:
    normalized = normalize(data)
    return normalized


class myART(BaseART):
    # template for ART module

    @staticmethod
    def validate_params(params: dict):
        raise NotImplementedError

    def validate_data(self, X: np.ndarray):
        raise NotImplementedError

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        raise NotImplementedError

    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        raise NotImplementedError

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        raise NotImplementedError

    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        raise NotImplementedError

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        raise NotImplementedError
