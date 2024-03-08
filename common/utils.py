import numpy as np

def normalize(data: np.ndarray) -> np.ndarray:
    normalized = (data-np.min(data))/(np.max(data)-np.min(data))
    return normalized

def compliment_code(data: np.ndarray) -> np.ndarray:
    cc_data = np.hstack([data, 1.0-data])
    return cc_data

def l1norm(x: np.ndarray) -> float:
    return float(np.sum(np.absolute(x)))

def l2norm2(data: np.ndarray) -> float:
    return float(np.matmul(data, data))

def fuzzy_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.minimum(x, y)
