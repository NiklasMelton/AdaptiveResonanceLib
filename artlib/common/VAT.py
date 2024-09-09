# Bezdek, J. C., & Hathaway, R. J. (2002). VAT: A tool for visual assessment of cluster tendency.
# Proceedings of the 2002 International Joint Conference on Neural Networks. doi:10.1109/IJCNN.2002.1007487
import numpy as np
from typing import Optional, Tuple, Callable
from scipy.spatial.distance import pdist, squareform


def VAT(data: np.ndarray, distance_metric: Optional[Callable] = lambda X: pdist(X, "euclidean")) -> Tuple[np.ndarray, np.ndarray]:
    if distance_metric is None:
        R = data
    else:
        R = squareform(distance_metric(data))

    dim = data.shape[0]
    P = np.zeros((dim,), dtype=int)

    i, j = np.unravel_index(np.argmax(R), R.shape)
    P[0] = i
    I = [i]
    J = set(range(dim)) - {i}

    for r in range(1, dim):
        J_list = list(J)
        R_sub = R[np.ix_(I, J_list)]
        _, j_ = np.unravel_index(np.argmin(R_sub), R_sub.shape)
        j = J_list[j_]
        P[r] = j
        I.append(j)
        J -= {j}

    return R[np.ix_(P, P)], P


