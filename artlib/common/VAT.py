"""VAT.

.. # Bezdek, J. C., & Hathaway, R. J. (2002). .. # VAT: A tool for visual assessment of
cluster tendency. .. # Proceedings of the 2002 International Joint Conference on Neural
Networks. .. # doi:10.1109/IJCNN.2002.1007487

"""
import numpy as np
from typing import Optional, Tuple, Callable
from scipy.spatial.distance import pdist, squareform


def VAT(
    data: np.ndarray,
    distance_metric: Optional[Callable] = lambda X: pdist(X, "euclidean"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Visual Assessment of Cluster Tendency (VAT) algorithm.

    VAT was originally designed as a visualization tool for clustering behavior of data.
    When the VAT-reordered distance matrix is plotted as an image, clusters will appear
    in visually distinct groups along the diagonal. However, it has since been
    discovered that the reordering significantly improves the results of order-dependent
    clustering methods like ART. It is therefore recommended to pre-process data with
    VAT prior to presentation when possible.

    .. # Bezdek, J. C., & Hathaway, R. J. (2002).
    .. # VAT: A tool for visual assessment of cluster tendency.
    .. # Proceedings of the 2002 International Joint Conference on Neural Networks.
    .. # doi:10.1109/IJCNN.2002.1007487

    .. bibliography:: ../../references.bib
       :filter: citation_key == "bezdek2002vat"

    Parameters
    ----------
    data : np.ndarray
        Input dataset as a 2D numpy array where each row is a sample.
    distance_metric : callable, optional
        Callable function to calculate pairwise distances. Defaults to Euclidean
        distance using `pdist`. If None, assumes data is a pre-computed distance matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Reordered distance matrix reflecting cluster structure.
        - Reordered list of indices indicating the optimal clustering order.

    """
    if distance_metric is None:
        pairwise_dist = data
    else:
        pairwise_dist = squareform(distance_metric(data))

    num_samples = data.shape[0]
    indicies = []
    remaining = list(range(num_samples))

    ix, jx = np.unravel_index(pairwise_dist.argmax(), pairwise_dist.shape)
    indicies.append(ix)
    remaining.pop(ix)

    while remaining:
        sub_matrix = pairwise_dist[np.ix_(indicies, remaining)]
        _, jx = np.unravel_index(sub_matrix.argmin(), sub_matrix.shape)
        indicies.append(remaining[jx])
        remaining.pop(jx)

    return pairwise_dist[np.ix_(indicies, indicies)], np.array(indicies)
