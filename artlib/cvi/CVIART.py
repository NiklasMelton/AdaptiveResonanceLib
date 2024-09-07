import numpy as np
import sklearn.metrics as metrics
from typing import Optional, List, Callable, Literal, Iterable
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART


class CVIART(BaseART):
    """CVI Art Classification

    Expanded version of Art that uses Cluster Validity Indicies to help with cluster selection.
    PBM is not implemented, can be seen here.
    https://git.mst.edu/acil-group/CVI-Fuzzy-ART/-/blob/master/PBM_index.m?ref_type=heads

    Note, the default step_fit function in base ART evaluates the matching function even if
    the other criteria has failed. This means it could run slower then it would otherwise.


    Parameters:
        rho: float [0,1] for the vigilance parameter.
        alpha: float choice parameter. 1e-7 recommended value.
        beta: float [0,1] learning parameters. beta = 1 is fast learning recommended value.
        validity: int the cluster validity index being used.
        W: list of weights, top down.
        labels: class labels for data set.
    """
    CALINSKIHARABASZ = 1
    DAVIESBOULDIN = 2
    SILHOUETTE = 3
    # PBM = 4

    def validate_params(self, params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        self.base_module.validate_params(params)
        assert 'validity' in params
        assert isinstance(params['validity'], int)
        assert params["validity"] in [CVIART.CALINSKIHARABASZ, CVIART.DAVIESBOULDIN, CVIART.SILHOUETTE]

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            normalized data
        """
        return self.base_module.prepare_data(X)

    def restore_data(self, X: np.ndarray) -> np.ndarray:
        """
        restore data to state prior to preparation

        Parameters:
        - X: data set

        Returns:
            restored data
        """
        return self.base_module.restore_data(X)

    @property
    def W(self):
        return self.base_module.W

    @W.setter
    def W(self, new_W):
        self.base_module.W = new_W

    @property
    def labels_(self):
        return self.base_module.labels_

    @labels_.setter
    def labels_(self, new_labels_):
        self.base_module.labels_ = new_labels_


    def __init__(self, base_module: BaseART, validity: int):
        self.base_module = base_module
        params = dict(base_module.params, **{"validity": validity})
        super().__init__(params)
        print(self.params)


    def CVI_match(self, x, w, c_, params, extra, cache):
        if len(self.W) < 2:
            return True

        if extra['validity'] == self.CALINSKIHARABASZ:
            valid_func = metrics.calinski_harabasz_score
        elif extra['validity'] == self.DAVIESBOULDIN:
            valid_func = metrics.davies_bouldin_score
        elif extra['validity'] == self.SILHOUETTE:
            valid_func = metrics.silhouette_score
        else:
            raise ValueError(f"Invalid Validity Parameter: {extra['validity']}")

        old_VI = valid_func(self.data, self.labels_)
        new_labels = np.copy(self.labels_)
        new_labels[extra["index"]] = c_
        new_VI = valid_func(self.data, new_labels)
        if extra['validity'] != self.DAVIESBOULDIN:
            return new_VI >= old_VI
        else:
            return new_VI <= old_VI

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, match_reset_func: Optional[Callable] = None,
            max_iter=1, match_reset_method: Literal["original", "modified"] = "original"):
        self.data = X
        self.base_module.validate_data(X)
        self.base_module.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ), dtype=int)
        for _ in range(max_iter):
            for index, x in enumerate(X):
                self.pre_step_fit(X)
                if match_reset_func is None:
                    cvi_match_reset_func = lambda i, w, cluster_a, params, cache: self.CVI_match(i, w, cluster_a, params, {"index": index, "validity":self.params["validity"]}, cache)
                else:
                    cvi_match_reset_func = lambda i, w, cluster_a, params, cache: (match_reset_func(i,w,cluster_a,params,cache) and self.CVI_match(i, w, cluster_a, params, {"index": index, "validity":self.params["validity"]}, cache))
                c = self.base_module.step_fit(x, match_reset_func=cvi_match_reset_func, match_reset_method=match_reset_method)
                self.labels_[index] = c
                self.post_step_fit(X)


    def pre_step_fit(self, X: np.ndarray):
        return self.base_module.pre_step_fit(X)

    def post_step_fit(self, X: np.ndarray):
        return self.base_module.post_step_fit(X)

    def step_fit(self, x: np.ndarray, match_reset_func: Optional[Callable] = None,
                 match_reset_method: Literal["original", "modified"] = "original") -> int:
        raise NotImplementedError

    def step_pred(self, x) -> int:
        return self.base_module.step_pred(x)

    def get_cluster_centers(self) -> List[np.ndarray]:
        return self.base_module.get_cluster_centers()

    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        return self.base_module.plot_cluster_bounds(ax, colors,linewidth)
