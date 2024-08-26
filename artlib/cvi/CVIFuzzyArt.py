import numpy as np
import sklearn.metrics as metrics
from artlib.elementary.FuzzyART import FuzzyART


class CVIFuzzyART(FuzzyART):
    """CVI Fuzzy Art Classification

    Expanded version of Fuzzy Art that uses Cluster Validity Indicies to help with cluster selection.
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

    def __init__(self, rho: float, alpha: float, beta: float, validity: int):
        super().__init__(rho, alpha, beta)
        # Because Fuzzy art doesn't accept validity, and makes the params the way it does, validations have to be done after init.
        self.params['validity'] = validity
        assert 'validity' in self.params
        assert isinstance(self.params['validity'], int)

    def CVI_match(self, x, w, c_, params, cache):
        if len(self.W) < 2:
            return True

        if params['validity'] == self.CALINSKIHARABASZ:
            valid_func = metrics.calinski_harabasz_score
        elif params['validity'] == self.DAVIESBOULDIN:
            valid_func = metrics.davies_bouldin_score
        elif params['validity'] == self.SILHOUETTE:
            valid_func = metrics.silhouette_score
        else:
            raise ValueError(f"Invalid Validity Parameter: {params['validity']}")

        old_VI = valid_func(self.data, self.labels_)
        new_labels = np.copy(self.labels_)
        new_labels[self.index] = c_
        new_VI = valid_func(self.data, new_labels)
        if params['validity'] != self.DAVIESBOULDIN:
            return new_VI > old_VI
        else:
            return new_VI < old_VI

    def fit(self, X: np.ndarray, max_iter: int = 1):
        self.data = X
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ), dtype=int)
        for _ in range(max_iter):
            print(_)
            for i, x in enumerate(X):
                self.pre_step_fit(X)
                self.index = i
                c = self.step_fit(x, match_reset_func=self.CVI_match)
                self.labels_[i] = c
                self.post_step_fit(X)
