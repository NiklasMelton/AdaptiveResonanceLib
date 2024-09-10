"""
Add Reference in correct format.
The original matlab code can be found at https://github.com/ACIL-Group/iCVI-toolbox/tree/master
The formulation is available at
https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=3833&context=doctoral_dissertations Pages 314-316 and 319-320
Extended icvi offline mode can be found at
https://ieeexplore.ieee.org/document/9745260
"""
import numpy as np
from typing import Optional, Literal, Callable
from artlib.elementary.FuzzyART import FuzzyART
from artlib.cvi.iCVIs.CalinkskiHarabasz import iCVI_CH


class iCVIFuzzyART(FuzzyART):
    """iCVI Fuzzy Art Classification

    Parameters:
        rho: float [0,1] for the vigilance parameter.
        alpha: float choice parameter. 1e-7 recommended value.
        beta: float [0,1] learning parameters. beta = 1 is fast learning recommended value.
        validity: int the cluster validity index being used.
        W: list of weights, top down.
        labels: class labels for data set.
    """
    CALINSKIHARABASZ = 1

    def __init__(self, rho: float, alpha: float, beta: float, validity: int, offline: bool = True):
        super().__init__(rho, alpha, beta)
        self.params['validity'] = validity  # Currently not used. Waiting for more algorithms.
        self.offline = offline
        assert 'validity' in self.params  # Because Fuzzy art doesn't accept validity, and makes the params the way it does, validations have to be done after init.
        assert isinstance(self.params['validity'], int)

    def iCVI_match(self, x, w, c_, params, cache):
        if self.offline:
            new = self.iCVI.switch_label(x, self.labels_[self.index], c_)
        else:
            new = self.iCVI.add_sample(x, c_)
        # Eventually this should be an icvi function that you pass the params, and it handles if this is true or false.
        return new['criterion_value'] > self.iCVI.criterion_value
        # return self.iCVI.evalLabel(x, c_) This except pass params instead.

    # Could add max epochs back in, but only if offline is true, or do something special...
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, match_reset_func: Optional[Callable] = None, max_iter=1, match_reset_method:Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+", epsilon: float = 0.0):
        """
        Fit the model to the data

        Parameters:
        - X: data set
        - y: not used. For compatibility.
        - match_reset_func: a callable accepting the data sample, a cluster weight, the params dict, and the cache dict
            Permits external factors to influence cluster creation.
            Returns True if the cluster is valid for the sample, False otherwise
        - max_iter: number of iterations to fit the model on the same data set
        - match_reset_method:
            "MT+": Original method, rho=M+epsilon
             "MT-": rho=M-epsilon
             "MT0": rho=M, using > operator
             "MT1": rho=1.0,  Immediately create a new cluster on mismatch
             "MT~": do not change rho

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0], ), dtype=int)

        self.iCVI = iCVI_CH(X[0])

        if self.offline:
            for x in X:
                params = self.iCVI.add_sample(x, 0)
                self.iCVI.update(params)

        for i, x in enumerate(X):
            self.pre_step_fit(X)
            self.index = i
            if match_reset_func is None:
                c = self.step_fit(x, match_reset_func=self.iCVI_match, match_reset_method=match_reset_method, epsilon=epsilon)
            else:
                match_reset_func = lambda x, w, c_, params, cache: (match_reset_func(x, w, c_, params, cache) & self.iCVI_match(x, w, c_, params, cache))
                c = self.step_fit(x, match_reset_func=match_reset_func, match_reset_method=match_reset_method, epsilon=epsilon)

            if self.offline:
                params = self.iCVI.switch_label(x, self.labels_[i], c)
            else:
                params = self.iCVI.add_sample(x, c)
            self.iCVI.update(params)

            self.labels_[i] = c
            self.post_step_fit(X)
