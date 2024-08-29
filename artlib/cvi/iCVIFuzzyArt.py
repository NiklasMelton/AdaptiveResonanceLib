"""
Add Reference in correct format.
The original matlab code can be found at https://github.com/ACIL-Group/iCVI-toolbox/tree/master
The formulation is available at
https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=3833&context=doctoral_dissertations Pages 314-316 and 319-320
Extended icvi offline mode can be found at
https://ieeexplore.ieee.org/document/9745260
"""
import numpy as np
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
    def fit(self, X: np.ndarray):
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
            c = self.step_fit(x, match_reset_func=self.iCVI_match)

            if self.offline:
                params = self.iCVI.switch_label(x, self.labels_[i], c)
            else:
                params = self.iCVI.add_sample(x, c)
            self.iCVI.update(params)

            self.labels_[i] = c
            self.post_step_fit(X)
