"""iCVI Fuzzy ART

.. # da Silva, Leonardo Enzo Brito, Nagasharath Rayapati, and Donald C. Wunsch.
.. # "iCVI-ARTMAP: using incremental cluster validity indices and adaptive resonance
.. # theory reset mechanism to accelerate validation and achieve multiprototype
.. # unsupervised representations."
.. # IEEE Transactions on Neural Networks and Learning Systems 34.12 (2022): 9757-9770.

The original matlab code can be found at
https://github.com/ACIL-Group/iCVI-toolbox/tree/master
 The formulation is available at
scholarsmine.mst.edu/cgi/viewcontent.cgi?article=3833&context=doctoral_dissertations
 Pages 314-316 and 319-320 Extended icvi offline mode can be found at
https://ieeexplore.ieee.org/document/9745260

.. bibliography:: ../../references.bib
   :filter: citation_key == "da2022icvi"

"""
import numpy as np
from typing import Optional, Literal, Callable
from artlib.elementary.FuzzyART import FuzzyART
from artlib.cvi.iCVIs.CalinkskiHarabasz import iCVI_CH


class iCVIFuzzyART(FuzzyART):
    """ICVI Fuzzy Art For Clustering.

    .. # da Silva, Leonardo Enzo Brito, Nagasharath Rayapati, and Donald C. Wunsch.
    .. # "iCVI-ARTMAP: using incremental cluster validity indices and adaptive resonance
    .. # theory reset mechanism to accelerate validation and achieve multiprototype
    .. # unsupervised representations."
    .. # IEEE Transactions on Neural Networks and Learning Systems
    .. # 34.12 (2022): 9757-9770.

    .. bibliography:: ../../references.bib
       :filter: citation_key == "da2022icvi"

    """

    CALINSKIHARABASZ = 1

    def __init__(
        self,
        rho: float,
        alpha: float,
        beta: float,
        validity: int,
        offline: bool = True,
    ):
        """Initialize the iCVIFuzzyART model.

        Parameters
        ----------
        rho : float
            Vigilance parameter in the range [0, 1].
        alpha : float
            Choice parameter. A value of 1e-7 is recommended.
        beta : float
            Learning parameter in the range [0, 1]. A value of 1 is recommended for
            fast learning.
        validity : int
            The cluster validity index being used.
        offline : bool, optional
            Whether to use offline mode for iCVI updates, by default True.

        """
        super().__init__(rho, alpha, beta)
        self.params[
            "validity"
        ] = validity  # Currently not used. Waiting for more algorithms.
        self.offline = offline
        assert "validity" in self.params
        assert isinstance(self.params["validity"], int)

    def iCVI_match(self, x, w, c_, params, cache):
        """Apply iCVI (incremental Cluster Validity Index) matching criteria.

        Parameters
        ----------
        x : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight.
        c_ : int
            Cluster index.
        params : dict
            Dictionary containing algorithm parameters.
        cache : dict
            Cache used for storing intermediate results.

        Returns
        -------
        bool
            True if the new criterion value is better than the previous one,
            False otherwise.

        """
        if self.offline:
            new = self.iCVI.switch_label(x, self.labels_[self.index], c_)
        else:
            new = self.iCVI.add_sample(x, c_)
        # Eventually this should be an icvi function that you pass the params,
        # and it handles if this is true or false.
        return new["criterion_value"] > self.iCVI.criterion_value
        # return self.iCVI.evalLabel(x, c_) This except pass params instead.

    # Could add max epochs back in, but only if offline is true,
    # or do something special...
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        match_reset_func: Optional[Callable] = None,
        max_iter=1,
        match_tracking: Literal["MT+", "MT-", "MT0", "MT1", "MT~"] = "MT+",
        epsilon: float = 0.0,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset.
        y : np.ndarray, optional
            Not used. For compatibility.
        match_reset_func : callable, optional
            A callable accepting the data sample, a cluster weight, the params dict,
            and the cache dict.
            Returns True if the cluster is valid for the sample, False otherwise.
        max_iter : int, optional
            Number of iterations to fit the model on the same dataset, by default 1.
        match_tracking : {"MT+", "MT-", "MT0", "MT1", "MT~"}, optional
            Method for resetting match criterion.
        epsilon : float, optional
            Epsilon value used for adjusting match criterion, by default 0.0.

        """
        self.validate_data(X)
        self.check_dimensions(X)
        self.is_fitted_ = True

        self.W: list[np.ndarray] = []
        self.labels_ = np.zeros((X.shape[0],), dtype=int)

        self.iCVI = iCVI_CH(X[0])

        if self.offline:
            for x in X:
                params = self.iCVI.add_sample(x, 0)
                self.iCVI.update(params)

        for i, x in enumerate(X):
            self.pre_step_fit(X)
            self.index = i
            if match_reset_func is None:
                c = self.step_fit(
                    x,
                    match_reset_func=self.iCVI_match,
                    match_tracking=match_tracking,
                    epsilon=epsilon,
                )
            else:
                match_reset_func_ = lambda x, w, c_, params, cache: (
                    match_reset_func(x, w, c_, params, cache)
                    & self.iCVI_match(x, w, c_, params, cache)
                )
                c = self.step_fit(
                    x,
                    match_reset_func=match_reset_func_,
                    match_tracking=match_tracking,
                    epsilon=epsilon,
                )

            if self.offline:
                params = self.iCVI.switch_label(x, self.labels_[i], c)
            else:
                params = self.iCVI.add_sample(x, c)
            self.iCVI.update(params)

            self.labels_[i] = c
            self.post_step_fit(X)
