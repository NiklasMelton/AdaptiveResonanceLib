import numpy as np
from typing import Optional, List, Tuple, Dict
from artlib.common.BaseART import BaseART


class IsingBernoullieART(BaseART):
    """Ising (autologistic) ART for binary data (x ∈ {0,1}^D).

    Conceptual analogue of BayesianART (full-covariance Gaussian) for Bernoulli:
      p(x) ∝ exp( h^T x + x^T J x ), with symmetric J and zero diagonal.

    We stay close to BayesianART's API:
      - category_choice: uses (pseudo-)likelihood (product of node-conditionals)
                         times cluster prior p(c_j) = n_j / Σ_k n_k
      - match_criterion: returns a stable match ∈ [0,1] based on mean conditional
                         probability of the observed bits (no need for match_criterion_bin)
      - update: one online SGD step on the (negative) log pseudo-likelihood for
                the winning cluster (with optional L2 regularization on J, h)
      - new_weight: initialize biases h from a clipped logit of the first sample,
                    couplings J = 0, count n = 1

    Notes
    -----
    • Pseudo-likelihood: for each dimension d,
        p(x_d=1 | x_{¬d}) = σ( η_d ),   η_d = h_d + Σ_{v≠d} J_{dv} x_v
      with σ(z) = 1 / (1+exp(-z)). The joint pseudo-likelihood is ∏_d p(x_d|x_{¬d}),
      and we work in log-space to avoid underflow.

    • Match: arithmetic mean of per-bit conditional probabilities of the observed x:
        m = (1/D) Σ_d [ x_d * σ(η_d) + (1-x_d) * (1-σ(η_d)) ] ∈ [0,1].
      This preserves ART’s “ρ ≤ M” semantics, so no match_criterion_bin is needed.

    • Regularization: optional L2 on J and h during updates for stability.

    Parameters
    ----------
    rho : float
        Vigilance parameter in [0,1].
    eta : float
        Learning rate for the online SGD update per winning sample.
    l2 : float
        L2 penalty coefficient (weight decay) for h and J.
    init_on : float
        Initial probability assigned to bits that are 1 in the first sample (e.g., 0.9).
    init_off : float
        Initial probability assigned to bits that are 0 in the first sample (e.g., 0.1).
    clip_logits : float
        Numerical clip for converting probabilities to logits (avoid inf).

    """

    def __init__(
        self,
        rho: float,
        eta: float = 0.05,
        l2: float = 0.0,
        init_on: float = 0.9,
        init_off: float = 0.1,
        clip_logits: float = 1e-6,
    ):
        params = {
            "rho": float(rho),
            "eta": float(eta),
            "l2": float(l2),
            "init_on": float(init_on),
            "init_off": float(init_off),
            "clip_logits": float(clip_logits),
        }
        super().__init__(params)

    # ---- utilities ----
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # stable sigmoid
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _logit(p: np.ndarray, eps: float) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p) - np.log(1.0 - p)

    @staticmethod
    def _ensure_binary(x: np.ndarray) -> np.ndarray:
        xb = np.asarray(x, dtype=float)
        if not np.all((xb == 0.0) | (xb == 1.0)):
            raise ValueError("IsingBernoullieART requires binary inputs in {0,1}.")
        return xb

    @staticmethod
    def validate_params(params: dict):
        assert (
            "rho" in params
            and isinstance(params["rho"], float)
            and 0.0 <= params["rho"] <= 1.0
        )
        assert "eta" in params and params["eta"] > 0.0
        assert "l2" in params and params["l2"] >= 0.0
        assert 0.0 < params["init_off"] < 1.0 and 0.0 < params["init_on"] < 1.0
        assert params["clip_logits"] > 0.0

    def check_dimensions(self, X: np.ndarray):
        if not hasattr(self, "dim_"):
            self.dim_ = X.shape[1]
        else:
            assert X.shape[1] == self.dim_

    # ---- weight layout ----
    # For each cluster weight vector w:
    #   w = [ h(0:D), J(0:D*D) row-major, n ]
    # with J symmetric and J_ii = 0 enforced in update/new_weight.

    # ---- core computations ----
    def _unpack(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        D = self.dim_
        h = w[:D]
        J = w[D : D + D * D].reshape((D, D))
        n = w[-1]
        return h, J, n

    def _pack(self, h: np.ndarray, J: np.ndarray, n: float) -> np.ndarray:
        return np.concatenate([h, J.reshape(-1), [n]])

    def _eta(self, h: np.ndarray, J: np.ndarray, x: np.ndarray) -> np.ndarray:
        # η = h + J x  (J_ii is maintained as 0)
        return h + J.dot(x)

    def _log_pseudolik(self, h: np.ndarray, J: np.ndarray, x: np.ndarray) -> float:
        # sum_d log p(x_d | x_¬d)  with p(x_d=1|...) = σ(η_d)
        eta = self._eta(h, J, x)
        p = self._sigmoid(eta)
        # per-bit conditional prob of the observed bit:
        px = np.where(x > 0.5, p, 1.0 - p)
        return float(np.sum(np.log(np.clip(px, 1e-12, 1.0))))  # stabilize

    def category_choice(
        self, i: np.ndarray, w: np.ndarray, params: dict
    ) -> tuple[float, Optional[dict]]:
        """Activation ~ pseudo-likelihood * cluster prior.

        Mirrors BayesianART (likelihood * prior), but uses pseudo-likelihood.

        """
        x = self._ensure_binary(i)
        h, J, n = self._unpack(w)

        log_pl = self._log_pseudolik(h, J, x)  # stable in log-space
        # Prior p(c_j) = n / Σ n
        total_n = sum(w_[-1] for w_ in self.W) if self.W else 1.0
        log_p_cj = np.log(n) - np.log(total_n)

        # Return activation in probability space to mirror BayesianART
        # (If you face underflow in high-D, switch to returning log_pl + log_p_cj)
        activation = float(np.exp(log_pl + log_p_cj))
        cache = {"log_pl": log_pl}
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        """Match ∈ [0,1]: arithmetic mean of per-bit conditional probabilities.

        of the observed bits under the current (h, J).   m = (1/D) Σ_d [ x_d * σ(η_d) +
        (1-x_d) * (1-σ(η_d)) ] This preserves the usual ART direction (require m ≥ ρ),
        so match_criterion_bin is NOT needed.

        """
        x = self._ensure_binary(i)
        h, J, _ = self._unpack(w)
        eta = self._eta(h, J, x)
        p = self._sigmoid(eta)
        m = float(np.mean(np.where(x > 0.5, p, 1.0 - p)))
        return m, {"mean_conditional": m}

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """One-step online SGD on negative log pseudo-likelihood for the winning
        cluster.

          gradient wrt h_d     : (x_d - σ(η_d))
          gradient wrt J_{dv}  : (x_d - σ(η_d)) * x_v   for v ≠ d
        We apply L2 weight decay on h and J if l2 > 0, enforce J symmetry and zero diag,
        then increment n.

        """
        x = self._ensure_binary(i)
        D = self.dim_
        h, J, n = self._unpack(w)
        eta_lr = params["eta"]
        l2 = params["l2"]

        # forward
        eta = self._eta(h, J, x)
        p = self._sigmoid(eta)
        resid = x - p  # shape (D,)

        # gradients
        # dh = resid
        # dJ = outer(resid, x), but zero diagonal; symmetrize
        dh = resid
        dJ = np.outer(resid, x)
        np.fill_diagonal(dJ, 0.0)
        # symmetrize update (since model uses symmetric J): average with its transpose update
        dJ = 0.5 * (dJ + dJ.T)

        # weight decay (L2): gradient += l2 * param
        if l2 > 0.0:
            dh -= l2 * h
            dJ -= l2 * J

        # SGD step
        h_new = h + eta_lr * dh
        J_new = J + eta_lr * dJ
        # enforce symmetry and zero diagonal explicitly
        J_new = 0.5 * (J_new + J_new.T)
        np.fill_diagonal(J_new, 0.0)

        n_new = n + 1.0
        return self._pack(h_new, J_new, n_new)

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """
        Initialize from a single sample:
          • h initialized from a per-bit probability that reflects the bit value:
                p_init_d = init_on if x_d=1 else init_off   (then logit)
          • J = 0 (no interactions yet), symmetric with zero diagonal
          • n = 1
        """
        x = self._ensure_binary(i)
        D = x.size
        p_init = np.where(x > 0.5, params["init_on"], params["init_off"])
        h0 = self._logit(p_init, params["clip_logits"])
        J0 = np.zeros((D, D), dtype=float)
        n0 = 1.0
        return self._pack(h0, J0, n0)

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Return per-cluster Bernoulli means implied by current h and J when
        conditioning on the cluster's own average signal.

        For a simple, stable proxy, we return σ(h) (i.e., ignoring interactions). This
        mirrors how GaussianART returns the mean vector.

        """
        centers = []
        for w in self.W:
            h, J, _ = self._unpack(w)
            centers.append(self._sigmoid(h))  # proxy mean
        return centers


if __name__ == "__main__":
    SEED = 1337
    np.random.seed(SEED)

    # Load data (use your existing helper)
    X_train, y_train, X_test, y_test = load_mnist_subset_binarized_split(
        n_per_class=100, threshold=0.2, seed=SEED, train_size=(6 / 7)
    )

    from artlib import SimpleARTMAP

    base_art = IsingBernoullieART(
        rho=0.10,  # vigilance
        eta=0.05,  # learning rate for pseudo-likelihood SGD
        l2=1e-2,  # mild weight decay for stability (tune as needed)
        init_on=0.9,  # initial prob for bits that are 1 in the first sample
        init_off=0.1,  # initial prob for bits that are 0 in the first sample
        clip_logits=1e-6,
    )
    model = SimpleARTMAP(base_art)

    # (If your SimpleARTMAP keeps its own vigilance on module A, mirror it)
    try:
        model.module_a.params["rho"] = base_art.params["rho"]
    except Exception:
        pass

    # Fit on training set
    model.fit(X_train, y_train, verbose=True)

    # Report
    from sklearn.metrics import accuracy_score

    print("Supervised IsingBernoullieART + SimpleARTMAP on binarized MNIST subset")
    print(
        f"  Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}, Dims: {X_train.shape[1]}"
    )
    print(f"  Vigilance (rho): {base_art.params.get('rho')}")
    print(f"  Clusters: {model.module_a.n_clusters}")

    # Evaluate on held-out test set
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"  Test Accuracy: {test_acc:.4f}")
