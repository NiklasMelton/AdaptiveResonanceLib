import numpy as np
from typing import Optional, List, Tuple, Dict
from artlib.common.BaseART import BaseART


class BernoulliART(BaseART):
    """Bernoulli ART for Clustering (binary inputs).

    Diagonal (independent-dimension) Bernoulli model:
        p(x | θ) = ∏_d θ_d^{x_d} (1 - θ_d)^{(1 - x_d)}  , x_d ∈ {0,1}

    We mirror GaussianART’s API:
      - category_choice: computes (unnormalized) likelihood * cluster prior,
        penalized by a “complexity term” analogous to sqrt(det Σ) via ∏_d sqrt(θ_d (1-θ_d)).
      - match_criterion: returns a value in [0, 1] akin to Gaussian’s exp(-½ …) term.
      - update: online update of θ via running mean of bits.
      - new_weight: initialize from the first sample (like GaussianART uses i as mean).
      - get_cluster_centers: returns θ (per-dim probability of 1).

    Stored weight layout w for each cluster (length = 2*D + 1 but we recompute one piece):
        [ θ(0:D), var_root(0:D), sqrt_det_var(1), n(1) ]
      - θ: per-dimension Bernoulli parameter (mean of bits)
      - var_root: per-dim sqrt(θ_d (1-θ_d)) — cached to avoid recomputing inside loops
      - sqrt_det_var: sqrt(∏_d θ_d (1-θ_d)) — complexity analogue to Gaussian sqrt(det Σ)
      - n: cluster sample count (for priors and online updates)

    Notes:
      * We clip θ to [eps, 1-eps] to avoid log(0) and zero variance.
      * “alpha” plays the same role as in GaussianART (a small stabilizer in the denominator).

    """

    def __init__(
        self,
        rho: float,
        alpha: float = 1e-10,
        eps: float = 1e-6,
        match_mode: str = "meanprob",
        tau: float = 0.05,
    ):
        """
        Parameters
        ----------
        rho : float
            Vigilance parameter.
        alpha : float, optional
            Small stabilizer to avoid division by zero in category choice, by default 1e-10.
        eps : float, optional
            Clipping for θ to avoid log(0) and zero variance, by default 1e-6.
        """
        params = {
            "rho": rho,
            "alpha": alpha,
            "eps": eps,
            "match_mode": match_mode,
            "tau": float(tau),
        }
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "alpha" in params
        assert "eps" in params
        assert "match_mode" in params
        assert 1.0 >= params["rho"] >= 0.0
        assert params["alpha"] > 0.0
        assert 0.0 < params["eps"] < 0.5
        assert isinstance(params["rho"], float)
        assert params["match_mode"] in ["geo", "masked_geo", "meanprob"]

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _clip01(x: np.ndarray, eps: float) -> np.ndarray:
        return np.clip(x, eps, 1.0 - eps)

    @staticmethod
    def _bernoulli_loglik(i: np.ndarray, theta: np.ndarray, eps: float) -> float:
        """Sum_d [ i_d log θ_d + (1-i_d) log(1-θ_d) ] (safe, no underflow)."""
        th = np.clip(theta, eps, 1.0 - eps)
        return float(np.sum(i * np.log(th) + (1.0 - i) * np.log1p(-th)))

    @staticmethod
    def _variance_terms(theta: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
        """Per-dim sqrt(θ(1-θ)) and global sqrt_det_var = sqrt(prod θ(1-θ))."""
        th = BernoulliART._clip01(theta, eps)
        var = th * (1.0 - th)  # per-dim variance
        var = np.maximum(var, eps * eps)  # floor
        var_root = np.sqrt(var)  # per-dim sqrt variance
        sqrt_det_var = float(np.sqrt(np.prod(var)))  # global complexity term
        return var_root, sqrt_det_var

    # -------------------------
    # ART interface
    # -------------------------

    # --- category_choice: return log-activation, keep cache as-is ---
    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict):
        D = self.dim_
        theta = w[:D]
        sqrt_det_var = w[-2]
        n = w[-1]

        loglik = self._bernoulli_loglik(i, theta, params["eps"])  # ≤ 0
        log_p_i_cj = loglik - np.log(params["alpha"] + sqrt_det_var)

        total_n = sum(w_[-1] for w_ in self.W) if self.W else 1.0
        log_p_cj = np.log(n) - np.log(total_n)

        log_activation = log_p_i_cj + log_p_cj
        activation = float(log_activation)  # return log-activation (preserves ordering)

        # cache avg loglik for match
        loglik_avg = loglik / D
        cache = {"loglik": loglik, "loglik_avg": loglik_avg}
        return activation, cache

    def match_criterion(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> Tuple[float, Optional[Dict]]:
        D = self.dim_
        theta = w[:D]
        th = np.clip(theta, params["eps"], 1.0 - params["eps"])
        mode = params.get("match_mode", "meanprob")

        if mode == "meanprob":
            # Arithmetic mean of per-bit success probabilities
            # p_d = θ_d if i_d=1 else (1-θ_d)
            p = i * th + (1.0 - i) * (1.0 - th)
            m = float(np.mean(p))
            return m, {"meanprob": m}

        elif mode == "masked_geo":
            # Geometric mean over informative dims: where θ far from 0.5 or i_d=1
            tau = params.get("tau", 0.05)
            mask = (np.abs(th - 0.5) >= tau) | (i >= 0.5)
            if not np.any(mask):
                return 0.5, {"geo_like_masked": 0.5}
            # geometric mean over mask
            logp = np.sum(
                np.log(i[mask] * th[mask] + (1.0 - i[mask]) * (1.0 - th[mask]))
            )
            m = float(np.exp(logp / np.count_nonzero(mask)))
            return m, {"geo_like_masked": m, "mask_frac": np.count_nonzero(mask) / D}

        else:  # "geo" (your current behavior)
            if cache is None:
                loglik = self._bernoulli_loglik(i, th, params["eps"])
                loglik_avg = loglik / D
            else:
                loglik_avg = (
                    cache["loglik"] / D
                    if "loglik" in cache
                    else cache.get("loglik_avg", -np.log(2.0))
                )
            m = float(np.exp(loglik_avg))
            return m, {"geo_like": m}

    def update(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
    ) -> np.ndarray:
        """Online update of θ via running mean of bits.

        θ_new = (1 - 1/n_new) * θ + (1/n_new) * i
        Recompute variance-derived caches.

        """
        D = self.dim_
        theta = w[:D]
        n = w[-1]

        n_new = n + 1.0
        theta_new = (1.0 - (1.0 / n_new)) * theta + (1.0 / n_new) * i
        var_root_new, sqrt_det_var_new = self._variance_terms(theta_new, params["eps"])

        return np.concatenate([theta_new, var_root_new, [sqrt_det_var_new], [n_new]])

    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        """Initialize from the first sample (like GaussianART uses i as mean).

        For stronger priors, you could add pseudo-counts, but we keep parity with
        GaussianART’s “use the sample” behavior.

        """
        i_bin = np.asarray(i, dtype=float)
        # Safety: enforce binary {0,1}, but allow arrays like {0.,1.}
        if not np.all((i_bin == 0.0) | (i_bin == 1.0)):
            raise ValueError("BernoulliART requires binary inputs in {0,1}.")
        var_root, sqrt_det_var = self._variance_terms(i_bin, params["eps"])
        return np.concatenate([i_bin, var_root, [sqrt_det_var], [1.0]])

    def get_cluster_centers(self) -> List[np.ndarray]:
        """Return θ for each cluster (probability-of-1 per dimension).

        If you need hard prototypes later, threshold at 0.5 externally.

        """
        return [w[: self.dim_] for w in self.W]


#!/usr/bin/env python3
# supervised_test_bernoulli_artmap_mnist_subset_train_test.py

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from artlib import SimpleARTMAP


def load_mnist_subset_binarized_split(
    n_per_class: int = 200,
    threshold: float = 0.5,
    seed: int = 1337,
    train_size: float = 0.5,  # 50/50 split of the 2000-sample subset
):
    """Load a stratified 2000-sample subset of MNIST (200 per class), binarize pixels,
    and produce a stratified train/test split.

    Returns
    -------
    X_train : (N_train, 784) float32 in {0.0, 1.0}
    y_train : (N_train,)
    X_test  : (N_test,  784) float32 in {0.0, 1.0}
    y_test  : (N_test,)

    """
    # Fetch MNIST
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    X = X[:, ~(np.all(X == X[0, :], axis=0))]

    # First: take a stratified 2000-sample subset (200/class)
    subset_size = n_per_class * 10
    strat_subset = StratifiedShuffleSplit(
        n_splits=1, train_size=subset_size, random_state=seed
    )
    idx_subset, _ = next(strat_subset.split(X, y))

    Xs = X[idx_subset]
    ys = y[idx_subset]

    # Binarize pixels
    X_bin = (Xs > threshold).astype(np.float32)

    # Second: stratified train/test split within that subset
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx_tr, idx_te = next(sss.split(X_bin, ys))

    X_train, y_train = X_bin[idx_tr], ys[idx_tr]
    X_test, y_test = X_bin[idx_te], ys[idx_te]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    SEED = 1337
    np.random.seed(SEED)

    # Load data (2000 total -> 1000 train / 1000 test by default)
    X_train, y_train, X_test, y_test = load_mnist_subset_binarized_split(
        n_per_class=6999, threshold=0.2, seed=SEED, train_size=(6 / 7)
    )

    # Initialize BaseART (Bernoulli) and wrap with SimpleARTMAP for supervised learning
    base_art = BernoulliART(
        rho=0.2, alpha=1e-10, eps=1e-6, match_mode="meanprob"
    )  # tweak rho as desired
    model = SimpleARTMAP(base_art)

    # Fit on training set
    model.fit(X_train, y_train, verbose=True)

    print("Supervised BernoulliART + SimpleARTMAP on MNIST 2000-sample subset")
    print(
        f"  Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}, Dims: {X_train.shape[1]}"
    )
    print(f"  Vigilance (rho): {base_art.params.get('rho')}")
    print(f"  Clusters: {model.module_a.n_clusters}")
    # Evaluate on test set (not the training set)
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"  Test Accuracy: {test_acc:.4f}")
