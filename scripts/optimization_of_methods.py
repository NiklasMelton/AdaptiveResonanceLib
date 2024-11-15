import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import artlib

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples,
    factor=0.5,
    noise=0.05,
    random_state=seed,
    shuffle=False,
)
noisy_moons = datasets.make_moons(
    n_samples=n_samples, noise=0.05, random_state=seed, shuffle=False
)
blobs = datasets.make_blobs(
    n_samples=n_samples, random_state=seed, shuffle=False
)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(
    n_samples=n_samples, random_state=random_state, shuffle=False
)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=random_state,
    shuffle=False,
)

# ============
# Set up cluster parameters
# ============

init_params = {
    "BayesianART": {
        "rho": 2e-5,
        "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
    },
    "DualVigilanceFuzzyART": {
        "FuzzyART": {"rho": 0.95, "alpha": 0.8, "beta": 1.0},
        "rho_lower_bound": 0.5,
        "base_module": artlib.FuzzyART(0.0, 0.0, 1.0),
    },
    "EllipsoidART": {
        "rho": 0.2,
        "alpha": 0.0,
        "beta": 1.0,
        "r_hat": 0.6,
        "mu": 0.8,
    },
    "FuzzyART": {"rho": 0.7, "alpha": 0.0, "beta": 1.0},
    "GaussianART": {
        "rho": 0.15,
        "sigma_init": np.array([0.5, 0.5]),
    },
    "HypersphereART": {"rho": 0.6, "alpha": 0.0, "beta": 1.0, "r_hat": 0.8},
    "QuadraticNeuronART": {
        "rho": 0.99,
        "s_init": 0.8,
        "lr_b": 0.2,
        "lr_w": 0.1,
        "lr_s": 0.2,
    },
}

rho_range = np.linspace(0.1, 1.0, 28)
rho_range[-1] = 0.99
alpha_range = np.linspace(0.0, 1.0, 21)


param_grids = {
    "BayesianART": {
        "rho": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5],
        "cov_init": [np.array([[1e-4, 0.0], [0.0, 1e-4]])],
    },
    "DualVigilanceFuzzyART": {
        "base_module": [artlib.FuzzyART(0.1, 0.0, 1.0)],
        "base_module__rho": rho_range,
        "base_module__alpha": alpha_range,
        "base_module__beta": [1.0],
        "rho_lower_bound": rho_range,
    },
    "EllipsoidART": {
        "rho": rho_range,
        "alpha": alpha_range,
        "beta": [1.0],
        "r_hat": np.linspace(0.1, 0.9, 17),
        "mu": np.linspace(0.1, 1.0, 10),
    },
    "FuzzyART": {"rho": rho_range, "alpha": alpha_range, "beta": [1.0]},
    "GaussianART": {
        "rho": rho_range,
        "sigma_init": [np.array([0.5, 0.5])],
    },
    "HypersphereART": {
        "rho": rho_range,
        "alpha": alpha_range,
        "beta": [1.0],
        "r_hat": np.linspace(0.1, 0.8, 22),
    },
    "QuadraticNeuronART": {
        "rho": rho_range,
        "s_init": np.linspace(0.1, 1.0, 10),
        "lr_b": np.linspace(0.1, 0.6, 6),
        "lr_w": np.linspace(0.1, 0.6, 6),
        "lr_s": np.linspace(0.1, 0.6, 6),
    },
}

plot_num = 1

datasets = [
    (
        "noisy_circles",
        noisy_circles,
    ),
    (
        "noisy_moons",
        noisy_moons,
    ),
    (
        "varied",
        varied,
    ),
    (
        "aniso",
        aniso,
    ),
    (
        "blobs",
        blobs,
    ),
    (
        "no_structure",
        no_structure,
    ),
]

for i_dataset, (dataset_name, dataset) in enumerate(datasets):
    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============
    print(dataset_name)
    hypersphere = artlib.HypersphereART(**init_params["HypersphereART"])
    ellipsoid = artlib.EllipsoidART(**init_params["EllipsoidART"])
    gaussian = artlib.GaussianART(**init_params["GaussianART"])
    bayesian = artlib.BayesianART(**init_params["BayesianART"])
    quadratic_neuron = artlib.QuadraticNeuronART(
        **init_params["QuadraticNeuronART"]
    )
    fuzzy = artlib.FuzzyART(**init_params["FuzzyART"])
    fuzzyDV = artlib.DualVigilanceART(
        artlib.FuzzyART(**init_params["FuzzyART"]),
        rho_lower_bound=init_params["DualVigilanceFuzzyART"]["rho_lower_bound"],
    )

    clustering_algorithms = (
        ("DualVigilanceFuzzyART", fuzzyDV),
        ("HypersphereART", hypersphere),
        ("EllipsoidART", ellipsoid),
        ("GaussianART", gaussian),
        ("BayesianART", bayesian),
        ("QuadraticNeuronART", quadratic_neuron),
        ("FuzzyART", fuzzy),
    )

    for name, algorithm in clustering_algorithms:
        print("\t", name)
        grid = GridSearchCV(
            algorithm,
            param_grids[name],
            refit=False,
            verbose=0,
            n_jobs=-1,
            scoring="adjusted_rand_score",
            cv=[(np.array(list(range(len(X)))), np.array(list(range(len(X)))))],
        )
        X_prepared = algorithm.prepare_data(X)
        grid.fit(X_prepared, y)
        print("\tBest parameters:")
        print("\t\t", grid.best_params_)
