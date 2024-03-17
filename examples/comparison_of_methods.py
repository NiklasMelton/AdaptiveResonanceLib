import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler


import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

import artlib

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed, shuffle=False
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed, shuffle=False)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, shuffle=False)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, shuffle=False)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state, shuffle=False
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

datasets = [
    (
        noisy_circles,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.95,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.9,

                },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.6,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.99,
                    "s_init": 0.8,
                    "lr_b": 0.2,
                    "lr_w": 0.1,
                    "lr_s": 0.2
                },
        },
    ),
    (
        noisy_moons,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.85,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.78,

                },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.7,
                    "alpha": 0.9,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.9,
                    "s_init": 0.9,
                    "lr_b": 0.1,
                    "lr_w": 0.1,
                    "lr_s": 0.1
                },
        },
    ),
    (
        varied,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.95,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.88,

                },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.66,
                    "alpha": 0.1,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.95,
                    "s_init": 0.8,
                    "lr_b": 0.1,
                    "lr_w": 0.1,
                    "lr_s": 0.1
                },
        },
    ),
    (
        aniso,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.95,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.9,

                 },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.97,
                    "s_init": 0.95,
                    "lr_b": 0.1,
                    "lr_w": 0.1,
                    "lr_s": 0.2
                },
        },
    ),
    (
        blobs,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.91,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.82,

                 },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.6,
                    "alpha": 0.0,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.9,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.95,
                    "s_init": 0.8,
                    "lr_b": 0.05,
                    "lr_w": 0.1,
                    "lr_s": 0.05
                },
        },
    ),
    (
        no_structure,
        {
            "BayesianART":
                {
                    "rho": 2e-5,
                    "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
                },
            "DualVigilanceART":
                {
                    "FuzzyART":
                        {
                            "rho": 0.85,

                            "alpha": 0.8,
                            "beta": 1.0
                        },
                    "rho_lower_bound": 0.78,

                },
            "EllipsoidART":
                {
                    "rho": 0.2,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.6,
                    "mu": 0.8
                },
            "FuzzyART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0
                },
            "GaussianART":
                {
                    "rho": 0.15,
                    "sigma_init": np.array([0.5, 0.5]),
                },
            "HypersphereART":
                {
                    "rho": 0.7,
                    "alpha": 0.0,
                    "beta": 1.0,
                    "r_hat": 0.8
                },
            "QuadraticNeuronART":
                {
                    "rho": 0.9,
                    "s_init": 0.9,
                    "lr_b": 0.1,
                    "lr_w": 0.1,
                    "lr_s": 0.1
                },
        },
    ),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = algo_params

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============
    print(i_dataset)
    hypersphere = artlib.HypersphereART(**params["HypersphereART"])
    ellipsoid = artlib.EllipsoidART(**params["EllipsoidART"])
    gaussian = artlib.GaussianART(**params["GaussianART"])
    bayesian = artlib.BayesianART(**params["BayesianART"])
    quadratic_neuron = artlib.QuadraticNeuronART(**params["QuadraticNeuronART"])
    fuzzy = artlib.FuzzyART(**params["FuzzyART"])
    fuzzyDV = artlib.DualVigilanceART(
        artlib.FuzzyART(**params["DualVigilanceART"]["FuzzyART"]),
        lower_bound=params["DualVigilanceART"]["rho_lower_bound"]
    )



    clustering_algorithms = (
        ("Hypersphere\nART", hypersphere),
        ("Ellipsoid\nART", ellipsoid),
        ("Gaussian\nART", gaussian),
        ("Bayesian\nART", bayesian),
        ("Quadratic Neuron\nART", quadratic_neuron),
        ("FuzzyART", fuzzy),
        ("Dual Vigilance\nFuzzyART", fuzzyDV),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            X_prepared = algorithm.prepare_data(X)
            algorithm.fit(X_prepared)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X_prepared)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1

plt.show()