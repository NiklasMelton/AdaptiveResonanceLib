from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.HullART import HullART
from artlib.common.VAT import VAT


def cluster_moons():
    data, target = make_moons(n_samples=1000, noise=0.08, random_state=10,
                              shuffle=False)
    # _, s = VAT(data)
    # data = np.array(data)[s, :]
    # target = target[s]
    print("Data has shape:", data.shape)

    params = {"rho": 0.0, "alpha": 1e-20, "alpha_hull": 8., "min_lambda": 1e-5,
              "max_lambda": 0.2}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    # cls = cls.fit_gif(X, filename="fit_gif_HullART.gif", n_cluster_estimate=10, verbose=True)
    cls = cls.fit(X, verbose=True)
    y = cls.labels_
    print(np.unique(y))

    cls.visualize(X, y)
    plt.show()


def cluster_3d_data():
    # Generate 3D dataset
    data, target = make_blobs(n_samples=1000, centers=4, n_features=3,
                              cluster_std=1.0, random_state=9, shuffle=False)
    print("Data has shape:", data.shape)

    # Parameters for HullART
    params = {"rho": 0.0, "alpha": 1e-20, "alpha_hull": 0., "min_lambda": 1e-1,
              "max_lambda": 0.4}
    cls = HullART(**params)

    # Prepare data
    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    # Fit the model
    cls = cls.fit(X, verbose=True)
    y = cls.labels_
    print(np.unique(y))


    # Visualization for 3D data
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    cls.visualize(X, y, ax=ax)
    plt.show()
    plt.show()


if __name__ == "__main__":
    cluster_moons()
    # cluster_3d_data()