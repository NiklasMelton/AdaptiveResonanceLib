from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.HullART import HullART
from artlib import VAT
from matplotlib.path import Path



def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=1,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    print("Data has shape:", data.shape)

    params = {"rho": 0.1, "alpha": 1e-3, "alpha_hat": 2.0, "min_lambda": 1e-10}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()

def cluster_moons():
    data, target = make_moons(n_samples=1000, noise=0.05, random_state=171,
                              shuffle=False)
    # vat, idx = VAT(data)
    # plt.figure()
    # plt.imshow(vat)
    #
    # data = data[idx,:]
    # target = target[idx]
    print("Data has shape:", data.shape)

    params = {"rho": 0.2, "alpha": 1e-20, "alpha_hat": 8., "min_lambda": 1e-5, "rho_lambda": 0.2}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    # cls = cls.fit_gif(X, filename="fit_gif_HullART.gif", n_cluster_estimate=10, verbose=True)
    cls = cls.fit(X, verbose=True)
    y = cls.labels_
    print(np.unique(y))

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    # cluster_blobs()
    cluster_moons()