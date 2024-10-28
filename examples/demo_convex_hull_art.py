from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.HullART import HullART, plot_polygon
from artlib import VAT
from alphashape import alphashape


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    print("Data has shape:", data.shape)

    params = {"rho": 0.6, "alpha": 1e-3, "alpha_hat": 1.0}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()

def cluster_moons():
    data, target = make_moons(n_samples=1000, noise=0.05, random_state=170,
                              shuffle=False)
    vat, idx = VAT(data)
    plt.figure()
    plt.imshow(vat)

    data = data[idx,:]
    target = target[idx]
    print("Data has shape:", data.shape)

    params = {"rho": 0.95, "alpha": 1e-10, "alpha_hat": 1.0}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()

if __name__ == "__main__":
    # cluster_blobs()
    cluster_moons()