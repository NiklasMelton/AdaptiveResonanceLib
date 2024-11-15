from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from artlib import GaussianART


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    print("Data has shape:", data.shape)

    params = {
        "rho": 0.05,
        "sigma_init": np.array([0.5, 0.5]),
    }
    cls = GaussianART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()
