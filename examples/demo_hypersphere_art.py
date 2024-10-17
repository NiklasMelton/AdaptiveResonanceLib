from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import HypersphereART


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    print("Data has shape:", data.shape)

    params = {"rho": 0.5, "alpha": 0.0, "beta": 1.0, "r_hat": 0.8}
    cls = HypersphereART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()
