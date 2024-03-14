from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import TopoART, FuzzyART


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = FuzzyART.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.6,
        "alpha": 0.8,
        "beta": 1.0
    }
    base_art = FuzzyART(**params)
    cls = TopoART(base_art, betta_lower=0.3, tau=150, phi=35)
    cls = cls.fit(X, max_iter=5)
    y = cls.labels_

    print(f"{cls.n_clusters} clusters found")
    print(f"{cls.base_module.n_clusters} internal clusters found")

    print("Adjacency Matrix:")
    print(cls.adjacency)

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()