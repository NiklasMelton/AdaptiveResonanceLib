from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from artlib import ART2A, normalize

"""
==================================================================
DISCLAIMER: DO NOT USE ART2!!!
IT DOES NOT WORK
It is provided for completeness only.
Stephan Grossberg himself has said ART2 does not work.
==================================================================
"""

def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=2, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = normalize(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.2,
        "alpha": 0.0,
        "beta": 1.0,
    }
    cls = ART2A(**params)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()
