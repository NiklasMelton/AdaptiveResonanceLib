from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from artlib import BayesianART
import numpy as np


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    params = {
        "rho": 7e-5,
        "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]]),
    }
    cls = BayesianART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)


    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()