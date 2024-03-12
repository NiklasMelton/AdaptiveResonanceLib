from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from elementary.BayesianART import BayesianART
from common.utils import normalize
import numpy as np


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = normalize(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.00002,
        "cov_init": np.array([[0.0001, 0.0], [0.0, 0.0001]]),
    }
    cls = BayesianART(params)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()