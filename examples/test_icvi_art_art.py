from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from artlib import iCVIART, FuzzyART
import cvi


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = FuzzyART.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.0,
        "alpha": 0.0,
        "beta": 1.0
    }
    base_art = FuzzyART(**params)
    icvi = cvi.DB()
    cls = iCVIART(base_art, icvi)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")
    print(f"{cls.base_module.n_clusters} internal clusters found")
    print(cls.iCVI._n_samples)

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()
