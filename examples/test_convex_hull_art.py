from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.ConvexHullART import ConvexHullART, plot_convex_polygon
from scipy.spatial import ConvexHull


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = ConvexHullART.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.75,
    }
    cls = ConvexHullART(**params)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()
