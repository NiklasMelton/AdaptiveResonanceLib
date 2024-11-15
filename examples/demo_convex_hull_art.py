from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.HullART import HullART, plot_polygon
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

def test():
    points = np.array(
        [(0.0, 0.0), (0.0, 1.0), (1.0,1.0), (1.0, 0.0)]
    )
    x = alphashape(points, alpha=1.0)
    print(x.length)

if __name__ == "__main__":
    cluster_blobs()
    # test()