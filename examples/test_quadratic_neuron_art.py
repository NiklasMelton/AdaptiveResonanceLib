from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from elementary.QuadraticNeuronART import QuadraticNeuronART
from common.utils import normalize


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = normalize(data)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.9,
        "s_init": 0.9,
        "lr_b": 0.1,
        "lr_w": 0.1,
        "lr_s": 0.1
    }
    cls = QuadraticNeuronART(**params)
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()