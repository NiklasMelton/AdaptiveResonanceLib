from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import BinaryFuzzyARTMAP
from artlib.common.binary_encodings import binarize_features_thermometer


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    print("Data has shape:", data.shape)

    params = {"rho": 0.9, "alpha": 1e-10}
    cls = BinaryFuzzyARTMAP(**params)

    X = binarize_features_thermometer(data, n_bits=4)
    X = cls.prepare_data(X)
    print("Prepared data has shape:", X.shape)

    cls = cls.fit(X, target)
    y = cls.labels_

    print(f"{cls.module_a.n_clusters} clusters found")

    cls.visualize(X, y)

    plt.show()


def main():
    # cluster_iris()
    cluster_blobs()


if __name__ == "__main__":
    main()
