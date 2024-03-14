import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import FusionART, FuzzyART


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    data_channel_a = data[:,0].reshape((-1,1))
    data_channel_b = data[:,1].reshape((-1,1))

    X_channel_a = FuzzyART.prepare_data(data_channel_a)
    X_channel_b = FuzzyART.prepare_data(data_channel_b)

    X = np.hstack([X_channel_a, X_channel_b])
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.5,
        "alpha": 0.0,
        "beta": 1.0
    }
    art_a = FuzzyART(**params)
    art_b = FuzzyART(**params)
    cls = FusionART([art_a, art_b], gamma_values=[0.5, 0.5], channel_dims=[2,2])
    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(data, y)
    plt.show()


if __name__ == "__main__":
    cluster_blobs()