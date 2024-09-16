from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import FuzzyART, SMART


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    base_params = {
        "alpha": 0.0,
        "beta": 1.0
    }
    rho_values = [0.5, 0.85, 0.9]
    cls = SMART(FuzzyART, rho_values, base_params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    cls = cls.fit(X)
    y = cls.layers[0].labels_

    cls.visualize(X, y)
    plt.show()


def main():
    cluster_blobs()



if __name__ == "__main__":
    main()