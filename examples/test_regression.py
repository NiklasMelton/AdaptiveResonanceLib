import numpy as np
from sklearn.datasets import load_iris, make_blobs
import matplotlib.pyplot as plt

from artlib import FuzzyART, ARTMAP, FusionART, normalize

def make_data():
    x_ = np.arange(0.0,4.0, 0.01)
    x = x_+0.5
    y = np.sin(x)+np.cos(x**2)+np.log(x**3)+np.cos(x)*np.log(x**2)

    x_norm = normalize(x)
    y_norm = normalize(y)
    return x_norm.reshape((-1,1)), y_norm.reshape((-1,1))


def fit_regression_ARTMAP():
    X_, y_ = make_data()
    print("Data has shape:", X_.shape)

    X = FuzzyART.prepare_data(X_)
    y = FuzzyART.prepare_data(y_)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.95,
        "alpha": 0.0,
        "beta": 1.0
    }
    art_a = FuzzyART(**params)
    art_b = FuzzyART(**params)

    cls = ARTMAP(art_a, art_b)

    cls = cls.fit(X, y)

    print(f"{len(np.unique(cls.labels_a))} clusters found")

    y_pred = cls.predict_regression(X)

    plt.plot(X_, y_, 'r-', label="original")
    plt.plot(X_, y_pred, "b-", label="ARTMAP")
    plt.legend()
    plt.show()

def fit_regression_FusionART():
    X_, y_ = make_data()
    print("Data has shape:", X_.shape)

    X = FuzzyART.prepare_data(X_)
    y = FuzzyART.prepare_data(y_)
    print("Prepared data has shape:", X.shape)

    params = {
        "rho": 0.95,
        "alpha": 0.0,
        "beta": 1.0
    }
    art_a = FuzzyART(**params)
    art_b = FuzzyART(**params)

    cls = FusionART([art_a, art_b], gamma_values=np.array([0.5, 0.5]), channel_dims=[2, 2])

    Xy = cls.join_channel_data(channel_data=[X, y])

    cls = cls.fit(Xy)

    print(f"{cls.n_clusters} clusters found")

    X_no_y = cls.join_channel_data(channel_data=[X], skip_channels=[-1])

    y_pred = cls.predict_regression(X_no_y, target_channels=[-1])

    plt.plot(X_, y_, 'r-', label="original")
    plt.plot(X_, y_pred, "b-", label="FusionART")
    plt.legend()
    plt.show()

def main():
    # fit_regression_ARTMAP()
    fit_regression_FusionART()



if __name__ == "__main__":
    main()