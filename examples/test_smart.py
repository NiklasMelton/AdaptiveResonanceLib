from sklearn.datasets import load_iris, make_blobs
import matplotlib.pyplot as plt

import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from elementary.FuzzyART import FuzzyART, prepare_data
from hierarchical.SMART import SMART


def cluster_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = prepare_data(data)
    print("Prepared data has shape:", X.shape)

    base_params = {
        "alpha": 0.0,
        "beta": 1.0
    }
    rho_values = [0.7, 0.85, 0.9]
    cls = SMART(FuzzyART, rho_values, base_params)

    cls = cls.fit(X)
    y = cls.layers[0].labels_

    cls.visualize(X, y)
    plt.show()


def main():
    cluster_blobs()



if __name__ == "__main__":
    main()