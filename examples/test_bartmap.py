
from sklearn.datasets import make_checkerboard
import matplotlib.pyplot as plt
import path
import sys
import numpy as np

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)

from biclustering.BARTMAP import BARTMAP
from elementary.FuzzyART import FuzzyART
from common.utils import normalize


def cluster_checkerboard():
    n_clusters = (4, 3)
    data, rows, columns = make_checkerboard(
        shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
    )
    print("Data has shape:", data.shape)

    X = normalize(data)
    print("Prepared data has shape:", X.shape)

    params_a = {
        "rho": 0.6,
        "alpha": 0.0,
        "beta": 1.0
    }
    params_b = {
        "rho": 0.6,
        "alpha": 0.0,
        "beta": 1.0
    }
    art_a = FuzzyART(params_a)
    art_b = FuzzyART(params_b)
    cls = BARTMAP(art_a, art_b, {"eta": -1.})
    cls.fit(X)

    print(f"{cls.n_row_clusters} row clusters found")
    print(f"{cls.n_column_clusters} column clusters found")

    cls.visualize()
    plt.show()


if __name__ == "__main__":
    cluster_checkerboard()
