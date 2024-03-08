import numpy as np
from sklearn.datasets import load_iris, make_blobs
import umap
import umap.plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


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
    n = len(np.unique(y))
    colors = cm.rainbow(np.linspace(0, 1, n))

    fig, ax = plt.subplots()

    for k, col in enumerate(colors):
        cluster_data = y == k
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], color=col, marker=".", s=10)

    print("Cluster Info:")
    for j in range(len(cls.modules)):
        print(f"\t{cls.modules[j].n_clusters} level-{j+1} clusters found")
        layer_colors = []
        for k in range(cls.modules[j].n_clusters):
            if j == 0:
                layer_colors.append(colors[k])
            else:
                layer_colors.append(colors[cls.map_deep(j-1, k)])
        cls.modules[j].plot_bounding_boxes(ax, layer_colors)

    plt.show()


def main():
    cluster_blobs()



if __name__ == "__main__":
    main()