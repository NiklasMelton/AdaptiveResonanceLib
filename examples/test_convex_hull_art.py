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

# points = np.array(
#     [
#         [0.0, 0.0],
#         [0.0, 0.5],
#         [0.1, 0.2],
#     ]
# )
# new_point = np.array([[0.5, 0.5]])
# hull = ConvexHull(points, incremental=True)
#
# fig, ax = plt.subplots()
#
# plot_convex_polygon(points, ax)
# plt.scatter(new_point[:,0], new_point[:, 1], c='k')
#
# fig, ax = plt.subplots()
# hull.add_points(new_point)
#
# plot_convex_polygon(hull.points[hull.vertices,:], ax)
# plt.show()
