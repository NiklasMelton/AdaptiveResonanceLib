from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np

from artlib.experimental.HullART import HullART
from artlib import VAT
from matplotlib.path import Path


def make_star(n_samples=500, noise=0.05):
    """
    Generates a random 2D dataset in the shape of a filled 5-pointed star.

    Parameters:
    - n_samples: int, number of points to generate
    - noise: float, standard deviation of noise to apply to each point

    Returns:
    - points: np.ndarray of shape (n_samples, 2), containing the x and y coordinates of points
    """
    # Star shape parameters
    star_radius_outer = 1.0
    star_radius_inner = 0.4

    # Calculate the 5-pointed star vertices
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    outer_points = [
        (star_radius_outer * np.cos(angle), star_radius_outer * np.sin(angle)) for angle
        in angles[::2]]
    inner_points = [
        (star_radius_inner * np.cos(angle), star_radius_inner * np.sin(angle)) for angle
        in angles[1::2]]
    star_vertices = np.array(outer_points + inner_points)

    # Create a star-shaped polygon path
    star_path = Path(star_vertices)

    # Generate random points and select only those within the star
    points = []
    while len(points) < n_samples:
        random_points = np.random.uniform(-1, 1, (
        n_samples, 2))  # Generate points in bounding box
        inside_points = random_points[
            star_path.contains_points(random_points)]  # Check if inside star
        points.extend(inside_points.tolist())

    # Limit to desired number of points and add noise
    points = np.array(points[:n_samples])
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)

    return points


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=1,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

    print("Data has shape:", data.shape)

    params = {"rho": 0.1, "alpha": 1e-3, "alpha_hat": 2.0, "min_lambda": 1e-10}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    y = cls.fit_predict(X)

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()

def cluster_moons():
    data, target = make_moons(n_samples=1000, noise=0.05, random_state=170,
                              shuffle=False)
    # vat, idx = VAT(data)
    # plt.figure()
    # plt.imshow(vat)
    #
    # data = data[idx,:]
    # target = target[idx]
    print("Data has shape:", data.shape)

    params = {"rho": 0.97, "alpha": 1e-10, "alpha_hat": 3., "min_lambda": 1e-10}
    cls = HullART(**params)

    X = cls.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    cls = cls.fit_gif(X, filename="fit_gif_HullART.gif", n_cluster_estimate=10, verbose=True)
    y = cls.labels_
    print(np.unique(y))

    print(f"{cls.n_clusters} clusters found")

    cls.visualize(X, y)
    plt.show()

# def cluster_star():
#     data = make_star(n_samples=1000, noise=0.05)
#
#     print("Data has shape:", data.shape)
#
#     params = {"rho": 0.5, "alpha": 1e-10, "alpha_hat": 0.0, "min_lambda": 1e-10}
#     cls = HullART(**params)
#
#     X = cls.prepare_data(data)
#     print("Prepared data has shape:", X.shape)
#
#     y = cls.fit_predict(X)
#     print(np.unique(y))
#
#     print(f"{cls.n_clusters} clusters found")
#
#     cls.visualize(X, y)
#     plt.show()

if __name__ == "__main__":
    # cluster_blobs()
    cluster_moons()
    # cluster_star()