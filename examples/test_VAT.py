# Bezdek, J. C., & Hathaway, R. J. (2002). VAT: A tool for visual assessment of cluster tendency.
# Proceedings of the 2002 International Joint Conference on Neural Networks. doi:10.1109/IJCNN.2002.1007487


import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from artlib import VAT
import time


def visualize_blobs():
    data, target = make_blobs(n_samples=2000, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    t0 = time.time()
    R, P = VAT(data)
    dt = time.time() - t0
    print("time:", dt)
    print(P.shape, np.unique(P))


    plt.figure()
    plt.scatter(data[:,0], data[:,1],c=P, cmap="jet",s=10)

    plt.figure()
    plt.imshow(R)
    plt.show()

if __name__ == "__main__":

    visualize_blobs()
