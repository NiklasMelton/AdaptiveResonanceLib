import numpy as np
from sklearn.datasets import load_iris, make_blobs
import matplotlib.pyplot as plt

from artlib import FuzzyART, SimpleARTMAP, QuadraticNeuronART


def cluster_iris():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import umap.plot

    data, target = load_iris(return_X_y=True)
    print("Data has shape:", data.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.33, random_state=0
    )

    params = {"rho": 0.0, "alpha": 0.0, "beta": 1.0}
    art = FuzzyART(**params)
    # params = {
    #     "rho": 0.0,
    #     "s_init": 0.9,
    #     "lr_b": 0.1,
    #     "lr_w": 0.1,
    #     "lr_s": 0.1
    # }
    # art = QuadraticNeuronART(**params)

    X = art.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    cls = SimpleARTMAP(art)

    cls = cls.fit(X_train, y_train)

    print(f"{len(np.unique(cls.labels_a))} clusters found")

    _, y_pred = cls.predict(X_test)

    print(classification_report(y_test, y_pred))

    mapper = umap.UMAP().fit(X_test)

    umap.plot.points(
        mapper, labels=y_pred, color_key_cmap="Paired", background="black"
    )
    umap.plot.plt.show()


def cluster_blobs():
    data, target = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )
    print("Data has shape:", data.shape)

    params = {"rho": 0.9, "alpha": 0.0, "beta": 1.0}
    art = FuzzyART(**params)

    # params = {
    #     "rho": 0.0,
    #     "s_init": 0.9,
    #     "lr_b": 0.1,
    #     "lr_w": 0.1,
    #     "lr_s": 0.1
    # }
    # art = QuadraticNeuronART(**params)

    X = art.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    cls = SimpleARTMAP(art)

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
