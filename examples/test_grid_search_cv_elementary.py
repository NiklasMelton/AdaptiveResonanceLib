import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import path
import sys

# directory reach
directory = path.Path(__file__).abspath()

print(directory.parent)
# setting path
sys.path.append(directory.parent.parent)
from artlib import FuzzyART


def grid_search_blobs():
    data, target = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    X = FuzzyART.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    base_params = {
        "rho": 0.7,
        "alpha": 0.0,
        "beta": 1.0,
    }
    cls = FuzzyART(**base_params)

    param_grid = {
        "rho": [0.0, 0.25, 0.5, 0.7, 0.9],
        "alpha": [0.0],
        "beta": [1.0],
    }

    grid = GridSearchCV(
        cls,
        param_grid,
        refit=True,
        verbose=3,
        n_jobs=1,
        scoring="adjusted_rand_score",
        cv=[
            (
                np.array(
                    list(range(len(X)))
                ),
                np.array(
                    list(range(len(X)))
                )
            )
        ]
    )

    grid.fit(X, target)

    # print best parameter after tuning
    print("Best parameters:")
    print(grid.best_params_)
    grid_predictions = grid.predict(X)

    # print classification report
    print(classification_report(target, grid_predictions))


if __name__ == "__main__":
    grid_search_blobs()
