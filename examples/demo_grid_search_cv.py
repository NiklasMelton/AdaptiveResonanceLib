from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from artlib import HypersphereART, SimpleARTMAP


def grid_search_blobs():
    data, target = load_iris(return_X_y=True)
    print("Data has shape:", data.shape)

    base_params = {"rho": 0.3, "alpha": 0.0, "beta": 1.0, "r_hat": 0.8}
    art = HypersphereART(**base_params)

    X = art.prepare_data(data)
    print("Prepared data has shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.33, random_state=0
    )

    cls = SimpleARTMAP(art)
    # print(cls.get_params(deep=False))
    # raise ValueError

    param_grid = {
        "module_a__rho": [0.0, 0.25, 0.5, 0.7, 0.9],
        "module_a__alpha": [0.0],
        "module_a__beta": [1.0],
        "module_a__r_hat": [0.8],
    }

    grid = GridSearchCV(cls, param_grid, refit=True, verbose=3, n_jobs=1)

    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print("Best parameters:")
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


if __name__ == "__main__":
    grid_search_blobs()
