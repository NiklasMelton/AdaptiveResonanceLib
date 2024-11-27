from artlib import FuzzyART, SimpleARTMAP, HypersphereART, VAT
from artlib.experimental.HullART import HullART
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pickle
import warnings
from tqdm import tqdm
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import concurrent.futures

warnings.filterwarnings("ignore")

datasets = [
    # (15, "breast cancer (original)"),
    (936, "NPHA"),
    (759, "glioma"),
    (17, "breast cancer (diagnostic)"),
    (14, "breast cancer (yugoslavia)"),
    (16, "breast cancer (prognostic)"),
    (451, "breast cancer (coimbra)"),
    (45, "heart disease"),
    (519, "heart failure"),
    (62, "lung cancer"),
    # (46, "hepatitis"),
    (174, "parkinsons"),
    (145, "statlog"),
    (33, "dermatology"),
    (30, "contraceptive choice"),
    (225, "ILPD"),
    # (571, "HCV"),
    # (43, "habermans"),
    (863, "maternal health"),
    (915, "differntiated thyroid cancer"),
    (890, "AIDS clinical trials 175"),
    (878, "cirrhosis"),
    (244, "fertility"),
    (83, "primary tumor"),
]

art_models = [
    (FuzzyART, {"rho": 0.0, "alpha": 1e-5, "beta": 1.0}),
    (HypersphereART, {"rho": 0.0, "alpha": 1e-5, "beta": 1.0, "r_hat": None}),
    (
        HullART,
        {
            "rho": 0.0,
            "alpha": 1e-5,
            "alpha_hull": 0.0,
            "min_lambda": 1e-5,
            "max_lambda": 0.2,
        },
    ),
]

MT_METHODS = ["MT1", "MT+", "MT-", "MT0", "MT~"]
EPSILONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, np.inf]


def verify_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


def load_mnist_as_numpy():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten the images from 28x28 into 784 (28*28) vectors
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Normalize the pixel values to the range [0, 1]
    x_train_flat = x_train_flat.astype("float32") / 255
    x_test_flat = x_test_flat.astype("float32") / 255

    # # Combine train and test data into one numpy array
    # X = np.concatenate([x_train_flat, x_test_flat])
    # y = np.concatenate([y_train, y_test]).reshape((-1, 1))

    # return X, y
    y = y_test.reshape((-1, 1))

    return x_test_flat, y


def data_loader():
    for id, name in datasets:
        try:
            if name == "MNIST":
                X, y = load_mnist_as_numpy()
                yield name, X, y
            else:
                if os.path.exists(f"data/uci_dataset_{name}_{id}.pickle"):
                    data = pickle.load(
                        (open(f"data/uci_dataset_{name}_{id}.pickle", "rb"))
                    )
                    yield data["name"], data["X"], data["y"]
                else:
                    data = fetch_ucirepo(id=id)

                    features = data.data.features
                    # features = features.fillna("missing")
                    if "id" in features.columns:
                        features = features.drop(["id"], axis=1)
                    if "Id" in features.columns:
                        features = features.drop(["Id"], axis=1)
                    if "ID" in features.columns:
                        features = features.drop(["ID"], axis=1)
                    if name == "mushroom":
                        X = pd.get_dummies(features).astype(float).to_numpy()
                    elif name == "NPHA":
                        X = (
                            pd.get_dummies(features.drop(columns=["Age"]))
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "abalone":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "Sex",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "glioma":
                        columns_to_encode = features.columns.difference(
                            ["Age_at_diagnosis"]
                        )
                        X = (
                            pd.get_dummies(features, columns=columns_to_encode)
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "breast cancer (yugoslavia)":
                        X = pd.get_dummies(features).astype(float).to_numpy()
                    elif name == "heart disease":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "sex",
                                    "cp",
                                    "fbs",
                                    "restecg",
                                    "exang",
                                    "slope",
                                    "thal",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "lung cancer":
                        X = pd.get_dummies(features).astype(float).to_numpy()
                    elif name == "hepatitis":
                        features = features.fillna("missing")
                        X = pd.get_dummies(features).astype(float).to_numpy()
                    elif name == "statlog":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=["chest-pain", "electrocardiographic", "thal"],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "myocardial infarction":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "INF_ANAM",
                                    "STENOK_AN",
                                    "FK_STENOK",
                                    "IBS_POST",
                                    "GB",
                                    "DLIT_AG",
                                    "ZSN_A",
                                    "ant_im",
                                    "lat_im",
                                    "inf_im",
                                    "post_im",
                                    "ROE",
                                    "TIME_B_S",
                                    "R_AB_1_n",
                                    "R_AB_2_n",
                                    "R_AB_3_n",
                                    "NOT_NA_1_n",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "contraceptive choice":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "wife_edu",
                                    "husband_edu",
                                    "husband_occupation",
                                    "standard_of_living_index",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "differntiated thyroid cancer":
                        columns_to_encode = features.columns.difference(["Age"])
                        X = (
                            pd.get_dummies(features, columns=columns_to_encode)
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "AIDS clinical trials 175":
                        # features = features.drop(["pidnum"], axis=1)
                        X = (
                            pd.get_dummies(features, columns=["trt", "strat"])
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "cirrhosis":
                        features[features == "NaNN"] = 0
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "Drug",
                                    "Sex",
                                    "Ascites",
                                    "Hepatomegaly",
                                    "Spiders",
                                    "Edema",
                                    "Stage",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "fertility":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "high_fevers",
                                    "alcohol",
                                    "smoking",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "echocardiogram":
                        features = features.drop(["name", "group"], axis=1)
                        X = features.astype(float).to_numpy()
                    elif name == "ILPD":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "Gender",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    elif name == "HCV":
                        X = (
                            pd.get_dummies(
                                features,
                                columns=[
                                    "Sex",
                                ],
                            )
                            .astype(float)
                            .to_numpy()
                        )
                    else:
                        X = features.astype(float).to_numpy()
                    targets = data.data.targets
                    targets.columns = ["class"]
                    y = pd.Categorical(targets["class"]).codes.reshape((-1, 1))
                    assert verify_numeric(X), f"{name} X is not numeric"
                    assert verify_numeric(y), f"{name} y is not numeric"
                    pickle.dump(
                        {"name": name, "X": X, "y": y},
                        open(f"data/uci_dataset_{name}_{id}.pickle", "wb"),
                    )
                    print(f"VAT soring: {name}")
                    R, P = VAT(X)
                    X = X[P, :]
                    y = y[P]
                    yield name, X, y
        except Exception as E:
            print(E, name)
            raise E


def experiment_loader():
    for model, params in art_models:
        for match_tracking in MT_METHODS:
            for dataset_name, X, y in data_loader():
                if match_tracking in ["MT+", "MT-"]:
                    for epsilon in EPSILONS:
                        for fold in range(5):
                            yield model, params, match_tracking, epsilon, dataset_name, X, y, fold
                else:
                    for fold in range(5):
                        yield model, params, match_tracking, 0.0, dataset_name, X, y, fold


def experiment_loader_minimum():
    for model, params in art_models:
        for dataset_name, X, y in data_loader():
            yield model, params, "MT0", 0.0, dataset_name, X, y, 0


def perform_test(model, params, match_tracking, epsilon, dataset_name, X_, y, fold):
    test_results = []
    params = dict(params)
    if "r_hat" in params:
        params["r_hat"] = 0.5 * np.sqrt(X_.shape[1])
    if "sigma_init" in params:
        params["sigma_init"] = 0.5 * np.ones((X_.shape[1],))
    # if "max_lambda" in params:
    #     params["max_lambda"] = 0.25*np.sqrt(X_.shape[1])
    for shuffle in range(1):
        test_data = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "clusters": None,
            "time": None,
            "params": params,
            "dataset": dataset_name,
            "match_tracking": match_tracking,
            "epsilon": epsilon,
            "model": str(model.__name__),
            "shuffle": shuffle,
            "fold": fold,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=shuffle)
        split_iterator = enumerate(skf.split(X_, y))

        for i, (train_index, test_index) in split_iterator:
            if i == fold:
                art = model(**params)
                cls = SimpleARTMAP(art)

                X_min = np.min(X_, axis=0)
                X_max = np.max(X_, axis=0)

                cls.d_min = X_min
                cls.d_max = X_max

                X = cls.prepare_data(X_)
                non_nan_columns = ~np.isnan(X).any(axis=0)
                X = X[:, non_nan_columns]

                X_train = X[train_index, :]
                X_test = X[test_index, :]
                y_train = y[train_index, :]
                y_test = y[test_index, :]

                # Test modified match reset method
                t0 = time.time()
                cls.fit(
                    X_train, y_train, match_tracking=match_tracking, epsilon=epsilon
                )
                dt = time.time() - t0

                y_pred = cls.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="macro")
                rec = recall_score(y_test, y_pred, average="macro")
                f1 = f1_score(y_test, y_pred, average="macro")

                test_data[f"accuracy"] = acc
                test_data[f"precision"] = prec
                test_data[f"recall"] = rec
                test_data[f"f1"] = f1
                test_data[f"time"] = dt
                test_data[f"clusters"] = int(cls.n_clusters_a)

        test_results.append(test_data)

    return test_results


def perform_tests(parquet_file="incremental.parquet", num_workers=None):
    # Initialize an empty Parquet file if it doesn't exist
    if not os.path.exists(parquet_file):
        pd.DataFrame().to_parquet(parquet_file)
        existing_args = set()
    else:
        existing_df = pd.read_parquet(parquet_file)
        existing_args = set(
            (
                row["model"],
                row["match_tracking"],
                row["epsilon"],
                row["dataset"],
                row["fold"],
            )
            for _, row in existing_df.iterrows()
        )

    print("Finding total jobs")
    n_jobs = 0
    for (
        model,
        params,
        match_tracking,
        epsilon,
        dataset_name,
        X_,
        y,
        fold,
    ) in experiment_loader():
        current_args = (
            str(model.__name__),
            match_tracking,
            epsilon,
            dataset_name,
            fold,
        )
        # Skip processing if the current_args are already in existing_args
        if current_args not in existing_args:
            print(current_args)
            n_jobs += 1
    print(f"Total Jobs: {n_jobs}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a generator that yields futures
        def future_generator():
            for (
                model,
                params,
                match_tracking,
                epsilon,
                dataset_name,
                X_,
                y,
                fold,
            ) in experiment_loader():
                current_args = (
                    str(model.__name__),
                    match_tracking,
                    epsilon,
                    dataset_name,
                    fold,
                )
                # Skip processing if the current_args are already in existing_args
                if current_args not in existing_args:
                    # time.sleep(2)
                    # wait_for_available_resources()
                    yield executor.submit(
                        perform_test,
                        model,
                        params,
                        match_tracking,
                        epsilon,
                        dataset_name,
                        X_,
                        y,
                        fold,
                    )

        # Initialize tqdm progress bar with an unknown total
        with tqdm(total=n_jobs) as pbar:
            # Use the generator in as_completed
            for future in concurrent.futures.as_completed(future_generator()):
                try:
                    result = future.result()
                    result_df = pd.DataFrame(result)  # Create DataFrame for the result

                    # Append the result to the Parquet file
                    existing_df = pd.read_parquet(parquet_file)
                    combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                    combined_df.to_parquet(parquet_file)

                except Exception as e:
                    print(f"Error occurred: {e}")
                finally:
                    # Update the progress bar after each job is completed
                    pbar.update(1)

    # Load and return the final combined DataFrame
    final_df = pd.read_parquet(parquet_file)
    return final_df


def test_datasets():
    for (
        model,
        params,
        match_tracking,
        epsilon,
        dataset_name,
        X_,
        y,
        fold,
    ) in experiment_loader_minimum():
        try:
            perform_test(
                model, params, match_tracking, epsilon, dataset_name, X_, y, fold
            )
        except Exception as E:
            print(str(model.__name__), dataset_name, E)


if __name__ == "__main__":
    num_workers = 8
    results = perform_tests(parquet_file="hull_art.parquet", num_workers=num_workers)
    # test_datasets()
