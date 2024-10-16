import pickle
from pathlib import Path
from sklearn.datasets import make_blobs
import numpy as np
from artlib import ART1, ART2A, BayesianART, DualVigilanceART, EllipsoidART, FuzzyART, GaussianART, HypersphereART, QuadraticNeuronART


def model_factory(model_class, params):
    """
    A factory function to initialize models, handling special cases.
    """
    if model_class.__name__ == "DualVigilanceART":
        # For DualVigilanceART, initialize with a base ART model
        base_art = FuzzyART(params["rho"], params["alpha"], params["beta"])
        return model_class(base_art, params["rho_lower_bound"])

    # Default case for other models
    return model_class(**params)


def cluster_and_store_results():
    # Generate blob data for clustering
    data, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)
    print("Data has shape:", data.shape)

    # Define the models and their parameters in a list of tuples
    models_with_params = [
        # (ART1, {"rho": 0.7, "beta": 1.0, "L": 1.0}),
        (ART2A, {"rho": 0.2, "alpha": 0.0, "beta": 1.0}),
        (BayesianART, {"rho": 7e-5, "cov_init": np.array([[1e-4, 0.0], [0.0, 1e-4]])}),
        (DualVigilanceART, {"rho": 0.85, "alpha": 0.8, "beta": 1.0, "rho_lower_bound": 0.78}),
        (EllipsoidART, {"rho": 0.01, "alpha": 0.0, "beta": 1.0, "r_hat": 0.65, "mu": 0.8}),
        (FuzzyART, {"rho": 0.5, "alpha": 0.0, "beta": 1.0}),
        (GaussianART, {"rho": 0.05, "sigma_init": np.array([0.5, 0.5])}),
        (HypersphereART, {"rho": 0.5, "alpha": 0.0, "beta": 1.0, "r_hat": 0.8}),
        (QuadraticNeuronART, {"rho": 0.9, "s_init": 0.9, "lr_b": 0.1, "lr_w": 0.1, "lr_s": 0.1}),
    ]

    results = {}

    for model_class, params in models_with_params:
        # Instantiate the model
        cls = model_factory(model_class, params)
        model_name = model_class.__name__

        # Prepare data
        X = cls.prepare_data(data)
        print(f"Prepared data for {model_name} has shape:", X.shape)

        # Fit the model and predict clusters
        labels = cls.fit_predict(X)

        # Store the labels and params in a dictionary keyed by the model name
        results[model_name] = {"params": params, "labels": labels}
        print(f"{cls.n_clusters} clusters found for {model_name}")


    # Save the results to a pickle file
    current_file_path = Path(__file__).resolve().parent.parent
    output_file = current_file_path / "unit_tests" / "cluster_results_snapshot.pkl"

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the results to the pickle file
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print("Results saved to cluster_results.pkl")

if __name__ == "__main__":
    cluster_and_store_results()
