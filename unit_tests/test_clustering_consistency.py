import pickle
import pytest
import numpy as np
from pathlib import Path
from artlib import ART1, ART2A, BayesianART, DualVigilanceART, EllipsoidART, FuzzyART, GaussianART, HypersphereART, \
    QuadraticNeuronART


# Factory function to initialize models, handling special cases like DualVigilanceART
def model_factory(model_class, params):
    if model_class.__name__ == "DualVigilanceART":
        base_art = FuzzyART(params["rho"], params["alpha"], params["beta"])
        return model_class(base_art, params["rho_lower_bound"])
    return model_class(**params)


# Load the clustering results from the pickle file
@pytest.fixture(scope="module")
def cluster_results():
    # Define the path to the pickle file
    current_file_path = Path(__file__).resolve().parent.parent
    pickle_file = current_file_path / "unit_tests" / "cluster_results_snapshot.pkl"

    # Load the results
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


# Dictionary of model classes to map model names to classes
model_classes = {
    "ART2A": ART2A,
    "BayesianART": BayesianART,
    "DualVigilanceART": DualVigilanceART,
    "EllipsoidART": EllipsoidART,
    "FuzzyART": FuzzyART,
    "GaussianART": GaussianART,
    "HypersphereART": HypersphereART,
    "QuadraticNeuronART": QuadraticNeuronART,
}


@pytest.mark.parametrize("model_name", model_classes.keys())
def test_clustering_consistency(model_name, cluster_results):
    # Get the stored params and labels for the model
    stored_data = cluster_results[model_name]
    stored_params = stored_data["params"]
    stored_labels = stored_data["labels"]

    # Instantiate the model using the stored parameters
    model_class = model_classes[model_name]
    model_instance = model_factory(model_class, stored_params)

    # Generate blob data (same data used when saving the pickle file)
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.50, random_state=0, shuffle=False)

    # Prepare the data
    X = model_instance.prepare_data(data)

    # Fit the model and predict the clusters
    predicted_labels = model_instance.fit_predict(X)

    # Check that the predicted labels match the stored labels
    assert np.array_equal(predicted_labels, stored_labels), f"Labels for {model_name} do not match!"

