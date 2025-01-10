
# AdaptiveResonanceLib

Welcome to AdaptiveResonanceLib, a comprehensive and modular Python library for Adaptive Resonance Theory (ART) algorithms. Based on scikit-learn, our library offers a wide range of ART models designed for both researchers and practitioners in the field of machine learning and neural networks. Whether you're working on classification, clustering, or pattern recognition, AdaptiveResonanceLib provides the tools you need to implement ART algorithms efficiently and effectively.

<!-- START available_models -->
## Available Models

AdaptiveResonanceLib includes implementations for the following ART models:

- #### Elementary Clustering
    - [ART1](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.ART1)
    - [ART2](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.ART2A)
    - [Bayesian ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.BayesianART)
    - [Gaussian ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.GaussianART)
    - [Hypersphere ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.HypersphereART)
    - [Ellipsoid ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.EllipsoidART)
    - [Fuzzy ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.FuzzyART)
    - [Quadratic Neuron ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.QuadraticNeuronART)
- #### Metric Informed
    - [CVI ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.CVIART)
    - [iCVI Fuzzy ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.iCVIFuzzyART)
- #### Topological
    - [Topo ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.TopoART)
    - [Dual Vigilance ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.DualVigilanceART)
- #### Classification
    - [Simple ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.SimpleARTMAP)
- #### Regression
    - [ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.ARTMAP)
    - [Fusion ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.FusionART)
- #### Hierarchical
    - [DeepARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.DeepARTMAP)
    - [SMART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.SMART)
- #### Data Fusion
    - [Fusion ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.FusionART)
- #### Reinforcement Learning
    - [FALCON](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.FALCON)
    - [TD-FALCON](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.TD_FALCON)
- #### Biclustering
    - [Biclustering ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.biclustering.html#artlib.biclustering.BARTMAP.BARTMAP)

<!-- END available_models -->

<!-- START comparison_of_elementary_models -->
## Comparison of Elementary Models

[comment]: <> (![Comparison of Elementary Images]&#40;https://github.com/NiklasMelton/AdaptiveResonanceLib/raw/main/docs/_static/comparison_of_elementary_methods.jpg?raw=true"&#41;)
![Comparison of Elementary Images](https://github.com/NiklasMelton/AdaptiveResonanceLib/raw/main/img/comparison_of_elementary_methods.jpg?raw=true")
<!-- END comparison_of_elementary_models -->

<!-- START installation -->
## Installation

To install AdaptiveResonanceLib, simply use pip:

```bash
pip install artlib
```

Or to install directly from the most recent source:

```bash
pip install git+https://github.com/NiklasMelton/AdaptiveResonanceLib.git@develop
```

Ensure you have Python 3.9 or newer installed.
<!-- END installation -->

<!-- START quick-start -->
## Quick Start

Here are some quick examples to get you started with AdaptiveResonanceLib:

### Clustering Data with the Fuzzy ART model

```python
from artlib import FuzzyART
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
n_dim = 28*28
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape((-1, n_dim)) # flatten images
X_test = X_test.reshape((-1, n_dim))

# Initialize the Fuzzy ART model
model = FuzzyART(rho=0.7, alpha = 0.0, beta=1.0)

# (Optional) Tell the model the data limits for normalization
lower_bounds = np.array([0.]*n_dim)
upper_bounds = np.array([255.]*n_dim)
model.set_data_bounds(lower_bounds, upper_bounds)

# Prepare Data
train_X_prep = model.prepare_data(X_train)
test_X_prep = model.prepare_data(X_test)

# Fit the model
model.fit(train_X_prep)

# Predict data labels
predictions = model.predict(test_X_prep)
```

### Fitting a Classification Model with SimpleARTMAP

```python
from artlib import GaussianART, SimpleARTMAP
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
n_dim = 28*28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, n_dim)) # flatten images
X_test = X_test.reshape((-1, n_dim))

# Initialize the Gaussian ART model
sigma_init = np.array([0.5]*X_train.shape[1]) # variance estimate for each feature
module_a = GaussianART(rho=0.0, sigma_init=sigma_init)

# (Optional) Tell the model the data limits for normalization
lower_bounds = np.array([0.]*n_dim)
upper_bounds = np.array([255.]*n_dim)
module_a.set_data_bounds(lower_bounds, upper_bounds)

# Initialize the SimpleARTMAP model
model = SimpleARTMAP(module_a=module_a)

# Prepare Data
train_X_prep = model.prepare_data(X_train)
test_X_prep = model.prepare_data(X_test)

# Fit the model
model.fit(train_X_prep, y_train)

# Predict data labels
predictions = model.predict(test_X_prep)
```

### Fitting a Regression Model with FusionART

```python
from artlib import FuzzyART, HypersphereART, FusionART
import numpy as np

# Your dataset
X_train = np.array([...]) # shape (n_samples, n_features_X)
y_train = np.array([...]) # shape (n_samples, n_features_y)
test_X = np.array([...])

# Initialize the Fuzzy ART model
module_x = FuzzyART(rho=0.0, alpha = 0.0, beta=1.0)

# Initialize the Hypersphere ART model
r_hat = 0.5*np.sqrt(X_train.shape[1]) # no restriction on hyperpshere size
module_y = HypersphereART(rho=0.0, alpha = 0.0, beta=1.0, r_hat=r_hat)

# Initialize the FusionARTMAP model
gamma_values = [0.5, 0.5] # eqaul weight to both channels
channel_dims = [
  2*X_train.shape[1], # fuzzy ART complement codes data so channel dim is 2*n_features
  y_train.shape[1]
]
model = FusionART(
  modules=[module_x, module_y],
  gamma_values=gamma_values,
  channel_dims=channel_dims
)

# Prepare Data
train_Xy = model.join_channel_data(channel_data=[X_train, y_train])
train_Xy_prep = model.prepare_data(train_Xy)
test_Xy = model.join_channel_data(channel_data=[X_train], skip_channels=[1])
test_Xy_prep = model.prepare_data(test_Xy)

# Fit the model
model.fit(train_Xy_prep)

# Predict y-channel values and clip X values outside previously observed ranges
pred_y = model.predict_regression(test_Xy_prep, target_channels=[1], clip=True)
```

### Data Normalization

AdaptiveResonanceLib models require feature data to be normalized between 0.0
and 1.0 inclusively. This requires identifying the boundaries of the data space.

If the first batch of your training data is representative of the entire data space,
you dont need to do anything and artlib will identify the data bounds automatically.
However, this will often not be sufficient and the following work-arounds will be
needed:

Users can manually set the bounds using the following code snippet or similar:
```python
# Set the boundaries of your data for normalization
lower_bounds = np.array([0.]*n_features)
upper_bounds = np.array([1.]*n_features)
model.set_data_bounds(lower_bounds, upper_bounds)
```

Or users can present all batches of data to the model for automatic
boundary identification:
```python
# Find the boundaries of your data for normalization
all_data = [train_X, test_X]
_, _ = model.find_data_bounds(all_data)
```

If only the boundaries of your testing data are unknown, you can call
`model.predict()` with `clip=True` to clip testing data to the bounds seen during
training. Only use this if you understand what you are doing.

<!-- END quick-start -->

<!-- START documentation -->
## Documentation

For more detailed documentation, including the full list of parameters for each model, visit our [Read the Docs page](https://adaptiveresonancelib.readthedocs.io/en/latest/index.html).
<!-- END documentation -->

<!-- START examples -->
## Examples

For examples of how to use each model in AdaptiveResonanceLib, check out the [`/examples`](https://github.com/NiklasMelton/AdaptiveResonanceLib/tree/develop/examples) directory in our repository.
<!-- END examples -->

<!-- START contributing -->
## Contributing

We welcome contributions to AdaptiveResonanceLib! If you have suggestions for improvements, or if you'd like to add more ART models, please see our `CONTRIBUTING.md` file for guidelines on how to contribute.

You can also join our [Discord server](https://discord.gg/E465HBwEuN) and participate directly in the discussion.
<!-- END contributing -->

<!-- START license -->
## License

AdaptiveResonanceLib is open source and available under the MIT license. See the [`LICENSE`](https://github.com/NiklasMelton/AdaptiveResonanceLib/blob/develop/LICENSE) file for more info.
<!-- END license -->

<!-- START contact -->
## Contact

For questions and support, please open an issue in the GitHub issue tracker or message us on our [Discord server](https://discord.gg/E465HBwEuN). We'll do our best to assist you.

Happy Modeling with AdaptiveResonanceLib!
<!-- END contact -->

<!-- START citation -->
## Citing this Repository
If you use this project in your research, please cite it as:

Melton, N. (2024). AdaptiveResonanceLib (Version 0.1.3)
<!-- END citation -->
