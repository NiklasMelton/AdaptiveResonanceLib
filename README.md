
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

# Your dataset
train_X = np.array([...]) # shape (n_samples, n_features)
test_X = np.array([...])

# Initialize the Fuzzy ART model
model = FuzzyART(rho=0.7, alpha = 0.0, beta=1.0)

# Prepare Data
train_X_prep = model.prepare_data(train_X)
test_X_prep = model.prepare_data(test_X)

# Fit the model
model.fit(train_X_prep)

# Predict data labels
predictions = model.predict(test_X_prep)
```

### Fitting a Classification Model with SimpleARTMAP

```python
from artlib import GaussianART, SimpleARTMAP
import numpy as np

# Your dataset
train_X = np.array([...]) # shape (n_samples, n_features)
train_y = np.array([...]) # shape (n_samples, ), must be integers
test_X = np.array([...])

# Initialize the Gaussian ART model
sigma_init = np.array([0.5]*train_X.shape[1]) # variance estimate for each feature
module_a = GaussianART(rho=0.0, sigma_init=sigma_init)

# Initialize the SimpleARTMAP model
model = SimpleARTMAP(module_a=module_a)

# Prepare Data
train_X_prep = model.prepare_data(train_X)
test_X_prep = model.prepare_data(test_X)

# Fit the model
model.fit(train_X_prep, train_y)

# Predict data labels
predictions = model.predict(test_X_prep)
```

### Fitting a Regression Model with FusionART

```python
from artlib import FuzzyART, HypersphereART, FusionART
import numpy as np

# Your dataset
train_X = np.array([...]) # shape (n_samples, n_features_X)
train_y = np.array([...]) # shape (n_samples, n_features_y)
test_X = np.array([...])

# Initialize the Fuzzy ART model
module_x = FuzzyART(rho=0.0, alpha = 0.0, beta=1.0)

# Initialize the Hypersphere ART model
r_hat = 0.5*np.sqrt(train_X.shape[1]) # no restriction on hyperpshere size
module_y = HypersphereART(rho=0.0, alpha = 0.0, beta=1.0, r_hat=r_hat)

# Initialize the FusionARTMAP model
gamma_values = [0.5, 0.5] # eqaul weight to both channels
channel_dims = [
  2*train_X.shape[1], # fuzzy ART complement codes data so channel dim is 2*n_features
  train_y.shape[1]
]
model = FusionART(
  modules=[module_x, module_y],
  gamma_values=gamma_values,
  channel_dims=channel_dims
)

# Prepare Data
train_Xy = model.join_channel_data(channel_data=[train_X, train_y])
train_Xy_prep = model.prepare_data(train_Xy)
test_Xy = model.join_channel_data(channel_data=[train_X], skip_channels=[1])
test_Xy_prep = model.prepare_data(test_Xy)

# Fit the model
model.fit(train_X_prep, train_y)

# Predict y-channel values
pred_y = model.predict_regression(test_Xy_prep, target_channels=[1])
```

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
