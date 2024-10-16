
# AdaptiveResonanceLib

Welcome to AdaptiveResonanceLib, a comprehensive and modular Python library for Adaptive Resonance Theory (ART) algorithms. Based on scikit-learn, our library offers a wide range of ART models designed for both researchers and practitioners in the field of machine learning and neural networks. Whether you're working on classification, clustering, or pattern recognition, AdaptiveResonanceLib provides the tools you need to implement ART algorithms efficiently and effectively.

<!-- START available_models -->
## Available Models

AdaptiveResonanceLib includes implementations for the following ART models:

- #### Elementary Clustering
    - ART1
    - ART2
    - Bayesian ART
    - Gaussian ART
    - Hypersphere ART
    - Ellipsoidal ART
    - Fuzzy ART
    - Quadratic Neuron ART
    - Dual Vigilance ART
- #### Metric Informed 
    - CVI ART  
    - iCVI Fuzzy ART  
- #### Topological
    - Topo ART
- #### Classification
    - Simple ARTMAP
- #### Regression
    - ARTMAP
    - Fusion ART
- #### Hierarchical
    - DeepARTMAP
    - SMART
- #### Data Fusion
    - Fusion ART
- #### Reinforcement Learning
    - FALCON
    - TD-FALCON
- #### Biclustering
    - Biclustering ARTMAP
  
<!-- END available_models -->

<!-- START comparison_of_elementary_models -->
## Comparison of Elementary Models

[comment]: <> (![Comparison of Elementary Images]&#40;https://github.com/NiklasMelton/AdaptiveResonanceLib/raw/main/docs/_static/comparison_of_elementary_methods.jpg?raw=true"&#41;)
![Comparison of Elementary Images](https://github.com/NiklasMelton/AdaptiveResonanceLib/raw/main/img/comparison_of_elementary_methods.jpg?raw=true")
<!-- END comparison_of_elementary_models -->

<!-- START installation -->
## Installation

To install AdaptiveResonanceLib, simply use pip:

[comment]: <> (```bash)

[comment]: <> (pip install AdaptiveResonanceLib)

[comment]: <> (```)

```bash
pip install artlib
```

Ensure you have Python 3.9 or newer installed.
<!-- END installation -->

<!-- START quick-start -->
## Quick Start

Here's a quick example of how to use AdaptiveResonanceLib with the Fuzzy ART model:

```python
from artlib import FuzzyART
import numpy as np

# Your dataset
train_X = np.array([...])
test_X = np.array([...])

# Initialize the Fuzzy ART model
model = FuzzyART(rho=0.7, alpha = 0.0, beta=1.0)

# Fit the model
model.fit(train_X)

# Predict new data points
predictions = model.predict(test_X)
```

<!-- END quick-start -->

<!-- START documentation -->
## Documentation

For more detailed documentation, including the full list of parameters for each model, visit our [documentation page](https://adaptiveresonancelib.readthedocs.io/en/latest/index.html).
<!-- END documentation -->

<!-- START examples -->
## Examples

For examples of how to use each model in AdaptiveResonanceLib, check out the `/examples` directory in our repository.
<!-- END examples -->

<!-- START contributing -->
## Contributing

We welcome contributions to AdaptiveResonanceLib! If you have suggestions for improvements, or if you'd like to add more ART models, please see our `CONTRIBUTING.md` file for guidelines on how to contribute.

You can also join our [Discord server](https://discord.gg/E465HBwEuN) and participate directly in the discussion.
<!-- END contributing -->

<!-- START license -->
## License

AdaptiveResonanceLib is open source and available under the MIT license. See the `LICENSE` file for more info.
<!-- END license -->

<!-- START contact -->
## Contact

For questions and support, please open an issue in the GitHub issue tracker or message us on our [Discord server](https://discord.gg/E465HBwEuN). We'll do our best to assist you.

Happy Modeling with AdaptiveResonanceLib!
<!-- END contact -->
