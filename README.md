
# AdaptiveResonanceLib

Welcome to AdaptiveResonanceLib, a comprehensive and modular Python library for Adaptive Resonance Theory (ART) algorithms. Based on scikit-learn, our library offers a wide range of ART models designed for both researchers and practitioners in the field of machine learning and neural networks. Whether you're working on classification, clustering, or pattern recognition, AdaptiveResonanceLib provides the tools you need to implement ART algorithms efficiently and effectively.

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
    - Gram ART
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
  
- #### Biclustering
    - Biclustering ARTMAP

## Comparison of Elementary Models
![Comparison of Elementary Images](./img/comparison_of_elementary_methods.jpg?raw=true")

## Installation

To install AdaptiveResonanceLib, simply use pip:

[comment]: <> (```bash)

[comment]: <> (pip install AdaptiveResonanceLib)

[comment]: <> (```)

```bash
pip install artlib
```

Ensure you have Python 3.9 or newer installed.

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

Replace `params` with the parameters appropriate for your use case.

## Documentation

For more detailed documentation, including the full list of parameters for each model, visit our [documentation page](https://github.com/NiklasMelton/AdaptiveResonanceLib).

## Examples

For examples of how to use each model in AdaptiveResonanceLib, check out the `/examples` directory in our repository.

## Contributing

We welcome contributions to AdaptiveResonanceLib! If you have suggestions for improvements, or if you'd like to add more ART models, please see our `CONTRIBUTING.md` file for guidelines on how to contribute.

You can also join our [Discord server](https://discord.gg/E465HBwEuN) and participate directly in the discussion.

## License

AdaptiveResonanceLib is open source and available under the MIT license. See the `LICENSE` file for more info.

## Contact

For questions and support, please open an issue in the GitHub issue tracker or message us on our [Discord server](https://discord.gg/E465HBwEuN). We'll do our best to assist you.

Happy Modeling with AdaptiveResonanceLib!
