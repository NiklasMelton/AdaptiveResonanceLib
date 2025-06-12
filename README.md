
# AdaptiveResonanceLib

Welcome to AdaptiveResonanceLib, a comprehensive and modular Python library for Adaptive Resonance Theory (ART) algorithms. Based on scikit-learn, our library offers a wide range of ART models designed for both researchers and practitioners in the field of machine learning and neural networks. Whether you're working on classification, clustering, or pattern recognition, AdaptiveResonanceLib provides the tools you need to implement ART algorithms efficiently and effectively.

[//]: # (# What is ART)

[//]: # (Adaptive Resonance Theory &#40;ART&#41; is a theory of how the brain learns as well as a )

[//]: # (family of algorithms based on the concept of pattern resonance. Originially )

[//]: # (Formulated by Steven Grossburg and Gail Carpenter in the 1970s, ART posits that )

[//]: # (the brain learns to identify patterns using top-down activations and bottom-up )

[//]: # (expectations working cohesively to identify familiar and novel stimuli. )

[//]: # ()
[//]: # (When presented with a pattern, an ART-like network activates prior memories relative)

[//]: # (to their similarity with the pattern. The most active prior memory is now a )

[//]: # (candidate for assignment, but it must first pass the resonance &#40;vigilance check&#41;. If )

[//]: # (the input pattern is fully familiar &#40;entirely contained in the active memory&#41;, )

[//]: # (nothing else needs to be done. However, if the input patter contains novel )

[//]: # (information, we must check if the prior memory is capable of incorporating the new )

[//]: # (information by performing a vigilance test. In this way, we limit the scope of a single )

[//]: # (prior memory. If the pattern passes the vigilance test for the prior memory, we say )

[//]: # (that the pattern and memory are resonant and we update the prior memory to )

[//]: # (incorporate the new pattern information. If the prior memory fails the vigilance )

[//]: # (test, we suppress that memory and check the next highest activated memory. If no )

[//]: # (prior memories pass the vigilance test, we create a new memory initialized using the )

[//]: # (input pattern. When clustering, the index of the winning memory is assigned as the )

[//]: # (cluster label of the input pattern.)

[//]: # ()
[//]: # (This process allows ART to solve the stability plasticity dillema with an explicit )

[//]: # (memory limiting &#40;vigilance&#41; parameter $ ρ$. By increasing $ ρ$, a practitioner )

[//]: # (can decrease the allowable size of individual memories, thereby creating more )

[//]: # (categories with more specificity. At the extreme ends, setting $ ρ=1.0$ will force )

[//]: # (ART models to memorize unique data samples, while setting $ ρ=0.0$ will allow ART )

[//]: # (models to group all data samples into a single category.)

<!-- START what is ART -->
## Adaptive Resonance Theory (ART)

**Adaptive Resonance Theory (ART)** is both
1. A neuroscientific theory of how the brain balances *plasticity* (learning new
   information) with *stability* (retaining what it already knows), and
2. A family of machine‑learning algorithms that operationalise this idea for
   clustering, classification, continual‑learning, and other tasks.

First proposed by Stephen Grossberg and Gail Carpenter in the mid‑1970s , ART models treat learning as an **interactive search** between *bottom‑up evidence* and *top‑down expectations*:

1. **Activation.**
   A new input pattern activates stored memories (categories) in proportion to their similarity to the input.

2. **Candidate selection.**
   The most active memory (call it *J*) is tentatively chosen to represent the input.

3. **Vigilance check (resonance test).**
   The match between the input and memory *J* is compared to a user‑chosen threshold \(ρ\) (the **vigilance parameter**).
   * If the match ≥ \(ρ\) → **Resonance.** The memory and input are deemed compatible; *J* is updated to incorporate the new information.
   * If the match < \(ρ\) → **Mismatch‑reset.** Memory *J* is temporarily inhibited, and the next best candidate is tested.
   * If no memory passes the test → a **new category** is created directly from the input.

4. **Output.**
   In clustering mode, the index of the resonant (or newly created) memory is returned as the cluster label.

### Vigilance

ρ sets an explicit upper bound on how dissimilar two inputs can be while still
ending up in the *same* category:

| Vigilance \(ρ\)             | Practical effect |
|-----------------------------|------------------|
| \(  ρ = 0 \)                | All inputs merge into a single, broad category |
| Moderate (\( 0 <  ρ < 1 \)) | Finer granularity as \(ρ\) increases |
| \(  ρ = 1 \)                | Every distinct input forms its own category (memorisation) |

This single knob lets practitioners trade off *specificity* against *generality* without retraining from scratch.

### Notable Variants

| Variant                          | Input type                                | Task                      | Trait                                                                                          |
|----------------------------------|-------------------------------------------|---------------------------|------------------------------------------------------------------------------------------------|
| **ART 1**                        | Binary                                    | Unsupervised clustering   | Original model                                                                                 |
| **Fuzzy ART**                    | Real‑valued \([0,1]\)                     | Unsupervised clustering   | Uses fuzzy AND operator for analog inputs, resulting in rectagular categories                  |
| **ARTMAP**         | Paired inputs \((X, y)\)                  | Supervised classification | Two ART modules linked by an associative map field                                             |
| **Gaussian ART** | Real‑valued                               | Clustering                | Replace rectangular category fields with Gaussian ones for smoother decision boundaries        |
| **FALCON**      | Paired inputs \((State, Action, Reward)\) | Reinforcement Learning    | Uses three ART modules to create a dynamic SARSA grid for solving reinforcement learning tasks |

All variants share the same resonance‑test backbone, so you can grasp one and quickly extend to the others.

### Strengths and Things to Watch

* **Online / incremental learning** – adapts one sample at a time without replay.
* **Explicit category prototypes** – easy to inspect and interpret.
* **Built‑in catastrophic‑forgetting control** via \(ρ\).
* **Parameter sensitivity** – vigilance (and, in many variants, the learning rate \(\beta\)) must be tuned to your data.
* **Order dependence** – the sequence of inputs can affect category formation; shuffling your training data is recommended for unbiased results.
<!-- END what is ART -->


<!-- START available_models -->
## Available Models

AdaptiveResonanceLib includes implementations for the following ART models:

- #### Elementary Clustering
    - [ART1](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.ART1.ART1)
    - [ART2](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.ART2A.ART2A)
    - [Bayesian ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.BayesianART.BayesianART)
    - [Gaussian ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.GaussianART.GaussianART)
    - [Hypersphere ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.HypersphereART.HypersphereART)
    - [Ellipsoid ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.EllipsoidART.EllipsoidART)
    - [Fuzzy ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.FuzzyART.FuzzyART)
    - [Quadratic Neuron ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.QuadraticNeuronART.QuadraticNeuronART)
    - [Binary Fuzzy ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.BinaryFuzzyART.BinaryFuzzyART)
- #### Metric Informed
    - [CVI ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cvi.html#artlib.cvi.CVIART.CVIART)
    - [iCVI Fuzzy ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cvi.html#artlib.cvi.iCVIFuzzyART.iCVIFuzzyART)
- #### Topological
    - [Topo ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.topological.html#artlib.topological.TopoART.TopoART)
    - [Dual Vigilance ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.topological.html#artlib.topological.DualVigilanceART.DualVigilanceART)
- #### Classification
    - [Simple ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.supervised.html#artlib.supervised.SimpleARTMAP.SimpleARTMAP)
- #### Regression
    - [ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.supervised.html#artlib.supervised.ARTMAP.ARTMAP)
    - [Fusion ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.fusion.html#artlib.fusion.FusionART.FusionART)
- #### Hierarchical
    - [DeepARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.hierarchical.html#artlib.hierarchical.DeepARTMAP.DeepARTMAP)
    - [SMART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.hierarchical.html#artlib.hierarchical.SMART.SMART)
- #### Data Fusion
    - [Fusion ART](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.fusion.html#artlib.fusion.FusionART.FusionART)
- #### Reinforcement Learning
    - [FALCON](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.reinforcement.html#artlib.reinforcement.FALCON.FALCON)
    - [TD-FALCON](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.reinforcement.html#artlib.reinforcement.TD_FALCON.TD_FALCON)
- #### Biclustering
    - [Biclustering ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.biclustering.html#artlib.biclustering.BARTMAP.BARTMAP)
- #### C++ Accelerated
    - [Fuzzy ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.FuzzyARTMAP.FuzzyARTMAP)
    - [Hypersphere ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.HypersphereARTMAP.HypersphereARTMAP)
    - [Gaussian ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.GaussianARTMAP.GaussianARTMAP)
    - [Binary Fuzzy ARTMAP](https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.BinaryFuzzyARTMAP.BinaryFuzzyARTMAP)
    -

<!-- END available_models -->

<!-- START comparison_of_elementary_models -->
## Comparison of Elementary Models

[comment]: <> (![Comparison of Elementary Images]&#40;https://github.com/NiklasMelton/AdaptiveResonanceLib/raw/main/docs/_static/comparison_of_elementary_methods.jpg?raw=true"&#41;)
![Comparison of Elementary Images](https://raw.githubusercontent.com/NiklasMelton/AdaptiveResonanceLib/main/docs/_static/img/comparison_of_elementary_methods.jpg?raw=true")
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

<!-- START cpp_optimized -->

[//]: # (## C++ Optimization)

[//]: # ()
[//]: # ( While all classes use NumPy and SciPy for mathematical operations to accelerate )

[//]: # ( execution times, Some classes have Numba optimized activation and )

[//]: # ( vigilance functions to further accelerate them. These include:)

[//]: # ()
[//]: # (- [ART1]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.ART1&#41;)

[//]: # (- [Fuzzy ART]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.FuzzyART&#41;)

[//]: # (- [Binary Fuzzy ART]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.BinaryFuzzyART&#41;)

[//]: # ()
[//]: # (For classification tasks specifically, several [Simple ARTMAP]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.html#artlib.SimpleARTMAP&#41; variations have been )

[//]: # (fully written in c++ with accompanying Python wrappers. These include:)

[//]: # ()
[//]: # (- [Fuzzy ARTMAP]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.FuzzyARTMAP.FuzzyARTMAP&#41;)

[//]: # (- [Hypersphere ARTMAP]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.HypersphereARTMAP.HypersphereARTMAP&#41;)

[//]: # (- [Gaussian ARTMAP]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.GaussianARTMAP.GaussianARTMAP&#41;)

[//]: # (- [Binary Fuzzy ARTMAP]&#40;https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.BinaryFuzzyARTMAP.BinaryFuzzyARTMAP&#41;)

[//]: # ()
[//]: # (These ARTMAP modules trade modularity for run time by executing all fit and predict )

[//]: # (operations in c++. However these models use their results to populate a their )

[//]: # (equivelant python classes allowing them to otherwise behave as the python-exclusive )

[//]: # (versions in all other ways. )

## C++ Optimizations

Most **ARTlib** classes rely on NumPy / SciPy for linear-algebra routines, but several go further:

| Level | Accelerated components | Implementations |
|-------|------------------------|-----------------|
| **Python (Numba JIT)** | Activation & vigilance kernels | [ART1][], [Fuzzy ART][], [Binary Fuzzy ART][] |
| **Native C++ (Pybind11)** | Entire **fit** / **predict** pipelines | [Fuzzy ARTMAP][], [Hypersphere ARTMAP][], [Gaussian ARTMAP][], [Binary Fuzzy ARTMAP][] |

[ART1]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.ART1.ART1
[Fuzzy ART]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.FuzzyART.FuzzyART
[Binary Fuzzy ART]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.elementary.html#artlib.elementary.BinaryFuzzyART.BinaryFuzzyART

[Fuzzy ARTMAP]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.FuzzyARTMAP.FuzzyARTMAP
[Hypersphere ARTMAP]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.HypersphereARTMAP.HypersphereARTMAP
[Gaussian ARTMAP]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.GaussianARTMAP.GaussianARTMAP
[Binary Fuzzy ARTMAP]: https://adaptiveresonancelib.readthedocs.io/en/latest/artlib.cpp_optimized.html#artlib.cpp_optimized.BinaryFuzzyARTMAP.BinaryFuzzyARTMAP

### How the C++ variants work

1. **End-to-end native execution** – Training and inference run entirely in C++, eliminating Python-level overhead.
2. **State hand-off** – After fitting, the C++ routine exports cluster weights and metadata back to the corresponding pure-Python class. You can therefore:
   • inspect attributes (`weights_`, `categories_`, …)
   • serialize with `pickle`
   • plug them into any downstream ARTlib or scikit-learn pipeline **exactly as you would with the Python-only models**.
3. **Trade-off** – The C++ versions sacrifice some modularity (you cannot swap out
   internal ART components) in exchange for significantly shorter run-times.

### C++ Acceleration Quick reference


| Class                 | Acceleration method       | Primary purpose |
|-----------------------|---------------------------|-----------------|
| [ART1]                | Numba JIT kernels         | Clustering      |
| [Fuzzy ART]           | Numba JIT kernels         | Clustering      |
| [Binary Fuzzy ART]    | Numba JIT kernels         | Clustering      |
|                       |                           |                 |
| [Fuzzy ARTMAP]        | Full C++ implementation   | Classification  |
| [Hypersphere ARTMAP]  | Full C++ implementation   | Classification  |
| [Gaussian ARTMAP]     | Full C++ implementation   | Classification  |
| [Binary Fuzzy ARTMAP] | Full C++ implementation   | Classification  |

### Example Usage
```python
from artlib import FuzzyARTMAP
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
n_dim = 28*28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, n_dim)) # flatten images
X_test = X_test.reshape((-1, n_dim))

# Initialize the Fuzzy ART model
model = FuzzyARTMAP(rho=0.7, alpha = 0.0, beta=1.0)

# (Optional) Tell the model the data limits for normalization
lower_bounds = np.array([0.]*n_dim)
upper_bounds = np.array([255.]*n_dim)
model.set_data_bounds(lower_bounds, upper_bounds)

# Prepare Data
train_X_prep = model.prepare_data(X_train)
test_X_prep = model.prepare_data(X_test)

# Fit the model
model.fit(train_X_prep, y_train)

# Predict data labels
predictions = model.predict(test_X_prep)
```

### Timing Comparison
The below figures demonstrate the acceleration seen by the C++ ARTMAP variants in
comparison to their baseline Python versions for a 1000 sample subset of the MNIST
dataset.

<p align="center">
  <img src="https://raw.githubusercontent.com/NiklasMelton/AdaptiveResonanceLib/main/docs/_static/img/mnist_art_fit_times.jpg?raw=true"
       alt="MNIST ART fit times" width="45%" />
  <img src="https://raw.githubusercontent.
com/NiklasMelton/AdaptiveResonanceLib/main/docs/_static/img/mnist_art_predict_times.jpg?raw=true"
       alt="MNIST ART predict times" width="45%" />
</p>

From the above plots, it becomes apparent that the C++ variants are superior in their
runtime performance and should be the default choice of practitioners wishing to work
with these specific compound models.

While the current selection remains limited, future releases will expand the native C++
implementation as user demand for them increases.

<!-- END cpp_optimized -->

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

Melton, N. (2024). AdaptiveResonanceLib (Version 0.1.4)
<!-- END citation -->
