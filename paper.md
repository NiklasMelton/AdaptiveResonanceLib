---
title: 'Adaptive Resonance Lib: A Python package for Adaptive Resonance Theory (ART) models'
tags:
  - Python
  - clustering
  - classification
  - regression
  - reinforcement learning
  - machine learning
authors:
  - name: Niklas M. Melton
    corresponding: true
    orcid: 0000-0001-9625-7086
    affiliation: 1
  - name: Dustin Tanksley
    orcid: 0000-0002-1677-0304
    affiliation: 1
  - name: Donald C. Wunsch II
    orcid: 0000-0002-9726-9051
    affiliation: 1
affiliations:
 - name: Missouri University of Science and Technology, USA
   index: 1
   ror: 00scwqd12
date: 18 October 2024
bibliography: references.bib
---

# Summary

The Adaptive Resonance Library (**artlib**) is a Python library that implements a wide
range of Adaptive Resonance Theory (ART) algorithms. **artlib** supports eight elementary
ART modules and 11 compound ART modules, including but not limited to Fuzzy ART
[@carpenter1991fuzzy], Hypersphere ART [@anagnostopoulos2000hypersphere], Gaussian ART
[@williamson1996gaussian], ARTMAP [@carpenter1991artmap], Fusion ART
[@tan2007intelligence], and FALCON [@tan2004falcon]. These models can be applied to
tasks like unsupervised clustering, supervised classification, regression, and
reinforcement learning [@da2019survey]. This library provides an extensible and
modular framework where users can integrate custom models or extend current
implementations, allowing for experimentation with existing and novel machine learning
techniques.

In addition to the diverse ART models, **artlib** offers implementations of
visualization methods for various cluster geometries, along with pre-processing
techniques such as Visual Assessment of Tendency (VAT) [@bezdek2002vat], data
normalization, and complement coding.

# Statement of Need

The Adaptive Resonance Library (**artlib**) is essential for researchers, developers,
and educators interested in adaptive neural networks, specifically ART algorithms.
While deep learning dominates machine learning, ART models offer unique advantages
in incremental and real-time learning environments due to their ability to learn new
data without forgetting previously learned information.

Currently, no comprehensive Python library implements a variety of ART models in an
open-source, modular, and extensible manner. **artlib** fills this gap by offering a
range of ART implementations that integrate seamlessly with machine learning workflows,
including scikit-learn Pipelines and GridSearchCV [@scikit-learn]. The library is
designed for ease of use and high performance, leveraging Python's scientific stack
(NumPy [@harris2020array], SciPy [@2020SciPy-NMeth], and scikit-learn [@scikit-learn])
for fast numerical computation.

The modular design of **artlib** also enables users to create novel compound ART models,
such as Dual Vigilance Fusion ART [@da2019dual; @tan2007intelligence] or
Quadratic Neuron SMART [@su2001application; @su2005new; @bartfai1994hierarchical].
This flexibility offers powerful experimental and time-saving benefits, allowing
researchers and practitioners to evaluate models on diverse datasets efficiently.

Additionally, the library serves as a valuable educational tool, providing
well-documented code and clear APIs to support hands-on experimentation with ART
models. It is ideal for academic courses or personal projects in artificial
intelligence and machine learning, making **artlib** a versatile resource.

**artlib** is actively maintained and designed for future extension, allowing users
to add new ART models, adjust parameters for specific applications, and explore ART's
potential for novel research problems. Its integration with popular Python libraries
ensures its adaptability to current machine learning challenges.

# Comparison to Existing Implementations

While there are a few libraries and toolboxes that provide implementations of specific
ART models [@birkjohann2023artpython; @aiopenlab2023art; @dilekman2022artificial;
@artpy2022; @dixit2020adaptive; @inyoot2021art; @valixandra2021adaptive;
@wan2022art2; @ray2023artpy],
they tend to be limited
in scope or
platform-specific. For instance,
MATLAB-based ART toolboxes often provide implementations of Fuzzy ART and ARTMAP models,
but they lack the flexibility and modularity required for broader experimentation, and
they are not easily accessible to Python-based workflows.

Other existing implementations of ART models may provide standalone versions of
individual models, but they are often not designed to integrate seamlessly with modern
Python libraries such as scikit-learn, NumPy, and SciPy. As a result, researchers and
developers working in Python-based environments face challenges when trying to
incorporate ART models into their machine learning pipelines.

**artlib** fills this gap by offering a comprehensive and modular collection of ART
models, including both elementary and compound ART architectures. It is designed for
interoperability with popular Python tools, enabling users to easily integrate ART
models into machine learning workflows, optimize models using scikit-learn's
GridSearchCV, and preprocess data using standard libraries. This flexibility and
integration make **artlib** a powerful resource for both research and practical
applications.

# Adaptive Resonance Theory (ART)

ART is a class of neural networks known for solving the stability-plasticity dilemma,
making it particularly effective for classification, clustering, and incremental
learning tasks [@grossberg1976a; @grossberg1976a; @Grossberg1980HowDA;
@grossberg2013adaptive; @da2019survey]. ART models are designed to dynamically learn
and adapt to new patterns without catastrophic forgetting, making them ideal for
real-time systems requiring continuous learning.

The ability of ART to preserve previously learned patterns while learning new data
in real-time has made it a powerful tool in domains such as robotics, medical
diagnosis, and adaptive control systems. **artlib** aims to extend the application of
these models in modern machine learning pipelines, offering a unique toolkit for
leveraging ART's strengths.

# Acknowledgements
This research was supported by the National Science Foundation (NSF) under Award
Number 2420248. The project titled EAGER: Solving Representation Learning and
Catastrophic Forgetting with Adaptive Resonance Theory provided essential funding for
the completion of this work.

We would also like to thank Gail Carpenter and Stephen Grossberg for their
feedback regarding this project and their immense contribution to machine learning by
pioneering Adaptive Resonance Theory.

# References
