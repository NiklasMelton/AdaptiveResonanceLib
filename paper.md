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
    orcid: 0000-0001-9625-7086
    affiliation: "1, 2"
  - name: Dustin Tanksley
    orcid: 0000-0002-1677-0304
    affiliation: "1, 2"
  - name: Donald C. Wunsch II
    orcid: 0000-0002-9726-9051
    affiliation: "1, 2, 3"

affiliations:
 - name: Missouri University of Science and Technology, USA
   index: 1
- name: National Science Foundation (NSF), USA
   index: 2
 - name: Kummer Institute Center for Artificial Intelligence and Autonomous Systems
   (KICAIAS), USA
   index: 3

date: 18 October 2024
bibliography: references.bib
---

# Summary

The Adaptive Resonance Library (**artlib**) is a Python library that implements a wide
range of Adaptive Resonance Theory (ART) algorithms. ART is a class of neural networks
that is known for addressing the stability-plasticity dilemma, making it particularly
effective for classification, clustering, and incremental learning tasks
[@grossberg1976a; @grossberg1976a; @Grossberg1980HowDA; @grossberg2013adaptive;
@da2019survey]. ART models are designed to dynamically learn and adapt to new patterns
without catastrophic forgetting, making them suitable for real-time systems that require
continuous learning.

**artlib** supports a variety of ART-based models including but not limited to
Fuzzy ART [@carpenter1991fuzzy], Hyperpshere ART [@anagnostopoulos2000hypersphere],
Gaussian ART [@williamson1996gaussian], ARTMAP [@carpenter1991artmap],
Fusion ART [@tan2007intelligence], and FALCON [@tan2004falcon]. These models
can be applied to tasks like unsupervised clustering, supervised classification or
regression, and reinforcement learning [@da2019survey]. This library provides an
extensible and modular framework where users can integrate custom models or extend
current implementations, which opens up opportunities for experimenting with existing
and novel machine learning techniques.

In addition to the zoo of ART models, we provide
implementations of visualization methods for the various cluster geometries as well as
pre-processing techniques such as Visual Assessment of Tendency (VAT)[@bezdek2002vat],
data normalization, and complement coding.

# Statement of need

The Adaptive Resonance Library (**artlib**) is essential for researchers, developers,
and educators interested in adaptive neural networks, specifically ART algorithms. While
the field of machine learning has been dominated by architectures like deep learning,
ART models offer unique advantages in incremental and real-time learning environments
due to their ability to learn new data without forgetting previously learned
information.

Currently, no comprehensive Python library exists that implements a variety of ART
models in an open-source, modular, and extensible manner. **artlib* fills this gap by
providing a range of ART implementations that are easy to integrate into various machine
learning workflows such as scikit-learn Pipelines and GridSearchCV [@scikit-learn].
The library is designed for ease of use and performance, offering users fast numerical
computation via Python’s scientific stack (NumPy[@harris2020array], SciPy
[@2020SciPy-NMeth], and scikit-learn [@scikit-learn]).

In particular, the modular nature of this library allows for the creation of
never-before published compound ART models such as Dual Vigilance
Fusion ART [@da2019dual; @tan2007intelligence] or Quadratic Neuron SMART
[@su2001application; @su2005new; @bartfai1994hierarchical]. Such flexibility offers
powerful experimental and time-saving advantages to researchers and practitioners when
evaluating models on their data.

Additionally, the library is a valuable educational tool for students learning about
neural networks and adaptive systems. With well-documented code and clear APIs, it
supports hands-on experimentation with ART models, making it an excellent resource for
academic courses or personal projects in artificial intelligence and machine learning.

Furthermore, **artlib** is actively maintained and designed to support further
extensions, allowing users to add new ART models, adjust parameters for specific use
cases, and leverage ART for novel research problems. Its integration with popular Python
libraries ensures that **artlib** remains adaptable and applicable to current machine
learning challenges.

# Acknowledgements
This research was supported by the National Science Foundation (NSF) under Award
Number 2420248. The project titled EAGER: Solving Representation Learning and
Catastrophic Forgetting with Adaptive Resonance Theory provided essential funding for
the completion of this work.

We would also like to thank Gail Carpenter and Stephen Grossberg for their
feedback regarding this project and their immense contribution to machine learning by
pioneering Adaptive Resonance Theory.

# References
