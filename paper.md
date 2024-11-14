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
range of Adaptive Resonance Theory (ART) algorithms. **artlib** currently supports eight
elementary ART models and 11 compound ART models

[comment]: <> (, including Fuzzy ART)

[comment]: <> ([@carpenter1991fuzzy], Hypersphere ART [@anagnostopoulos2000hypersphere], Ellipsoid ART)

[comment]: <> ([@anagnostopoulos2001a; @anagnostopoulos2001b], Gaussian ART)

[comment]: <> ([@williamson1996gaussian], Bayesian ART [@vigdor2007bayesian], Quadratic Neuron)

[comment]: <> (ART [@su2001application; @su2005new], ART1 [@carpenter1987massively], ART2)

[comment]: <> ([@carpenter1987art; @carpenter1991art], ARTMAP [@carpenter1991artmap], Simplified)

[comment]: <> (ARTMAP [@gotarredona1998adaptive], SMART [@bartfai1994hierarchical], TopoART)

[comment]: <> ([@tscherepanow2010topoart], Dual Vigilance ART [@da2019dual], CVIART [@da2022icvi],)

[comment]: <> (BARTMAP [@xu2011bartmap; @xu2012biclustering], Fusion ART [@tan2007intelligence],)

[comment]: <> (FALCON [@tan2004falcon], and TD-FALCON [@tan2008integrating]. )
These models can be
applied to tasks such as unsupervised clustering, supervised classification, regression,
and reinforcement learning [@da2019survey]. This library provides an extensible and
modular framework where users can integrate custom models or extend current
implementations, allowing for experimentation with existing and novel machine learning
techniques.

In addition to the diverse set of ART models, **artlib** offers implementations of
visualization methods for various cluster geometries, along with pre-processing
techniques such as Visual Assessment of Tendency (VAT) [@bezdek2002vat], data
normalization, and complement coding.

## Elementary Models Provided

1. ART1 [@carpenter1987massively]: ART1 was the first ART clustering algorithm to be
   developed. It clusters binary vectors using a similarity metric based on the Hamming
   distance.

2. ART2 [@carpenter1987art; @carpenter1991art]: ART2 was the first attempt to extend
   ART1 to the domain of continuous-valued data. ART2-A was developed shortly after and
   improved the algorithmic complexity while retaining the properties of the original
   ART2 implementation. ART2 more closely resembles a multi-layer perceptron network as
   it uses an adaptive weight vector in the activation layer and a Heaviside function
   for the resonance layer. ART2 is widely considered to not work and is not
   recommended for use. It is included here for historical purposes.

3. Fuzzy ART [@carpenter1991fuzzy]: is the most cited and arguably most widely used
   ART variant at this time. Fuzzy ART is a hyper-box based clustering method,
   capable of clustering continuous-valued data. Data is pre-processed into zero-volume
   hyper-boxes through the process of complement coding before being used to
   initialize or expand the volume of a cluster hyper-box. In the fast-learning regime,
   Fuzzy ART suffers no catastrophic forgetting. It is exceptionally fast and
   explainable.

4. Hyperpshere ART [@anagnostopoulos2000hypersphere]: Hypersphere ART was designed
   to succeed Fuzzy ART with a more efficient internal knowledge representation.
   Categories are hyperpspheres and require less internal memory however computational
   complexity is increased relative to Fuzzy ART.

5. Ellipsoid ART[@anagnostopoulos2001a; @anagnostopoulos2001b]: Ellipsoid ART is a
   generalization of Hyperpshere ART which permits ellipsoids with arbitrary
   rotation. Ellipsoid ART is highly order dependent as the second sample added
   to any cluster sets the axes orientations.

6. Guassian ART [@williamson1996gaussian]: clusters data in Gaussian Distributions
   (Hyper-ellipsoids) and is similar to Bayesian ART but differs in that the
   hyper-ellipsoid always have their principal axes square to the coordinate frame.
   It is also faster than Bayesian ART.

7. Bayesian ART [@vigdor2007bayesian]: clusters data in Bayesian Distributions
   (Hyper-ellipsoids) and is similar to Gaussian ART but differs in that it allows
   arbitrary rotation of the hyper-ellipsoid.

8. Quadratic Neuron ART [@su2001application; @su2005new]: QN-ART utilizes a weight
   vector and a quadratic term to create clusters in a hyper-ellipsoid structure. It
   is superficially similar to ART2-A but is more sophisticated in that neurons also
   learn a bias and quadratic activation term.

## Compound Models Provided

1. ARTMAP [@carpenter1991artmap]: ARTMAP uses two ART modules to separately cluster
   two parallel data streams (A and B). An inter-ART module regulates the clustering
   such that clusters in the `module_A` maintain a many-to-one mapping with the
   clusters in the `module_B` by using a match-tracking function. When the data stream
   are independent and dependent variable for the A and B side respectively, ARTMAP
   learns a functional mapping the describes their relationship. ARTMAP can
   therefore be used for both classification and regression tasks. However, ARTMAP
   does not perform as well as Fusion ART for regression tasks where data is not
   monotonic.

2. Simple ARTMAP [@gotarredona1998adaptive]: Simple ARTMAP (or Simplified ARTMAP)
   was developed to streamline the ARTMAP algorith for classification task. As most
   classification problems provide discrete labels, it is possible to replace the
   B-side of the ARTMAP algorithm with the class labels directly. Simple ARTMAP does
   this and creates a mapping from B-side class labels to A-side cluster labels. The
   many-to-one mapping property is preserved, but the learning process is
   significantly faster and less computationally intensive. Simple ARTMAP is the
   recommended model for classification tasks.

3. Fusion ART [@tan2007intelligence]: Fusion ART allows for the fusion of
   multi-channel data by leveraging a discrete ART module for each channel.
   Activation for each category is calculated as a weighted sum of all channel
   activations and resonance only occurs if all channels are simultaneously resonant.
   This allows Fusion ART to learn mappings across all channels. Fusion ART works
   exceptionally well for regression problems and is the recommended model for this
   task.

4. FALCON [@tan2004falcon]: The Reactive FALCON model is a special case of Fusion
   ART designed for reinforcement learning. A Fusion ART model is used for learning
   the mapping between state, action, and reward channels. Special functions are
   implemented for selecting the best action and for predicting reward values.

5. TD-FALCON [@tan2008integrating]: Temporal Difference FALCON is an extension of
   Reactive FALCON which utilizes the SARSA method for temporal difference
   reinforcement learning. TD-FALCON is the recommended model for reinforcement
   learning tasks.

6. Dual Vigilance ART [@da2019dual]: Dual Vigilance ART utilizes an elementary ART
   module with a second, less restrictive vigilance parameter. Clusters are formed
   using the typical process for the underlying art module unless no resonant
   category is found. When this occurs, the less-restrictive vigilance parameter is
   used to determine if a near-resonant category can be found. If one can be found,
   a new cluster is formed, and the near-resonant category label is copied to the new
   cluster. If neither resonant nor near resonant categories can be found, a new
   cluster and new category label are both created. In this way, Dual Vigilance ART
   is capable of finding arbitrarily shaped structures as composites of the
   underlying ART geometry (i.e Hyper-ellipsoids or Hyper-boxes).

7. SMART [@bartfai1994hierarchical]: Self-consistent Modular ART is a special case
   of Deep ARTMAP and an extension of ARTMAP. SMART permits n-many modules (in
   contrast to ARTMAPS 2-modules) which passes the same sample vector to each
   module. Each module has a vigilance parameter monotonically increasing with depth.
   This permits SMART to create self-consistent hierarchies of clusters through a
   divisive clustering approach. The number of modules and granularity at each module
   are both parameterizable.

8. Deep ARTMAP: Deep ARTMAP is a novel contribution of this library. It generalizes
   SMART by permitting each module to accept some function $f^i(x)$. This
   generalization allows the user to find hierarchical relationships between an
   abritrary number of functional transformations of some input data. When only two
   modules are used and $f^1(x) = target$ and $f^2(x) = x$ Deep ARTMAP reduces to
   standard ARTMAP.

9. Topo ART [@tscherepanow2010topoart]: Topo ART is a topological clustering
   approach which uses an elementary ART module to learn a distributed cluster graph
   where samples can belong to multiple distinct clusters. The co-resonant clusters are
   tracked using an adjacency matrix which describes the cluster relationships of
   the entire model.

10. CVI ART [@da2022icvi]: CVI ART maps the clusters of an elementary ART module to
    category label identified by the optimal Cluster Validity Index (CV). This
    mapping occurs similarly to simplified ARTMAP. An iterative implementation (iCVI
    ART) is also provided, however it is currently only compatible with Fuzzy ART.

11. BARTMAP [@xu2011bartmap; @xu2012biclustering]: BARTMAP is a Biclustering
    algorithm based loosely on ARTMAP. The algorithm accepts two instantiated
    elementary ART modules `module_A` and `module_B` which cluster the rows (samples)
    and columns (features) respectively. The features are clustered independently,
    but the samples are clustered by considering samples already within a row
    cluster as well as the candidate sample and enforcing a minimum Pearson correlation
    within the subset of features belonging to at least one of the feature clusters.

# Statement of Need

The Adaptive Resonance Library (**artlib**) is essential for researchers, developers,
and educators interested in adaptive neural networks, specifically ART algorithms.
While deep learning dominates machine learning, ART models offer unique advantages
in incremental and real-time learning environments due to their ability to learn new
data without forgetting previously learned information.

Currently, no comprehensive Python library implements a variety of ART models in an
open-source, modular, and extensible manner. **artlib** fills this gap by offering a
range of ART implementations that integrate seamlessly with machine learning workflows,
including scikit-learn's `Pipeline` and `GridSearchCV` [@scikit-learn]. The library is
designed for ease of use and high performance, leveraging Python's scientific stack
(NumPy [@harris2020array], SciPy [@2020SciPy-NMeth], and scikit-learn [@scikit-learn])
for fast numerical computation.

The modular design of **artlib** enables users to create novel compound ART models,
such as Dual Vigilance Fusion ART [@da2019dual; @tan2007intelligence] or
Quadratic Neuron SMART [@su2001application; @su2005new; @bartfai1994hierarchical].
This flexibility offers powerful experimental and time-saving benefits, allowing
researchers and practitioners to evaluate models on diverse datasets efficiently.

Additionally, the library serves as a valuable educational tool, providing
well-documented code and familiar APIs to support hands-on experimentation with ART
models. It is ideal for academic courses or personal projects in artificial
intelligence and machine learning, making **artlib** a versatile resource.

**artlib** is actively maintained and designed for future extension, allowing users
to create new ART models, adjust parameters for specific applications, and explore ART's
potential for novel research problems. Its integration with popular Python libraries
ensures its adaptability to current machine learning challenges.

# Comparison to Existing Implementations

While there are several open-source repositories that provide
python implementations of specific ART models [@birkjohann2023artpython;
@aiopenlab2023art; @dilekman2022artificial; @artpy2022; @dixit2020adaptive;
@inyoot2021art; @valixandra2021adaptive; @wan2022art2; @ray2023artpy], they lack
modularity and are limited in scope, often implementing just one or two models. For
instance, MATLAB-based ART toolboxes [@mathworks_art1s; @mathworks_fuzzyart_fuzzyartmap;
@mathworks_topoart; @mathworks_art_fuzzyart_artmap] provide implementations of
Fuzzy ART, TopoART, ART1, and ARTMAP models, but they lack the flexibility and
modularity required for broader experimentation. The most significant existing ART
implementation exists in julia and provides just five models
[@Petrenko_AdaptiveResonance_jl_A_Julia_2022] but, like the previously listed
MATLAB-based toolboxes, it is not easily accessible to Python-based work flows and
lacks a modular design.

These existing implementations of ART models may provide standalone versions of
individual models, but they are often not designed to integrate seamlessly with modern
Python libraries such as scikit-learn, NumPy, and SciPy. As a result, researchers and
developers working in Python-based environments face challenges when trying to
incorporate ART models into their machine learning pipelines.

In contrast, **artlib** offers a comprehensive and modular collection of ART models,
including both elementary and compound ART architectures. It is designed for
interoperability with popular Python tools, enabling users to easily integrate ART
models into machine learning workflows, optimize models using scikit-learn's
`GridSearchCV`, and preprocess data using standard libraries. Further, **artlib**
provides users the flexibility to construct their own compound ART modules (those
art modules deriving properties from other, elementary modules) which
may or may not exist in published literature. **artlib** also provides a template
in the source code to encourage users to develop and experiment with their own custom
ART algorithms. This flexibility and integration make **artlib** a powerful resource
for both research and practical applications.

# Adaptive Resonance Theory (ART)

ART is a class of neural networks known for solving the stability-plasticity dilemma,
making it particularly effective for classification, clustering, and incremental
learning tasks [@grossberg1976a; @grossberg1976a; @Grossberg1980HowDA;
@grossberg2013adaptive; @da2019survey]. ART models are designed to dynamically learn
and adapt to new patterns without catastrophic forgetting, making them ideal for
real-time systems requiring continuous learning.

Over the years, dozens of ART variations have been published [@da2019survey],
extending the applicability of ART to nearly all learning regimes, including
reinforcement learning [@tan2004falcon; @tan2008integrating], hierarchical
clustering [@bartfai1994hierarchical], topological clustering
[@tscherepanow2010topoart], and biclustering [@xu2011bartmap; @xu2012biclustering].
These numerous models provide an ART-based solution for most machine learning use cases.
However, the rapid pace of bespoke model development, coupled with the challenges
students face in learning ART's foundational principles, has contributed to a
scarcity of open-source, user-friendly implementations for most ART variants.

The ability of ART to preserve previously learned patterns while learning new data in
real-time has made it a powerful tool in domains such as robotics, medical diagnosis,
and adaptive control systems. **artlib** aims to extend the application of these models
in modern machine learning pipelines, offering a unique and approachable toolkit for
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
