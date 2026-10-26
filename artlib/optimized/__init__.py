"""This module provides factories for creating optimized versions of common ARTMAP
modules. Currently, the supported ARTMAP variants and their available backends are as
follows:

FuzzyARTMAP:
    c++ (default)
    torch
    python

HypersphereARTMAP:
    c++ (default)
    torch
    python

GaussianARTMAP:
    c++ (default)
    torch
    python

BinaryFuzzyARTMAP:
    c++ (default)
    torch
    python

C++ is the fastest backend except for large models on an NVIDIA GPU, where
backend="torch" with device="cuda" can be faster. See "Choosing a backend" in the
README. The torch backend raises a BackendPerformanceWarning when another backend is
likely much faster.


Additionally, a c++ implementation of a rational fraction sorting algorithm
"fracsort" is provided.
"""
