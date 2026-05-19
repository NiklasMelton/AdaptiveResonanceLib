"""This module implements ARTMAP models with hybridized backends for optimized runtimes.

For the implemented models, you can select from torch, c++, and python for your
backends. Torch further enhances selection by allowing users to provide a device
selection of "cuda" or "cpu". The hybrid methods enable users to use different
combinations of backend/device at different stages of the model life. C++ can be used
for fitting while torch+gpu can be used for prediction.

"""
