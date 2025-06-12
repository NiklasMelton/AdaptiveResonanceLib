"""This module implements several common ARTMAP models in native C++, dramatically
improving runtime performance of these modules.

By sacrificing some modularity, we are able to create compiled version of the fit and
predict methods and accelerate the model performance. Aside from these functions, the
model memory remains stored in a python class, the user to interact with these classes
as if they were standard Python implementations. While the current selection of
accelerated models is limited, this module will continue to expand in future versions.

"""
