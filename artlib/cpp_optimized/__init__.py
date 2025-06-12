"""This module implements several common ARTMAP models in native C++, dramatically
improving runtime performance of these modules.

By sacrificing some modularity, we are able to create compiled versions of the fit,
predict, and partial_fit methods in order to accelerate the model performance. The
remaining class methods are implemented in python while the model memory remains stored
in a python class. Indeed, these classes are all derived version of the
:class:`~artlib.supervised.SimpleARTMAP.SimpleARTMAP` class and thus retain all of its
core functionality. This allows the user to interact with these classes as if they were
standard Python implementations.

Due to their accelerated performance, these classes should be a user's default choice if
the desired compound module is implemented here.

While the current selection of accelerated models is limited, this module will continue
to expand in future versions.

"""
