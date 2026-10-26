"""Warnings about backend choice for the optimized ARTMAP models."""


class BackendPerformanceWarning(UserWarning):
    """A backend that is likely much slower than another one for this model.

    Raised by the torch backend when it runs on CPU or on an Apple GPU (MPS), where
    the C++ backend was faster in every benchmark, or when a model trained on CUDA is
    small enough that the C++ backend is usually much faster. Silence it with::

        import warnings
        from artlib import BackendPerformanceWarning

        warnings.filterwarnings("ignore", category=BackendPerformanceWarning)

    """
