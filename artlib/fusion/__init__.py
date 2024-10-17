"""Data fusion is the process of integrating multiple data sources to produce more
consistent, accurate, and useful information than would be possible when using the
sources independently. It is used in a wide range of fields, including sensor networks,
image processing, and decision-making systems.

The ART module contained herein allows for the fusion of an arbitrary number of data
channels. This structure not only supports classification tasks but also enables it to
be used for regression on polytonic (as opposed to monotonic) problems. By leveraging
data from multiple channels, the module improves regression accuracy by combining
diverse information sources, making it particularly suited for complex problems where
single-channel approaches fall short.

This is the recommended module for such regression problems.

`Data fusion <https://en.wikipedia.org/wiki/Data_fusion>`_

"""
