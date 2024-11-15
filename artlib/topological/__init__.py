"""Topological clustering is a method of grouping data points based on their topological
structure, capturing the shape or connectivity of the data rather than relying solely on
distance measures. This approach is particularly useful when the data has a non-linear
structure or when traditional clustering algorithms fail to capture the intrinsic
geometry of the data. Topological clustering techniques, such as hierarchical clustering
and Mapper, are often used in fields like data analysis and computational topology.

The two modules herein provide contrasting advantages.
:class:`~artlib.topological.TopoART.TopoART` allows for the creation of an adjacency
matrix which can be useful when clusters overlap or are in close proximity.
:class:`~artlib.topological.DualVigilanceART.DualVigilanceART` allows for the abstract
merging of many smaller clusters and is well suited to problems where the clusters
take-on complex geometries where other clustering approaches would fail.

`Topological clustering <https://en.wikipedia.org/wiki/Topological_data_analysis>`_

"""
