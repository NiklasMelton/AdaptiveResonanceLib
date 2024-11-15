"""Cluster validity indices are metrics used to evaluate the quality of clustering
results. These indices help to determine the optimal number of clusters and assess the
performance of clustering algorithms by measuring the compactness and separation of the
clusters. Common cluster validity indices include the Silhouette score, Davies-Bouldin
index, and Dunn index. These indices play an important role in unsupervised learning
tasks where true labels are not available for evaluation.

This module implements CVI-driven ART modules which utilize the CVI to inform
clustering; often resulting in objectively superior results.

`Cluster validity indices
<https://en.wikipedia.org/wiki/Cluster_analysis#Cluster_validity>`_

"""
