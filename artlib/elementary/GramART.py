import numpy as np
from typing import Optional, Iterable, Tuple, Union, List
from matplotlib.axes import Axes
from artlib.common.BaseART import BaseART
from artlib.common.utils import normalize, compliment_code, l1norm, fuzzy_and
from collections import defaultdict, deque, defaultdict
from copy import deepcopy

class ProtoNode:
    def __init__(self):
        self.distribution = defaultdict(lambda: 0)
        self.counter = defaultdict(lambda: 0)

TreeType = Tuple[Iterable, np.ndarray]
ProtoTreeType = List[ProtoNode]

def topological_sort(symbols, matrix):
    N = len(symbols)
    # Step 1: Create the adjacency list
    adj_list = defaultdict(list)
    in_degree = [0] * N

    for x in range(N):
        for y in range(N):
            if matrix[x][y] == 1:
                adj_list[y].append(x)
                in_degree[x] += 1

    # Step 2: Initialize the queue with nodes having in-degree 0
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(symbols[node])

        for successor in adj_list[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # If result contains all symbols, return the sorted order
    if len(result) == N:
        return result
    else:
        raise ValueError("The graph contains a cycle; topological sort is not possible.")


class GramART(BaseART):

    def __init__(self, rho: float):
        params ={
            "rho": rho
        }
        super().__init__(params)

    @staticmethod
    def prepare_data(x: TreeType) -> List:
        """
        prepare data for clustering

        Parameters:
        - X: data set

        Returns:
            normalized and compliment coded data
        """
        return topological_sort(x[0], x[1])

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        pass

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        pass

    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert isinstance(params["rho"], float)

    def category_choice(self, i: List, w: ProtoTreeType, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        tree_intersection = sum([p.distribution[s] for s,p in zip(i, w)])
        cache = {"tree_intersection": tree_intersection}
        return tree_intersection/len(w), cache

    def match_criterion(self, i: List, w: ProtoTreeType, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        """
        get the match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion, cache used for later processing

        """
        return cache["tree_intersection"]/len(i), cache

    def match_criterion_bin(self, i: List, w: ProtoTreeType, params: dict, cache: Optional[dict] = None) -> tuple[bool, dict]:
        """
        get the binary match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion binary, cache used for later processing

        """
        M, cache = self.match_criterion(i, w, params, cache)
        return M >= params["rho"], cache

    def update(self, i: List, w: ProtoTreeType, params: dict, cache: Optional[dict] = None) -> ProtoTreeType:
        """
        get the updated cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            updated cluster weight, cache used for later processing

        """
        w_new = deepcopy(w)
        for j in range(len(i)):
            if j >= len(w):
                w_new.append(ProtoNode())
            for s in w_new[j].distribution.keys():
                if s == i[j]:
                    d = 1
                else:
                    d = 0
                w_new[j].distribution[s] = (w_new[j].distribution[s]*w_new[j].counter[s] + d)/ (w_new[j].counter[s] + 1)
                w_new[j].counter[s] += 1

        return w_new

    def new_weight(self, i: List, params: dict) -> ProtoTreeType:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        w_new = []
        for s in i:
            p = ProtoNode()
            p.distribution[s] = 1.0
            p.counter[s] = 1.0
            w_new.append(p)

        return w_new

