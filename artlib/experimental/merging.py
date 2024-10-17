from typing import List, Callable


def find(parent: List[int], i: int) -> int:
    """
    Find the root of the set containing element i using path compression.

    Parameters
    ----------
    parent : list
        List representing the parent of each element.
    i : int
        The element to find the root of.

    Returns
    -------
    int
        The root of the set containing element i.

    """
    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]


def union(parent: List[int], rank: list[int], x: int, y: int):
    """
    Perform union of two sets containing elements x and y using union by rank.

    Parameters
    ----------
    parent : list
        List representing the parent of each element.
    rank : list
        List representing the rank (depth) of each tree.
    x : int
        The first element.
    y : int
        The second element.

    """
    root_x = find(parent, x)
    root_y = find(parent, y)

    if root_x != root_y:
        # Union by rank to keep the tree flat
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1


def merge_objects(objects: List, can_merge: Callable):
    """
    Merge objects into groups based on a merge condition function using Union-Find algorithm.

    Parameters
    ----------
    objects : list
        List of objects to be merged.
    can_merge : callable
        A function that takes two objects and returns True if they can be merged.

    Returns
    -------
    list of list
        A list of merged groups, where each group is a list of object indices.

    """
    # Initialize Union-Find structure
    n = len(objects)
    parent = list(range(n))
    rank = [0] * n

    # Check each pair of objects
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge(objects[i], objects[j]):
                union(parent, rank, i, j)

    # Collect merged groups by indices
    groups = {}
    for i in range(n):
        root = find(parent, i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())
