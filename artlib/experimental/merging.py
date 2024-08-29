def find(parent, i):
    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]


def union(parent, rank, x, y):
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


def merge_objects(objects, can_merge):
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
