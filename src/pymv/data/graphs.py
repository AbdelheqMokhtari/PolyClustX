"""Convert graph objects and edge lists to sample-aligned sparse adjacency."""

import numpy as np
from scipy import sparse


def graph_adjacency(graph, sample_ids):
    """Accept a NetworkX-compatible graph; node IDs must match sample IDs exactly.

    No NetworkX dependency is needed for conversion. Directed edges retain their
    direction; parallel weights are summed, self-loops counted once. Isolated
    nodes must be present, preventing silent sample-order assumptions.
    """
    ids = list(sample_ids)
    if set(graph.nodes) != set(ids):
        raise ValueError("Graph node IDs must match dataset sample_ids, including isolated nodes")
    positions = {node: i for i, node in enumerate(ids)}
    rows, cols, values = [], [], []
    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Graph edge weights must be finite and nonnegative")
        i, j = positions[source], positions[target]
        rows.append(i)
        cols.append(j)
        values.append(weight)
        if not graph.is_directed() and i != j:
            rows.append(j)
            cols.append(i)
            values.append(weight)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(ids), len(ids)))
