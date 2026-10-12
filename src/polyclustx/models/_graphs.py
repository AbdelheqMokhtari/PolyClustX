"""Validated sparse graphs shared by feature-input graph estimators."""
import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def cosine_graph(x, neighbors):
    n = len(x)
    count = min(neighbors + 1, n)
    distances, indices = NearestNeighbors(n_neighbors=count, metric="cosine").fit(x).kneighbors(x)
    weights = np.maximum(1 - distances, 0)
    graph = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(n), count), indices.ravel())), shape=(n, n))
    graph.setdiag(0)
    graph.eliminate_zeros()
    return (graph + graph.T).tocsr()


def input_graphs(As, n, views):
    if As is None:
        return []
    if not isinstance(As, (list, tuple)) or len(As) != views:
        raise ValueError("As must have one adjacency (or None) per feature view")
    result = []
    for a in As:
        if a is None:
            continue
        a = sparse.csr_matrix(a, dtype=np.float64)
        if a.shape != (n, n) or not np.isfinite(a.data).all() or (a.data < 0).any():
            raise ValueError("Graphs must be finite, nonnegative, sample-aligned square matrices")
        result.append(((a + a.T) * .5).tocsr())
    return result


def normalize_graph(graph, self_loops=False):
    if self_loops:
        graph = graph + sparse.eye(graph.shape[0], format="csr")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inverse, where=degree > 0)
    return (sparse.diags(inverse) @ graph @ sparse.diags(inverse)).tocsr()


def torch_sparse(graph, device):
    coo = graph.tocoo()
    return torch.sparse_coo_tensor(np.stack([coo.row, coo.col]), coo.data,
                                   graph.shape, dtype=torch.float32, device=device).coalesce()
