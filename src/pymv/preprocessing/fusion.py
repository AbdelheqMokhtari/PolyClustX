"""Sparse-preserving early fusion of relations belonging to one view."""

import numpy as np
from scipy import sparse


def fuse_graphs(graphs, *, method="sum", weights=None):
    """Fuse same-order adjacencies by weighted sum or elementwise product.

    Sum weights default to one, not an average. Hadamard fusion does not accept
    weights. Returns CSR if any input is sparse, otherwise a dense array.
    """
    if method not in {"sum", "hadamard"}:
        raise ValueError("method must be 'sum' or 'hadamard'")
    if not graphs:
        raise ValueError("At least one graph is required for fusion")
    arrays = []
    use_sparse = any(sparse.issparse(g) for g in graphs)
    for graph in graphs:
        if graph is None:
            raise ValueError("Relations cannot contain None; omit missing relations")
        a = sparse.csr_matrix(graph, dtype=float) if use_sparse else np.asarray(graph, dtype=float)
        values = a.data if use_sparse else a
        if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Graphs must be finite, nonnegative square matrices")
        if arrays and a.shape != arrays[0].shape:
            raise ValueError("Relations must use the same shape and sample order")
        arrays.append(a)
    if method == "hadamard":
        if weights is not None:
            raise ValueError("Hadamard fusion does not accept weights")
        result = arrays[0].copy()
        for a in arrays[1:]:
            result = result.multiply(a) if use_sparse else result * a
    else:
        w = np.ones(len(arrays)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(arrays),) or not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
            raise ValueError("Provide one finite nonnegative weight per relation, at least one positive")
        result = arrays[0] * w[0]
        for a, weight in zip(arrays[1:], w[1:]):
            result = result + a * weight
    return result.tocsr() if use_sparse else result
