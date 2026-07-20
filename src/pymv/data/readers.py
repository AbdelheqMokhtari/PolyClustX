"""
Readers for the ``.mat`` layouts used by the dataset catalog.

Every reader has the same signature::

    reader(data: dict, **kwargs) -> (Xs, As, labels)

where ``data`` is the dictionary returned by :func:`scipy.io.loadmat`.
The catalog (:mod:`pymv.data.registry`) pairs each dataset with the
reader that understands its file layout:

- :func:`read_standard`      — views in a cell array ``X``, labels in ``y``.
- :func:`read_feature_views` — one matrix per named key, no graphs.
- :func:`read_graph`         — shared features + one adjacency per view.
- :func:`read_multimodal`    — ``X_image`` / ``X_text`` + shared graph ``A``.
- :func:`read_wiki`, :func:`read_amazon_photos` — dataset-specific layouts.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def _to_dense(arr):
    """Convert to a dense numpy array regardless of input type."""
    if sp.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def _to_sparse(arr) -> sp.csr_matrix:
    """Convert to CSR float64 regardless of input type."""
    if sp.issparse(arr):
        return arr.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(arr, dtype=np.float64))


def _labels(arr) -> np.ndarray:
    return _to_dense(arr).reshape(-1)


def read_standard(data, n_views: int):
    """Views stored in a cell array ``X``, labels in ``y``.

    Different ``.mat`` files use either a ``(1, n_views)`` or a
    ``(n_views, 1)`` cell layout, so the cell array is raveled to a
    flat sequence before indexing.
    """
    cells = data["X"].ravel()
    Xs = [_to_dense(cells[v]).astype(np.float64) for v in range(n_views)]
    return Xs, [None] * n_views, _labels(data["y"])


def read_feature_views(data, keys: tuple, label_key: str):
    """One feature matrix per named key; no graphs."""
    Xs = [_to_dense(data[k]).astype(np.float64) for k in keys]
    return Xs, [None] * len(Xs), _labels(data[label_key])


def read_graph(data, feature_key: str, adj_keys: tuple, label_key: str):
    """A shared feature matrix with one adjacency matrix per view."""
    X = _to_dense(data[feature_key]).astype(np.float64)
    As = [_to_dense(data[k]).astype(np.float64) for k in adj_keys]
    Xs = [X for _ in As]
    return Xs, As, _labels(data[label_key])


def read_multimodal(data):
    """``X_image`` / ``X_text`` feature views sharing one graph ``A``.

    The adjacency is kept SPARSE — these multimodal graphs can have
    tens of thousands of nodes, so densifying A (n x n) would need
    tens of GiB.  The downstream preprocessing accepts sparse input.
    """
    X_image = _to_dense(data["X_image"]).astype(np.float64)
    X_text = _to_dense(data["X_text"]).astype(np.float64)
    A = _to_sparse(data["A"])
    return [X_image, X_text], [A, A], _labels(data["labels"])


def read_wiki(data):
    """Wiki layout: raw + log-scaled features, link graph + kNN graph."""
    X = _to_dense(data["fea"]).astype(np.float64)
    Xs = [X, np.log2(1 + X)]
    As = [data["W"], kneighbors_graph(X, 5, metric="cosine")]
    return Xs, As, _labels(data["gnd"])


def read_amazon_photos(data):
    """Amazon photos layout: raw + similarity features, shared graph."""
    X = data["features"]
    A = _to_dense(data["adj"]).astype(np.float64)
    return [X, X @ X.T], [A, A], _labels(data["label"])
