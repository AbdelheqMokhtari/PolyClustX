"""
Feature preprocessing utilities for pymv.

This module provides functions for preparing single-view and
multi-view feature matrices before they are fed to clustering
models.

Public API (intended for direct use by end-users)
-------------------------------------------------
- :func:`get_propagated_features` — full per-view pipeline:
  normalize → (optionally) propagate through graph.
- :func:`preprocess_features` — centering / scaling across one
  or more feature views.

Internal helpers (used by the public API; usable but rarely needed)
-------------------------------------------------------------------
- :func:`preprocess_graph` — builds the row-normalized
  propagation matrix  S = D⁻¹(A + βI).
- :func:`normalize_features` — TF-IDF or L2 normalization on a
  single feature matrix.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize

# Graph preprocessing


def preprocess_graph(adj: Union[np.ndarray, sp.spmatrix], beta: float = 1.0):
    r"""Build the row-normalized propagation matrix.

    .. note:: **Internal helper** — called by
       :func:`get_propagated_features`.  You normally do not need
       to call this directly.

    Computes

    .. math::
        S = D^{-1}(A + \beta\,I)

    where *D* is the diagonal degree matrix of :math:`A + \beta I`.

    Parameters
    ----------
    adj : np.ndarray or scipy.sparse matrix of shape (n, n)
        Adjacency matrix (binary or weighted).  Converted to
        CSR sparse format internally if dense.

    beta : float, default=1.0
        Self-loop weight.  A value of 0 means no self-loops are
        added; 1.0 is the standard choice.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n, n)
        Row-normalized propagation matrix *S*.  Each row sums
        to 1 (or 0 for isolated nodes after adding self-loops,
        which is extremely rare when ``beta > 0``).

    Examples
    --------
    >>> import numpy as np
    >>> from pymv.preprocessing import preprocess_graph
    >>> A = np.array([[0, 1, 0],
    ...               [1, 0, 1],
    ...               [0, 1, 0]])
    >>> S = preprocess_graph(A, beta=1.0)
    >>> S.toarray().round(3)
    array([[0.5  , 0.5  , 0.   ],
           [0.333, 0.333, 0.333],
           [0.   , 0.5  , 0.5  ]])
    """

    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # Add self-loops
    adj = adj + beta * sp.eye(adj.shape[0])

    rowsum = np.array(adj.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0

    r_mat_inv = sp.diags(r_inv)

    S = r_mat_inv.dot(adj)

    return S


# Feature normalization

def normalize_features(
    features: Union[np.ndarray, sp.spmatrix],
    tf_idf: bool = False
):
    r"""Normalize a single feature matrix with L2 or TF-IDF + L2.

    .. note:: **Internal helper** — called by
       :func:`get_propagated_features`.  You normally do not need
       to call this directly unless you want fine-grained control
       over the normalization of an individual view.

    When ``tf_idf=False`` (default), each row is L2-normalized:

    .. math::
        \mathbf{x}_i \leftarrow
        \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}

    When ``tf_idf=True``, a TF-IDF transformation is applied first
    (via :class:`~sklearn.feature_extraction.text.TfidfTransformer`
    with ``norm='l2'``), which inherently includes L2 normalization.

    Parameters
    ----------
    features : np.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
        Raw feature matrix for **one** view.

    tf_idf : bool, default=False
        If ``True``, apply TF-IDF weighting before L2 normalization.
        Useful when the feature matrix contains raw term-frequency
        counts (e.g. bag-of-words).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Dense, row-normalized feature matrix.

    See Also
    --------
    get_propagated_features : High-level function that calls this
        internally then optionally propagates through a graph.

    Examples
    --------
    >>> import numpy as np
    >>> from pymv.preprocessing import normalize_features
    >>> X = np.array([[3.0, 4.0], [0.0, 1.0]])
    >>> normalize_features(X).round(3)
    array([[0.6, 0.8],
           [0. , 1. ]])
    """

    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else:
        features = normalize(features, norm="l2")

    if not isinstance(features, np.ndarray):
        features = features.toarray()

    return features


# Multi-view preprocessing

def preprocess_features(
    features: Union[np.ndarray, sp.spmatrix, List[Union[np.ndarray, sp.spmatrix]]],
    center: bool = False,
    scale: bool = False
):
    """Apply optional centering and/or scaling to feature views.

    **Public API** — use this when you want to standardize one or
    more feature matrices *before* passing them to a model (or
    independently of any model).

    Each view is processed with a **separate**
    :class:`~sklearn.preprocessing.StandardScaler`, so centering
    and scaling statistics are computed per-view.

    Parameters
    ----------
    features : np.ndarray, scipy.sparse matrix, or list thereof
        A single feature matrix or a list of feature matrices.
        If a single matrix is passed, a single matrix is returned
        (not wrapped in a list).

    center : bool, default=False
        Subtract the column mean from each feature.

        .. warning::
           Centering destroys sparsity.  If a sparse matrix is
           passed and ``center=True``, centering is **silently
           skipped** for that view to avoid a dense conversion.
           Convert to dense first if centering is required.

    scale : bool, default=False
        Divide each feature by its standard deviation (unit
        variance scaling).

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Processed features.  Returns the same type that was
        passed in: a single array if a single array was given,
        a list if a list was given.

    Examples
    --------
    Single view:

    >>> import numpy as np
    >>> from pymv.preprocessing import preprocess_features
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> preprocess_features(X, center=True).round(3)
    array([[-1.225, -1.225],
           [ 0.   ,  0.   ],
           [ 1.225,  1.225]])

    Multiple views:

    >>> X1 = np.random.randn(100, 10)
    >>> X2 = np.random.randn(100, 20)
    >>> Xs = preprocess_features([X1, X2], center=True, scale=True)
    >>> len(Xs)
    2

    See Also
    --------
    get_propagated_features : Full per-view pipeline (normalize →
        graph propagation) that should be called **before** this
        function if graph structure is available.
    """

    single_input = not isinstance(features, list)

    if single_input:
        features = [features]

    if not (center or scale):
        return features

    processed = []

    for X in features:

        local_center = center

        if sp.issparse(X) and center:
            # Avoid breaking sparsity
            local_center = False

        scaler = StandardScaler(
            with_mean=local_center,
            with_std=scale
        )

        X_proc = scaler.fit_transform(X)

        processed.append(X_proc)

    return processed[0] if single_input else processed


# Graph feature propagation

def get_propagated_features(
    features: Union[np.ndarray, sp.spmatrix],
    adj: Union[np.ndarray, sp.spmatrix, None] = None,
    beta: float = 1.0,
    tf_idf: bool = False,
    center: bool = False,
    scale: bool = False,
):
    r"""Center/scale, normalize, and (optionally) propagate features.

    **Public API** — this is the recommended entry point for
    preparing a **single view** before clustering.

    The pipeline is:

    1. **Center / scale** the feature matrix column-wise via
       :func:`preprocess_features` (only when requested).
       Column-wise operations must come first so that the
       subsequent row normalization sees standardized columns.
    2. **Normalize** the feature matrix via
       :func:`normalize_features` (L2 or TF-IDF + L2).
    3. **Propagate** through the graph (if an adjacency matrix is
       provided):

       .. math::
           H = S\,X, \quad S = D^{-1}(A + \beta\,I)

       If ``adj`` is ``None``, propagation is skipped and the
       normalized features are returned directly.

    .. note::
       **Why this order?**  Centering and scaling are *column-wise*
       statistics that align feature distributions across samples.
       Row normalization (L2) then projects each sample onto the
       unit sphere.  Reversing the two (normalizing first, then
       centering) would destroy the unit-norm property and produce
       geometrically inconsistent representations.

    Parameters
    ----------
    features : np.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
        Raw feature matrix for **one** view.

    adj : np.ndarray, scipy.sparse matrix, or None, default=None
        Adjacency matrix of shape ``(n_samples, n_samples)``.
        Pass ``None`` when there is no graph structure for this
        view (the normalization step is still applied).

    beta : float, default=1.0
        Self-loop weight used by :func:`preprocess_graph`.

    tf_idf : bool, default=False
        If ``True``, apply TF-IDF + L2 normalization instead of
        plain L2 normalization.  See :func:`normalize_features`.

    center : bool, default=False
        Subtract the column mean from each feature **before**
        row normalization.  See :func:`preprocess_features`.

    scale : bool, default=False
        Divide each feature by its column standard deviation
        **before** row normalization.  See :func:`preprocess_features`.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Dense feature matrix — optionally centered/scaled,
        normalized, and (if ``adj`` was provided) propagated.

    See Also
    --------
    preprocess_features : Column-wise centering / scaling.
    normalize_features : Internal — normalization only.
    preprocess_graph : Internal — builds the propagation matrix.

    Examples
    --------
    Without graph (pure normalization):

    >>> import numpy as np
    >>> from pymv.preprocessing import get_propagated_features
    >>> X = np.array([[3.0, 4.0], [0.0, 1.0]])
    >>> get_propagated_features(X).round(3)
    array([[0.6, 0.8],
           [0. , 1. ]])

    With graph propagation:

    >>> A = np.array([[0, 1], [1, 0]])
    >>> H = get_propagated_features(X, adj=A, beta=1.0)
    >>> H.shape
    (2, 2)
    """

    # Step 1: column-wise centering / scaling (before row normalization)
    if center or scale:
        features = preprocess_features(features, center=center, scale=scale)

    # Step 2: row-wise L2 (or TF-IDF + L2) normalization
    X = normalize_features(features, tf_idf)

    # Step 3: graph propagation (if adjacency is provided)
    if adj is None:
        return X

    S = preprocess_graph(adj, beta)

    H = S.dot(X)

    if sp.issparse(H):
        H = H.toarray()

    return H
