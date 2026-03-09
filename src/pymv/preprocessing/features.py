"""
Feature preprocessing utilities for Pymv.

This module contains functions for preparing multi-view feature
matrices before they are passed to clustering models.

It includes:

- Feature normalization
- Graph propagation
- Multi-view preprocessing
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from typing import Union, List

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer


# ------------------------------------------------------------------
# Graph preprocessing
# ------------------------------------------------------------------

def preprocess_graph(adj: Union[np.ndarray, sp.spmatrix], beta: float = 1.0):
    """
    Compute the linear propagation matrix S.

    S = D^{-1}(A + beta * I)

    Parameters
    ----------
    adj : ndarray or sparse matrix
        Adjacency matrix.

    beta : float
        Self-loop weight.

    Returns
    -------
    sparse matrix
        Row-normalized propagation matrix.
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


# ------------------------------------------------------------------
# Feature normalization
# ------------------------------------------------------------------

def normalize_features(
    features: Union[np.ndarray, sp.spmatrix],
    tf_idf: bool = False
):
    """
    Normalize feature vectors.

    Optionally applies TF-IDF transformation before L2 normalization.

    Parameters
    ----------
    features : ndarray or sparse matrix
        Feature matrix (n_samples, n_features).

    tf_idf : bool
        Whether to apply TF-IDF transformation.

    Returns
    -------
    ndarray
        Normalized dense feature matrix.
    """

    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else:
        features = normalize(features, norm="l2")

    if not isinstance(features, np.ndarray):
        features = features.toarray()

    return features


# ------------------------------------------------------------------
# Multi-view preprocessing
# ------------------------------------------------------------------

def preprocess_features(
    features: List[Union[np.ndarray, sp.spmatrix]],
    center: bool = False,
    scale: bool = False
):
    """
    Apply optional centering and scaling to multiple feature views.

    Parameters
    ----------
    features : list
        List of feature matrices.

    center : bool
        Whether to center features.

    scale : bool
        Whether to scale features.

    Returns
    -------
    list
        List of processed feature matrices.
    """

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

    return processed


# ------------------------------------------------------------------
# Graph feature propagation
# ------------------------------------------------------------------

def get_propagated_features(
    features: Union[np.ndarray, sp.spmatrix],
    adj: Union[np.ndarray, sp.spmatrix, None] = None,
    beta: float = 1.0,
    tf_idf: bool = False
):
    """
    Apply graph-based feature propagation.

    H = S X

    where S = D^{-1}(A + beta I)

    If no adjacency matrix is provided, the normalized features
    are returned unchanged.

    Parameters
    ----------
    features : ndarray or sparse matrix
        Feature matrix.

    adj : ndarray or sparse matrix or None
        Adjacency matrix.

    beta : float
        Self-loop weight.

    tf_idf : bool
        Apply TF-IDF normalization.

    Returns
    -------
    ndarray
        Propagated feature matrix.
    """

    X = normalize_features(features, tf_idf)

    if adj is None:
        return X

    S = preprocess_graph(adj, beta)

    H = S.dot(X)

    if sp.issparse(H):
        H = H.toarray()

    return H