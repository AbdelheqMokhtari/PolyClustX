"""
Per-view feature normalization strategies.

Each strategy transforms **one** view; the model instantiates (or
receives) an independent strategy per view, so heterogeneous
modalities can be normalized differently::

    LMGECplus(..., normalization=["tfidf", "zscore"])   # view 0, view 1

The original LMGEC pipeline (optional column standardisation followed
by row-wise L2 / TF-IDF) is reproduced by composing
:class:`ZScoreNormalization` and :class:`L2Normalization` in a
:class:`NormalizationPipeline` — the facade does this automatically
from the legacy ``center`` / ``scale`` / ``tf_idf`` flags.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pymv.preprocessing import normalize_features, preprocess_features

from .factory import register_component
from .interfaces import Normalization

_EPS = 1e-12


def _densify(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


@register_component("normalization", "none", aliases=("identity",))
class NoNormalization(Normalization):
    """Pass features through unchanged."""

    def transform(self, X):
        return X


@register_component("normalization", "l2")
class L2Normalization(Normalization):
    """Row-wise L2 normalization (each sample on the unit sphere).

    This is LMGEC's default normalization
    (:func:`pymv.preprocessing.normalize_features` with
    ``tf_idf=False``).
    """

    def transform(self, X):
        return normalize_features(X, tf_idf=False)


@register_component("normalization", "tfidf", aliases=("tf_idf",))
class TfIdfNormalization(Normalization):
    """TF-IDF weighting + L2 normalization (for count features)."""

    def transform(self, X):
        return normalize_features(X, tf_idf=True)


@register_component("normalization", "zscore", aliases=("standard",))
class ZScoreNormalization(Normalization):
    """Column-wise standardisation (centering and/or unit variance).

    Wraps :func:`pymv.preprocessing.preprocess_features` for a single
    view, inheriting its sparse-safety (centering is skipped on sparse
    input to preserve sparsity).

    Parameters
    ----------
    center : bool, default=True
        Subtract the column mean.
    scale : bool, default=True
        Divide by the column standard deviation.
    """

    def __init__(self, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale

    def transform(self, X):
        if not (self.center or self.scale):
            return X
        return preprocess_features([X], center=self.center, scale=self.scale)[0]


@register_component("normalization", "minmax")
class MinMaxNormalization(Normalization):
    """Column-wise min-max scaling to ``[0, 1]``.

    Constant columns are mapped to 0 (zero range is guarded).
    """

    def transform(self, X):
        X = _densify(X)
        col_min = X.min(axis=0, keepdims=True)
        col_rng = X.max(axis=0, keepdims=True) - col_min
        col_rng = np.where(col_rng < _EPS, 1.0, col_rng)
        return (X - col_min) / col_rng


@register_component("normalization", "whitening", aliases=("whiten",))
class WhiteningNormalization(Normalization):
    """PCA whitening: decorrelate features and give them unit variance.

    Centers the view, rotates it onto its principal axes and divides
    by the singular values, so the transformed covariance is (close
    to) the identity.  Directions with negligible variance are left
    unscaled to avoid amplifying numerical noise.

    Parameters
    ----------
    eps : float, default=1e-8
        Variance floor below which a direction is not rescaled.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def transform(self, X):
        X = _densify(X)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        n = X.shape[0]
        sigma = S / np.sqrt(max(n - 1, 1))
        keep = sigma > self.eps
        scale = np.where(keep, np.sqrt(max(n - 1, 1)), 1.0)
        return U * scale


class NormalizationPipeline(Normalization):
    """Apply several normalization steps in sequence (composition).

    Used by the facade to reproduce the original LMGEC order
    (column standardisation → row normalization) and available to
    users who need custom chains::

        NormalizationPipeline([ZScoreNormalization(), L2Normalization()])
    """

    def __init__(self, steps: list):
        self.steps = list(steps)

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def __repr__(self) -> str:
        inner = " -> ".join(repr(s) for s in self.steps)
        return f"NormalizationPipeline({inner})"
