"""
Graph propagation strategies.

Propagation smooths a (normalized) view over its graph before the
alternating optimisation.  Strategies:

* ``"none"``       — no propagation (feature-only views).
* ``"standard"``   — original LMGEC: ``H = S X`` with
  ``S = D⁻¹(A + βI)``.
* ``"controlled"`` — partial propagation ``H = ((1−γ)I + γS) X`` with
  a configurable (or data-driven ``"auto"``) strength γ.

Every strategy returns ``(H, gamma_used)`` so the model can report the
effective propagation strength per view.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pymv.preprocessing import preprocess_graph

from . import components as C
from .factory import register_component
from .interfaces import GraphPropagation


def _densify(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


@register_component("propagation", "none", aliases=("disabled",))
class NoPropagation(GraphPropagation):
    """Skip propagation entirely (ignores any adjacency)."""

    def propagate(self, X, adj):
        return _densify(X), 0.0


@register_component("propagation", "standard", aliases=("original",))
class StandardPropagation(GraphPropagation):
    r"""Original LMGEC propagation ``H = S X``.

    ``S = D⁻¹(A + βI)`` is the row-normalized adjacency with
    β-weighted self-loops (see
    :func:`pymv.preprocessing.preprocess_graph`).

    Parameters
    ----------
    beta : float, default=1.0
        Self-loop weight.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def propagate(self, X, adj):
        if adj is None:
            return _densify(X), 0.0
        S = preprocess_graph(adj, beta=self.beta)
        return _densify(S @ X), 1.0


@register_component("propagation", "controlled")
class ControlledPropagation(GraphPropagation):
    r"""Partial propagation ``H = ((1−γ)I + γS) X``.

    γ interpolates between raw features (γ=0) and full LMGEC
    propagation (γ=1), mitigating over-smoothing on dense graphs.

    Parameters
    ----------
    beta : float, default=1.0
        Self-loop weight for building ``S``.
    gamma : float or "auto", default=1.0
        Propagation strength γ ∈ [0, 1].  ``"auto"`` derives γ from
        the graph-signal smoothness of the view (rough signals get
        more propagation; see
        :func:`~pymv.models.lmgecplus.components.graph_smoothness`).
    """

    def __init__(self, beta: float = 1.0, gamma: float | str = 1.0):
        self.beta = beta
        self.gamma = gamma

    def propagate(self, X, adj):
        X = _densify(X)
        if adj is None:
            return X, 0.0
        S = preprocess_graph(adj, beta=self.beta)
        g = self.gamma
        if isinstance(g, str):  # "auto"
            g = C.smoothness_to_gamma(C.graph_smoothness(X, S))
        g = float(np.clip(g, 0.0, 1.0))
        return _densify(C.controlled_propagation(X, S, g)), g
