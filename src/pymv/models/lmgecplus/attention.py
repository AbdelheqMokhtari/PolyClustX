"""
Attention mechanisms for cross-view fusion.

An attention mechanism converts per-view reconstruction errors into
fusion weights β (one ``(n,)`` vector per view, summing to 1 across
views for every node).  Mechanisms are interchangeable classes; the
model consumes them through
:class:`~pymv.models.lmgecplus.weighting.AttentionWeighting`.

Available mechanisms
--------------------
* ``"node"``    — node-level: every node distributes its own weight
  across views (extension 2 of LMGECplus).
* ``"view"``    — view-level: one scalar weight per view, computed
  from the mean reconstruction error and broadcast to all nodes.
* ``"dynamic"`` — alias for node-level attention recomputed at every
  iteration (the historical ``dynamic_attention=True``).
* ``"static"``  — alias for node-level attention computed once after
  initialisation and frozen.

All mechanisms share the regularised-allocation closed form of
:func:`~pymv.models.lmgecplus.components.view_attention`
(``reg=None``/``"entropy"`` → softmax, ``reg="l2"`` → sparse simplex
projection).
"""

from __future__ import annotations

import numpy as np

from . import components as C
from .factory import register_component
from .interfaces import Attention, FitState


@register_component("attention", "node", aliases=("dynamic", "node_dynamic"))
class NodeLevelAttention(Attention):
    """Node-level cross-view attention (LMGECplus extensions 2–5).

    Parameters
    ----------
    dynamic : bool, default=True
        Recompute at every iteration (``True``) or freeze the weights
        computed from the initial reconstruction (``False``).
    dim_normalized : bool, default=False
        Divide each view's errors by its feature dimension before the
        allocation (prevents bias toward low-dimensional modalities).
    regularization : {None, "entropy", "l2"}, default=None
        Allocation regulariser (see
        :func:`~pymv.models.lmgecplus.components.view_attention`).
    lam : float, default=1.0
        Regularisation strength λ (softmax temperature).
    """

    def __init__(
        self,
        dynamic: bool = True,
        dim_normalized: bool = False,
        regularization: str | None = None,
        lam: float = 1.0,
    ):
        self.dynamic = dynamic
        self.dim_normalized = dim_normalized
        self.regularization = regularization
        self.lam = lam

    def _normalized_errors(self, errors: list, state: FitState) -> list:
        if not self.dim_normalized:
            return list(errors)
        return [e / d for e, d in zip(errors, state.dims)]

    def compute(self, errors: list, state: FitState) -> list:
        errs = self._normalized_errors(errors, state)
        return C.view_attention(errs, lam=self.lam, reg=self.regularization)


@register_component("attention", "static")
class StaticAttention(NodeLevelAttention):
    """Node-level attention computed once and frozen (preset).

    Identical to ``NodeLevelAttention(dynamic=False)``; registered
    under its own name so configs read naturally
    (``attention: static``).
    """

    def __init__(self, dim_normalized: bool = False,
                 regularization: str | None = None, lam: float = 1.0):
        super().__init__(dynamic=False, dim_normalized=dim_normalized,
                         regularization=regularization, lam=lam)


@register_component("attention", "view", aliases=("view_level",))
class ViewLevelAttention(Attention):
    """View-level attention: one scalar weight per view.

    Aggregates the per-node errors of each view into a mean error and
    runs the same regularised allocation on the resulting ``(1, V)``
    problem; the scalar weights are broadcast to every node.  This is
    the natural middle ground between LMGEC's fixed a-priori weights
    and full node-level attention.

    Parameters
    ----------
    dynamic, dim_normalized, regularization, lam
        As in :class:`NodeLevelAttention`.
    """

    def __init__(
        self,
        dynamic: bool = True,
        dim_normalized: bool = False,
        regularization: str | None = None,
        lam: float = 1.0,
    ):
        self.dynamic = dynamic
        self.dim_normalized = dim_normalized
        self.regularization = regularization
        self.lam = lam

    def compute(self, errors: list, state: FitState) -> list:
        errs = list(errors)
        if self.dim_normalized:
            errs = [e / d for e, d in zip(errs, state.dims)]
        # one "virtual node" holding each view's mean error
        mean_errs = [np.array([float(np.mean(e))]) for e in errs]
        scalar = C.view_attention(mean_errs, lam=self.lam,
                                  reg=self.regularization)
        n = state.n_samples
        return [np.full(n, float(b[0])) for b in scalar]
