"""
View-weighting strategies — how per-view projections are fused.

The consensus embedding is the per-node convex combination
``Z = Σ_v β_v ⊙ (H_v W_v)``.  A :class:`ViewWeighting` strategy decides
the β's at three moments of the LMGEC life-cycle (see
:class:`~pymv.models.lmgecplus.interfaces.ViewWeighting`):

* ``initial_weights`` — before centroids exist (initial consensus),
* ``start_weights``   — for the first optimisation iteration,
* ``update``          — once per iteration.

Strategies
----------
* ``"uniform"``   — equal weight 1/V for every view.
* ``"inertia"``   — original LMGEC: softmax of the per-view K-Means
  inertia, computed once a priori and kept fixed.
* ``"entropy"``   — view-level entropy-regularised weights recomputed
  every iteration from the mean reconstruction errors.
* ``"attention"`` — delegate to an interchangeable
  :class:`~pymv.models.lmgecplus.interfaces.Attention` mechanism.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans

from .attention import ViewLevelAttention
from .factory import register_component, resolve_component
from .interfaces import Attention, FitState, ViewWeighting

logger = logging.getLogger(__name__)


@register_component("weighting", "uniform")
class UniformWeighting(ViewWeighting):
    """Every view receives the constant weight ``1 / n_views``."""

    def initial_weights(self, state: FitState) -> list:
        n, V = state.n_samples, state.n_views
        return [np.full(n, 1.0 / V) for _ in range(V)]


@register_component("weighting", "inertia", aliases=("original", "lmgec"))
class InertiaWeighting(ViewWeighting):
    r"""Original LMGEC a-priori weights, fixed for the whole run.

    Each view is clustered on its own initial projection; the weight is
    a softmax of the (size-normalized) K-Means inertia:

    .. math::
        \alpha_v \propto
        \exp\!\left(-\frac{\|Z_v - F_v[G_v]\|_F}{\sqrt{n}\;\tau}\right)

    The :math:`\sqrt{n}` normalization keeps the scale independent of
    the dataset size, preventing the softmax from collapsing to a
    single view on large datasets.

    Parameters
    ----------
    temperature : float, default=1.0
        Softmax temperature τ (lower → sharper weighting).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def initial_weights(self, state: FitState) -> list:
        """Raw (unnormalized) a-priori scores ``exp(-inertia/τ)``."""
        n, V = state.n_samples, state.n_views
        alphas = np.zeros(V)
        for v in range(V):
            Zv = state.Zs[v]
            km = KMeans(n_clusters=state.n_clusters, n_init=10,
                        random_state=state.random_state)
            km.fit(Zv)
            Fv = km.cluster_centers_
            inertia = np.linalg.norm(Zv - Fv[km.labels_]) / np.sqrt(n)
            logger.debug("[View %d] Normalized Inertia: %.4f", v, inertia)
            alphas[v] = np.exp(-inertia / self.temperature)
        self._raw = alphas
        return [np.full(n, alphas[v]) for v in range(V)]

    def start_weights(self, state: FitState) -> list:
        """Normalized fixed weights α used for every iteration."""
        V = state.n_views
        asum = self._raw.sum()
        alphas = self._raw / asum if asum > 0 else np.full(V, 1.0 / V)
        self._alphas = alphas
        n = state.n_samples
        return [np.full(n, alphas[v]) for v in range(V)]


@register_component("weighting", "attention")
class AttentionWeighting(ViewWeighting):
    """Fusion weights produced by an attention mechanism (composition).

    Wraps any :class:`~pymv.models.lmgecplus.interfaces.Attention`:

    * the *initial* consensus (before any reconstruction errors exist)
      is fused with a bootstrap strategy — by default the original
      inertia weights, matching historical LMGECplus behaviour;
    * a **static** mechanism is evaluated once on the initialisation
      errors and frozen;
    * a **dynamic** mechanism is re-evaluated every iteration.

    Parameters
    ----------
    attention : Attention | str | dict, default="node"
        The mechanism (spec resolved through the component registry).
    init_weighting : ViewWeighting | str | dict, default="inertia"
        Bootstrap strategy for the initial consensus.
    """

    provides_attention = True

    def __init__(self, attention: Attention | str | dict = "node",
                 init_weighting: ViewWeighting | str | dict = "inertia"):
        self.attention = resolve_component("attention", attention)
        self.init_weighting = resolve_component("weighting", init_weighting)

    def initial_weights(self, state: FitState) -> list:
        return self.init_weighting.initial_weights(state)

    def start_weights(self, state: FitState) -> list:
        if self.attention.dynamic:
            # dynamic: first W-update still uses the bootstrap weights
            return self.init_weighting.start_weights(state)
        # static: compute once from the initialisation errors, freeze
        self._static = self.attention.compute(state.errors, state)
        return self._static

    def update(self, state: FitState) -> list:
        if self.attention.dynamic:
            return self.attention.compute(state.errors, state)
        return self._static


@register_component("weighting", "entropy")
class EntropyWeighting(AttentionWeighting):
    """View-level entropy-regularised weighting (preset).

    Equivalent to ``AttentionWeighting`` over a dynamic
    :class:`~pymv.models.lmgecplus.attention.ViewLevelAttention` with
    entropy regularisation — one adaptive scalar weight per view,
    recomputed every iteration.

    Parameters
    ----------
    lam : float, default=1.0
        Entropy regularisation strength (softmax temperature).
    """

    def __init__(self, lam: float = 1.0):
        super().__init__(
            attention=ViewLevelAttention(dynamic=True,
                                         regularization="entropy", lam=lam),
        )
        self.lam = lam
