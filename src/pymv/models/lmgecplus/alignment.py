"""
Cross-view alignment strategies.

LMGEC's W-update is a closed-form orthogonal Procrustes problem, so
alignment objectives are expressed as **target augmentations**: view
*v* regresses onto ``T_v = T_shared + μ · a(Z_prev)`` instead of the
shared clustering target alone.  This keeps every update closed-form
while pulling the per-view projections toward (a function of) the
previous consensus.

Strategies
----------
* ``"none"``        — disabled.
* ``"consensus"``   — original LMGECplus extension 8:
  ``a(Z) = Z`` (plain drift penalty toward the consensus).
* ``"cosine"``      — direction-only pull: ``a(Z)`` is the row-wise
  L2-normalized consensus rescaled to the consensus' mean row norm,
  i.e. views are aligned in *angle* but not in magnitude.
* ``"contrastive"`` — pull toward the own consensus row while pushing
  away from the batch mean (a single shared negative):
  ``a(Z) = Z − mean(Z)``.  A first-order, closed-form surrogate of a
  contrastive objective — documented as such, not a full InfoNCE.

Custom alignments only need to subclass
:class:`~pymv.models.lmgecplus.interfaces.Alignment` and implement
``augment_target`` (and optionally ``penalty`` for loss reporting).
"""

from __future__ import annotations

import numpy as np

from .factory import register_component
from .interfaces import Alignment, FitState

_EPS = 1e-12


@register_component("alignment", "none", aliases=("disabled",))
class NoAlignment(Alignment):
    """Alignment disabled — the shared target is used unchanged."""

    def augment_target(self, T_shared, state, view):
        return T_shared


@register_component("alignment", "consensus", aliases=("original",))
class ConsensusAlignment(Alignment):
    """Original cross-view alignment: ``T_v = T_shared + μ Z_prev``.

    Penalises per-view representation drift away from the previous
    consensus (LMGECplus extension 8).

    Parameters
    ----------
    mu : float, default=0.1
        Alignment strength μ.
    """

    def __init__(self, mu: float = 0.1):
        self.mu = mu

    def augment_target(self, T_shared, state, view):
        if state.Z_prev is None:
            return T_shared
        return T_shared + self.mu * state.Z_prev


@register_component("alignment", "cosine")
class CosineAlignment(Alignment):
    r"""Direction-only alignment toward the previous consensus.

    Augments the target with the row-normalized consensus, rescaled by
    the consensus' mean row norm so the augmentation lives on the same
    scale as ``Z_prev``:

    .. math::
        T_v = T + \mu\,\bar{z}\,\frac{Z_{prev}}{\|Z_{prev}\|_{row}}

    Compared to :class:`ConsensusAlignment`, nodes with large-magnitude
    consensus rows do not dominate the pull — only the *direction*
    (cosine) is aligned.

    Parameters
    ----------
    mu : float, default=0.1
        Alignment strength μ.
    """

    def __init__(self, mu: float = 0.1):
        self.mu = mu

    def augment_target(self, T_shared, state, view):
        Z = state.Z_prev
        if Z is None:
            return T_shared
        norms = np.sqrt((Z ** 2).sum(axis=1, keepdims=True))
        mean_norm = float(norms.mean())
        directions = Z / np.maximum(norms, _EPS)
        return T_shared + self.mu * mean_norm * directions

    def penalty(self, state: FitState) -> float:
        """Report μ · Σ_v mean(1 − cos(Z_v, Z_prev))."""
        Z = state.Z_prev
        if Z is None or not state.Zs:
            return 0.0
        zn = Z / np.maximum(
            np.sqrt((Z ** 2).sum(1, keepdims=True)), _EPS)
        total = 0.0
        for Zv in state.Zs:
            vn = Zv / np.maximum(
                np.sqrt((Zv ** 2).sum(1, keepdims=True)), _EPS)
            total += float(np.mean(1.0 - (zn * vn).sum(1)))
        return self.mu * total


@register_component("alignment", "contrastive")
class ContrastiveAlignment(Alignment):
    r"""Pull toward the own consensus row, push from the batch mean.

    .. math::
        T_v = T + \mu\,(Z_{prev} - \bar{Z}_{prev})

    The subtracted batch mean acts as one shared negative sample: rows
    are attracted to their own consensus position and repelled from
    the global centroid, sharpening between-node separation.  This is
    a linear, closed-form surrogate of a contrastive loss (a full
    InfoNCE with per-node negatives has no closed-form W-update).

    Parameters
    ----------
    mu : float, default=0.1
        Alignment strength μ.
    """

    def __init__(self, mu: float = 0.1):
        self.mu = mu

    def augment_target(self, T_shared, state, view):
        Z = state.Z_prev
        if Z is None:
            return T_shared
        return T_shared + self.mu * (Z - Z.mean(axis=0, keepdims=True))

    def penalty(self, state: FitState) -> float:
        """Report the (negated) spread gain: −μ · Σ_v var(Z_v)."""
        if not state.Zs:
            return 0.0
        total = 0.0
        for Zv in state.Zs:
            centered = Zv - Zv.mean(axis=0, keepdims=True)
            total += float((centered ** 2).mean())
        return -self.mu * total
