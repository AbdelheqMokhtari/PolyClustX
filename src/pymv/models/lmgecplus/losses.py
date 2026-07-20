"""
Reconstruction losses for the modular LMGEC framework.

Each loss is defined by its penalty ρ(r) on per-node residual norms
and the matching IRLS node weights u(r) (see
:class:`~pymv.models.lmgecplus.interfaces.ReconstructionLoss`).
Because LMGEC's W/centroid updates are weighted least-squares
problems, *any* IRLS-representable loss plugs in without touching the
optimizer:

=========  ==================  ==========================  =============
name       ρ(r)                u(r)                        behaviour
=========  ==================  ==========================  =============
``l2``     r²                  1                           original LMGEC
``l21``    r                   1 / (2r)                    outlier-robust
``huber``  r²/2 or δ(r−δ/2)    min(1, δ/r)                 soft-robust
=========  ==================  ==========================  =============
"""

from __future__ import annotations

import numpy as np

from . import components as C
from .factory import register_component
from .interfaces import ReconstructionLoss

EPS = C.EPS


@register_component("loss", "l2", aliases=("frobenius",))
class L2Loss(ReconstructionLoss):
    r"""Squared Frobenius loss ``Σ_i r_i²`` — the original LMGEC loss.

    Produces unit IRLS weights, so the optimizer follows the plain
    least-squares path (bit-for-bit identical to classic LMGEC).
    """

    reweights_nodes = False

    def penalty(self, residual_norms):
        return residual_norms ** 2

    def irls_node_weights(self, residual_norms):
        return np.ones_like(residual_norms)


@register_component("loss", "l21", aliases=("robust",))
class L21Loss(ReconstructionLoss):
    r"""ℓ2,1 loss ``Σ_i r_i`` (row-wise ℓ2, column-wise ℓ1).

    Robust to outlier *nodes*: the IRLS weight ``u_i = 1/(2 r_i)``
    down-weights samples with large residuals.

    Parameters
    ----------
    normalize : bool, default=True
        Rescale the weights to mean 1 so the overall problem scale is
        preserved (matches the historical ``robust_loss=True`` flag).
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def penalty(self, residual_norms):
        return residual_norms

    def irls_node_weights(self, residual_norms):
        u = 1.0 / (2.0 * residual_norms + EPS)
        if self.normalize:
            u = u / max(u.mean(), EPS)
        return u


@register_component("loss", "huber")
class HuberLoss(ReconstructionLoss):
    r"""Huber loss — quadratic near zero, linear in the tails.

    .. math::
        \rho(r) = \begin{cases}
            r^2 / 2               & r \le \delta \\
            \delta (r - \delta/2) & r > \delta
        \end{cases}
        \qquad
        u(r) = \min(1,\; \delta / r)

    Parameters
    ----------
    delta : float or "auto", default="auto"
        Transition point between the quadratic and linear regimes.
        ``"auto"`` sets δ to the median residual norm of the current
        batch (scale-adaptive, a common robust-statistics default).
    normalize : bool, default=True
        Rescale the IRLS weights to mean 1 (scale preservation, as in
        :class:`L21Loss`).
    """

    def __init__(self, delta: float | str = "auto", normalize: bool = True):
        self.delta = delta
        self.normalize = normalize

    def _delta(self, residual_norms) -> float:
        if isinstance(self.delta, str):  # "auto"
            positive = residual_norms[residual_norms > 0]
            return float(np.median(positive)) if positive.size else 1.0
        return float(self.delta)

    def penalty(self, residual_norms):
        d = self._delta(residual_norms)
        r = residual_norms
        return np.where(r <= d, 0.5 * r ** 2, d * (r - 0.5 * d))

    def irls_node_weights(self, residual_norms):
        d = self._delta(residual_norms)
        u = np.minimum(1.0, d / np.maximum(residual_norms, EPS))
        if self.normalize:
            u = u / max(u.mean(), EPS)
        return u
