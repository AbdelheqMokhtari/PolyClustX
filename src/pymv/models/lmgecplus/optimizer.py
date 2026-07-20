"""
Optimisation logic for the modular LMGEC framework.

The optimizer owns *all* training-loop mechanics — initialisation,
the alternating updates, loss monitoring and convergence — and talks
to the algorithmic ingredients exclusively through the component
interfaces.  The model class contains no optimisation code.

One outer iteration of :class:`AlternatingOptimizer` (identical maths
to the historical ``LMGECplus.fit`` loop):

1. ``T ← clustering.shared_target(state)``            (``F[G]`` / ``R F``)
2. per view: ``W_v ← procrustes(H_v, alignment(T))``  weighted by
   ``β_v ⊙ u_v`` (fusion × IRLS weights)
3. ``β ← weighting.update(state)``                    (attention, …)
4. ``Z ← Σ_v β_v ⊙ H_v W_v``                          (consensus)
5. ``G, F, R ← clustering.refit(state)``
6. monitored loss ``Σ_v Σ_i β_{v,i} ρ(r_{v,i}) + alignment.penalty``
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from . import components as C
from .factory import register_component
from .interfaces import ComponentBundle, FitState, OptimizationInfo, Optimizer

_EPS = 1e-12


@register_component("optimizer", "alternating")
class AlternatingOptimizer(Optimizer):
    """Block-coordinate (alternating) optimizer — the LMGEC scheme.

    Parameters
    ----------
    max_iter : int, default=30
        Maximum number of outer iterations.
    tol : float or None, default=None
        Early-stopping tolerance on the *relative* loss change.
        ``None`` (default) always runs ``max_iter`` iterations —
        exactly the historical behaviour.
    """

    def __init__(self, max_iter: int = 30, tol: float | None = None):
        self.max_iter = max_iter
        self.tol = tol

    # ── initialisation ────────────────────────────────────────────

    def _init_projections(self, state: FitState) -> None:
        """Per-view SVD projections W and representations Z = H W."""
        m = state.embedding_dim
        state.Ws, state.Zs = [], []
        for H in state.Hs:
            m_v = min(m, min(H.shape) - 1)
            svd = TruncatedSVD(n_components=m_v,
                               random_state=state.random_state)
            svd.fit(H)
            W = svd.components_.T
            state.Ws.append(W)
            state.Zs.append(H @ W)

    def _init_errors(self, state: FitState, bundle: ComponentBundle) -> None:
        """Residuals of the initial reconstruction (for static attention)."""
        T = bundle.clustering.shared_target(state)
        state.residual_norms, state.errors = [], []
        for v in range(state.n_views):
            R = state.Hs[v] - T @ state.Ws[v].T
            sq = (R ** 2).sum(1)
            state.residual_norms.append(np.sqrt(sq))
            state.errors.append(sq)

    def _initialize(self, state: FitState, bundle: ComponentBundle) -> None:
        self._init_projections(state)

        # fuse the initial projections with the strategy's a-priori
        # weights; initial weights may be unnormalized (raw scores), so
        # the fusion is renormalized per node.
        state.view_node_weights = bundle.weighting.initial_weights(state)
        consensus = np.zeros_like(state.Zs[0])
        denom = np.zeros(state.n_samples)
        for v in range(state.n_views):
            consensus += state.view_node_weights[v][:, None] * state.Zs[v]
            denom += state.view_node_weights[v]
        denom = np.where(denom > _EPS, denom, float(state.n_views))
        consensus = consensus / denom[:, None]
        state.consensus = consensus
        state.Z_prev = consensus

        state.G, state.F, state.R = bundle.clustering.initialize(
            consensus, state)

        self._init_errors(state, bundle)
        state.us = [np.ones(state.n_samples) for _ in range(state.n_views)]
        state.view_node_weights = bundle.weighting.start_weights(state)

    # ── main loop ─────────────────────────────────────────────────

    def run(self, state: FitState, bundle: ComponentBundle) -> OptimizationInfo:
        """Run the alternating optimisation, mutating *state* in place."""
        loss = bundle.loss
        weighting = bundle.weighting
        alignment = bundle.alignment
        clustering = bundle.clustering

        self._initialize(state, bundle)

        loss_history: list[float] = []
        converged = False
        n_iter = 0

        for t in range(self.max_iter):
            state.iteration = t
            T_shared = clustering.shared_target(state)

            # -- per-view projection update + residuals --
            new_Ws, new_Zs = [], []
            residual_norms, errors, new_us = [], [], []
            for v in range(state.n_views):
                w_v = state.view_node_weights[v] * state.us[v]
                T_v = alignment.augment_target(T_shared, state, v)
                Wv = C.procrustes_W(state.Hs[v], T_v, node_w=w_v)
                new_Ws.append(Wv)
                new_Zs.append(state.Hs[v] @ Wv)

                Rres = state.Hs[v] - T_shared @ Wv.T
                sq = (Rres ** 2).sum(1)
                rown = np.sqrt(sq)
                residual_norms.append(rown)
                errors.append(sq)
                new_us.append(loss.irls_node_weights(rown))

            state.Ws, state.Zs = new_Ws, new_Zs
            state.residual_norms, state.errors = residual_norms, errors

            # -- fusion-weight update (attention / static / …) --
            state.view_node_weights = weighting.update(state)
            state.us = new_us

            # -- consensus (per-node convex fusion across views) --
            consensus = np.zeros_like(state.Zs[0])
            for v in range(state.n_views):
                consensus += (
                    state.view_node_weights[v][:, None] * state.Zs[v]
                )
            state.consensus = consensus

            # -- clustering update --
            node_w = None
            if loss.reweights_nodes:
                node_w = sum(
                    state.view_node_weights[v] * state.us[v]
                    for v in range(state.n_views)
                )
            state.G, state.F, state.R = clustering.refit(state, node_w)

            state.Z_prev = consensus

            # -- monitored loss  Σ_v Σ_i β_{v,i} ρ(r_{v,i}) --
            total = 0.0
            for v in range(state.n_views):
                rho = loss.penalty(state.residual_norms[v])
                total += float((state.view_node_weights[v] * rho).sum())
            total += alignment.penalty(state)
            loss_history.append(total)
            n_iter = t + 1

            # -- convergence --
            if len(loss_history) > 1:
                prev = loss_history[-2]
                rel = abs(total - prev) / max(abs(prev), _EPS)
                if self.tol is not None and rel <= self.tol:
                    converged = True
                    break
                converged = rel <= 1e-4  # informational when tol is None

        return OptimizationInfo(
            loss_history=np.asarray(loss_history),
            converged=converged,
            n_iter=n_iter,
        )
