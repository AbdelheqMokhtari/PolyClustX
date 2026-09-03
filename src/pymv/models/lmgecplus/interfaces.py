"""
Component interfaces (contracts) for the modular LMGEC framework.

Every algorithmic ingredient of LMGEC/LMGECplus is expressed as a small
strategy object implementing one of the abstract base classes below.
The model class (:class:`~pymv.models.lmgecplus.LMGECplus`) only *wires*
components together; the optimizer only *calls* them through these
interfaces.  A researcher can therefore replace any single ingredient
(a loss, a normalization, an attention mechanism, …) by writing **one
new class** — nothing else in the library needs to change.

Data flow
---------
::

    Xs, As ──▶ Normalization ──▶ GraphPropagation ──▶ Hs (prepared views)
                                                        │
                                                        ▼
                 ┌──────────────── Optimizer loop ────────────────┐
                 │  W-update   (ReconstructionLoss node weights,  │
                 │              Alignment target augmentation)    │
                 │  weights    (ViewWeighting / Attention)        │
                 │  consensus  Z = Σ_v β_v ⊙ (H_v W_v)            │
                 │  clustering (ClusteringHead)                   │
                 └────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                                  LMGECResult

All mutable quantities travel inside a single :class:`FitState` object,
so component signatures stay stable when new state is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

# ── Shared optimisation state ─────────────────────────────────────

@dataclass(eq=False)
class FitState:
    """Mutable state shared by all components during one ``fit``.

    Attributes
    ----------
    Hs : list[np.ndarray]
        Prepared (normalized + propagated) dense views, one ``(n, d_v)``
        matrix per view.
    n_clusters : int
        Number of clusters *k*.
    embedding_dim : int
        Rank-clamped shared latent dimension *m*.
    random_state : int or None
        Seed forwarded to stochastic components (K-Means, GMM, …).
    Ws : list[np.ndarray]
        Per-view orthonormal projections ``(d_v, m)``.
    Zs : list[np.ndarray]
        Per-view projected representations ``H_v @ W_v`` of shape ``(n, m)``.
    consensus : np.ndarray
        Fused embedding ``(n, m)`` — the per-node convex combination of
        the ``Zs`` under the current view weights.
    Z_prev : np.ndarray
        Consensus of the *previous* iteration (used by alignment).
    F : np.ndarray
        Cluster centroids ``(k, m)``.
    G : np.ndarray
        Hard labels ``(n,)`` (int32).
    R : np.ndarray or None
        Soft responsibilities ``(n, k)`` — ``None`` for hard heads.
    residual_norms : list[np.ndarray]
        Per-view per-node residual row norms ``‖(H_v − T Wᵥᵀ)_i‖₂``.
    errors : list[np.ndarray]
        Per-view per-node **squared** reconstruction errors
        (``residual_norms**2``) consumed by attention / weighting.
    us : list[np.ndarray]
        Per-view IRLS node weights produced by the reconstruction loss.
    view_node_weights : list[np.ndarray]
        Per-view per-node fusion weights β (each ``(n,)``; for every
        node the weights sum to 1 across views).
    iteration : int
        Current outer iteration (0-based).
    """

    Hs: list
    n_clusters: int
    embedding_dim: int
    random_state: int | None = None

    Ws: list = field(default_factory=list)
    Zs: list = field(default_factory=list)
    consensus: np.ndarray | None = None
    Z_prev: np.ndarray | None = None
    F: np.ndarray | None = None
    G: np.ndarray | None = None
    R: np.ndarray | None = None
    residual_norms: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    us: list = field(default_factory=list)
    view_node_weights: list = field(default_factory=list)
    iteration: int = 0

    @property
    def n_views(self) -> int:
        return len(self.Hs)

    @property
    def n_samples(self) -> int:
        return self.Hs[0].shape[0]

    @property
    def dims(self) -> list[int]:
        """Per-view feature dimensionality ``d_v``."""
        return [H.shape[1] for H in self.Hs]


# ── Base class with uniform repr / introspection ──────────────────

class Component:
    """Mixin giving every component a readable repr and param dump.

    ``get_params()`` reflects the public constructor attributes and is
    used by the experiment logger so that the *exact* configuration of
    every module ends up in the experiment report.
    """

    def get_params(self) -> dict:
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_")
        }

    def __repr__(self) -> str:  # e.g.  L21Loss(normalize=True)
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


# ── 1. Reconstruction loss ────────────────────────────────────────

class ReconstructionLoss(Component, ABC):
    r"""Penalty ρ applied to per-node residual row norms.

    LMGEC's alternating updates stay closed-form for any loss that can
    be minimised by *iteratively re-weighted least squares* (IRLS):
    the loss only has to supply, for residual norms
    :math:`r_i = \|(H - T W^\top)_i\|_2`,

    * ``penalty(r)``   — the monitored objective value ρ(rᵢ), and
    * ``irls_node_weights(r)`` — the IRLS weights uᵢ such that
      minimising ``Σ_i u_i r_i²`` locally minimises ``Σ_i ρ(r_i)``.
    """

    #: whether the loss produces non-trivial node weights (False → the
    #: optimizer can skip node re-weighting entirely, exactly matching
    #: the plain least-squares path).
    reweights_nodes: bool = True

    @abstractmethod
    def penalty(self, residual_norms: np.ndarray) -> np.ndarray:
        """Return ρ(r) for an array of per-node residual norms."""

    @abstractmethod
    def irls_node_weights(self, residual_norms: np.ndarray) -> np.ndarray:
        """Return IRLS node weights u(r) (positive, same shape)."""


# ── 2. Feature normalization ──────────────────────────────────────

class Normalization(Component, ABC):
    """Per-view feature transform applied *before* graph propagation.

    Operates on a single view; the model applies one (independently
    configurable) instance per view.
    """

    @abstractmethod
    def transform(self, X):
        """Return the normalized feature matrix (dense or sparse)."""


# ── 3. Graph propagation ──────────────────────────────────────────

class GraphPropagation(Component, ABC):
    """Smooths a (normalized) view over its graph."""

    @abstractmethod
    def propagate(self, X, adj) -> tuple[np.ndarray, float]:
        """Return ``(H, gamma_used)``.

        ``adj`` may be ``None`` (no graph for this view) in which case
        implementations must return ``X`` unchanged with ``gamma=0``.
        """


# ── 4. Attention ──────────────────────────────────────────────────

class Attention(Component, ABC):
    """Maps per-view reconstruction errors to fusion weights β.

    Returns one ``(n,)`` weight vector per view with
    ``Σ_v β_{v,i} = 1`` for every node *i*.
    """

    #: recompute every iteration (True) or freeze after init (False).
    dynamic: bool = True

    @abstractmethod
    def compute(self, errors: list, state: FitState) -> list:
        """Return per-view weight vectors from per-view ``(n,)`` errors."""


# ── 5. View weighting ─────────────────────────────────────────────

class ViewWeighting(Component, ABC):
    """Strategy producing the per-view fusion weights β at each stage.

    Three hooks cover the LMGEC life-cycle:

    * :meth:`initial_weights` — before any centroids exist; used to
      build the *initial* consensus (original LMGEC uses the K-Means
      inertia weights here).  **May be unnormalized** — the optimizer
      renormalizes the initial consensus per node, so strategies can
      return raw scores.
    * :meth:`start_weights` — after initial centroids / errors are
      available; weights for the first optimisation iteration
      (static attention hooks in here).  Must be normalized
      (per node, the weights sum to 1 across views).
    * :meth:`update` — called once per iteration after fresh
      reconstruction errors were computed.  Must be normalized.

    The default implementations keep the current weights fixed, which
    is correct for every *static* strategy.
    """

    #: True when the weights come from an attention mechanism (the
    #: model then exposes them as ``attention_weights``).
    provides_attention: bool = False

    @abstractmethod
    def initial_weights(self, state: FitState) -> list:
        """Weights used to fuse the initial per-view projections."""

    def start_weights(self, state: FitState) -> list:
        """Weights for iteration 0 (default: the initial weights)."""
        return state.view_node_weights

    def update(self, state: FitState) -> list:
        """Weights for the next fusion (default: unchanged)."""
        return state.view_node_weights


# ── 6. Cross-view alignment ───────────────────────────────────────

class Alignment(Component, ABC):
    """Augments each view's regression target to align views.

    LMGEC's W-update solves an orthogonal Procrustes problem, so
    alignment strategies are expressed as *target augmentations*
    ``T_v = T_shared + μ · a(Z_prev)`` — this keeps the update
    closed-form regardless of the alignment used.
    """

    @abstractmethod
    def augment_target(self, T_shared: np.ndarray, state: FitState,
                       view: int) -> np.ndarray:
        """Return the (possibly augmented) target for *view*."""

    def penalty(self, state: FitState) -> float:
        """Optional additive term for the monitored loss (default 0)."""
        return 0.0


# ── 7. Clustering head ────────────────────────────────────────────

class ClusteringHead(Component, ABC):
    """Assignment + centroid strategy on the consensus embedding."""

    #: whether :attr:`FitState.R` (soft responsibilities) is populated.
    provides_soft: bool = False

    def initialize(self, Z: np.ndarray, state: FitState):
        """Return initial ``(G, F, R)`` from the fused embedding.

        The default (shared by all heads for comparability) is a
        K-Means initialisation, as in the original LMGEC.
        """
        km = KMeans(n_clusters=state.n_clusters, n_init=10,
                    random_state=state.random_state)
        km.fit(Z)
        return km.labels_.astype(np.int32), km.cluster_centers_, None

    @abstractmethod
    def shared_target(self, state: FitState) -> np.ndarray:
        """Return the ``(n, m)`` reconstruction target (e.g. ``F[G]``)."""

    @abstractmethod
    def refit(self, state: FitState, node_w: np.ndarray | None):
        """Re-assign and re-estimate centroids; return ``(G, F, R)``."""


# ── 8. Optimizer ──────────────────────────────────────────────────

class Optimizer(Component, ABC):
    """Owns the training loop; the model class owns none of it."""

    @abstractmethod
    def run(self, state: FitState, components) -> "OptimizationInfo":
        """Run the optimisation, mutating *state* in place."""


@dataclass
class OptimizationInfo:
    """What an :class:`Optimizer` reports back after running."""

    loss_history: np.ndarray
    converged: bool
    n_iter: int

    @property
    def final_loss(self) -> float:
        return float(self.loss_history[-1]) if len(self.loss_history) else float("nan")

    def as_dict(self) -> dict:
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "final_loss": self.final_loss,
        }


# ── Component bundle (what the optimizer receives) ────────────────

@dataclass
class ComponentBundle:
    """Resolved strategy objects wired together for one ``fit`` call.

    ``normalizations`` and ``propagations`` hold **one instance per
    view** (views are preprocessed independently); the remaining
    components are shared across views.
    """

    loss: ReconstructionLoss
    normalizations: list
    propagations: list
    weighting: ViewWeighting
    alignment: Alignment
    clustering: ClusteringHead
    optimizer: Optimizer

    def describe(self) -> dict:
        """Human-readable summary used for experiment logging."""
        def _one(c):
            return repr(c)

        return {
            "reconstruction_loss": _one(self.loss),
            "normalization": [_one(c) for c in self.normalizations],
            "graph_propagation": [_one(c) for c in self.propagations],
            "view_weighting": _one(self.weighting),
            "alignment": _one(self.alignment),
            "clustering": _one(self.clustering),
            "optimizer": _one(self.optimizer),
        }
