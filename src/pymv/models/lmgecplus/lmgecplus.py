"""
LMGECplus — a modular research framework for LMGEC-style multi-view
clustering.

The model class is a thin **facade**: every algorithmic ingredient is
an interchangeable strategy object (see
:mod:`~pymv.models.lmgecplus.interfaces`), and the class only

1. resolves the requested components (strings / dicts / instances),
2. prepares the views (normalization → propagation, per view),
3. hands everything to an :class:`~pymv.models.lmgecplus.interfaces.Optimizer`,
4. collects the outputs into an
   :class:`~pymv.models.lmgecplus.result.LMGECResult`.

Two equivalent configuration surfaces are supported:

**Component API** (research / ablations)::

    LMGECplus(
        n_clusters=3, embedding_dim=8,
        reconstruction_loss="l21",         # or L21Loss(), or "huber"
        normalization="zscore",            # per-view list allowed
        graph_propagation={"enabled": True, "gamma": 0.4},
        attention="dynamic",               # or NodeLevelAttention(...)
        clustering="soft",                 # or {"method": "gmm"}
        alignment="cosine",
    )

**Legacy flag API** (backward compatible — the historical boolean
extension flags)::

    LMGECplus(n_clusters=3, embedding_dim=8,
              robust_loss=True, node_attention=True)

With *every* extension disabled and no component overridden (the
defaults), LMGECplus reproduces the original LMGEC algorithm — same
orthogonal-Procrustes projections, fixed a-priori view weights, hard
K-Means assignment, and segment-mean centroids.

The original ``LMGEC`` class is untouched.
"""

from __future__ import annotations

import logging
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp

from pymv.core import BaseModel
from pymv.models.registry import register_model

from .alignment import ConsensusAlignment, NoAlignment
from .attention import NodeLevelAttention
from .clustering import HardKMeansHead, SoftKMeansHead
from .factory import resolve_component
from .interfaces import ComponentBundle, FitState
from .losses import L2Loss, L21Loss
from .normalization import (
    L2Normalization,
    NoNormalization,
    NormalizationPipeline,
    TfIdfNormalization,
    ZScoreNormalization,
)
from .optimizer import AlternatingOptimizer
from .propagation import ControlledPropagation, NoPropagation, StandardPropagation
from .result import LMGECResult
from .weighting import AttentionWeighting, InertiaWeighting

logger = logging.getLogger(__name__)

_PROPAGATION_WITH_BETA = {"standard", "original", "controlled"}


@register_model("lmgecplus", defaults={
    "embedding_dim": 16,
    "temperature": 1.0,
    "robust_loss": False,
    "node_attention": False,
    "dim_normalized_attention": False,
    "dynamic_attention": True,
    "attention_regularization": None,
    "attention_lambda": None,
    "controlled_propagation": False,
    "gamma": 1.0,
    "soft_clustering": False,
    "soft_beta": 50.0,
    "cross_view_alignment": False,
    "align_mu": 0.1,
    "beta": 1.0,
    "tf_idf": False,
    "center": False,
    "scale": False,
    "preprocess": True,
    "max_iter": 30,
    "reconstruction_loss": None,
    "normalization": None,
    "graph_propagation": None,
    "attention": None,
    "view_weighting": None,
    "alignment": None,
    "clustering": None,
    "optimizer": None,
    "tol": None,
    "track_memory": False,
})
class LMGECplus(BaseModel):
    """Modular LMGEC — every algorithmic component is replaceable.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.
    embedding_dim : int
        Latent dimension *m* (auto-clamped to the views' rank).
    temperature : float, default=1.0
        Softmax temperature for the a-priori (inertia) view weights
        and default attention λ.

    robust_loss : bool, default=False
        Legacy flag — ℓ2,1 (IRLS) reconstruction loss.
        Equivalent to ``reconstruction_loss="l21"``.
    node_attention : bool, default=False
        Legacy flag — node-level cross-view attention.
        Equivalent to ``attention="node"``.
    dim_normalized_attention : bool, default=False
        Legacy flag — divide per-node errors by the view dimension
        (only with ``node_attention``).
    dynamic_attention : bool, default=True
        Legacy flag — recompute attention every iteration (``True``)
        vs once after initialisation (``False``).
    attention_regularization : {None, "entropy", "l2"}, default=None
        Legacy flag — attention allocation regulariser.
    attention_lambda : float or None, default=None
        Regularisation strength λ.  ``None`` → use ``temperature``.

    controlled_propagation : bool, default=False
        Legacy flag — partial propagation ``((1−γ)I + γS)X``.
        Equivalent to ``graph_propagation={"name": "controlled", ...}``.
    gamma : float or "auto", default=1.0
        Propagation strength (only with ``controlled_propagation``).

    soft_clustering : bool, default=False
        Legacy flag — soft K-Means head.
        Equivalent to ``clustering="soft"``.
    soft_beta : float, default=50.0
        Soft-assignment sharpness β.

    cross_view_alignment : bool, default=False
        Legacy flag — consensus-drift alignment.
        Equivalent to ``alignment="consensus"``.
    align_mu : float, default=0.1
        Alignment strength μ.

    beta, tf_idf, center, scale, preprocess, max_iter
        Same meaning as in :class:`~pymv.models.lmgec.LMGEC`.
    random_state : int or None, default=None
        Seed for the stochastic components (K-Means, GMM, SVD).

    reconstruction_loss : str | dict | ReconstructionLoss | None
        Component override — ``"l2"``, ``"l21"``, ``"huber"`` or a
        custom instance.  ``None`` → derived from ``robust_loss``.
    normalization : spec | list[spec] | None
        Component override — ``"none"``, ``"l2"``, ``"tfidf"``,
        ``"zscore"``, ``"minmax"``, ``"whitening"`` or instances.  A
        list configures each view independently.  ``None`` → the
        original pipeline derived from ``center``/``scale``/``tf_idf``.
    graph_propagation : spec | list[spec] | None
        Component override — ``"none"``, ``"standard"``,
        ``"controlled"`` or dicts like
        ``{"enabled": True, "gamma": 0.4}``.  ``None`` → derived from
        ``controlled_propagation``/``beta``/``gamma``.
    attention : spec | None
        Attention mechanism — ``"static"``, ``"dynamic"``, ``"node"``,
        ``"view"`` or an :class:`~.interfaces.Attention` instance.
        Shorthand for ``view_weighting=AttentionWeighting(attention)``.
    view_weighting : spec | None
        Fusion strategy — ``"uniform"``, ``"inertia"``, ``"entropy"``,
        ``"attention"`` or a custom instance.  Takes precedence over
        ``attention`` and the legacy attention flags.
    alignment : spec | None
        ``"none"``, ``"consensus"``, ``"cosine"``, ``"contrastive"``
        or a custom instance.  ``None`` → derived from
        ``cross_view_alignment``.
    clustering : spec | None
        ``"kmeans"``, ``"soft"``, ``"gmm"`` (also as
        ``{"method": ...}``) or a custom head.  ``None`` → derived
        from ``soft_clustering``.
    optimizer : spec | None
        Optimisation strategy (default: alternating optimizer with
        ``max_iter``/``tol``).
    tol : float or None, default=None
        Early-stopping tolerance on the relative loss change
        (``None`` → always run ``max_iter`` iterations, the
        historical behaviour).
    track_memory : bool, default=False
        Record the peak Python heap allocation of ``fit`` in
        ``results_.memory_usage`` (small tracing overhead).

    Attributes
    ----------
    labels_, centroids_, embedding_, loss_history_, alphas_
        As in LMGEC (backward compatible).
    responsibilities_ : (n, k) or None
        Soft assignments (soft / GMM heads only).
    attention_ : (n_views, n) or None
        Final per-node attention weights (attention weighting only).
    gammas_ : list[float] or None
        Effective per-view propagation strength (controlled /
        component-specified propagation only).
    results_ : LMGECResult
        Structured result object with every intermediate output
        (embedding, weights, losses, convergence, timings, …).

    Examples
    --------
    Ablation of the reconstruction loss — nothing else changes::

        for loss in ("l2", "l21", "huber"):
            model = LMGECplus(n_clusters=3, embedding_dim=8,
                              reconstruction_loss=loss)
            model.fit(Xs, As)

    A custom component is one class away::

        from pymv.models.lmgecplus import ReconstructionLoss, register_component

        @register_component("loss", "cauchy")
        class CauchyLoss(ReconstructionLoss):
            def penalty(self, r): ...
            def irls_node_weights(self, r): ...

        LMGECplus(n_clusters=3, embedding_dim=8,
                  reconstruction_loss="cauchy")
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 1.0,
        robust_loss: bool = False,
        node_attention: bool = False,
        dim_normalized_attention: bool = False,
        dynamic_attention: bool = True,
        attention_regularization: str | None = None,
        attention_lambda: float | None = None,
        controlled_propagation: bool = False,
        gamma: float | str = 1.0,
        soft_clustering: bool = False,
        soft_beta: float = 50.0,
        cross_view_alignment: bool = False,
        align_mu: float = 0.1,
        beta: float | list[float] = 1.0,
        tf_idf: bool | list[bool] = False,
        center: bool = False,
        scale: bool = False,
        preprocess: bool = True,
        max_iter: int = 30,
        random_state: int | None = None,
        reconstruction_loss=None,
        normalization=None,
        graph_propagation=None,
        attention=None,
        view_weighting=None,
        alignment=None,
        clustering=None,
        optimizer=None,
        tol: float | None = None,
        track_memory: bool = False,
    ):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        # legacy extension flags
        self.robust_loss = robust_loss
        self.node_attention = node_attention
        self.dim_normalized_attention = dim_normalized_attention
        self.dynamic_attention = dynamic_attention
        self.attention_regularization = attention_regularization
        self.attention_lambda = attention_lambda
        self.controlled_propagation = controlled_propagation
        self.gamma = gamma
        self.soft_clustering = soft_clustering
        self.soft_beta = soft_beta
        self.cross_view_alignment = cross_view_alignment
        self.align_mu = align_mu
        # shared preprocessing / loop settings
        self.beta = beta
        self.tf_idf = tf_idf
        self.center = center
        self.scale = scale
        self.preprocess = preprocess
        self.max_iter = max_iter
        self.random_state = random_state
        # component overrides (None → derive from the flags above)
        self.reconstruction_loss = reconstruction_loss
        self.normalization = normalization
        self.graph_propagation = graph_propagation
        self.attention = attention
        self.view_weighting = view_weighting
        self.alignment = alignment
        self.clustering = clustering
        self.optimizer = optimizer
        self.tol = tol
        self.track_memory = track_memory

        # filled after fit()
        self.labels_ = None
        self.centroids_ = None
        self.embedding_ = None
        self.responsibilities_ = None
        self.attention_ = None
        self.alphas_ = None
        self.gammas_ = None
        self.loss_history_ = None
        self.results_: LMGECResult | None = None

    # ── framework plumbing ────────────────────────────────────────

    @property
    def name(self) -> str:
        return getattr(self, "_name", self.__class__.__name__)

    @name.setter
    def name(self, value: str):
        self._name = value

    def _resolve_per_view(self, param, n_views, param_name: str):
        """Broadcast a scalar parameter to a per-view list."""
        if isinstance(param, list):
            if len(param) != n_views:
                raise ValueError(
                    f"len({param_name})={len(param)} but got {n_views} views."
                )
            return param
        return [param] * n_views

    @staticmethod
    def _densify(X):
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    # ── component resolution (flags → strategies, overrides win) ──

    def _attention_lambda(self) -> float:
        return (self.attention_lambda
                if self.attention_lambda is not None else self.temperature)

    def _build_loss(self):
        default = L21Loss() if self.robust_loss else L2Loss()
        return resolve_component("loss", self.reconstruction_loss, default)

    def _build_normalizations(self, n_views: int) -> list:
        if self.normalization is not None:
            specs = self._resolve_per_view(
                self.normalization if isinstance(self.normalization, list)
                else [self.normalization] * n_views,
                n_views, "normalization",
            )
            return [resolve_component("normalization", s) for s in specs]
        if not self.preprocess:
            return [NoNormalization() for _ in range(n_views)]
        # legacy pipeline: column standardisation → row normalization
        tf_idfs = self._resolve_per_view(self.tf_idf, n_views, "tf_idf")
        norms = []
        for v in range(n_views):
            steps = []
            if self.center or self.scale:
                steps.append(ZScoreNormalization(center=self.center,
                                                 scale=self.scale))
            steps.append(TfIdfNormalization() if tf_idfs[v]
                         else L2Normalization())
            norms.append(steps[0] if len(steps) == 1
                         else NormalizationPipeline(steps))
        return norms

    def _build_propagations(self, n_views: int) -> list:
        betas = self._resolve_per_view(self.beta, n_views, "beta")

        if self.graph_propagation is not None:
            specs = (self.graph_propagation
                     if isinstance(self.graph_propagation, list)
                     else [self.graph_propagation] * n_views)
            specs = self._resolve_per_view(specs, n_views, "graph_propagation")
            return [self._resolve_propagation(s, betas[v])
                    for v, s in enumerate(specs)]
        if not self.preprocess:
            return [NoPropagation() for _ in range(n_views)]
        if self.controlled_propagation:
            return [ControlledPropagation(beta=betas[v], gamma=self.gamma)
                    for v in range(n_views)]
        return [StandardPropagation(beta=betas[v]) for v in range(n_views)]

    @staticmethod
    def _resolve_propagation(spec, beta):
        """Resolve one propagation spec, defaulting β when it applies."""
        if isinstance(spec, str) and spec.lower() in _PROPAGATION_WITH_BETA:
            return resolve_component("propagation", spec, beta=beta)
        if isinstance(spec, dict):
            merged = dict(spec)
            name = str(merged.get("name", merged.get("method",
                       merged.get("type", "")))).lower()
            enabled = merged.get("enabled", True)
            if enabled and (not name or name in _PROPAGATION_WITH_BETA):
                merged.setdefault("beta", beta)
            return resolve_component("propagation", merged)
        return resolve_component("propagation", spec)

    def _build_weighting(self):
        if self.view_weighting is not None:
            return resolve_component("weighting", self.view_weighting)

        attention_spec = self.attention
        if attention_spec == "none":
            attention_spec = None

        if attention_spec is not None:
            mechanism = resolve_component("attention", attention_spec)
        elif self.node_attention:
            mechanism = NodeLevelAttention(
                dynamic=self.dynamic_attention,
                dim_normalized=self.dim_normalized_attention,
                regularization=self.attention_regularization,
                lam=self._attention_lambda(),
            )
        else:
            return InertiaWeighting(temperature=self.temperature)

        return AttentionWeighting(
            attention=mechanism,
            init_weighting=InertiaWeighting(temperature=self.temperature),
        )

    def _build_alignment(self):
        default = (ConsensusAlignment(mu=self.align_mu)
                   if self.cross_view_alignment else NoAlignment())
        return resolve_component("alignment", self.alignment, default)

    def _build_clustering(self):
        default = (SoftKMeansHead(beta=self.soft_beta)
                   if self.soft_clustering else HardKMeansHead())
        return resolve_component("clustering", self.clustering, default)

    def _build_optimizer(self):
        default = AlternatingOptimizer(max_iter=self.max_iter, tol=self.tol)
        return resolve_component("optimizer", self.optimizer, default)

    def _build_components(self, n_views: int) -> ComponentBundle:
        """Resolve every strategy for this fit (overrides win over flags)."""
        return ComponentBundle(
            loss=self._build_loss(),
            normalizations=self._build_normalizations(n_views),
            propagations=self._build_propagations(n_views),
            weighting=self._build_weighting(),
            alignment=self._build_alignment(),
            clustering=self._build_clustering(),
            optimizer=self._build_optimizer(),
        )

    # ── preprocessing ─────────────────────────────────────────────

    def _prepare_views(self, Xs, As, bundle: ComponentBundle):
        """Normalize then propagate every view independently."""
        n_views = len(Xs)

        if As is None or not self.preprocess:
            adj_list = [None] * n_views
        else:
            adj_list = list(As)
            while len(adj_list) < n_views:  # reuse last adjacency
                adj_list.append(adj_list[-1])

        Hs, used_gammas = [], []
        for v in range(n_views):
            Xn = bundle.normalizations[v].transform(Xs[v])
            H, g = bundle.propagations[v].propagate(Xn, adj_list[v])
            Hs.append(self._densify(H))
            used_gammas.append(float(g))
        return Hs, used_gammas

    # ── main entry point ──────────────────────────────────────────

    def fit(self, Xs, As=None):
        """Fit the model.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices — one per view (raw when
            ``preprocess=True``, already prepared otherwise).
        As : list[np.ndarray | sp.spmatrix] or None, default=None
            Adjacency matrices — one per view (the last one is reused
            when fewer are given).  Ignored when ``preprocess=False``
            or the propagation component is ``"none"``.

        Returns
        -------
        self
            The fitted model.  All intermediate outputs are exposed on
            :attr:`results_`; the historical ``labels_`` /
            ``embedding_`` / … attributes are also populated.
        """
        t0 = time.perf_counter()
        mem_started = False
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            mem_started = True

        try:
            n_views = len(Xs)
            bundle = self._build_components(n_views)
            logger.debug("LMGECplus components: %s", bundle.describe())

            Hs, used_gammas = self._prepare_views(Xs, As, bundle)

            # rank-safe shared latent dimension
            m = self.embedding_dim
            for H in Hs:
                m = min(m, min(H.shape) - 1)
            if m < self.embedding_dim:
                logger.warning(
                    "embedding_dim=%d exceeds a view's rank — clamping to %d.",
                    self.embedding_dim, m,
                )

            state = FitState(
                Hs=Hs,
                n_clusters=self.n_clusters,
                embedding_dim=m,
                random_state=self.random_state,
            )
            info = bundle.optimizer.run(state, bundle)

            memory_peak = None
            if mem_started:
                memory_peak = int(tracemalloc.get_traced_memory()[1])
        finally:
            if mem_started:
                tracemalloc.stop()

        self._collect_results(
            state, bundle, info, used_gammas,
            training_time=time.perf_counter() - t0,
            memory_peak=memory_peak,
        )
        return self

    def fit_predict(self, Xs, As=None):
        """Fit the model and return the hard cluster labels."""
        return self.fit(Xs, As).labels_

    # ── result collection ─────────────────────────────────────────

    def _hyperparameters(self) -> dict:
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_") and not k.endswith("_")
        }

    def _collect_results(self, state, bundle, info, used_gammas,
                         training_time, memory_peak) -> None:
        weights = state.view_node_weights
        is_attention = bundle.weighting.provides_attention
        report_gammas = (
            self.controlled_propagation or self.graph_propagation is not None
        )

        # backward-compatible attributes
        self.labels_ = np.asarray(state.G, dtype=np.int32)
        self.centroids_ = np.asarray(state.F)
        self.embedding_ = np.asarray(state.consensus)
        self.responsibilities_ = (
            state.R if bundle.clustering.provides_soft else None
        )
        self.attention_ = (
            np.stack(weights, axis=0) if is_attention else None
        )
        # constant weight vectors (global strategies) report the exact
        # weight; node-varying weights (attention) report their mean
        self.alphas_ = np.array([
            float(b[0]) if np.ptp(b) == 0 else float(b.mean())
            for b in weights
        ])
        self.gammas_ = used_gammas if report_gammas else None
        self.loss_history_ = np.asarray(info.loss_history)

        # structured result object
        self.results_ = LMGECResult(
            labels=self.labels_,
            embedding=self.embedding_,
            centroids=self.centroids_,
            view_weights=self.alphas_,
            attention_weights=self.attention_,
            cluster_probabilities=self.responsibilities_,
            reconstruction_errors=[np.asarray(e) for e in state.errors],
            loss_history=self.loss_history_,
            convergence=info.as_dict(),
            view_projections=[np.asarray(W) for W in state.Ws],
            gammas=self.gammas_,
            components=bundle.describe(),
            params=self._hyperparameters(),
            training_time=training_time,
            memory_usage=memory_peak,
        )
        logger.debug(
            "LMGECplus fit done: %d iterations, final loss %.6g, %.2fs",
            info.n_iter, info.final_loss, training_time,
        )
