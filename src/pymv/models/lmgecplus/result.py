"""
Structured fit result for the modular LMGEC framework.

Instead of scattering intermediate quantities across model attributes,
every useful output of a ``fit`` is collected in one
:class:`LMGECResult` — embeddings, weights, losses, convergence and
resource usage — so visualization / analysis code has a single, stable
surface::

    model.fit(Xs, As)
    res = model.results_
    res.embedding, res.labels, res.view_weights, res.loss_history, ...

(The model still populates the historical ``labels_`` / ``embedding_``
/ … attributes for backward compatibility.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LMGECResult:
    """Everything produced by one ``LMGECplus.fit`` call.

    Attributes
    ----------
    labels : np.ndarray of shape (n,)
        Hard cluster assignment per sample.
    embedding : np.ndarray of shape (n, m)
        Final consensus embedding.
    centroids : np.ndarray of shape (k, m)
        Final cluster centroids.
    view_weights : np.ndarray of shape (n_views,)
        Mean fusion weight per view (α for global strategies, mean
        node attention otherwise).
    attention_weights : np.ndarray of shape (n_views, n) or None
        Per-node attention weights — only when the view weighting is
        attention-based.
    cluster_probabilities : np.ndarray of shape (n, k) or None
        Soft assignments — only for soft clustering heads.
    reconstruction_errors : list[np.ndarray]
        Final per-view per-node squared reconstruction errors.
    loss_history : np.ndarray
        Monitored loss per iteration.
    convergence : dict
        ``{"converged": bool, "n_iter": int, "final_loss": float}``.
    view_projections : list[np.ndarray]
        Final per-view orthonormal projections ``W_v``.
    gammas : list[float] or None
        Effective propagation strength per view (``None`` when the
        legacy full-propagation path was used).
    components : dict
        Human-readable description of the selected modules.
    params : dict
        Constructor hyper-parameters of the model.
    training_time : float
        Wall-clock ``fit`` duration in seconds.
    memory_usage : int or None
        Peak Python heap allocation during ``fit`` in bytes
        (``None`` when memory tracking is disabled).
    """

    labels: np.ndarray
    embedding: np.ndarray
    centroids: np.ndarray
    view_weights: np.ndarray
    attention_weights: np.ndarray | None = None
    cluster_probabilities: np.ndarray | None = None
    reconstruction_errors: list = field(default_factory=list)
    loss_history: np.ndarray = field(default_factory=lambda: np.array([]))
    convergence: dict = field(default_factory=dict)
    view_projections: list = field(default_factory=list)
    gammas: list | None = None
    components: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    training_time: float = 0.0
    memory_usage: int | None = None

    # ── convenience ───────────────────────────────────────────────

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def n_views(self) -> int:
        return int(len(self.view_weights))

    def to_dict(self) -> dict:
        """Compact, JSON-friendly summary (no large arrays).

        Suited for experiment logging / tracking.
        """
        return {
            "n_samples": int(self.labels.shape[0]),
            "n_clusters": self.n_clusters,
            "n_views": self.n_views,
            "view_weights": [float(a) for a in self.view_weights],
            "gammas": self.gammas,
            "loss_history": [float(x) for x in self.loss_history],
            "convergence": dict(self.convergence),
            "components": dict(self.components),
            "params": {k: str(v) for k, v in self.params.items()},
            "training_time": round(float(self.training_time), 4),
            "memory_usage": self.memory_usage,
        }

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        conv = self.convergence or {}
        lines = [
            "LMGECResult",
            f"  samples/views/clusters : "
            f"{self.labels.shape[0]} / {self.n_views} / {self.n_clusters}",
            f"  embedding              : {self.embedding.shape}",
            f"  view weights           : "
            f"[{', '.join(f'{a:.3f}' for a in self.view_weights)}]",
            f"  iterations             : {conv.get('n_iter', '?')} "
            f"(converged={conv.get('converged', '?')})",
            f"  final loss             : {conv.get('final_loss', float('nan')):.6g}",
            f"  training time          : {self.training_time:.3f}s",
        ]
        if self.memory_usage is not None:
            lines.append(
                f"  peak memory            : {self.memory_usage / 1e6:.1f} MB")
        if self.attention_weights is not None:
            lines.append(
                f"  attention weights      : {self.attention_weights.shape}")
        if self.cluster_probabilities is not None:
            lines.append(
                f"  cluster probabilities  : "
                f"{self.cluster_probabilities.shape}")
        for name, desc in (self.components or {}).items():
            lines.append(f"  {name:<22} : {desc}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        conv = self.convergence or {}
        return (
            f"LMGECResult(n={self.labels.shape[0]}, k={self.n_clusters}, "
            f"views={self.n_views}, n_iter={conv.get('n_iter', '?')}, "
            f"time={self.training_time:.2f}s)"
        )
