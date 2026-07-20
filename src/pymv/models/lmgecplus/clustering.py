"""
Clustering heads — assignment + centroid strategies.

A :class:`~pymv.models.lmgecplus.interfaces.ClusteringHead` owns three
decisions: the *reconstruction target* the views regress onto
(``F[G]`` for hard heads, ``R F`` for soft heads), the *assignment*
rule, and the *centroid* update.  All heads share the same K-Means
initialisation so heads remain comparable in ablations.

Heads
-----
* ``"kmeans"`` — hard nearest-centroid assignment (original LMGEC).
* ``"soft"``   — soft K-Means (softmax responsibilities).
* ``"gmm"``    — Gaussian Mixture Model responsibilities (EM refit on
  the consensus embedding each iteration, warm-started at the current
  centroids).
"""

from __future__ import annotations

import numpy as np

from . import components as C
from .factory import register_component
from .interfaces import ClusteringHead, FitState


@register_component("clustering", "kmeans", aliases=("hard",))
class HardKMeansHead(ClusteringHead):
    """Hard K-Means head — the original LMGEC clustering.

    Assignment is the nearest centroid; centroids are (optionally
    node-weighted) per-cluster means of the consensus embedding.
    """

    provides_soft = False

    def shared_target(self, state: FitState):
        return state.F[state.G]

    def refit(self, state: FitState, node_w):
        G = C.hard_assign(state.consensus, state.F)
        F = C.centroids_from_Z(state.consensus, G, state.n_clusters,
                               node_w=node_w)
        return G, F, None


@register_component("clustering", "soft", aliases=("soft_kmeans",))
class SoftKMeansHead(ClusteringHead):
    r"""Soft K-Means head — softmax responsibilities.

    .. math::
        R_{ij} \propto \exp(-\beta \|z_i - f_j\|^2)

    The reconstruction target becomes the responsibility-weighted
    centroid mix ``R F``.

    Parameters
    ----------
    beta : float, default=50.0
        Assignment sharpness (large β → near-hard).
    """

    provides_soft = True

    def __init__(self, beta: float = 50.0):
        self.beta = beta

    def shared_target(self, state: FitState):
        if state.R is None:  # first iteration: only hard init exists
            return state.F[state.G]
        return state.R @ state.F

    def refit(self, state: FitState, node_w):
        R = C.soft_assign(state.consensus, state.F, beta=self.beta)
        F = C.centroids_from_Z(state.consensus, R, state.n_clusters,
                               node_w=node_w)
        G = np.argmax(R, axis=1).astype(np.int32)
        return G, F, R


@register_component("clustering", "gmm")
class GMMHead(ClusteringHead):
    """Gaussian Mixture Model head.

    Each outer iteration re-fits a GMM on the consensus embedding,
    warm-started at the current centroids; responsibilities come from
    the E-step (``predict_proba``).  IRLS node weights are not folded
    into the M-step (the GMM has its own weighting via
    responsibilities), which is the main documented difference from
    the K-Means heads.

    Parameters
    ----------
    covariance_type : {"diag", "full", "tied", "spherical"}, default="diag"
        Component covariance structure.
    em_iter : int, default=10
        EM iterations per outer refit.
    reg_covar : float, default=1e-6
        Covariance regularisation (numerical stability).
    """

    provides_soft = True

    def __init__(self, covariance_type: str = "diag", em_iter: int = 10,
                 reg_covar: float = 1e-6):
        self.covariance_type = covariance_type
        self.em_iter = em_iter
        self.reg_covar = reg_covar

    def shared_target(self, state: FitState):
        if state.R is None:
            return state.F[state.G]
        return state.R @ state.F

    def refit(self, state: FitState, node_w):
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=state.n_clusters,
            covariance_type=self.covariance_type,
            max_iter=self.em_iter,
            n_init=1,
            means_init=state.F,
            reg_covar=self.reg_covar,
            random_state=state.random_state,
        )
        gm.fit(state.consensus)
        R = gm.predict_proba(state.consensus)
        G = np.argmax(R, axis=1).astype(np.int32)
        return G, gm.means_, R
