
"""
LMGEC: Linear Multi-view Graph Embedding Clustering

Implementation of the LMGEC algorithm for multi-view clustering
with optional graph-based feature propagation.
"""

import warnings
import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from pymv.preprocessing import (
    get_propagated_features,
    preprocess_features
)
from pymv.core import BaseModel


# ================================================================
# Update rules
# ================================================================

def _update_rule_F(XW, G, K):
    """Update cluster centroids."""
    return tf.math.unsorted_segment_mean(XW, G, num_segments=K)


def _update_rule_W(X, F, G):
    """Update projection matrix."""
    _, U, V = tf.linalg.svd(
        tf.transpose(X) @ tf.gather(F, G),
        full_matrices=False
    )
    return U @ tf.transpose(V)


def _update_rule_G(XW, F):
    """Assign cluster labels."""
    centroids_expanded = F[:, None, ...]
    distances = tf.reduce_mean(
        tf.math.squared_difference(XW, centroids_expanded),
        axis=2
    )

    return tf.math.argmin(
        distances,
        axis=0,
        output_type=tf.int32
    )


# ================================================================
# Training loop
# ================================================================

@tf.function
def _train_loop(Xs, F, G, alphas, k, max_iter):

    n_views = len(Xs)

    losses = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True
    )

    XW_consensus = tf.zeros(
        (tf.shape(Xs[0])[0], tf.shape(F)[1]),
        dtype=tf.float32
    )

    for i in tf.range(max_iter):

        loss = tf.constant(0.0, dtype=tf.float32)

        XW_consensus = tf.zeros_like(XW_consensus)

        for v in range(n_views):

            Wv = _update_rule_W(Xs[v], F, G)

            XWv = tf.linalg.matmul(Xs[v], Wv)

            XW_consensus += alphas[v] * XWv

            loss_v = tf.linalg.norm(
                Xs[v] - tf.gather(F @ tf.transpose(Wv), G)
            )

            loss += alphas[v] * loss_v

        G = _update_rule_G(XW_consensus, F)

        F = _update_rule_F(XW_consensus, G, k)

        losses = losses.write(i, loss)

    return G, F, XW_consensus, losses.stack()


# ================================================================
# LMGEC model
# ================================================================

class LMGEC(BaseModel):
    """
    Linear Multi-view Graph Embedding Clustering (LMGEC).
    """

    def __init__(
        self,
        n_clusters,
        embedding_dim,
        temperature=10.0,
        max_iter=30,
        tol=1e-4,
        beta=1.0,
        center_data=True,
        tf_idf=False,
        scale=False,
    ):

        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.center_data = center_data
        self.tf_idf = tf_idf
        self.scale = scale

        # learned attributes
        self.labels_ = None
        self.embedding_ = None
        self.centroids_ = None
        self.alphas_ = None
        self.loss_history_ = None

    # ------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------

    def _init_W(self, X):

        n_features = X.shape[1]

        if self.embedding_dim >= n_features:

            warnings.warn(
                f"View has {n_features} features but embedding_dim "
                f"is {self.embedding_dim}. Skipping SVD reduction.",
                UserWarning
            )

            return np.eye(n_features)

        svd = TruncatedSVD(
            n_components=self.embedding_dim
        ).fit(X)

        return svd.components_.T

    def _init_G_F(self, XW):

        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=10
        ).fit(XW)

        G = km.labels_.astype(np.int32)
        F = km.cluster_centers_

        return G, F

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def fit(self, Xs, adjs=None):

        if adjs is None:
            adjs = [None] * len(Xs)

        if len(Xs) != len(adjs):
            raise ValueError(
                "Number of feature matrices must match adjacency matrices."
            )

        # --------------------------------------------------------
        # Graph propagation
        # --------------------------------------------------------

        propagated_views = []

        for X, A in zip(Xs, adjs):

            H = get_propagated_features(
                features=X,
                adj=A,
                beta=self.beta,
                tf_idf=self.tf_idf,
            )

            propagated_views.append(H)

        Xs = preprocess_features(
            propagated_views,
            center=self.center_data,
            scale=self.scale
        )

        n_views = len(Xs)

        # --------------------------------------------------------
        # Initialization
        # --------------------------------------------------------

        alphas = np.zeros(n_views)

        XW_consensus = 0

        for v in range(n_views):

            Wv = self._init_W(Xs[v])

            XWv = Xs[v] @ Wv

            Gv, Fv = self._init_G_F(XWv)

            inertia = np.linalg.norm(XWv - Fv[Gv])

            alphas[v] = np.exp(-inertia / self.temperature)

            XW_consensus += alphas[v] * XWv

        alpha_sum = alphas.sum()

        if alpha_sum == 0:
            alphas[:] = 1.0 / n_views
            alpha_sum = 1.0

        XW_consensus /= alpha_sum

        G, F = self._init_G_F(XW_consensus)

        # --------------------------------------------------------
        # TensorFlow training
        # --------------------------------------------------------

        Xs_tf = [
            tf.convert_to_tensor(X, dtype=tf.float32)
            for X in Xs
        ]

        F_tf = tf.convert_to_tensor(F, dtype=tf.float32)
        G_tf = tf.convert_to_tensor(G, dtype=tf.int32)
        alphas_tf = tf.convert_to_tensor(alphas, dtype=tf.float32)

        G, F, XW_consensus, loss_history = _train_loop(
            Xs_tf,
            F_tf,
            G_tf,
            alphas_tf,
            self.n_clusters,
            self.max_iter
        )

        # --------------------------------------------------------
        # Store results
        # --------------------------------------------------------

        self.labels_ = G.numpy()
        self.embedding_ = XW_consensus.numpy()
        self.centroids_ = F.numpy()
        self.alphas_ = alphas
        self.loss_history_ = loss_history.numpy()

        return self