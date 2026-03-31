"""
LMGEC — Linear Multi-view Graph Embedding and Clustering.

"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from pymv.core import BaseModel
from pymv.models.registry import register_model
from pymv.preprocessing import (
    get_propagated_features,
    preprocess_features,
)

logger = logging.getLogger(__name__)


# GPU memory configuration — runs once at import time

def _configure_gpu():
    """Enable memory growth on all visible GPUs.

    Prevents TensorFlow from pre-allocating the entire GPU memory,
    which causes OOM on devices with limited VRAM (e.g. laptops).
    Must run before any TF operation.
    """
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # already initialized skip silently


_configure_gpu()

# update rules  F, W, And G

def _update_rule_F(XW, G, K):
    return tf.math.unsorted_segment_mean(XW, G, num_segments=K)


def _update_rule_W(X, F, G):
    F_g = tf.gather(F, G)
    M = tf.linalg.matmul(X, F_g, transpose_a=True)
    s, u, v = tf.linalg.svd(M)
    W = tf.linalg.matmul(u, v, adjoint_b=True)
    return W


def _update_rule_G(XW, F):
    diff = tf.expand_dims(XW, 1) - tf.expand_dims(F, 0)
    dists = tf.reduce_sum(tf.square(diff), axis=-1)
    return tf.cast(tf.argmin(dists, axis=1), tf.int32)


# method to initialise W, G, and F

def _init_W(X, n_components):
    n_components = min(n_components, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    return svd.components_.T


def _init_G_F(XW, k):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(XW)
    return km.labels_.astype(np.int32), km.cluster_centers_


# Training loop (one per number of views — traced separately)

def _make_train_loop(n_views):
    """
    Build a tf.function-traced training loop for exactly *n_views*
    views.  Python-level indexing (``Xs[0]``, ``Xs[1]``, …) is
    unrolled at trace time so TensorFlow never sees a Python list
    indexed by a symbolic tensor.
    """

    @tf.function
    def _train_loop(Xs, F, G, alphas, k, max_iter):
        m = tf.shape(F)[1]
        n = tf.shape(Xs[0])[0]
        losses = tf.TensorArray(dtype=tf.float64, size=max_iter)

        for t in tf.range(max_iter):
            XW_consensus = tf.zeros((n, m), dtype=tf.float64)
            loss_total = tf.constant(0.0, dtype=tf.float64)

            # --- unrolled at trace time (Python for over a fixed range) ---
            for v in range(n_views):
                Wv = _update_rule_W(Xs[v], F, G)
                XWv = tf.linalg.matmul(Xs[v], Wv)
                XW_consensus = XW_consensus + alphas[v] * XWv

                recon = tf.gather(
                    tf.linalg.matmul(F, Wv, transpose_b=True), G
                )
                loss_total = loss_total + tf.linalg.norm(Xs[v] - recon)

            G = _update_rule_G(XW_consensus, F)
            F = _update_rule_F(XW_consensus, G, k)
            losses = losses.write(t, loss_total)

        return G, F, XW_consensus, losses.stack()

    return _train_loop


# Cache so we don't re-trace for the same view count
_train_loop_cache: dict[int, callable] = {}


def _get_train_loop(n_views):
    if n_views not in _train_loop_cache:
        _train_loop_cache[n_views] = _make_train_loop(n_views)
    return _train_loop_cache[n_views]


# Public model class

@register_model("lmgec", defaults={
    "embedding_dim": 16,
    "temperature": 1.0,
    "beta": 1.0,
    "tf_idf": False,
    "center": False,
    "scale": False,
    "preprocess": True,
    "max_iter": 30,
})
class LMGEC(BaseModel):
    """Linear Multi-view Graph Embedding Clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.

    embedding_dim : int
        Latent dimension *m* for the per-view projections W.

    temperature : float, default=1.0
        Softmax temperature for computing per-view weights α.
        Lower → sharper weighting; higher → more uniform.

    beta : float or list[float], default=1.0
        Self-loop weight for graph propagation.  A single float
        applies the same value to every view.  A list sets a
        per-view value.  **Ignored when ``preprocess=False``.**

    tf_idf : bool or list[bool], default=False
        Feature normalization mode.  ``True`` → TF-IDF + L2;
        ``False`` → plain L2.  A list sets a per-view choice.
        **Ignored when ``preprocess=False``.**

    center : bool, default=False
        Center features (subtract column mean) after propagation.
        **Ignored when ``preprocess=False``.**

    scale : bool, default=False
        Scale features (unit variance) after propagation.
        **Ignored when ``preprocess=False``.**

    preprocess : bool, default=True
        Master switch for internal preprocessing.

        * ``True``  — the model normalizes features, builds the
          propagation matrix, computes H = S·X, and optionally
          centers/scales.  This is the default and matches the
          original LMGEC paper.
        * ``False`` — **all** internal preprocessing is skipped.
          The model treats each element of *Xs* as an
          already-prepared view and feeds it directly into the
          optimisation loop.  In this mode *As* is ignored even
          if provided, and *beta*, *tf_idf*, *center*, *scale*
          have no effect.

    max_iter : int, default=30
        Maximum number of alternating-optimisation iterations.

    Examples
    --------
    **Let the model preprocess (default):**

    >>> model = LMGEC(n_clusters=3, embedding_dim=4)
    >>> model.fit(Xs, As)          # raw features + adjacencies
    >>> model.labels_
    array([0, 2, 1, ...])

    **Per-view preprocessing options:**

    >>> model = LMGEC(
    ...     n_clusters=3,
    ...     embedding_dim=4,
    ...     tf_idf=[True, False],  # TF-IDF on view 0, L2 on view 1
    ...     beta=[1.0, 0.5],       # different self-loop weights
    ... )
    >>> model.fit(Xs, As)

    **User handles preprocessing externally:**

    >>> from pymv.preprocessing import get_propagated_features
    >>> H0 = get_propagated_features(X0, A0, beta=1.0, tf_idf=True)
    >>> H1 = get_propagated_features(X1, A1, beta=0.5, tf_idf=False)
    >>> model = LMGEC(n_clusters=3, embedding_dim=4, preprocess=False)
    >>> model.fit([H0, H1])        # As not needed

    **No graph at all (feature-only views):**

    >>> model = LMGEC(n_clusters=3, embedding_dim=4, preprocess=False)
    >>> model.fit([X1, X2])        # just feature matrices
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 1.0,
        beta: float | list[float] = 1.0,
        tf_idf: bool | list[bool] = False,
        center: bool = False,
        scale: bool = False,
        preprocess: bool = True,
        max_iter: int = 30,
    ):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.beta = beta
        self.tf_idf = tf_idf
        self.center = center
        self.scale = scale
        self.preprocess = preprocess
        self.max_iter = max_iter

        # Filled after fit()
        self.labels_ = None
        self.centroids_ = None
        self.embedding_ = None
        self.loss_history_ = None

    # helpers

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
                    f"len({param_name})={len(param)} but got "
                    f"{n_views} views."
                )
            return param
        return [param] * n_views

    # main entry point

    def fit(self, Xs, As=None):
        """Fit the model.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices — one per view.

            * When ``preprocess=True``: raw features; each will be
              normalized and (if *As* is given) propagated through
              its adjacency.
            * When ``preprocess=False``: already-prepared views
              that go directly into the optimisation loop.

        As : list[np.ndarray | sp.spmatrix] or None, default=None
            Adjacency matrices — one per view (or one shared).

            * Ignored entirely when ``preprocess=False``.
            * When ``preprocess=True`` and ``As is None``: features
              are normalized but **not** propagated (no graph).
            * When ``preprocess=True`` and ``As`` has fewer entries
              than ``Xs``: the last adjacency is reused for the
              remaining views.

        Returns
        -------
        self
            The fitted model with ``labels_``, ``centroids_``,
            ``embedding_``, and ``loss_history_`` attributes set.
        """

        n_views = len(Xs)

        # Prepare H views

        if self.preprocess:
            betas = self._resolve_per_view(self.beta, n_views, "beta")
            tf_idfs = self._resolve_per_view(self.tf_idf, n_views, "tf_idf")

            # Build adjacency list 
            if As is None:
                adj_list = [None] * n_views
            else:
                adj_list = list(As)
                # Reuse last adjacency if fewer were provided
                while len(adj_list) < n_views:
                    adj_list.append(adj_list[-1])

            Hs = []
            for v in range(n_views):
                H = get_propagated_features(
                    Xs[v],
                    adj=adj_list[v],
                    beta=betas[v],
                    tf_idf=tf_idfs[v],
                )
                Hs.append(H)

            # centering / scaling
            if self.center or self.scale:
                Hs = preprocess_features(
                    Hs, center=self.center, scale=self.scale
                )

        else:
            # User already prepared the views — use as-is
            Hs = list(Xs)

        # Ensure dense float64 arrays

        for i in range(len(Hs)):
            if sp.issparse(Hs[i]):
                Hs[i] = Hs[i].toarray()
            Hs[i] = np.asarray(Hs[i], dtype=np.float64)

        # Initialisation

        m = self.embedding_dim
        k = self.n_clusters

        alphas = np.zeros(n_views, dtype=np.float64)
        XW_consensus = np.zeros((Hs[0].shape[0], m), dtype=np.float64)

        for v in range(n_views):
            Wv = _init_W(Hs[v], m)
            XWv = Hs[v] @ Wv
            Gv, Fv = _init_G_F(XWv, k)
            inertia = np.linalg.norm(XWv - Fv[Gv])
            alphas[v] = np.exp(-inertia / self.temperature)
            XW_consensus += alphas[v] * XWv

        alpha_sum = alphas.sum()
        if alpha_sum > 0:
            XW_consensus /= alpha_sum
        else:
            XW_consensus /= n_views

        G_init, F_init = _init_G_F(XW_consensus, k)

        # run training loop (with automatic GPU → CPU fallback on OOM)

        alphas_norm = alphas / (alpha_sum if alpha_sum > 0 else n_views)
        train_loop = _get_train_loop(n_views)

        def _run_on_device(device: str):
            """Create tensors and run the loop on the given device."""
            with tf.device(device):
                _Hs = tuple(tf.constant(H, dtype=tf.float64) for H in Hs)
                _F = tf.constant(F_init, dtype=tf.float64)
                _G = tf.constant(G_init, dtype=tf.int32)
                _a = tf.constant(alphas_norm, dtype=tf.float64)
                return train_loop(_Hs, _F, _G, _a, k, self.max_iter)

        has_gpu = bool(tf.config.list_physical_devices("GPU"))

        if has_gpu:
            try:
                G_out, F_out, XW_out, losses = _run_on_device("/GPU:0")
            except (tf.errors.ResourceExhaustedError,
                    tf.errors.InternalError):
                logger.warning(
                    "GPU out of memory — falling back to CPU. "
                    "This run will be slower but will complete."
                )
                tf.keras.backend.clear_session()
                G_out, F_out, XW_out, losses = _run_on_device("/CPU:0")
        else:
            G_out, F_out, XW_out, losses = _run_on_device("/CPU:0")

        # Store results & free GPU memory

        self.labels_ = G_out.numpy()
        self.centroids_ = F_out.numpy()
        self.embedding_ = XW_out.numpy()
        self.loss_history_ = losses.numpy()

        # Release all TF tensors so GPU VRAM is freed for the next call
        del G_out, F_out, XW_out, losses
        tf.keras.backend.clear_session()

        return self

    def fit_predict(self, Xs, As=None):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        Xs : list[np.ndarray]
            Feature matrices (see :meth:`fit`).

        As : list or None, default=None
            Adjacency matrices (see :meth:`fit`).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster label for each sample.
        """
        return self.fit(Xs, As).labels_