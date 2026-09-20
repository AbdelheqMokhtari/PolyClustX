"""Simple Contrastive Multi-view Clustering with data-level fusion.

This module adapts the reference SCM implementation to pymv's ``BaseModel``
contract.  Views are concatenated, augmented with missing-view masks and
Gaussian noise, and encoded with a shared autoencoder/projector.  The two
augmented representations are trained with the symmetric feature-level
contrastive objective proposed by SCM; K-means produces the final clusters.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.cluster import KMeans

from pymv.core import BaseModel
from pymv.models.registry import register_model

logger = logging.getLogger(__name__)


class _SCMNetwork(tf.keras.Model):
    """Encoder/decoder and feature projection from the SCM reference code."""

    def __init__(self, input_dim, feature_dim, projection_dim, hidden_dims):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            *[
                layer
                for width in hidden_dims
                for layer in (tf.keras.layers.Dense(width), tf.keras.layers.ReLU())
            ],
            tf.keras.layers.Dense(feature_dim),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(feature_dim,)),
            *[
                layer
                for width in reversed(hidden_dims)
                for layer in (tf.keras.layers.Dense(width), tf.keras.layers.ReLU())
            ],
            tf.keras.layers.Dense(input_dim),
        ])
        self.projector = tf.keras.layers.Dense(projection_dim)
        self.cluster_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.Softmax(),
        ])

    def call(self, x, training=False):
        h = self.encoder(x, training=training)
        z = tf.math.l2_normalize(self.projector(h), axis=1)
        q = self.cluster_head(z, training=training)
        xr = self.decoder(h, training=training)
        return xr, h, z, q


@register_model("scm", description="Contrastive clustering with concatenated feature views and a shared encoder.",
                paper="https://www.ijcai.org/proceedings/2024/519", uses_graphs=False, defaults={
    "embedding_dim": 256,
    "projection_dim": 128,
    "temperature": 0.5,
    "learning_rate": 0.0003,
    "weight_decay": 0.0,
    "batch_size": 256,
    "miss_rate": 0.25,
    "noise_rate": 0.25,
    "gaussian_noise": 0.4,
    "reconstruction_weight": 0.0,
    "hidden_dims": (500, 500, 2000),
    "preprocess": True,
    "max_iter": 50,
    "random_state": None,
})
class SCM(BaseModel):
    """Simple Contrastive Multi-view Clustering (SCM).

    Parameters follow the authors' reference implementation.  ``embedding_dim``
    is the autoencoder feature size and ``projection_dim`` is the consensus
    representation clustered after training.  ``preprocess=True`` independently
    min-max scales each feature column; adjacency matrices are accepted for
    framework compatibility but are not used by this feature-fusion algorithm.
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.5,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        miss_rate: float = 0.25,
        noise_rate: float = 0.25,
        gaussian_noise: float = 0.4,
        reconstruction_weight: float = 0.0,
        hidden_dims: tuple[int, ...] = (500, 500, 2000),
        preprocess: bool = True,
        max_iter: int = 50,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.miss_rate = miss_rate
        self.noise_rate = noise_rate
        self.gaussian_noise = gaussian_noise
        self.reconstruction_weight = reconstruction_weight
        self.hidden_dims = tuple(hidden_dims)
        self.preprocess = preprocess
        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_ = None
        self.embedding_ = None
        self.centroids_ = None
        self.loss_history_ = None
        self.embedding_dim_ = None
        self.model_ = None

    @property
    def name(self) -> str:
        return getattr(self, "_name", self.__class__.__name__)

    @name.setter
    def name(self, value: str):
        self._name = value

    def _validate(self, Xs):
        if not isinstance(Xs, (list, tuple)) or not Xs:
            raise ValueError("Xs must be a non-empty list of feature matrices.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.embedding_dim < 1 or self.projection_dim < 1:
            raise ValueError("embedding dimensions must be positive.")
        if self.batch_size < 2 or self.max_iter < 1:
            raise ValueError("batch_size must be >= 2 and max_iter must be positive.")
        for name, value in (("miss_rate", self.miss_rate), ("noise_rate", self.noise_rate)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        arrays = []
        n_samples = None
        for view in Xs:
            if sp.issparse(view):
                view = view.toarray()
            view = np.asarray(view, dtype=np.float32)
            if view.ndim != 2:
                raise ValueError("Every view must be a two-dimensional matrix.")
            if n_samples is None:
                n_samples = view.shape[0]
            elif view.shape[0] != n_samples:
                raise ValueError("All views must contain the same number of samples.")
            if not np.all(np.isfinite(view)):
                raise ValueError("Xs contains NaN or infinite values.")
            if self.preprocess:
                low = view.min(axis=0, keepdims=True)
                span = view.max(axis=0, keepdims=True) - low
                view = (view - low) / np.where(span > 0, span, 1.0)
            arrays.append(view)
        if n_samples < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        return arrays

    @staticmethod
    def _contrastive_loss(left, right, temperature):
        """NT-Xent loss used by ``Loss.forward_feature`` in SCM."""
        h = tf.concat((left, right), axis=0)
        logits = tf.matmul(h, h, transpose_b=True) / temperature
        n = tf.shape(left)[0]
        size = 2 * n
        positives = tf.concat((tf.range(n, size), tf.range(0, n)), axis=0)
        diagonal = tf.eye(size, dtype=tf.bool)
        logits = tf.where(diagonal, tf.cast(-1e9, logits.dtype), logits)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=positives, logits=logits
            )
        )

    def _augment(self, batch, view_widths, rng):
        rows = batch.shape[0]
        n_views = len(view_widths)
        view_mask = np.ones((rows, n_views), dtype=np.float32)
        affected = rng.choice(
            rows, size=int(rows * self.miss_rate), replace=False
        )
        for row in affected:
            # SCM keeps at least one view and masks at least one view.
            if n_views > 1:
                keep = rng.integers(1, n_views)
                kept = rng.choice(n_views, size=keep, replace=False)
                view_mask[row] = 0
                view_mask[row, kept] = 1
        feature_mask = np.repeat(view_mask, view_widths, axis=1)
        masked = batch * feature_mask

        noisy = batch.copy()
        noise_rows = rng.random(rows) < self.noise_rate
        noisy[noise_rows] += rng.normal(
            0.0, self.gaussian_noise, size=noisy[noise_rows].shape
        ).astype(np.float32)
        return masked, noisy

    def fit(self, Xs, As=None, *, observed_mask=None):
        """Fit SCM to a list of aligned feature views."""
        if observed_mask is not None and not np.asarray(observed_mask, dtype=bool).all():
            raise ValueError("SCM does not support missing views; explicitly impute first")
        del As  # SCM is a feature-level fusion method.
        views = self._validate(Xs)
        fused = np.concatenate(views, axis=1).astype(np.float32, copy=False)
        rng = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            tf.keras.utils.set_random_seed(self.random_state)

        model = _SCMNetwork(
            fused.shape[1], self.embedding_dim, self.projection_dim, self.hidden_dims
        )
        model(tf.convert_to_tensor(fused[:1]), training=False)
        if self.weight_decay:
            adamw = getattr(tf.keras.optimizers, "AdamW", None)
            if adamw is None:  # TensorFlow 2.10 compatibility
                adamw = tf.keras.optimizers.experimental.AdamW
            optimizer = adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        losses = []
        batch_size = min(self.batch_size, fused.shape[0])
        indices = np.arange(fused.shape[0])

        for _ in range(self.max_iter):
            rng.shuffle(indices)
            epoch_losses = []
            for start in range(0, fused.shape[0], batch_size):
                selected = indices[start:start + batch_size]
                if selected.size < 2:
                    continue
                clean = fused[selected]
                masked, noisy = self._augment(clean, [v.shape[1] for v in views], rng)
                clean_t = tf.convert_to_tensor(clean)
                masked_t = tf.convert_to_tensor(masked)
                noisy_t = tf.convert_to_tensor(noisy)
                with tf.GradientTape() as tape:
                    clean_xr, _, _, _ = model(clean_t, training=True)
                    mask_xr, _, mask_z, _ = model(masked_t, training=True)
                    noise_xr, _, noise_z, _ = model(noisy_t, training=True)
                    contrastive = self._contrastive_loss(noise_z, mask_z, self.temperature)
                    contrastive += self._contrastive_loss(mask_z, noise_z, self.temperature)
                    reconstruction = (
                        tf.reduce_mean(tf.square(clean_t - clean_xr))
                        + tf.reduce_mean(tf.square(clean_t - mask_xr))
                        + tf.reduce_mean(tf.square(clean_t - noise_xr))
                    )
                    loss = contrastive + self.reconstruction_weight * reconstruction
                gradients = tape.gradient(loss, model.trainable_variables)
                trainable = (
                    (gradient, variable)
                    for gradient, variable in zip(gradients, model.trainable_variables)
                    if gradient is not None
                )
                optimizer.apply_gradients(trainable)
                epoch_losses.append(float(loss.numpy()))
            losses.append(float(np.mean(epoch_losses)))

        _, _, embedding, _ = model(tf.convert_to_tensor(fused), training=False)
        self.embedding_ = embedding.numpy()
        clustering = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        ).fit(self.embedding_)
        self.labels_ = clustering.labels_.astype(np.int32)
        self.centroids_ = clustering.cluster_centers_
        self.inertia_ = float(clustering.inertia_)
        self.loss_history_ = np.asarray(losses)
        self.embedding_dim_ = self.projection_dim
        self.model_ = model
        return self
