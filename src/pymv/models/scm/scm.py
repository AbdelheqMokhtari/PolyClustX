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
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from pymv.core import BaseModel
from pymv.models._torch_utils import initialization_seed, resolve_device
from pymv.models.registry import register_model
from pymv.progress import iterations

logger = logging.getLogger(__name__)


class _SCMNetwork(nn.Module):
    """Encoder/decoder and feature projection from the SCM reference code."""

    def __init__(self, input_dim, feature_dim, projection_dim, hidden_dims):
        super().__init__()
        def mlp(widths):
            layers = []
            for index, (left, right) in enumerate(zip(widths, widths[1:])):
                layers.append(nn.Linear(left, right, device="cpu", dtype=torch.float32))
                if index < len(widths) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.encoder = mlp([input_dim, *hidden_dims, feature_dim])
        self.decoder = mlp([feature_dim, *reversed(hidden_dims), input_dim])
        self.projector = nn.Linear(feature_dim, projection_dim, device="cpu", dtype=torch.float32)
        # The reference's 64-way head is unused by the feature-clustering route.
        self.cluster_head = nn.Sequential(
            nn.Linear(projection_dim, 64, device="cpu", dtype=torch.float32), nn.Softmax(dim=1))

    def encode(self, x):
        h = self.encoder(x)
        return h, F.normalize(self.projector(h), dim=1)

    def forward(self, x):
        h, z = self.encode(x)
        q = self.cluster_head(z)
        xr = self.decoder(h)
        return xr, h, z, q


@register_model("scm", description="Contrastive clustering with concatenated feature views and a shared encoder.",
                implementation="https://github.com/SubmissionsIn/SCM",
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
    "device": "auto",
})
class SCM(BaseModel):
    """Simple Contrastive Multi-view Clustering (SCM).

    Parameters follow the authors' reference implementation.  ``embedding_dim``
    is the autoencoder feature size and ``projection_dim`` is the consensus
    representation clustered after training.  ``preprocess=True`` independently
    min-max scales each feature column; adjacency matrices are accepted for
    framework compatibility but are not used by this feature-fusion algorithm.
    ``max_iter`` counts full epochs, not the reference script's single-batch
    iterations. Adam uses coupled L2 ``weight_decay``, as in the PyTorch reference.
    ``device`` accepts auto, cpu or cuda[:index]; auto retries on CPU if CUDA
    runs out of memory. The fitted ``model_`` is an eval-mode PyTorch module on
    CPU, and public result attributes remain NumPy arrays.
    """

    backend = "pytorch"

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
        device: str = "auto",
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
        self.device = device

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
        for name in ("n_clusters", "embedding_dim", "projection_dim", "batch_size", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temperature", "learning_rate", "weight_decay", "gaussian_noise", "reconstruction_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in {"temperature", "learning_rate"} and value == 0):
                bound = "positive" if name in {"temperature", "learning_rate"} else "nonnegative"
                raise ValueError(f"{name} must be finite and {bound}")
        if any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1 for d in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
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
            if view.ndim != 2 or view.shape[0] < 2 or view.shape[1] < 1:
                raise ValueError("Every view must have at least two samples and one feature.")
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
        h = torch.cat((left, right), dim=0)
        logits = (h @ h.T) / temperature
        n = left.shape[0]
        positives = (torch.arange(2 * n, device=h.device) + n) % (2 * n)
        logits = logits.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=h.device), float("-inf"))
        return F.cross_entropy(logits, positives)

    def _augment(self, batch, view_widths, rng):
        rows = batch.shape[0]
        n_views = len(view_widths)
        view_mask = np.ones((rows, n_views), dtype=np.float32)
        affected = rng.choice(
            rows, size=int(rows * self.miss_rate), replace=False
        )
        if n_views > 1:
            # Reference distribution: uniform nonempty, non-full binary masks.
            masks = rng.integers(0, 2, size=(len(affected), n_views))
            bad = (masks.sum(axis=1) == 0) | (masks.sum(axis=1) == n_views)
            while bad.any():
                masks[bad] = rng.integers(0, 2, size=(int(bad.sum()), n_views))
                bad = (masks.sum(axis=1) == 0) | (masks.sum(axis=1) == n_views)
            view_mask[affected] = masks
        feature_mask = np.repeat(view_mask, view_widths, axis=1)
        masked = batch * feature_mask

        noisy = batch.copy()
        offset = 0
        for width in view_widths:
            # The reference selects noisy rows independently for each view.
            noise_rows = rng.random(rows) < self.noise_rate
            noisy[noise_rows, offset:offset + width] += rng.normal(
                0.0, self.gaussian_noise, size=(int(noise_rows.sum()), width)
            ).astype(np.float32)
            offset += width
        return masked, noisy

    def fit(self, Xs, As=None, *, observed_mask=None):
        """Fit SCM to a list of aligned feature views."""
        if observed_mask is not None and not np.asarray(observed_mask, dtype=bool).all():
            raise ValueError("SCM does not support missing views; explicitly impute first")
        del As  # SCM is a feature-level fusion method.
        views = self._validate(Xs)
        fused = np.concatenate(views, axis=1).astype(np.float32, copy=False)
        device = resolve_device(self.device)
        output = None
        try:
            output = self._fit_on_device(fused, [v.shape[1] for v in views], device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "auto" or device.type != "cuda":
                raise
            logger.warning("CUDA out of memory; restarting SCM on CPU")
        if output is None:
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            output = self._fit_on_device(fused, [v.shape[1] for v in views], device)
        self.model_, self.embedding_, self.loss_history_ = output
        self.device_ = str(device)
        clustering = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        ).fit(self.embedding_)
        self.labels_ = clustering.labels_.astype(np.int32)
        self.centroids_ = clustering.cluster_centers_
        self.inertia_ = float(clustering.inertia_)
        self.embedding_dim_ = self.projection_dim
        return self

    def _fit_on_device(self, fused, view_widths, device):
        rng = np.random.default_rng(self.random_state)
        with initialization_seed(self.random_state):
            model = _SCMNetwork(fused.shape[1], self.embedding_dim, self.projection_dim, self.hidden_dims)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        losses = []
        batch_size = min(self.batch_size, len(fused))
        indices = np.arange(len(fused))
        for _ in iterations(self.max_iter, "Contrastive epochs"):
            rng.shuffle(indices)
            epoch_losses = []
            for start in range(0, len(fused), batch_size):
                selected = indices[start:start + batch_size]
                if selected.size < 2:
                    continue  # A singleton has no contrastive negatives.
                clean = fused[selected]
                masked, noisy = self._augment(clean, view_widths, rng)
                masked_t = torch.as_tensor(masked, device=device)
                noisy_t = torch.as_tensor(noisy, device=device)
                optimizer.zero_grad(set_to_none=True)
                if self.reconstruction_weight:
                    clean_t = torch.as_tensor(clean, device=device)
                    clean_xr, _, _, _ = model(clean_t)
                    mask_xr, _, mask_z, _ = model(masked_t)
                    noise_xr, _, noise_z, _ = model(noisy_t)
                    reconstruction = (F.mse_loss(clean_xr, clean_t) + F.mse_loss(mask_xr, clean_t)
                                      + F.mse_loss(noise_xr, clean_t))
                else:
                    _, mask_z = model.encode(masked_t)
                    _, noise_z = model.encode(noisy_t)
                    reconstruction = 0.0
                loss = (self._contrastive_loss(noise_z, mask_z, self.temperature)
                        + self._contrastive_loss(mask_z, noise_z, self.temperature)
                        + self.reconstruction_weight * reconstruction)
                if not torch.isfinite(loss):
                    raise ValueError("SCM loss became nonfinite; check learning rate, temperature and feature scale")
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.detach().item())
            losses.append(float(np.mean(epoch_losses)))

        # Never put the entire dataset on the accelerator for inference.
        model.eval()
        embedding = np.empty((len(fused), self.projection_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(fused), batch_size):
                batch = torch.as_tensor(fused[start:start + batch_size], device=device)
                _, encoded = model.encode(batch)
                embedding[start:start + batch_size] = encoded.cpu().numpy()
        return model.cpu(), embedding, np.asarray(losses)
