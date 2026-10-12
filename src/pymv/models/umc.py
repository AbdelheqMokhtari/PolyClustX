"""UMC curriculum clustering on user-supplied feature matrices (ACL 2024)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from pymv.progress import iterations

from ._common import apply_scalers, batches, fit_scalers, integer, memory_check, number, optimize, validate_views
from ._deep import DeepModel, linear, mlp, parameter_budget
from ._torch_utils import initialization_seed
from .registry import register_model


def supervised_contrastive(representations, labels, temperature):
    """All augmentations are positives for their sample (or shared pseudo-label)."""
    z = F.normalize(torch.cat(representations), dim=1)
    groups = labels.repeat(len(representations))
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = (groups[:, None] == groups[None, :]) & ~diagonal
    logits = z @ z.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    log_probability = logits - denominator
    return -(log_probability.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1)).mean()


def high_quality_indices(embedding, labels, fraction, max_memory_mb=1024):
    """Density ranking with per-cluster neighborhood selection; no true labels.

    Candidate quality follows the reference's maximum nearest-neighbor-distance
    criterion. Singleton clusters are retained; tied/constant inputs remain valid.
    """
    chosen = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        x = embedding[indices]
        count = max(1, int(len(x) * fraction))
        if count == len(x) or len(x) == 1:
            chosen.extend(indices)
            continue
        neighbors = sorted({min(len(x) - 1, max(1, int(len(x) * p))) for p in np.arange(.1, .32, .02)})
        memory_check(len(x) * (max(neighbors) + 1) * 24, max_memory_mb)
        distances = NearestNeighbors(n_neighbors=max(neighbors) + 1).fit(x).kneighbors(x)[0][:, 1:]
        best_score, best_indices = -np.inf, None
        for k in neighbors:
            selected = np.argsort(distances[:, :k].mean(1), kind="stable")[:count]
            score = 0. if count == 1 else NearestNeighbors(n_neighbors=2).fit(x[selected]).kneighbors(
                x[selected])[0][:, 1].mean()
            if score > best_score:
                best_score, best_indices = score, selected
        chosen.extend(indices[best_indices])
    return np.asarray(chosen, dtype=int)


class FeatureFusion(nn.Module):
    def __init__(self, widths, hidden, dim):
        super().__init__()
        self.encoders = nn.ModuleList([mlp([w, *hidden, dim]) for w in widths])
        self.fusion = nn.Sequential(nn.GELU(), linear(len(widths) * dim, dim))
        self.unsupervised = mlp([dim, dim, dim])
        self.supervised = mlp([dim, dim, dim])

    def encode(self, xs):
        return [encoder(x) for encoder, x in zip(self.encoders, xs)]

    def forward(self, xs):
        return self.fusion(torch.cat(self.encode(xs), dim=1))


@register_model("umc", description="Feature-input UMC: modality masking and density-guided contrastive curriculum.",
                paper="https://aclanthology.org/2024.acl-long.2/",
                implementation="https://github.com/thuiar/UMC")
class UMC(DeepModel):
    """Feature-input adaptation without pretrained text/audio/video encoders.

    Keep ``core_view`` and mask each other view in turn. A single view uses
    independent feature dropout augmentations. Pretraining is followed by a
    density-guided curriculum, supervised contrastive learning on pseudo-labels
    of selected samples, and instance contrast on the remainder. Ground-truth
    labels are never consumed. Requires complete (or explicitly imputed) views.
    """

    def __init__(self, n_clusters, *, embedding_dim=128, hidden_dims=(128,), pretrain_epochs=20,
                 max_iter=20, batch_size=128, learning_rate=.001, temperature=.1,
                 initial_fraction=.25, core_view=0, augmentation_rate=.1, n_init=10,
                 preprocess=True, random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.pretrain_epochs = pretrain_epochs
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.initial_fraction = initial_fraction
        self.core_view = core_view
        self.augmentation_rate = augmentation_rate
        self.n_init = n_init
        self.preprocess = preprocess
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, mask = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        self._check_deep(len(xs[0]))
        integer("pretrain_epochs", self.pretrain_epochs)
        integer("core_view", self.core_view, 0)
        number("temperature", self.temperature, positive=True)
        number("initial_fraction", self.initial_fraction, positive=True)
        number("augmentation_rate", self.augmentation_rate)
        if self.core_view >= len(xs) or self.initial_fraction > 1 or self.augmentation_rate >= 1:
            raise ValueError("Choose an existing core_view, 0 < initial_fraction <= 1 and 0 <= augmentation_rate < 1")
        paths = [[x.shape[1], *self.hidden_dims, self.embedding_dim] for x in xs]
        paths += [[len(xs) * self.embedding_dim, self.embedding_dim],
                  [self.embedding_dim] * 3, [self.embedding_dim] * 3]
        parameter_budget(paths, self.max_memory_mb)
        memory_check((min(self.batch_size + 1, len(xs[0])) * max(3, len(xs)))**2 * 4 * 12, self.max_memory_mb)
        scalers = fit_scalers(xs, mask, self.preprocess)
        self._execute(lambda device: self._train(apply_scalers(xs, mask, scalers), device))
        self.scalers_, self.n_features_in_ = scalers, [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_dim
        return self

    def _augment(self, model, tensors, rng, head):
        views = model.encode(tensors)
        representations = [head(model.fusion(torch.cat(views, dim=1)))]
        if len(views) == 1:
            for _ in range(2):
                mask = torch.as_tensor(rng.random(views[0].shape) >= self.augmentation_rate,
                                       device=views[0].device)
                representations.append(head(model.fusion(views[0] * mask / (1 - self.augmentation_rate))))
        else:
            for v in range(len(views)):
                if v != self.core_view:
                    masked = [torch.zeros_like(z) if j == v else z for j, z in enumerate(views)]
                    representations.append(head(model.fusion(torch.cat(masked, dim=1))))
        return representations

    def _train(self, xs, device):
        with initialization_seed(self.random_state):
            model = FeatureFusion([x.shape[1] for x in xs], self.hidden_dims, self.embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.random_state)
        history, stages, selections = [], [], []

        def train_subset(indices, labels, head):
            values = []
            model.train()
            for positions in batches(len(indices), self.batch_size, rng):
                idx = indices[positions]
                tensors = [torch.as_tensor(x[idx], device=device) for x in xs]
                groups = torch.arange(len(idx), device=device) if labels is None \
                    else torch.as_tensor(labels[idx], device=device)
                loss = supervised_contrastive(self._augment(model, tensors, rng, head), groups, self.temperature)
                values.append(optimize(loss, optimizer))
            return float(np.mean(values)) if values else 0.

        for _ in iterations(self.pretrain_epochs, "Modality-mask pretraining"):
            history.append(train_subset(np.arange(len(xs[0])), None, model.unsupervised))
            stages.append("pretrain")
        centers = None
        for epoch in iterations(self.max_iter, "Density-guided curriculum"):
            embedding = self._encode_batches(model, xs, device, lambda m, x: m(x))
            km = KMeans(self.n_clusters, init="k-means++" if centers is None else centers,
                        n_init=self.n_init if centers is None else 1, random_state=self.random_state).fit(embedding)
            centers, labels = km.cluster_centers_, km.labels_
            fraction = self.initial_fraction + (1 - self.initial_fraction) * epoch / max(self.max_iter - 1, 1)
            selected = high_quality_indices(embedding, labels, fraction, self.max_memory_mb)
            remaining = np.setdiff1d(np.arange(len(labels)), selected)
            supervised = train_subset(selected, labels, model.supervised)
            unsupervised = train_subset(remaining, None, model.unsupervised)
            history.append(supervised + unsupervised)
            stages.append("curriculum")
            selections.append(len(selected))
        embedding = self._encode_batches(model, xs, device, lambda m, x: m(x))
        km = KMeans(self.n_clusters, init=centers, n_init=1, random_state=self.random_state).fit(embedding)
        return dict(model_=model.cpu(), embedding_=embedding, labels_=km.labels_, centroids_=km.cluster_centers_,
                    kmeans_=km, inertia_=km.inertia_, loss_history_=np.asarray(history), loss_stages_=stages,
                    selected_counts_=np.asarray(selections))

    def transform(self, Xs):
        xs, _ = self._prediction_views(Xs)
        return self._encode_batches(self.model_, xs, "cpu", lambda m, x: m(x))

    def predict(self, Xs):
        return self.kmeans_.predict(self.transform(Xs))
