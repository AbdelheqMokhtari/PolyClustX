"""Spectral co-training, exchanging embeddings rather than cluster labels."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn.functional import normalize

from pymv.progress import iterations

from ._common import TorchModel, integer, memory_check, number, validate_views
from .registry import register_model


def spectral_components(similarity, k):
    degree = similarity.sum(1).abs()
    inv = torch.where(degree > torch.finfo(degree.dtype).eps, degree.clamp_min(1e-15).rsqrt(), 0)
    normalized = inv[:, None] * similarity * inv[None, :]
    values, vectors = torch.linalg.eigh((normalized + normalized.T) * .5)
    # Matches mvlearn's leading singular subspace, including signed refinements.
    return vectors[:, values.abs().argsort(descending=True)[:k]]


def refine_similarities(affinities, embeddings):
    refined = []
    for v, kernel in enumerate(affinities):
        # Avoid materializing each n-by-n projection U_j U_j.T.
        constrained = sum(u @ (u.T @ kernel) for j, u in enumerate(embeddings) if j != v)
        refined.append((constrained + constrained.T) * .5)
    return refined


@register_model("multiview_spectral", description="Co-training of spectral similarity graphs across views.",
                paper="https://icml.cc/2011/papers/272_icmlpaper.pdf",
                implementation="https://mvlearn.github.io/references/cluster.html")
class MultiViewSpectralClustering(TorchModel):
    """Spectral co-training for any positive number of complete feature views.

    A single view skips co-training. Each iteration uses all *previous*
    embeddings and the original affinities. Final row-normalized embeddings
    are concatenated unless ``info_view`` selects a zero-based view index.
    This is transductive: no out-of-sample ``predict`` is supplied.
    Dense affinity memory is checked before allocation; time is cubic in N.
    """

    def __init__(self, n_clusters, *, affinity="rbf", gamma=None, n_neighbors=10,
                 degree=3, coef0=1., info_view=None, max_iter=10, n_init=10,
                 random_state=None, device="auto", max_memory_mb=512):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.degree = degree
        self.coef0 = coef0
        self.info_view = info_view
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, _ = validate_views(Xs, observed_mask, dtype=np.float64, max_memory_mb=self.max_memory_mb)
        self._check_common(len(xs[0]))
        integer("n_init", self.n_init)
        integer("n_neighbors", self.n_neighbors)
        integer("degree", self.degree)
        number("coef0", self.coef0)
        if self.gamma is not None:
            number("gamma", self.gamma, positive=True)
        if self.affinity not in {"rbf", "nearest_neighbors", "poly"}:
            raise ValueError("affinity must be rbf, nearest_neighbors or poly")
        if self.info_view is not None:
            integer("info_view", self.info_view, 0)
            if self.info_view >= len(xs):
                raise ValueError("info_view is outside the available views")
        if self.affinity == "nearest_neighbors" and self.n_neighbors > len(xs[0]):
            raise ValueError("n_neighbors cannot exceed the number of samples")
        memory_check((2 * len(xs) + 8) * len(xs[0])**2 * 8, self.max_memory_mb)
        self._execute(lambda device: self._fit_views(xs, device))
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_.shape[1]
        return self

    @torch.no_grad()
    def _fit_views(self, xs, device):
        affinities, gammas = [], []
        for array in xs:
            x = torch.as_tensor(array, device=device)
            distance = torch.cdist(x, x)
            distance.fill_diagonal_(0.)
            median = float(torch.quantile(distance.flatten(), .5))
            gamma = self.gamma if self.gamma is not None else (1 / (2 * median**2) if median > 0 else 1.)
            gammas.append(gamma)
            if self.affinity == "rbf":
                kernel = (-gamma * distance.square()).exp()
            elif self.affinity == "poly":
                kernel = (gamma * (x @ x.T) + self.coef0).pow(self.degree)
            else:
                distance.fill_diagonal_(-1.)  # Include self even when duplicate rows tie at distance zero.
                neighbors = distance.topk(self.n_neighbors, largest=False).indices
                kernel = torch.zeros_like(distance).scatter_(1, neighbors, 1.)
                kernel = (kernel + kernel.T) * .5
            if not torch.isfinite(kernel).all():
                raise ValueError("Affinity overflow; scale features or reduce gamma/degree")
            affinities.append(kernel)
        embeddings = [spectral_components(a, self.n_clusters) for a in affinities]
        changes = []
        if len(xs) > 1:
            for _ in iterations(self.max_iter, "Spectral co-training"):
                updated = [spectral_components(a, self.n_clusters)
                           for a in refine_similarities(affinities, embeddings)]
                changes.append(sum(float((2 * self.n_clusters - 2 * (a.T @ b).square().sum()).clamp_min(0))
                                   for a, b in zip(embeddings, updated)))
                embeddings = updated
        normalized = [normalize(u, dim=1).cpu().numpy() for u in embeddings]
        embedding = np.concatenate(normalized, axis=1) if self.info_view is None else normalized[self.info_view]
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(embedding)
        return dict(labels_=km.labels_.astype(np.int32), embedding_=embedding, centroids_=km.cluster_centers_,
                    inertia_=km.inertia_, view_embeddings_=normalized, gammas_=np.asarray(gammas),
                    subspace_change_history_=np.asarray(changes), loss_history_=None, n_iter_=len(changes))


MultiviewSpectralClustering = MultiViewSpectralClustering
