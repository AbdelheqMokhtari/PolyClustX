"""Co-EM clustering inspired by mvlearn; see THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from pymv.progress import iterations

from ._common import TorchModel, integer, memory_check, number, validate_views
from .registry import register_model


def distances(x, centers, batch_size=1024):
    return torch.cat([torch.cdist(chunk, centers) for chunk in x.split(batch_size)])


def em_step(x, partition, centers):
    updated = centers.clone()
    for k in range(len(centers)):
        members = partition == k
        if members.any():
            updated[k] = x[members].mean(dim=0)
    d = distances(x, updated)
    return d.argmin(dim=1), updated, d.min(dim=1).values.sum()


@register_model("multiview_kmeans", description="Co-EM K-means; ordinary K-means for one view.",
                paper="https://doi.org/10.1109/ICDM.2004.10095",
                implementation="https://mvlearn.github.io/references/cluster.html")
class MultiViewKMeans(TorchModel):
    """One-view K-means, two-view Co-EM, or a cyclic multi-view extension.

    ``max_iter`` counts complete sweeps. Restarts are selected using the sum
    of per-view minimum Euclidean distances (not squared distances). Final
    centers use samples agreeing across all views; absent consensus keeps the
    previous center. ``predict`` sums view-wise distances. Single-view fitting
    delegates to sklearn KMeans and reports its squared-distance inertia.
    """

    def __init__(self, n_clusters, *, n_init=10, max_iter=100, tol=1e-4, patience=5,
                 init="k-means++", random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.init = init
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, _ = validate_views(Xs, observed_mask, dtype=np.float64, max_memory_mb=self.max_memory_mb)
        self._check_common(len(xs[0]))
        memory_check(len(xs[0]) * self.n_clusters * (2 * len(xs) + 4) * 8, self.max_memory_mb)
        integer("n_init", self.n_init)
        integer("patience", self.patience)
        number("tol", self.tol)
        if self.init not in ("k-means++", "random"):
            raise ValueError("init must be k-means++ or random")
        if len(xs) == 1:
            km = KMeans(self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter,
                        tol=self.tol, random_state=self.random_state).fit(xs[0])
            self.centroids_ = [km.cluster_centers_]
            self.labels_ = km.labels_.astype(np.int32)
            self.embedding_ = xs[0]
            self.inertia_ = km.inertia_
            self.loss_history_ = np.array([km.inertia_])
            self.n_iter_ = km.n_iter_
            self.device_ = "cpu"
        else:
            self._execute(lambda device: self._fit_views(xs, device))
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_.shape[1]
        self.n_views_ = len(xs)
        return self

    @torch.no_grad()
    def _fit_views(self, xs, device):
        views = [torch.as_tensor(x, device=device) for x in xs]
        rng = np.random.default_rng(self.random_state)
        best = None
        objectives = []
        for _ in iterations(self.n_init, "Co-EM initializations"):
            seed = int(rng.integers(2**31))
            indices = (kmeans_plusplus(xs[-1], self.n_clusters, random_state=seed)[1]
                       if self.init == "k-means++" else rng.choice(len(xs[0]), self.n_clusters, replace=False))
            centers = [x[indices].clone() for x in views]
            parts = [distances(x, c).argmin(1) for x, c in zip(views, centers)]
            history, previous, stall = [], np.inf, 0
            for _ in range(self.max_iter):
                old = [p.clone() for p in parts]
                objective = 0.
                for v, x in enumerate(views):
                    parts[v], centers[v], loss = em_step(x, parts[(v - 1) % len(views)], centers[v])
                    objective += float(loss)
                history.append(objective)
                improved = not np.isfinite(previous) or previous - objective > self.tol * max(abs(previous), 1e-12)
                stall = 0 if improved else stall + 1
                previous = min(previous, objective)
                if stall >= self.patience or all(torch.equal(a, b) for a, b in zip(old, parts)):
                    break
            objectives.append(objective)
            if best is None or objective < best[0]:
                best = objective, centers, parts, history
        objective, centers, parts, history = best
        counts = []
        for k in range(self.n_clusters):
            consensus = torch.stack([p == k for p in parts]).all(dim=0)
            counts.append(int(consensus.sum()))
            if consensus.any():
                for v in range(len(views)):
                    centers[v][k] = views[v][consensus].mean(0)
        if 0 in counts:
            warnings.warn("Some clusters have no consensus samples; their previous centers are retained",
                          ConvergenceWarning, stacklevel=2)
        combined = sum(distances(x, c) for x, c in zip(views, centers))
        return dict(centroids_=[c.cpu().numpy() for c in centers], labels_=combined.argmin(1).cpu().numpy(),
                    embedding_=np.concatenate(xs, axis=1), loss_history_=np.asarray(history),
                    inertia_=float(combined.min(1).values.sum()), coem_objective_=objective,
                    restart_objectives_=np.asarray(objectives), consensus_counts_=np.asarray(counts),
                    view_labels_=[p.cpu().numpy() for p in parts], n_iter_=len(history))

    def predict(self, Xs, As=None):
        if not hasattr(self, "centroids_"):
            raise NotFittedError("Call fit before predict")
        xs, _ = validate_views(Xs, dtype=np.float64, max_memory_mb=self.max_memory_mb)
        if [x.shape[1] for x in xs] != self.n_features_in_:
            raise ValueError("Prediction needs the fitted view order and feature dimensions")
        with torch.no_grad():
            combined = sum(distances(torch.from_numpy(x), torch.from_numpy(c))
                           for x, c in zip(xs, self.centroids_))
        return combined.argmin(1).numpy()


MultiviewKMeans = MultiViewKMeans
