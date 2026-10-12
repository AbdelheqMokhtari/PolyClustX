"""Spectrum-guided Laplacian aggregation, independently implemented from Eq. 5–6."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from polyclustx.progress import iterations

from ._common import TorchModel, integer, memory_check, number, validate_views
from ._graphs import cosine_graph, input_graphs, normalize_graph
from .registry import register_model


def spectral_objective(eigenvalues, weights, k, gamma):
    """Relative eigengap, connectivity reward, and weight regularization."""
    return eigenvalues[k - 1].clamp_min(0) / eigenvalues[k].clamp_min(1e-12) \
        - eigenvalues[1] + gamma * weights.square().sum()


@register_model("sgla", description="Spectrum-guided aggregation of graph and feature-view Laplacians.",
                paper="https://arxiv.org/abs/2508.09452", uses_graphs=True,
                implementation="https://github.com/CyanideCentral/SGLA")
class SGLA(TorchModel):
    """Optimize simplex weights with COBYLA and a PyTorch symmetric eigensolver.

    Each feature view contributes a cosine kNN graph, in addition to supplied
    graphs. Negative cosine similarities are clipped to preserve valid degrees.
    Final clustering uses K-means on row-normalized bottom eigenvectors.
    This bounded dense implementation is not the reference's scalable SGLA+
    or its separate NetMF embedding pipeline. It is transductive.
    """

    def __init__(self, n_clusters, *, n_neighbors=10, gamma=.5, max_iter=50, tol=.001,
                 n_init=10, random_state=None, device="auto", max_memory_mb=512):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, _ = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        n = len(xs[0])
        self._check_common(n)
        if self.n_clusters >= n:
            raise ValueError("SGLA needs n_clusters < n_samples for its relative eigengap")
        integer("n_neighbors", self.n_neighbors)
        integer("n_init", self.n_init)
        number("gamma", self.gamma)
        number("tol", self.tol, positive=True)
        supplied = input_graphs(As, n, len(xs))
        memory_check(n * n * 8 * (len(supplied) + len(xs) + 8), self.max_memory_mb)
        graphs = supplied + [cosine_graph(x, self.n_neighbors) for x in xs]
        normalized = [normalize_graph(g) for g in graphs]
        self._execute(lambda device: self._solve(normalized, device))
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.graph_sources_ = [f"supplied_graph_{v}" for v in range(len(supplied))] \
            + [f"feature_view_{v}" for v in range(len(xs))]
        self.embedding_dim_ = self.n_clusters
        return self

    def _solve(self, graphs, device):
        matrices = [torch.as_tensor(g.toarray(), dtype=torch.float64, device=device) for g in graphs]
        eye = torch.eye(graphs[0].shape[0], dtype=torch.float64, device=device)
        count = len(matrices)
        history, best = [], [np.inf, np.ones(count) / count]
        progress = iter(iterations(max(self.max_iter, count + 1), "Laplacian objective evaluations"))

        def evaluate(free):
            next(progress, None)
            weights = np.r_[free, 1 - np.sum(free)]
            # COBYLA probes outside the simplex; never eigendecompose an invalid Laplacian.
            if (weights < -1e-10).any():
                return 1e6 + float(np.maximum(-weights, 0).sum())
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            w = torch.as_tensor(weights, device=device)
            lap = eye - sum(weight * matrix for weight, matrix in zip(w, matrices))
            value = float(spectral_objective(torch.linalg.eigvalsh(lap), w, self.n_clusters, self.gamma))
            history.append(value)
            if value < best[0]:
                best[:] = [value, weights.copy()]
            return value

        if count > 1:
            initial = np.full(count - 1, 1 / count)
            evaluate(initial)
            minimize(evaluate, initial, method="COBYLA", tol=self.tol,
                     constraints=[{"type": "ineq", "fun": lambda w: w},
                                  {"type": "ineq", "fun": lambda w: 1 - np.sum(w)}],
                     options={"maxiter": max(self.max_iter, count + 1), "rhobeg": .05})
        else:
            evaluate(np.empty(0))
        lap = eye - sum(float(w) * matrix for w, matrix in zip(best[1], matrices))
        values, vectors = torch.linalg.eigh(lap)
        embedding = normalize(vectors[:, :self.n_clusters].cpu().numpy())
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(embedding)
        return dict(embedding_=embedding, labels_=km.labels_, centroids_=km.cluster_centers_,
                    view_weights_=best[1], loss_history_=np.asarray(history), eigenvalues_=values.cpu().numpy(),
                    inertia_=km.inertia_, n_iter_=len(history))
