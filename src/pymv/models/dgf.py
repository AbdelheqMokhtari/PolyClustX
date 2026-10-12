"""Dual graph filtering with cross-modal, graph and community contrast (DGF)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from pymv.progress import iterations

from ._common import TorchModel, batches, integer, memory_check, number, optimize, validate_views
from ._deep import linear, parameter_budget
from ._graphs import cosine_graph, input_graphs, normalize_graph, torch_sparse
from ._torch_utils import initialization_seed
from .registry import register_model


class DualFilter(nn.Module):
    def __init__(self, widths, dim, alpha, beta, layers):
        super().__init__()
        self.projections = nn.ModuleList([linear(w, dim) for w in widths])
        self.alpha, self.beta, self.layers = alpha, beta, layers

    def forward(self, xs, graph):
        views = [F.normalize(layer(x), dim=1) for layer, x in zip(self.projections, xs)]
        fused = torch.stack(views).mean(0)
        with torch.no_grad():
            feature_graphs = []
            for z in views:
                similarity = z.T @ z / math.sqrt(len(z))
                affinity = torch.exp(similarity - similarity.max())
                degree = affinity.sum(1).clamp_min(1e-12).rsqrt()
                feature_graphs.append(degree[:, None] * affinity * degree[None, :])
            feature_graph = torch.stack(feature_graphs).mean(0)
        propagated = current = fused
        for _ in range(self.layers):
            current = self.alpha / (self.alpha + 1) * torch.sparse.mm(graph, current)
            propagated = propagated + current
        power = torch.eye(fused.shape[1], device=fused.device, dtype=fused.dtype)
        feature_filter = torch.zeros_like(power)
        for _ in range(self.layers):
            feature_filter = feature_filter + power
            power = power @ (self.beta / (self.beta + 1) * feature_graph)
        embedding = F.normalize(propagated @ feature_filter / ((self.alpha + 1) * (self.beta + 1)), dim=1)
        return views, embedding


def walk_contexts(graph, walks, length, context, rng, negative=False):
    """Uniform CSR walks; isolated nodes stay put instead of producing -1 IDs."""
    starts = np.tile(np.arange(graph.shape[0]), walks)
    paths = [starts]
    for _ in range(length):
        current = paths[-1]
        if negative:
            next_nodes = rng.integers(graph.shape[0], size=len(starts))
        else:
            degrees = np.diff(graph.indptr)[current]
            offsets = (rng.random(len(current)) * degrees).astype(int)
            next_nodes = current.copy()
            valid = degrees > 0
            next_nodes[valid] = graph.indices[graph.indptr[current[valid]] + offsets[valid]]
        paths.append(next_nodes)
    paths = np.stack(paths, axis=1)
    return np.concatenate([paths[:, start:start + context] for start in range(length + 2 - context)])


def community_loss(h, centers, labels, fraction):
    def standardize(x):
        return F.normalize((x - x.mean(0)) / x.std(0, unbiased=False).clamp_min(1e-8), dim=1)
    scores = standardize(h) @ standardize(centers).T
    loss = scores.sum() * 0
    for k in range(len(centers)):
        selected = scores[labels == k]
        if len(selected):
            count = max(1, math.ceil(len(selected) * fraction))
            numerator = torch.logsumexp(selected[:, k].topk(count).values, dim=0)
            loss = loss + torch.logsumexp(selected.flatten(), dim=0) - numerator
    return loss / len(h)


@register_model("dgf", description="Dual graph filtering with cross-modal, random-walk and community contrast.",
                paper="https://arxiv.org/abs/2511.20030", uses_graphs=True,
                implementation="https://github.com/HaoranZ99/DGF")
class DGF(TorchModel):
    """Train linear projections and node/feature filters on complete views.

    Supplied graphs are averaged into one common graph. With no graph, build
    cosine kNN graphs from the supplied features. More than two views use mean
    feature-graph fusion and all-pairs cross-modal contrast; one view retains
    feature-to-graph contrast. These are explicit extensions of the two-view
    reference. Final weights are used, never the epoch with best true-label NMI.
    """

    def __init__(self, n_clusters, *, embedding_dim=64, alpha=1., beta=1., num_layers=10,
                 max_iter=100, learning_rate=.001, weight_decay=.00001, batch_size=256,
                 walks_per_node=2, walk_length=5, context_size=3, community_fraction=.3,
                 attribute_filter=False, filter_std=1., n_neighbors=10, n_init=10,
                 random_state=None, device="auto", max_memory_mb=1024):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.num_layers = num_layers
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.community_fraction = community_fraction
        self.attribute_filter = attribute_filter
        self.filter_std = filter_std
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.random_state = random_state
        self.device = device
        self.max_memory_mb = max_memory_mb

    def fit(self, Xs, As=None, *, observed_mask=None):
        xs, _ = validate_views(Xs, observed_mask, max_memory_mb=self.max_memory_mb)
        n = len(xs[0])
        self._check_common(n)
        for name in ("embedding_dim", "num_layers", "walks_per_node", "walk_length", "n_neighbors", "n_init"):
            integer(name, getattr(self, name))
        integer("batch_size", self.batch_size, 2)
        integer("context_size", self.context_size, 2)
        for name in ("alpha", "beta", "weight_decay", "filter_std", "community_fraction"):
            number(name, getattr(self, name))
        number("learning_rate", self.learning_rate, positive=True)
        if self.context_size > self.walk_length + 1 or self.community_fraction > 1:
            raise ValueError("context_size must be <= walk_length + 1; community_fraction must be in [0,1]")
        paths = n * self.walks_per_node * (self.walk_length + 2 - self.context_size)
        memory_check(paths * self.context_size * self.embedding_dim * 4 * 8
                     + n * min(self.batch_size + 1, n) * (len(xs) + 1)**2 * 4 * 8
                     + n * self.embedding_dim * (len(xs) + self.num_layers) * 32
                     + self.embedding_dim**2 * (len(xs) + self.num_layers) * 32, self.max_memory_mb)
        parameter_budget([[x.shape[1], self.embedding_dim] for x in xs], self.max_memory_mb)
        supplied = input_graphs(As, n, len(xs))
        graphs = supplied or [cosine_graph(x, self.n_neighbors) for x in xs]
        graph = sum(graphs) / len(graphs)
        self._execute(lambda device: self._train(xs, graph.tocsr(), device))
        self.n_features_in_ = [x.shape[1] for x in xs]
        self.embedding_dim_ = self.embedding_dim
        self.graph_source_ = "supplied" if supplied else "feature_knn"
        return self

    def _train(self, xs, graph, device):
        with initialization_seed(self.random_state):
            model = DualFilter([x.shape[1] for x in xs], self.embedding_dim,
                               self.alpha, self.beta, self.num_layers).to(device)
        tensors = [torch.as_tensor(x, device=device) for x in xs]
        adjacency = torch_sparse(normalize_graph(graph, self_loops=True), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state)
        history, components = [], []
        for _ in iterations(self.max_iter, "Dual graph filtering epochs"):
            views, h = model(tensors, adjacency)
            cross = h.sum() * 0
            batch_count = 0
            for idx in batches(len(h), self.batch_size, rng):
                target = torch.arange(len(idx), device=device)
                for left, right in itertools.combinations([*views, h], 2):
                    scores = left[idx] @ right[idx].T - .001 * torch.eye(len(idx), device=device)
                    cross = cross + F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)
                batch_count += 1
            cross = cross / batch_count
            sampling_graph = graph
            if self.attribute_filter:
                z = [view.detach().cpu().numpy() for view in views]
                left, right = z[0], z[-1]
                rows, cols = graph.nonzero()
                if len(rows):
                    random_scores = (left[rng.integers(len(h), size=len(rows))]
                                     * right[rng.integers(len(h), size=len(rows))]).sum(1)
                    threshold = random_scores.mean() + self.filter_std * random_scores.std()
                    keep = (left[rows] * right[cols]).sum(1) >= threshold
                    from scipy import sparse
                    sampling_graph = sparse.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), graph.shape)
            positive, negative = [torch.as_tensor(walk_contexts(sampling_graph, self.walks_per_node,
                                      self.walk_length, self.context_size, rng, negative=negative), device=device)
                                  for negative in (False, True)]
            scores = [torch.logsumexp((h[p[:, :1]] * h[p[:, 1:]]).sum(-1), dim=1) for p in (positive, negative)]
            structural = F.softplus(scores[1] - scores[0]).mean()
            km = KMeans(self.n_clusters, n_init=self.n_init,
                        random_state=self.random_state).fit(h.detach().cpu().numpy())
            community = community_loss(h, torch.as_tensor(km.cluster_centers_, device=device),
                                       torch.as_tensor(km.labels_, device=device), self.community_fraction) \
                if self.community_fraction else h.sum() * 0
            terms = [cross, structural, community]
            loss = sum(term / term.detach().abs().clamp_min(1e-8) for term in terms)
            optimize(loss, optimizer)
            components.append([float(term.detach()) for term in terms])
            # Self-normalized loss is nearly constant; retain raw component sum for diagnostics.
            history.append(sum(components[-1]))
        model.eval()
        with torch.no_grad():
            _, h = model(tensors, adjacency)
        embedding = h.cpu().numpy()
        km = KMeans(self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(embedding)
        return dict(model_=model, labels_=km.labels_, embedding_=embedding, centroids_=km.cluster_centers_,
                    loss_history_=np.asarray(history), loss_components_=np.asarray(components),
                    loss_component_names_=["cross_modal", "graph", "community"], inertia_=km.inertia_)
