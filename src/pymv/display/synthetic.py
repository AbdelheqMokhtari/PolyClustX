"""Configurable synthetic multi-view datasets for interface examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn import config_context
from sklearn.neighbors import kneighbors_graph

from pymv.data import MultiViewDataset


@dataclass
class SyntheticGraphSpec:
    """One sparse relation: binary kNN, mutual kNN, or RBF-weighted kNN.

    ``bandwidth`` is the RBF length scale in feature-space units. Neighbors
    exclude the sample itself; graphs are undirected and have zero diagonal.
    """

    kind: str = "knn"
    neighbors: int = 10
    bandwidth: float = 1.0


@dataclass
class SyntheticSpec:
    n_samples: int = 300
    n_classes: int = 3
    dimensions: list[int] = field(default_factory=lambda: [10, 10])
    distributions: list[str] = field(default_factory=lambda: ["Gaussian", "Gaussian"])
    class_balance: str = "Balanced"
    separation: float = 3.0
    noise: float = 0.7
    same_modality: bool = True
    labeled: bool = True
    graph_relations: int = 0
    graph_neighbors: int = 10
    random_state: int = 7
    graph_specs: list[SyntheticGraphSpec] | None = None
    graph_views: list[int] | None = None
    shared_graph_view: int | None = None


def _graph_plan(spec):
    """Validate before allocating features or nearest-neighbor arrays."""
    if not isinstance(spec.graph_relations, int) or spec.graph_relations < 0:
        raise ValueError("graph_relations must be a non-negative integer")
    if not isinstance(spec.graph_neighbors, int) or spec.graph_neighbors < 1:
        raise ValueError("graph_neighbors must be a positive integer")
    graphs = spec.graph_specs if spec.graph_specs is not None else [
        SyntheticGraphSpec(neighbors=min(spec.graph_neighbors + i * 2, spec.n_samples - 1))
        for i in range(spec.graph_relations)
    ]
    targets = list(range(len(spec.dimensions))) if spec.graph_views is None else list(spec.graph_views)
    for view in targets + ([] if spec.shared_graph_view is None else [spec.shared_graph_view]):
        if not isinstance(view, int) or not 0 <= view < len(spec.dimensions):
            raise ValueError("Graph view indices must identify existing views (zero-based)")
    if len(set(targets)) != len(targets):
        raise ValueError("Graph target views must be unique")
    if graphs and not targets:
        raise ValueError("Select at least one graph target view")
    for graph in graphs:
        if not isinstance(graph, SyntheticGraphSpec) or graph.kind not in {"knn", "mutual_knn", "rbf_knn"}:
            raise ValueError("Graph kind must be knn, mutual_knn, or rbf_knn")
        if not isinstance(graph.neighbors, int) or not 1 <= graph.neighbors < spec.n_samples:
            raise ValueError("Graph neighbors must be an integer between 1 and n_samples - 1")
        if not np.isfinite(graph.bandwidth) or graph.bandwidth <= 0:
            raise ValueError("Graph bandwidth must be finite and positive")
    if spec.n_samples * sum(graph.neighbors for graph in graphs) * len(targets) > 2_000_000:
        raise ValueError("Synthetic graphs exceed the 2,000,000 directed-neighbor budget; "
                         "reduce samples, neighbors, relations, or target views")
    return graphs, targets


def _make_graph(view, spec):
    mode = "distance" if spec.kind == "rbf_knn" else "connectivity"
    # Bound distance scratch space; never construct a dense adjacency matrix.
    with config_context(working_memory=64):
        graph = kneighbors_graph(view, spec.neighbors, mode=mode, include_self=False).tocsr()
    if spec.kind == "rbf_knn":
        with np.errstate(over="ignore", under="ignore"):
            graph.data = np.exp(-0.5 * (graph.data / spec.bandwidth) ** 2)
    graph = graph.minimum(graph.T) if spec.kind == "mutual_knn" else graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph.tocsr()


def _class_sizes(spec, rng):
    if spec.class_balance == "Balanced":
        weights = np.ones(spec.n_classes)
    elif spec.class_balance == "Moderately imbalanced":
        weights = np.linspace(1.0, 0.45, spec.n_classes)
    elif spec.class_balance == "Strongly imbalanced":
        weights = np.geomspace(1.0, 0.12, spec.n_classes)
    else:
        raise ValueError("Unknown class-balance option")
    probabilities = weights / weights.sum()
    sizes = rng.multinomial(spec.n_samples - spec.n_classes, probabilities) + 1
    return sizes


def generate_synthetic(spec: SyntheticSpec):
    """Generate aligned features and optional sparse, feature-derived graphs.

    ``graph_specs`` defines independently configurable relations and overrides
    legacy ``graph_relations`` / ``graph_neighbors``. By default graphs are
    built separately for every view. ``graph_views`` restricts their targets;
    ``shared_graph_view`` builds graphs from that source view and copies them
    to the targets. Indices are zero-based. Labels never influence graph edges.
    """
    if spec.n_samples < spec.n_classes or spec.n_classes < 2:
        raise ValueError("Synthetic data needs at least two classes and one sample per class")
    if not spec.dimensions or any(d < 2 for d in spec.dimensions):
        raise ValueError("Each synthetic view needs at least two features")
    if len(spec.distributions) != len(spec.dimensions):
        raise ValueError("Provide one distribution for each view")
    graph_specs, graph_views = _graph_plan(spec)
    rng = np.random.default_rng(spec.random_state)
    sizes = _class_sizes(spec, rng)
    labels = np.concatenate([np.full(size, label) for label, size in enumerate(sizes)])
    rng.shuffle(labels)
    latent_dim = max(3, min(12, max(spec.dimensions)))
    centers = rng.normal(size=(spec.n_classes, latent_dim))
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    centers *= spec.separation

    views = []
    for dimension, distribution in zip(spec.dimensions, spec.distributions):
        if distribution == "Gaussian":
            residual = rng.normal(size=(spec.n_samples, latent_dim))
        elif distribution == "Heavy-tailed":
            residual = rng.standard_t(df=3, size=(spec.n_samples, latent_dim)) / np.sqrt(3)
        elif distribution == "Anisotropic":
            residual = rng.normal(size=(spec.n_samples, latent_dim))
            residual *= np.geomspace(0.25, 1.75, latent_dim)
        elif distribution == "Nonlinear":
            residual = rng.normal(size=(spec.n_samples, latent_dim))
        else:
            raise ValueError(f"Unknown synthetic distribution: {distribution}")
        latent = centers[labels] + spec.noise * residual
        projection = rng.normal(size=(latent_dim, dimension)) / np.sqrt(latent_dim)
        view = latent @ projection
        if distribution == "Nonlinear":
            view = np.sin(view) + 0.15 * view**2
        view += rng.normal(scale=spec.noise * 0.1, size=view.shape)
        views.append(view)

    relations = [None] * len(views)
    shared = None
    if graph_specs and spec.shared_graph_view is not None:
        shared = [_make_graph(views[spec.shared_graph_view], graph) for graph in graph_specs]
    for index in graph_views:
        relations[index] = (
            [graph.copy() for graph in shared] if shared is not None
            else [_make_graph(views[index], graph) for graph in graph_specs]
        ) or None
    modalities = (
        ["features"] * len(views)
        if spec.same_modality
        else [f"modality_{index + 1}" for index in range(len(views))]
    )
    return MultiViewDataset(
        views,
        relations,
        labels if spec.labeled else None,
        name="Synthetic dataset",
        view_names=[f"view_{index + 1}" for index in range(len(views))],
        modalities=modalities,
        metadata={"synthetic": asdict(spec)},
    )
