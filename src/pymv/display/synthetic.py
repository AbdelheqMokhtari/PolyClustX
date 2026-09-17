"""Configurable synthetic multi-view datasets for interface examples."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import kneighbors_graph

from pymv.data import MultiViewDataset


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
    """Generate aligned views with per-view distributions and optional kNN relations."""
    if spec.n_samples < spec.n_classes or spec.n_classes < 2:
        raise ValueError("Synthetic data needs at least two classes and one sample per class")
    if not spec.dimensions or any(d < 2 for d in spec.dimensions):
        raise ValueError("Each synthetic view needs at least two features")
    if len(spec.distributions) != len(spec.dimensions):
        raise ValueError("Provide one distribution for each view")
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

    relations = []
    for view in views:
        view_relations = []
        for relation in range(spec.graph_relations):
            neighbors = min(spec.graph_neighbors + relation * 2, spec.n_samples - 1)
            graph = kneighbors_graph(view, neighbors, mode="connectivity", include_self=False)
            view_relations.append(graph.maximum(graph.T).tocsr())
        relations.append(view_relations or None)
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
        metadata={"synthetic": spec.__dict__.copy()},
    )
