"""Validated, aligned datasets for single- and multi-view learning."""

from __future__ import annotations

import copy

import numpy as np
from scipy import sparse


def _matrix(value, name):
    result = value.tocsr().astype(float, copy=False) if sparse.issparse(value) else np.asarray(value, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    return result


class MultiViewDataset:
    """Aligned features, graph relations, optional labels and observation mask.

    Each feature view has shape (n_samples, n_features). Each entry in As is
    None, one adjacency, or a list of relations for that view. All matrices
    share the global sample order. A boolean observed_mask of shape (n, views)
    distinguishes observations from placeholders; zeros never imply missingness.
    At least one view must be observed for every sample.

    Labels are optional and only used for evaluation. Modalities names each
    view's modality; omitted modalities are considered the same, unspecified
    modality. Tuple unpacking is retained but discards masks and IDs: use
    the dataset's fit method to preserve missing-view safeguards.
    """

    def __init__(self, Xs, As=None, labels=None, name=None, *, sample_ids=None,
                 observed_mask=None, view_names=None, modalities=None, metadata=None):
        if not isinstance(Xs, (list, tuple)) or not Xs:
            raise ValueError("Xs must contain at least one feature matrix")
        features = {}
        self.Xs = []
        for v, x in enumerate(Xs):
            if id(x) not in features:
                features[id(x)] = _matrix(x, f"Xs[{v}]")
            self.Xs.append(features[id(x)])
        n, v = self.Xs[0].shape[0], len(self.Xs)
        if any(x.shape[0] != n for x in self.Xs):
            raise ValueError("Views must share a sample order; use align_views with sample IDs")
        self.observed_mask = (np.ones((n, v), dtype=bool) if observed_mask is None
                              else np.asarray(observed_mask))
        if self.observed_mask.shape != (n, v) or self.observed_mask.dtype != bool:
            raise ValueError("observed_mask must be a boolean (n_samples, n_views) array")
        self.observed_mask = self.observed_mask.copy()
        if not self.observed_mask.any(axis=1).all():
            raise ValueError("Every sample must have at least one observed view")
        for j, x in enumerate(self.Xs):
            observed = x[self.observed_mask[:, j]]
            values = observed.data if sparse.issparse(observed) else observed
            if not np.isfinite(values).all():
                raise ValueError(f"Observed features in view {j} must be finite")
            if not self.observed_mask[:, j].all():
                x = x.tolil() if sparse.issparse(x) else x.copy()
                x[~self.observed_mask[:, j]] = 0
                self.Xs[j] = x.tocsr() if sparse.issparse(x) else x
        self.As = [None] * v if As is None else list(As)
        if len(self.As) != v:
            raise ValueError("As must contain exactly one entry per view (use None for no graph)")
        graphs = {}
        for j, entry in enumerate(self.As):
            relations = entry if isinstance(entry, (list, tuple)) else [entry]
            converted = []
            if isinstance(entry, (list, tuple)) and (not entry or any(a is None for a in entry)):
                raise ValueError("Relation lists must be nonempty and cannot contain None")
            for adjacency in relations:
                if adjacency is None:
                    converted.append(None)
                    continue
                if hasattr(adjacency, "nodes") and hasattr(adjacency, "edges"):
                    from .graphs import graph_adjacency
                    adjacency = graph_adjacency(adjacency, np.arange(n) if sample_ids is None else sample_ids)
                if id(adjacency) not in graphs:
                    graphs[id(adjacency)] = _matrix(adjacency, f"As[{j}]")
                a = graphs[id(adjacency)]
                values = a.data if sparse.issparse(a) else a
                if a.shape != (n, n) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("Graphs must be finite nonnegative (n_samples, n_samples) matrices")
                converted.append(a)
            self.As[j] = converted if isinstance(entry, (list, tuple)) else converted[0]
        self.labels = None if labels is None else np.asarray(labels).reshape(-1)
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("Labels must have one entry per aligned sample")
        self.sample_ids = np.arange(n) if sample_ids is None else np.asarray(sample_ids).reshape(-1)
        if len(self.sample_ids) != n or len(set(self.sample_ids.tolist())) != n:
            raise ValueError("sample_ids must contain one unique ID per sample")
        if any(i is None or (isinstance(i, (float, np.floating)) and not np.isfinite(i))
               for i in self.sample_ids):
            raise ValueError("Sample IDs must not be missing or nonfinite")
        self.view_names = list(view_names) if view_names is not None else [f"view_{j + 1}" for j in range(v)]
        self.modalities = list(modalities) if modalities is not None else ["unspecified"] * v
        if len(self.view_names) != v or len(set(self.view_names)) != v or len(self.modalities) != v:
            raise ValueError("Provide one unique view name and one modality per view")
        self.name = name
        self.metadata = copy.deepcopy(metadata or {})

    @property
    def n_samples(self):
        return self.Xs[0].shape[0]

    @property
    def n_views(self):
        return len(self.Xs)

    @property
    def n_clusters(self):
        """Number of ground-truth classes, or None without labels."""
        return None if self.labels is None else int(len(np.unique(self.labels)))

    @property
    def feature_dims(self):
        return [x.shape[1] for x in self.Xs]

    @property
    def has_graphs(self):
        return any(any(a is not None for a in entry) if isinstance(entry, (list, tuple))
                   else entry is not None for entry in self.As)

    @property
    def is_complete(self):
        return bool(self.observed_mask.all())

    @property
    def category(self):
        kind = "multimodal" if len(set(self.modalities)) > 1 else "multiview"
        return f"attributed_graph_{kind}" if self.has_graphs else kind

    def fused_graphs(self, method="sum", weights=None):
        """Return one graph per view; weights maps view indices to relation weights."""
        from polyclustx.preprocessing.fusion import fuse_graphs
        weights = weights or {}
        if method not in {"sum", "hadamard"}:
            raise ValueError("method must be 'sum' or 'hadamard'")
        if method == "hadamard" and weights:
            raise ValueError("Hadamard fusion does not accept weights")
        if any(not isinstance(j, int) or j < 0 or j >= self.n_views for j in weights):
            raise ValueError("Relation weights must refer to existing zero-based view indices")
        if any(not isinstance(self.As[j], (list, tuple)) for j in weights):
            raise ValueError("Weights require a list of relations for the selected view")
        return [fuse_graphs(a, method=method, weights=weights.get(j))
                if isinstance(a, (list, tuple)) else a for j, a in enumerate(self.As)]

    def fit(self, model, *, fusion="sum", weights=None):
        """Fit without labels, rejecting unsupported missing views.

        Missing-aware models declare supports_missing_views = True and accept
        ``fit(Xs, As=None, *, observed_mask=...)``. The mask is never inferred.
        """
        kwargs = {}
        if not self.is_complete:
            if not getattr(model, "supports_missing_views", False):
                raise ValueError(f"{type(model).__name__} does not support missing views; "
                                 "explicitly impute_missing(data) or select a missing-aware model")
            kwargs["observed_mask"] = self.observed_mask.copy()
        return model.fit(self.Xs, self.fused_graphs(fusion, weights), **kwargs)

    def __iter__(self):
        yield self.Xs
        yield self.As
        yield self.labels

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.Xs, self.As, self.labels)[idx]

    def __repr__(self):
        return (f"MultiViewDataset({self.name!r}, n_samples={self.n_samples}, n_views={self.n_views}, "
                f"n_clusters={self.n_clusters}, complete={self.is_complete}, category={self.category!r})")

    def summary(self):
        return (f"Dataset: {self.name or '(unnamed)'}\nSamples: {self.n_samples}\nViews: {self.n_views}\n"
                f"Clusters: {self.n_clusters}\nFeature dims: {self.feature_dims}\n"
                f"Category: {self.category}\nComplete: {self.is_complete}")
