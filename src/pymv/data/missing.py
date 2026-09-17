"""Missing-view alignment, simulation and explicit baseline imputation."""

from __future__ import annotations

import copy

import numpy as np
from scipy import sparse

from .dataset import MultiViewDataset


def align_views(Xs, view_ids, *, graphs=None, labels=None, label_ids=None, **kwargs):
    """Align per-view IDs by union in first-occurrence order, preserving missingness.

    Graphs use each view's local ID order and are expanded to the union order.
    Global graphs can instead be attached after alignment. Labels require
    explicit label_ids covering the union exactly.
    """
    if not Xs or len(Xs) != len(view_ids):
        raise ValueError("Provide an ID vector for every view")
    ids = [np.asarray(i).reshape(-1) for i in view_ids]
    for x, index in zip(Xs, ids):
        if len(index) != x.shape[0] or len(set(index.tolist())) != len(index):
            raise ValueError("Each view needs unique IDs matching its rows")
    union = list(dict.fromkeys(i for index in ids for i in index.tolist()))
    positions = {sample: i for i, sample in enumerate(union)}
    n = len(union)
    mask = np.zeros((n, len(Xs)), dtype=bool)
    aligned, aligned_graphs = [], []
    graphs = [None] * len(Xs) if graphs is None else graphs
    if len(graphs) != len(Xs):
        raise ValueError("Provide one graph entry per view")
    for j, (x, index, relations) in enumerate(zip(Xs, ids, graphs)):
        rows = np.array([positions[i] for i in index.tolist()])
        mask[rows, j] = True
        selector = sparse.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(n, len(rows)))
        if sparse.issparse(x):
            aligned.append(selector @ x)
        else:
            a = np.zeros((n, x.shape[1]), dtype=float)
            a[rows] = x
            aligned.append(a)

        def expand(graph):
            if graph is None:
                return None
            if graph.shape != (len(rows), len(rows)):
                raise ValueError("Local graph shape must match the corresponding view IDs")
            return (selector @ sparse.csr_matrix(graph) @ selector.T).tocsr()

        aligned_graphs.append([expand(g) for g in relations] if isinstance(relations, (list, tuple))
                              else expand(relations))
    y = None
    if labels is not None:
        if label_ids is None:
            raise ValueError("label_ids are required when aligning labeled views")
        flat_labels, flat_ids = np.asarray(labels).reshape(-1), np.asarray(label_ids).reshape(-1)
        if len(flat_labels) != len(flat_ids) or len(set(flat_ids.tolist())) != len(flat_ids):
            raise ValueError("Labels require unique matching label_ids")
        mapping = dict(zip(flat_ids.tolist(), flat_labels.tolist()))
        if set(mapping) != set(union):
            raise ValueError("label_ids must cover exactly the union of sample IDs")
        y = np.asarray([mapping[i] for i in union])
    return MultiViewDataset(aligned, aligned_graphs, y, sample_ids=np.asarray(union, dtype=object),
                            observed_mask=mask, **kwargs)


def simulate_missing_views(data, missing_rate=0.2, *, random_state=0):
    """Remove a fraction of currently observed sample-view pairs on a copy.

    Constrained MCAR-style sampling chooses uniformly among removable pairs,
    never removing a sample's last view. This retention constraint means the
    process is not independent Bernoulli MCAR. The exact target count is
    floor(missing_rate * observed_pairs). Impossible rates raise.
    Graphs and labels remain unchanged: this simulates missing features only.
    """
    if not np.isfinite(missing_rate) or not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    result = copy.deepcopy(data)
    mask = result.observed_mask.copy()
    count = int(np.floor(missing_rate * mask.sum()))
    capacity = int(mask.sum() - data.n_samples)
    if missing_rate > 0 and data.n_views == 1:
        raise ValueError("Cannot remove the only view while retaining every sample")
    if count > capacity:
        raise ValueError(f"Requested rate exceeds removable capacity ({capacity} sample-view pairs)")
    rng = np.random.default_rng(random_state)
    removed = 0
    remaining = mask.sum(axis=1)
    for i, j in rng.permutation(np.argwhere(mask)):
        if removed == count:
            break
        if remaining[i] > 1:
            mask[i, j] = False
            remaining[i] -= 1
            removed += 1
    result.metadata["missing_simulation"] = {
        "distribution": "constrained_mcar", "requested_rate": float(missing_rate),
        "removed_pairs": removed, "achieved_rate": removed / int(data.observed_mask.sum()),
        "random_state": random_state,
    }
    return MultiViewDataset(result.Xs, result.As, result.labels, result.name,
                            sample_ids=result.sample_ids, observed_mask=mask,
                            view_names=result.view_names, modalities=result.modalities, metadata=result.metadata)


def impute_missing(data, strategy="mean"):
    """Explicit per-view mean or zero baseline, returning a complete dataset copy.

    Means use observed rows only. The original mask is retained in metadata.
    This is preprocessing, not native incomplete-view learning. Mean imputation
    can densify sparse features.
    """
    if strategy not in {"mean", "zero"}:
        raise ValueError("strategy must be 'mean' or 'zero'")
    result = copy.deepcopy(data)
    for j, x in enumerate(result.Xs):
        observed = data.observed_mask[:, j]
        if observed.all():
            continue
        if strategy == "mean" and not observed.any():
            raise ValueError(f"Cannot mean-impute entirely missing view {j}")
        if strategy == "mean":
            x = x.toarray() if sparse.issparse(x) else x.copy()
            x[~observed] = np.asarray(x[observed].mean(axis=0)).reshape(-1)
            result.Xs[j] = x
    result.metadata["imputation"] = {"strategy": strategy, "observed_mask": data.observed_mask.tolist()}
    result.observed_mask[:] = True
    return result
