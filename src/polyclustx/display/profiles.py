"""Compact dataset diagnostics for model selection in the display."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _bytes(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return np.asarray(matrix).nbytes


def _matrix_profile(matrix, observed, *, sample_rows=5000):
    matrix = matrix[observed]
    rows, columns = matrix.shape
    total = max(rows * columns, 1)
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        values = matrix.data
        total_sum = float(matrix.sum())
        total_square = float(matrix.multiply(matrix).sum())
        mean = total_sum / total
        variance = max(total_square / total - mean**2, 0.0)
        minimum = min(float(values.min()), 0.0) if values.size else 0.0
        maximum = max(float(values.max()), 0.0) if values.size else 0.0
        density = matrix.nnz / total
        column_mean = np.asarray(matrix.mean(axis=0)).ravel()
        column_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        feature_variance = np.maximum(column_square - column_mean**2, 0)
    else:
        values = np.asarray(matrix)
        if rows > sample_rows:
            indices = np.linspace(0, rows - 1, sample_rows, dtype=int)
            values = values[indices]
        mean = float(np.mean(values))
        variance = float(np.var(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        density = float(np.count_nonzero(values) / values.size)
        feature_variance = np.var(values, axis=0)
    quantiles = np.quantile(feature_variance, [0, 0.25, 0.5, 0.75, 1]).tolist()
    return {
        "shape": (rows, columns),
        "dtype": str(matrix.dtype),
        "mean": mean,
        "variance": variance,
        "std": variance**0.5,
        "min": minimum,
        "max": maximum,
        "density": density,
        "zero_fraction": 1 - density,
        "feature_variance_quantiles": quantiles,
        "memory_bytes": _bytes(matrix),
    }


def _graph_profile(graph):
    graph = sparse.csr_matrix(graph) if sparse.issparse(graph) else np.asarray(graph)
    n = graph.shape[0]
    nonzero = graph.nnz if sparse.issparse(graph) else int(np.count_nonzero(graph))
    if sparse.issparse(graph):
        symmetric = (graph - graph.T).nnz == 0
        values = graph.data
        diagonal = graph.diagonal()
    else:
        symmetric = bool(np.allclose(graph, graph.T))
        values = graph[graph != 0]
        diagonal = np.diag(graph)
    return {
        "nodes": n,
        "edges_nonzero": int(nonzero),
        "density": nonzero / max(n * n, 1),
        "symmetric": symmetric,
        "self_loops": int(np.count_nonzero(diagonal)),
        "weighted": bool(values.size and not np.all(np.isin(values, [0, 1]))),
        "memory_bytes": _bytes(graph),
    }


def dataset_profile(data):
    """Return label, view, graph, missingness, and memory diagnostics."""
    views = []
    graphs = []
    seen = set()
    graph_objects = set()
    memory = data.observed_mask.nbytes + data.sample_ids.nbytes
    for index, (name, matrix) in enumerate(zip(data.view_names, data.Xs)):
        profile = _matrix_profile(matrix, data.observed_mask[:, index])
        profile.update(
            index=index,
            name=name,
            modality=data.modalities[index],
            observed=int(data.observed_mask[:, index].sum()),
            missing=int((~data.observed_mask[:, index]).sum()),
        )
        feature, graph_keys = data.metadata.get("view_selection", {}).get(name, (name, []))
        profile.update(source_feature=feature,
                       description=data.metadata.get("feature_descriptions", {}).get(feature, ""))
        views.append(profile)
        if id(matrix) not in seen:
            memory += _bytes(matrix)
            seen.add(id(matrix))
        relations = data.As[index] if isinstance(data.As[index], (list, tuple)) else [data.As[index]]
        graph_rows = []
        for relation_index, graph in enumerate(relations):
            if graph is not None:
                graph_objects.add(id(graph))
                item = _graph_profile(graph)
                item.update(view=index, view_name=name, relation=relation_index + 1)
                keys = (graph_keys if len(relations) == 1 else graph_keys[relation_index:relation_index + 1])
                item.update(source_graph=" + ".join(keys), description="; ".join(
                    data.metadata.get("graph_descriptions", {}).get(key, key) for key in keys))
                graph_rows.append(item)
                if id(graph) not in seen:
                    memory += item["memory_bytes"]
                    seen.add(id(graph))
        graphs.extend(graph_rows)

    labels = None
    if data.labels is not None:
        values, counts = np.unique(data.labels, return_counts=True)
        probabilities = counts / counts.sum()
        labels = {
            "n_classes": len(values),
            "counts": [(str(value), int(count)) for value, count in zip(values, counts)],
            "smallest_class": int(counts.min()),
            "largest_class": int(counts.max()),
            "imbalance_ratio": float(counts.max() / counts.min()),
            "normalized_entropy": float(
                -(probabilities * np.log(probabilities)).sum() / np.log(len(values)) if len(values) > 1 else 1.0
            ),
        }
        memory += data.labels.nbytes
    observed_per_sample = data.observed_mask.sum(axis=1)
    return {
        "name": data.name or "Dataset",
        "n_samples": data.n_samples,
        "n_views": data.n_views,
        "n_graphs": len(graph_objects),
        "category": data.category,
        "complete": data.is_complete,
        "observed_pairs": int(data.observed_mask.sum()),
        "missing_pairs": int((~data.observed_mask).sum()),
        "minimum_views_per_sample": int(observed_per_sample.min()),
        "maximum_views_per_sample": int(observed_per_sample.max()),
        "memory_bytes": int(memory),
        "views": views,
        "graphs": graphs,
        "labels": labels,
    }


def format_bytes(size):
    """Human-readable binary size."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
