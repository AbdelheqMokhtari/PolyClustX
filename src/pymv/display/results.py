"""Aggregation and scalable result summaries for the interactive display."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def aggregate_results(record):
    """Aggregate repeated successful trials as mean ± standard deviation."""
    groups = {}
    for row in record.rows:
        params = {k: v for k, v in row.get("params", {}).items() if k not in {"random_state", "n_clusters"}}
        key = (row["K"], json.dumps(params, sort_keys=True, default=list))
        groups.setdefault(key, []).append(row)
    output = []
    reserved = {"K", "repeat", "seed", "params", "status", "error", "configuration"}
    for (clusters, params_json), rows in groups.items():
        successful = [row for row in rows if row.get("status") == "complete"]
        item = {
            "K": clusters,
            **json.loads(params_json),
            "successful runs": len(successful),
            "failed runs": len(rows) - len(successful),
        }
        numeric = sorted({key for row in successful for key, value in row.items()
                          if key not in reserved and isinstance(value, (int, float, np.number))})
        for metric in numeric:
            values = np.asarray([row[metric] for row in successful if metric in row], dtype=float)
            finite = values[np.isfinite(values)]
            item[metric] = "—" if not finite.size else f"{finite.mean():.4f} ± {finite.std(ddof=0):.4f}"
        output.append(item)
    return output


def compact_confusion(true_labels, predicted_labels, max_labels=30):
    """Contingency matrix with optimally aligned columns and bounded labels.

    Cluster IDs are arbitrary: Hungarian matching reorders columns to match
    classes, without relabeling unmatched clusters or losing sample counts.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.ndim != 1 or predicted_labels.shape != true_labels.shape:
        raise ValueError("True and predicted labels must be aligned one-dimensional arrays")
    if not isinstance(max_labels, int) or max_labels < 1:
        raise ValueError("max_labels must be a positive integer")

    def top(values):
        labels, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        return labels[order[:max_labels]]

    rows, columns = top(true_labels), top(predicted_labels)
    # Match only displayed labels: allocation remains bounded for thousands of classes.
    matching = np.zeros((len(rows), len(columns)), dtype=int)
    row_index = {value: index for index, value in enumerate(rows.tolist())}
    col_index = {value: index for index, value in enumerate(columns.tolist())}
    for truth, prediction in zip(true_labels.tolist(), predicted_labels.tolist()):
        if truth in row_index and prediction in col_index:
            matching[row_index[truth], col_index[prediction]] += 1
    matched_rows, matched_cols = linear_sum_assignment(-matching)
    row_order = list(matched_rows) + [i for i in range(len(rows)) if i not in matched_rows]
    rows = rows[row_order]
    ordered = list(matched_cols) + [i for i in range(len(columns)) if i not in matched_cols]
    columns = columns[ordered]
    row_overflow = len(np.unique(true_labels)) > len(rows)
    column_overflow = len(np.unique(predicted_labels)) > len(columns)
    matrix = np.zeros((len(rows) + row_overflow, len(columns) + column_overflow), dtype=int)
    row_map = {value: index for index, value in enumerate(rows.tolist())}
    column_map = {value: index for index, value in enumerate(columns.tolist())}
    for truth, prediction in zip(true_labels.tolist(), predicted_labels.tolist()):
        row = row_map.get(truth, len(rows))
        column = column_map.get(prediction, len(columns))
        matrix[row, column] += 1
    row_names = [str(value) for value in rows] + (["Other classes"] if row_overflow else [])
    column_names = [str(value) for value in columns] + (["Other clusters"] if column_overflow else [])
    return matrix, row_names, column_names


def detailed_rows(record):
    """Flatten parameters into columns without exposing nested diagnostics in tables."""
    return [{**{k: v for k, v in row.items() if k not in {"params", "diagnostics"}},
             **{f"param.{k}": v for k, v in row.get("params", {}).items()}}
            for row in record.rows]


def cluster_details(predicted_labels, true_labels=None):
    """Size and optional majority-label purity for each predicted cluster."""
    predicted_labels = np.asarray(predicted_labels)
    true_labels = None if true_labels is None else np.asarray(true_labels)
    rows = []
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        row = {"cluster": str(cluster), "samples": int(mask.sum()), "fraction": float(mask.mean())}
        if true_labels is not None:
            labels, counts = np.unique(true_labels[mask], return_counts=True)
            best = int(np.argmax(counts))
            row.update(majority_label=str(labels[best]), purity=float(counts[best] / counts.sum()))
        rows.append(row)
    return rows
