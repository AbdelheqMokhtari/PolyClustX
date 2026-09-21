"""Aggregation and scalable result summaries for the interactive display."""

from __future__ import annotations

import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def record_label(record):
    """Human-facing experiment identity; session IDs are implementation details."""
    return (f"{record.spec.name} · {record.dataset_summary['name']} · "
            f"{record.spec.model_name or record.spec.model.upper()}")


def run_label(record, index):
    row = record.rows[index]
    return f"Configuration {row['configuration']} · K={row['K']} · Run {row['repeat']}"


def download_stem(*parts):
    """Make a portable filename from names, not opaque session identifiers."""
    limit = 160 if len(parts) == 1 else 60
    words = [re.sub(r"[^\w-]+", "-", str(part), flags=re.UNICODE).strip("-_")[:limit] for part in parts]
    stem = "_".join(word for word in words if word)
    return stem.encode("utf-8")[:160].decode("utf-8", errors="ignore") or "results"


def record_download_stem(record):
    return download_stem(record.spec.name, record.dataset_summary["name"],
                         record.spec.model_name or record.spec.model.upper(), record.created_at[:19])


def aggregate_results(record):
    """Aggregate repeated successful trials as mean ± standard deviation."""
    groups = {}
    for row in record.rows:
        params = {k: v for k, v in row.get("params", {}).items() if k not in {"random_state", "n_clusters"}}
        key = (row.get("configuration"), row["K"], json.dumps(params, sort_keys=True, default=list))
        groups.setdefault(key, []).append(row)
    output = []
    reserved = {"K", "repeat", "seed", "params", "status", "error", "configuration"}
    for (configuration, clusters, params_json), rows in groups.items():
        successful = [row for row in rows if row.get("status") == "complete"]
        item = {
            "configuration": configuration,
            "K": clusters,
            **json.loads(params_json),
            "successful runs": len(successful),
            "failed runs": len(rows) - len(successful),
        }
        item.update({f"requested {key}": value["requested"]
                     for key, value in rows[0].get("parameter_adjustments", {}).items()})
        numeric = sorted({key for row in successful for key, value in row.items()
                          if key not in reserved and isinstance(value, (int, float, np.number))})
        for metric in numeric:
            values = np.asarray([row[metric] for row in successful if metric in row], dtype=float)
            finite = values[np.isfinite(values)]
            item[metric] = "—" if not finite.size else f"{finite.mean():.4f} ± {finite.std(ddof=0):.4f}"
        output.append(item)
    return output


def compact_confusion(true_labels, predicted_labels, max_labels=30):
    """Match all classes before display compaction, preserving every sample count.

    Predicted IDs are canonical 0..k-1; only their column order changes.
    Ground-truth names remain unchanged. Large displays keep matched pairs
    together and aggregate omitted rows/columns into separate Other buckets.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.ndim != 1 or predicted_labels.shape != true_labels.shape:
        raise ValueError("True and predicted labels must be aligned one-dimensional arrays")
    if not isinstance(max_labels, int) or max_labels < 1:
        raise ValueError("max_labels must be a positive integer")

    rows, truth_codes = np.unique(true_labels, return_inverse=True)
    columns, prediction_codes = np.unique(predicted_labels, return_inverse=True)
    if len(rows) * len(columns) > 4_000_000:
        raise ValueError("Too many classes for an interactive full Hungarian matrix; export predictions for analysis")
    matching = np.zeros((len(rows), len(columns)), dtype=np.int64)
    np.add.at(matching, (truth_codes, prediction_codes), 1)
    matched_rows, matched_cols = linear_sum_assignment(-matching)
    row_order = list(matched_rows) + [i for i in range(len(rows)) if i not in matched_rows]
    column_order = list(matched_cols) + [i for i in range(len(columns)) if i not in matched_cols]
    if max(len(rows), len(columns)) > max_labels:
        # Keep large matched pairs together, rather than matching only top classes.
        mass = [int(matching[row_order[i]].sum()) if i < len(row_order) else 0
                for i in range(max(len(rows), len(columns)))]
        for i, col in enumerate(column_order):
            mass[i] += int(matching[:, col].sum())
        chosen = sorted(np.argsort(-np.asarray(mass), kind="stable")[:max_labels])
        row_order = [row_order[i] for i in chosen if i < len(row_order)]
        column_order = [column_order[i] for i in chosen if i < len(column_order)]
    row_overflow, column_overflow = len(rows) > len(row_order), len(columns) > len(column_order)
    matrix = np.zeros((len(row_order) + row_overflow, len(column_order) + column_overflow), dtype=np.int64)
    row_map = np.full(len(rows), len(row_order))
    col_map = np.full(len(columns), len(column_order))
    row_map[row_order] = np.arange(len(row_order))
    col_map[column_order] = np.arange(len(column_order))
    np.add.at(matrix, (row_map[truth_codes], col_map[prediction_codes]), 1)
    row_names = [str(rows[i]) for i in row_order] + (["Other classes"] if row_overflow else [])
    column_names = [str(i) for i in column_order] + (["Other clusters"] if column_overflow else [])
    return matrix, row_names, column_names


def detailed_rows(record):
    """Flatten parameters into columns without exposing nested diagnostics in tables."""
    return [{**{k: v for k, v in row.items()
                if k not in {"params", "diagnostics", "requested_params", "parameter_adjustments"}},
             **{f"requested {k}": v["requested"] for k, v in row.get("parameter_adjustments", {}).items()},
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
