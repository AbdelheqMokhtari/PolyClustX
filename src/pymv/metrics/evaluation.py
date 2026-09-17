"""Shared, label-safe scoring for command-line and interactive experiments."""

from collections import OrderedDict

import numpy as np

from .clustering import calinski_harabasz_score, davies_bouldin_score, silhouette_score

INTERNAL_METRICS = {silhouette_score, davies_bouldin_score, calinski_harabasz_score}


def is_internal(metric):
    return metric in INTERNAL_METRICS or getattr(metric, "requires_features", False)


def higher_is_better(metric):
    return getattr(metric, "higher_is_better", metric is not davies_bouldin_score)


def validate_metrics(metrics, labels, *, benchmark=True):
    if (labels is None or not benchmark) and any(not is_internal(fn) for fn in metrics.values()):
        raise ValueError("External metrics require labels and benchmark=True; select internal metrics instead")


def evaluate_metrics(metrics, labels, predictions, embedding=None, *, benchmark=True):
    """Internal scores use the learned embedding; labels are only external targets.

    Degenerate clusterings (one cluster or one per sample) have undefined
    internal indices and return NaN. No fabricated zero scores are reported.
    """
    validate_metrics(metrics, labels, benchmark=benchmark)
    result = OrderedDict()
    predictions = np.asarray(predictions).reshape(-1)
    for name, fn in metrics.items():
        if is_internal(fn):
            if embedding is None:
                raise ValueError("Internal metrics require model.embedding_")
            if not 1 < len(np.unique(predictions)) < len(predictions):
                result[name] = float("nan")
                continue
            result[name] = float(fn(embedding, predictions))
        else:
            result[name] = float(fn(labels, predictions))
    return result
