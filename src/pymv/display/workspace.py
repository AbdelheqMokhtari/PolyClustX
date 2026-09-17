"""Testable workbench state and execution, with no Streamlit dependency."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import io
import json
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import sparse

from pymv.data import impute_missing, simulate_missing_views
from pymv.metrics import (
    ari_score,
    calinski_harabasz_score,
    clustering_accuracy,
    davies_bouldin_score,
    nmi_score,
    silhouette_score,
)
from pymv.metrics.evaluation import evaluate_metrics, validate_metrics
from pymv.models import get_model

METRICS = {
    "Silhouette": silhouette_score,
    "Davies–Bouldin": davies_bouldin_score,
    "Calinski–Harabasz": calinski_harabasz_score,
    "ACC": clustering_accuracy,
    "NMI": nmi_score,
    "ARI": ari_score,
}
INTERNAL_NAMES = list(METRICS)[:3]
EXTERNAL_NAMES = list(METRICS)[3:]


@dataclass
class ExperimentSpec:
    """A reproducible recipe; K is always explicit and independent of labels."""

    dataset_id: str
    model: str = "lmgec"
    name: str = "Experiment"
    params: dict = field(default_factory=dict)
    clusters: list = field(default_factory=lambda: [2, 3, 4])
    repeats: int = 1
    seed: int = 0
    benchmark: bool = False
    metrics: list = field(default_factory=lambda: list(INTERNAL_NAMES))
    fusion: str = "sum"
    relation_weights: dict = field(default_factory=dict)
    missing_rate: float = 0.0
    imputation: str | None = None
    center: bool = False
    scale: bool = False
    search_space: dict = field(default_factory=dict)
    trials: int = 1


@dataclass
class ExperimentRecord:
    id: str
    created_at: str
    spec: ExperimentSpec
    dataset_summary: dict
    rows: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)


def fingerprint(data):
    """Hash aligned values, graph relations, IDs, masks and labels for provenance."""
    digest = hashlib.sha256()
    digest.update(json.dumps(data.sample_ids.tolist(), default=str).encode())
    digest.update(data.observed_mask.tobytes())
    if data.labels is not None:
        digest.update(json.dumps(data.labels.tolist(), default=str).encode())

    def update(value):
        if value is None:
            digest.update(b"none")
        elif isinstance(value, (list, tuple)):
            digest.update(b"relations")
            for item in value:
                update(item)
        elif sparse.issparse(value):
            a = value.tocsr(copy=True)
            a.sort_indices()
            digest.update(str(a.shape).encode())
            for item in (a.data, a.indices, a.indptr):
                digest.update(item.tobytes())
        else:
            a = np.ascontiguousarray(value)
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(a.tobytes())

    update(data.Xs)
    update(data.As)
    return digest.hexdigest()


class Workspace:
    """Session-local dataset snapshots and independent, newest-first experiments."""

    def __init__(self):
        self.datasets = {}
        self.records = []

    def add_dataset(self, data):
        key = uuid.uuid4().hex
        self.datasets[key] = copy.deepcopy(data)
        return key

    def delete_record(self, record_id):
        self.records = [r for r in self.records if r.id != record_id]

    def run(self, spec, *, progress=None):
        """Append one independent record, retaining per-trial failures and artifacts.

        Structural/configuration errors raise before history changes. Training
        failures are recorded per trial so completed trials remain available.
        Inputs are copied per fit to protect other experiments from model mutation.
        """
        spec = copy.deepcopy(spec)
        if not spec.clusters or any(isinstance(k, bool) or not isinstance(k, int) for k in spec.clusters):
            raise ValueError("Cluster counts must be a nonempty list of integers")
        if spec.repeats < 1 or spec.trials < 1:
            raise ValueError("Repeats and trials must be positive")
        source = self.datasets[spec.dataset_id]
        if any(k < 1 or k > source.n_samples for k in spec.clusters):
            raise ValueError("Each K must be between 1 and the number of samples")
        if "n_clusters" in spec.params or "n_clusters" in spec.search_space:
            raise ValueError("Use the explicit cluster-count field for K, not model/search parameters")
        cls = get_model(spec.model)
        parameters = inspect.signature(cls.__init__).parameters
        if (set(spec.params) | set(spec.search_space)) - set(parameters):
            raise ValueError("Unknown model parameter; consult the model API")
        if any(not isinstance(v, list) or not v for v in spec.search_space.values()):
            raise ValueError("Each random-search parameter needs a nonempty JSON list")
        metric_fns = {name: METRICS[name] for name in spec.metrics}
        validate_metrics(metric_fns, source.labels, benchmark=spec.benchmark)
        data = (
            simulate_missing_views(source, spec.missing_rate, random_state=spec.seed)
            if spec.missing_rate
            else copy.deepcopy(source)
        )
        if spec.imputation:
            data = impute_missing(data, spec.imputation)
        if not data.is_complete and not getattr(cls, "supports_missing_views", False):
            raise ValueError(f"{cls.__name__} does not support missing views. Choose explicit mean/zero imputation.")
        original_mask = source.observed_mask.tolist()
        if "missing_simulation" in data.metadata:
            original_mask = data.metadata.get("imputation", {}).get("observed_mask", data.observed_mask.tolist())
        data.As = data.fused_graphs(spec.fusion, {int(k): v for k, v in spec.relation_weights.items()})
        if spec.center or spec.scale:
            if not data.is_complete:
                raise ValueError("Generic scaling requires explicit imputation for incomplete data")
            from pymv.preprocessing import preprocess_features

            data.Xs = preprocess_features(data.Xs, center=spec.center, scale=spec.scale)
        summary = {
            "name": source.name,
            "fingerprint": fingerprint(source),
            "n_samples": source.n_samples,
            "n_views": source.n_views,
            "category": source.category,
            "feature_dims": source.feature_dims,
            "view_names": source.view_names,
            "modalities": source.modalities,
            "metadata": data.metadata,
            "sample_ids": source.sample_ids.tolist(),
            "observed_mask": original_mask,
            "evaluation_space": "learned embedding (Euclidean)",
            "labels_used": spec.benchmark,
        }
        record = ExperimentRecord(uuid.uuid4().hex, datetime.now(timezone.utc).isoformat(), spec, summary)
        self.records.insert(0, record)
        rng = np.random.default_rng(spec.seed)
        configurations = []
        for _ in range(spec.trials if spec.search_space else 1):
            params = copy.deepcopy(spec.params)
            params.update({k: values[int(rng.integers(len(values)))] for k, values in spec.search_space.items()})
            configurations.append(params)
        total = len(configurations) * len(spec.clusters) * spec.repeats
        for params in configurations:
            for k in spec.clusters:
                for repeat in range(spec.repeats):
                    run_seed = spec.seed + repeat
                    kwargs = {**params, "n_clusters": k}
                    if "random_state" in parameters:
                        kwargs["random_state"] = run_seed
                    row = {"K": k, "repeat": repeat + 1, "seed": run_seed, "params": kwargs, "status": "complete"}
                    artifact = {}
                    started = time.perf_counter()
                    try:
                        model = cls(**kwargs)
                        copy.deepcopy(data).fit(model)
                        embedding = getattr(model, "embedding_", None)
                        row.update(
                            evaluate_metrics(
                                metric_fns, data.labels, model.labels_, embedding, benchmark=spec.benchmark
                            )
                        )
                        artifact = {"labels": np.asarray(model.labels_).copy()}
                        if embedding is not None:
                            artifact["embedding"] = np.asarray(embedding).copy()
                        loss = getattr(model, "loss_history_", None)
                        if loss is not None:
                            artifact["loss"] = np.asarray(loss)
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    row["seconds"] = time.perf_counter() - started
                    record.rows.append(row)
                    record.artifacts.append(artifact)
                    if progress:
                        progress(len(record.rows) / total)
        return record


def export_record(record):
    """Downloadable ZIP: JSON recipe/provenance, CSV scores, per-trial NPZ outputs.

    Original feature data and ground-truth labels are not exported. Reload the
    original dataset (checking its fingerprint) to reproduce a run. Session IDs
    are local identifiers, not paths to persisted datasets.
    """
    from pymv import __version__

    buffer = io.BytesIO()
    payload = {
        "pymv_version": __version__,
        "id": record.id,
        "created_at": record.created_at,
        "spec": asdict(record.spec),
        "dataset": record.dataset_summary,
        "results": record.rows,
    }

    def json_safe(value):
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return json_safe(value.item())
        return value

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("experiment.json", json.dumps(json_safe(payload), indent=2, allow_nan=False))
        table = io.StringIO()
        fields = list(dict.fromkeys(k for row in record.rows for k in row))
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record.rows)
        archive.writestr("metrics.csv", table.getvalue())
        for i, artifact in enumerate(record.artifacts):
            output = io.BytesIO()
            np.savez_compressed(output, **artifact)
            archive.writestr(f"trial_{i + 1}.npz", output.getvalue())
    return buffer.getvalue()
