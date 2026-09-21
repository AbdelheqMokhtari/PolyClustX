"""Testable workbench state and execution, with no Streamlit dependency."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import itertools
import json
import math
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from scipy import sparse

from pymv.experiment.exports import (
    aggregate_runs,
    diagnostics,
    json_safe,
    model_details,
    new_report,
    report_csv,
    report_json,
    save_report,
    timestamp,
)
from pymv.metrics import (
    ari_score,
    calinski_harabasz_score,
    clustering_accuracy,
    clustering_f1_score,
    davies_bouldin_score,
    nmi_score,
    silhouette_score,
)
from pymv.metrics.evaluation import evaluate_metrics, higher_is_better, validate_metrics
from pymv.models import get_model

from .artifacts import StoredArtifact
from .preprocessing import PreprocessingSpec, prepare_dataset

METRICS = {
    "Silhouette": silhouette_score,
    "Davies–Bouldin": davies_bouldin_score,
    "Calinski–Harabasz": calinski_harabasz_score,
    "ACC": clustering_accuracy,
    "NMI": nmi_score,
    "ARI": ari_score,
    "F1": clustering_f1_score,
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
    preprocessing: PreprocessingSpec | None = None
    search_strategy: str = "random"
    objective: str | None = None
    kind: str = "experiment"
    model_name: str | None = None


@dataclass
class ExperimentRecord:
    id: str
    created_at: str
    spec: ExperimentSpec
    dataset_summary: dict
    rows: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    completed_at: str | None = None
    json_path: str | None = None
    model_details: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    _artifact_directory: object | None = field(default=None, repr=False)


@dataclass
class DatasetResource:
    """A session dataset that can release its materialized matrices from RAM."""

    id: str
    name: str
    loader: Callable
    source: str = "memory"
    hint: dict = field(default_factory=dict)
    cleanup: Callable | None = field(default=None, repr=False)
    _data: object | None = field(default=None, repr=False)

    @property
    def loaded(self):
        return self._data is not None

    def load(self):
        if self._data is None:
            self._data = self.loader()
        self._data.name = self.name
        return self._data

    def unload(self):
        self._data = None

    def __getattr__(self, name):
        # Keeps read-only compatibility with the former datasets mapping.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.load(), name)

    def __setattr__(self, name, value):
        fields = {"id", "name", "loader", "source", "hint", "cleanup", "_data"}
        data = self.__dict__.get("_data")
        if name not in fields and data is not None and hasattr(data, name):
            setattr(data, name, value)
            return
        object.__setattr__(self, name, value)


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
        self.experiment_counter = 0
        self.preprocessing = {}
        self.models = {}
        self.benchmark = {}
        self.batches = []
        self.name_counters = {}
        self.pending_datasets = set()
        self.plots = {}

    @staticmethod
    def unique_name(name, existing):
        name = str(name).strip()
        if not name:
            raise ValueError("Choose a nonempty name")
        candidate, index = name, 2
        while candidate in existing:
            candidate = f"{name}#{index}"
            index += 1
        return candidate

    def suggested_name(self, prefix):
        """Human-readable, monotonic defaults; deleting items never reuses a number."""
        return f"{prefix}#{self.name_counters.get(prefix, 0) + 1}"

    def advance_name(self, prefix):
        self.name_counters[prefix] = self.name_counters.get(prefix, 0) + 1

    def apply_dataset(self, dataset_id, recipe):
        """Validate preparation before making a newly loaded dataset available to jobs."""
        prepared = prepare_dataset(self.load_dataset(dataset_id), recipe)
        resource = self.datasets[dataset_id]
        resource.hint["prepared_revision"] = resource.hint.get("prepared_revision", 0) + 1
        from .profiles import dataset_profile

        profile = dataset_profile(prepared)
        self.preprocessing[dataset_id] = copy.deepcopy(recipe)
        self.datasets[dataset_id].hint.update(prepared_complete=prepared.is_complete,
                                               prepared_profile=profile,
                                               prepared_observed=prepared.observed_mask.sum(axis=0).tolist())
        self.pending_datasets.discard(dataset_id)

    def rename_dataset(self, dataset_id, name):
        resource = self.datasets[dataset_id]
        resource.name = self.unique_name(name, {r.name for key, r in self.datasets.items() if key != dataset_id})
        if resource.loaded:
            resource._data.name = resource.name

    def compatible_pairs(self, dataset_ids, model_ids, *, clusters_by_dataset=None):
        """Find valid pairs without fitting; materialize at most one source at a time."""
        pairs = []
        for dataset_id in dataset_ids:
            if dataset_id in self.pending_datasets:
                continue
            resource = self.datasets[dataset_id]
            if "complete" not in resource.hint or "n_samples" not in resource.hint:
                data = self.load_dataset(dataset_id)
                resource.hint.update(complete=data.is_complete, n_samples=data.n_samples)
            recipe = self.preprocessing.get(dataset_id, PreprocessingSpec())
            complete = recipe.will_be_complete(resource.hint["complete"])
            for model_id in model_ids:
                settings = self.models[model_id]
                cls = get_model(settings["model"])
                if not complete and not getattr(cls, "supports_missing_views", False):
                    continue
                clusters = (clusters_by_dataset or {}).get(dataset_id, settings["clusters"])
                if all(1 <= k <= resource.hint["n_samples"] for k in clusters):
                    pairs.append((dataset_id, model_id))
        return pairs

    def add_model(self, name, model, params, clusters):
        """Save an independent configuration without retaining a fitted model."""
        key = uuid.uuid4().hex
        name = self.unique_name(name, {item["name"] for item in self.models.values()})
        self.models[key] = {"name": name, "model": model, "params": copy.deepcopy(params),
                            "clusters": list(clusters)}
        return key

    def benchmark_mode(self, dataset_id):
        if dataset_id not in self.benchmark:
            self.benchmark[dataset_id] = self.load_dataset(dataset_id).labels is not None
        return self.benchmark[dataset_id]

    def run_batch(self, specs, *, name=None, output_dir=None, progress=None):
        """Run selected combinations sequentially and persist one combined report."""
        specs = list(specs)
        if not specs:
            raise ValueError("Select at least one dataset/model combination")
        if any(spec.kind != "experiment" for spec in specs):
            raise ValueError("Run searches independently; run_batch accepts experiment configurations")
        if len({spec.benchmark for spec in specs}) > 1:
            raise ValueError("Benchmark and real-world datasets cannot be combined in one experiment")
        # Validate all combinations before fitting anything. Only one source is loaded at a time.
        for spec in specs:
            self.run(spec, validate_only=True)
        report = new_report(name=name)
        record_ids = []
        counter = self.experiment_counter
        for i, spec in enumerate(specs):
            callback = (lambda value, i=i: progress((i + value) / len(specs))) if progress else None
            record = self.run(spec, progress=callback, save_json=False)
            record_ids.append(record.id)
            report["experiments"].extend(record.report["experiments"])
        report["completed_at"] = timestamp()
        self.experiment_counter = counter + 1
        path = save_report(report, output_dir)
        self.batches.insert(0, {"name": name, "report": report, "record_ids": record_ids, "json_path": str(path)})
        return report

    def add_dataset(self, data, *, source="memory"):
        key = uuid.uuid4().hex
        snapshot = copy.deepcopy(data)
        self.datasets[key] = DatasetResource(
            key,
            self.unique_name(data.name or "Dataset", {r.name for r in self.datasets.values()}),
            loader=lambda value=snapshot: copy.deepcopy(value),
            source=source,
            hint={
                "n_samples": data.n_samples,
                "n_views": data.n_views,
                "n_clusters": data.n_clusters,
                "complete": data.is_complete,
                "category": data.category,
            },
            _data=snapshot,
        )
        return key

    def add_lazy_dataset(self, name, loader, *, source="catalog", hint=None):
        key = uuid.uuid4().hex
        name = self.unique_name(name, {r.name for r in self.datasets.values()})
        self.datasets[key] = DatasetResource(key, name, loader, source=source, hint=hint or {})
        return key

    def load_dataset(self, dataset_id, *, unload_others=True):
        if dataset_id not in self.datasets:
            raise KeyError("This dataset is no longer in the session")
        if unload_others:
            for key, resource in self.datasets.items():
                if key != dataset_id:
                    resource.unload()
        return self.datasets[dataset_id].load()

    def unload_dataset(self, dataset_id):
        self.datasets[dataset_id].unload()

    def remove_dataset(self, dataset_id):
        self.pending_datasets.discard(dataset_id)
        self.preprocessing.pop(dataset_id, None)
        self.benchmark.pop(dataset_id, None)
        resource = self.datasets.pop(dataset_id)
        resource.unload()
        if resource.cleanup:
            resource.cleanup()

    def delete_record(self, record_id):
        for record in self.records:
            if record.id == record_id and record._artifact_directory is not None:
                record._artifact_directory.cleanup()
        self.records = [r for r in self.records if r.id != record_id]
        for batch in self.batches:
            batch["record_ids"] = [key for key in batch["record_ids"] if key != record_id]
            batch["report"]["experiments"] = [entry for entry in batch["report"]["experiments"]
                                               if entry.get("record_id") != record_id]
        self.batches = [batch for batch in self.batches if batch["record_ids"]]

    def run(self, spec, *, progress=None, output_dir=None, save_json=True, validate_only=False):
        """Append one independent record, retaining per-trial failures and artifacts.

        Structural/configuration errors raise before history changes. Training
        failures are recorded per trial so completed trials remain available.
        Inputs are copied per fit to protect other experiments from model mutation.
        """
        spec = copy.deepcopy(spec)
        if spec.dataset_id in self.pending_datasets:
            raise ValueError("Apply changes on the Dataset page before running this dataset")
        if not spec.clusters or any(isinstance(k, bool) or not isinstance(k, int) for k in spec.clusters):
            raise ValueError("Cluster counts must be a nonempty list of integers")
        if (not isinstance(spec.repeats, int) or isinstance(spec.repeats, bool) or not 1 <= spec.repeats <= 20
                or not isinstance(spec.trials, int) or isinstance(spec.trials, bool) or spec.trials < 1):
            raise ValueError("Repeats must be between 1 and 20 and trials must be positive")
        if (not isinstance(spec.seed, int) or isinstance(spec.seed, bool)
                or not 0 <= spec.seed <= 2**32 - spec.repeats):
            raise ValueError("Choose a nonnegative seed below 2**32 minus the number of repeats")
        if spec.search_strategy not in {"random", "grid"}:
            raise ValueError("Unknown search strategy")
        source = self.load_dataset(spec.dataset_id)
        if any(k < 1 or k > source.n_samples for k in spec.clusters):
            raise ValueError("Each K must be between 1 and the number of samples")
        if "n_clusters" in spec.params or "n_clusters" in spec.search_space:
            raise ValueError("Use the explicit cluster-count field for K, not model/search parameters")
        cls = get_model(spec.model)
        parameters = inspect.signature(cls.__init__).parameters
        if (set(spec.params) | set(spec.search_space)) - set(parameters):
            raise ValueError("Unknown model parameter; consult the model API")
        if any(not isinstance(v, list) or not v for v in spec.search_space.values()):
            raise ValueError("Each search parameter needs a nonempty candidate list")
        if spec.kind == "search" and spec.objective not in spec.metrics:
            raise ValueError("Choose a search objective from the selected metrics")
        metric_fns = {name: METRICS[name] for name in spec.metrics}
        validate_metrics(metric_fns, source.labels, benchmark=spec.benchmark)
        recipe = spec.preprocessing or PreprocessingSpec(
            missing_rate=spec.missing_rate,
            missing_seed=spec.seed,
            imputation=spec.imputation,
            fusion=spec.fusion,
            relation_weights={int(k): v for k, v in spec.relation_weights.items()},
            mode="all" if spec.center or spec.scale else "none",
        )
        if spec.preprocessing is None:
            recipe.shared.center = spec.center
            recipe.shared.scale = spec.scale
        data = prepare_dataset(source, recipe)
        if not data.is_complete and not getattr(cls, "supports_missing_views", False):
            raise ValueError(f"{cls.__name__} does not support missing views. Choose explicit mean/zero imputation.")
        count = math.prod(len(v) for v in spec.search_space.values()) if spec.search_strategy == "grid" else spec.trials
        total = (count if spec.search_space else 1) * len(spec.clusters) * spec.repeats
        if total > 2000:
            raise ValueError("This session job exceeds 2,000 fits. Reduce the grid, K values or repeated runs.")
        if validate_only:
            return None
        original_mask = source.observed_mask.tolist()
        if "missing_simulation" in data.metadata:
            original_mask = data.metadata.get("imputation", {}).get("observed_mask", data.observed_mask.tolist())
        summary = {
            "name": source.name,
            "fingerprint": fingerprint(source),
            "n_samples": source.n_samples,
            "n_views": source.n_views,
            "category": source.category,
            "complete": data.is_complete,
            "n_classes": source.n_clusters if spec.benchmark else None,
            "observed_per_view": data.observed_mask.sum(axis=0).tolist(),
            "graph_relations_per_view": [len(a) if isinstance(a, (list, tuple)) else int(a is not None)
                                         for a in source.As],
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
        record.model_details = model_details(cls)
        if spec.model_name:
            record.model_details["display_name"] = spec.model_name
        record._artifact_directory = tempfile.TemporaryDirectory(prefix="pymv-runs-")
        self.records.insert(0, record)
        if spec.kind == "experiment":
            self.experiment_counter += 1
        rng = np.random.default_rng(spec.seed)
        if spec.search_strategy == "grid":
            configurations = (dict(zip(spec.search_space, values))
                              for values in itertools.product(*spec.search_space.values()))
        else:
            configurations = ({k: values[int(rng.integers(len(values)))] for k, values in spec.search_space.items()}
                              for _ in range(spec.trials if spec.search_space else 1))
        configuration = 0
        for candidates in configurations:
            params = {**spec.params, **candidates}
            for k in spec.clusters:
                configuration += 1
                for repeat in range(spec.repeats):
                    run_seed = spec.seed + repeat
                    kwargs = {**params, "n_clusters": k}
                    if recipe.changes_features and "preprocess" in parameters:
                        kwargs["preprocess"] = False
                    if "random_state" in parameters:
                        kwargs["random_state"] = run_seed
                    row = {"K": k, "repeat": repeat + 1, "seed": run_seed, "params": kwargs, "status": "complete",
                           "configuration": configuration, "started_at": timestamp()}
                    artifact = {}
                    started = time.perf_counter()
                    try:
                        requested = dict(kwargs)
                        resolver = getattr(cls, "resolve_fit_parameters", None)
                        if resolver is not None:
                            kwargs = resolver(data.Xs, kwargs)
                        row["params"] = kwargs
                        row["requested_params"] = requested
                        row["parameter_adjustments"] = {
                            key: {"requested": requested[key], "effective": value}
                            for key, value in kwargs.items() if key in requested and value != requested[key]
                        }
                        model = cls(**kwargs)
                        copy.deepcopy(data).fit(model)
                        row["diagnostics"] = diagnostics(model)
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
                        if spec.benchmark and data.labels is not None:
                            artifact["true_labels"] = np.asarray(data.labels).copy()
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    row["seconds"] = time.perf_counter() - started
                    row["completed_at"] = timestamp()
                    record.rows.append(row)
                    if artifact:
                        from pathlib import Path
                        path = Path(record._artifact_directory.name) / f"run-{len(record.rows)}.npz"
                        record.artifacts.append(StoredArtifact(path, artifact))
                    else:
                        record.artifacts.append({})
                    if progress:
                        progress(len(record.rows) / total)
        record.completed_at = timestamp()
        record.report = record_report(record)
        if save_json:
            record.json_path = str(save_report(record.report, output_dir))
        return record


def record_report(record):
    """Build the same per-configuration report used by the Python runners."""
    report = new_report(record.spec.kind, record.spec.name)
    report.update(id=record.id, started_at=record.created_at, completed_at=record.completed_at)
    groups = {}
    for row in record.rows:
        groups.setdefault(row["configuration"], []).append(row)
    for configuration, rows in groups.items():
        params = {k: v for k, v in rows[0]["params"].items() if k != "random_state"}
        runs = [{"run": row["repeat"], "status": row["status"], "seconds": row["seconds"],
                 "started_at": row["started_at"], "completed_at": row["completed_at"],
                 "parameters": row["params"], "requested_parameters": row.get("requested_params", row["params"]),
                 "parameter_adjustments": row.get("parameter_adjustments", {}), "error": row.get("error"),
                 "metrics": {key: row.get(key) for key in record.spec.metrics}, **row.get("diagnostics", {})}
                for row in rows]
        report["experiments"].append({"record_id": record.id, "configuration": configuration,
                                      "dataset": record.dataset_summary,
                                      "model": record.model_details, "parameters": params, "runs": runs,
                                      "aggregate": aggregate_runs(runs),
                                      "preprocessing": asdict(record.spec.preprocessing)
                                      if record.spec.preprocessing else None})
    if record.spec.kind == "search":
        objective = record.spec.objective
        maximize = higher_is_better(METRICS[objective])
        best, best_config, best_params = None, None, {}
        steps = []
        for entry in report["experiments"]:
            score = entry["aggregate"]["metrics"].get(objective, {}).get("mean")
            if entry["aggregate"]["failed_runs"]:
                score = None
            if score is not None and (best is None or (score > best if maximize else score < best)):
                best, best_config, best_params = score, entry["configuration"], entry["parameters"]
            steps.append({"step": entry["configuration"], "parameters": entry["parameters"],
                          "score": score, "best_score_so_far": best})
        report["search"] = {"strategy": record.spec.search_strategy, "objective": objective,
                            "direction": "maximize" if maximize else "minimize", "space": record.spec.search_space,
                            "clusters": record.spec.clusters, "seed": record.spec.seed,
                            "runs_per_configuration": record.spec.repeats, "steps": steps,
                            "best_configuration": best_config, "best_parameters": best_params, "best_score": best}
    return json_safe(report)


def export_record(record):
    """Downloadable ZIP: JSON recipe/provenance, CSV scores, per-trial NPZ outputs.

    Original feature matrices are not exported. Benchmark artifacts include the
    evaluation labels needed for confusion matrices. Reload the original dataset
    (checking its fingerprint) to reproduce a run. Session IDs are local
    identifiers, not paths to persisted datasets.
    """
    buffer = io.BytesIO()
    payload = {
        **record.report,
        "id": record.id,
        "created_at": record.created_at,
        "spec": asdict(record.spec),
        "dataset": record.dataset_summary,
        "results": record.rows,
    }

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("experiment.json", report_json(payload))
        archive.writestr("metrics.csv", report_csv(record.report))
        for i, artifact in enumerate(record.artifacts):
            if isinstance(artifact, StoredArtifact):
                archive.write(artifact.path, f"trial_{i + 1}.npz")
                continue
            output = io.BytesIO()
            np.savez_compressed(output, **artifact)
            archive.writestr(f"trial_{i + 1}.npz", output.getvalue())
    return buffer.getvalue()
