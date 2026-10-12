"""Portable experiment reports shared by Python runners and the display.

Reports deliberately exclude feature matrices and embeddings. Those potentially
large arrays belong in optional downloadable artifacts, not the JSON manifest.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalized_loss(values):
    """Min–max normalization per trajectory; a constant trajectory becomes zero."""
    if values is None:
        return None
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        return None
    finite = np.isfinite(a)
    result = np.full(a.shape, np.nan)
    if finite.any():
        low, high = a[finite].min(), a[finite].max()
        result[finite] = (a[finite] - low) / (high - low) if high > low else 0.0
    return json_safe(result)


def model_parameters(model):
    return json_safe({k: v for k, v in vars(model).items() if not k.startswith("_") and not k.endswith("_")})


def model_details(model):
    cls = model if isinstance(model, type) else type(model)
    details = {"name": cls.__name__, "module": cls.__module__,
               "supports_missing_views": bool(getattr(cls, "supports_missing_views", False))}
    if getattr(cls, "backend", None):
        details["backend"] = cls.backend
    from pymv.models.registry import list_models_info

    for info in list_models_info().values():
        if info["class"] is cls:
            details.update({key: info[key] for key in ("description", "paper", "implementation") if key in info})
            break
    return details


def diagnostics(model):
    loss = getattr(model, "loss_history_", None)
    return json_safe({
        "device": getattr(model, "device_", None),
        "loss_history": loss,
        "normalized_loss_history": normalized_loss(loss),
        "view_weights": getattr(model, "view_weights_", getattr(model, "alphas_", None)),
        "inertia": getattr(model, "inertia_", None),
        "per_view_results": getattr(model, "per_view_results_", None),
        "per_view_loss_history": getattr(model, "per_view_loss_history_", None),
    })


def dataset_details(data, *, benchmark):
    return json_safe({
        "name": data.name, "n_samples": data.n_samples, "n_views": data.n_views,
        "feature_dims": data.feature_dims, "view_names": data.view_names,
        "modalities": data.modalities, "category": data.category, "complete": data.is_complete,
        "observed_per_view": data.observed_mask.sum(axis=0), "has_graphs": data.has_graphs,
        "n_classes": data.n_clusters if benchmark else None, "benchmark": benchmark,
        "metadata": data.metadata,
    })


def aggregate_runs(runs):
    good = [r for r in runs if r.get("status") == "complete"]
    metrics = sorted({key for r in good for key in r.get("metrics", {})})
    result = {"successful_runs": len(good), "failed_runs": len(runs) - len(good), "metrics": {}}
    for name in metrics + ["seconds"]:
        values = [r.get("seconds") if name == "seconds" else r.get("metrics", {}).get(name) for r in good]
        values = np.asarray([v for v in values if v is not None], dtype=float)
        values = values[np.isfinite(values)]
        stats = {"mean": float(values.mean()) if values.size else None,
                 "std": float(values.std(ddof=0)) if values.size else None, "count": int(values.size)}
        if name == "seconds":
            result[name] = stats
        else:
            result["metrics"][name] = stats
    return result


def new_report(kind="experiment", name=None):
    from pymv import __version__

    return {"schema_version": "1.0", "pymv_version": __version__, "id": uuid.uuid4().hex,
            "kind": kind, "name": name, "started_at": timestamp(), "completed_at": None,
            "loss_normalization": "per-run min-max; constant history is zero; unavailable values are null",
            "std_definition": "population standard deviation (ddof=0), finite successful scores only",
            "experiments": []}


def report_json(report):
    return json.dumps(json_safe(report), indent=2, allow_nan=False).encode("utf-8")


def save_report(report, output_dir=None):
    """Write one uniquely named JSON under output_dir (default ./experiments)."""
    directory = Path(output_dir) if output_dir is not None else Path.cwd() / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{report['kind']}-{report['id']}.json"
    destination.write_bytes(report_json(report))
    return destination


def report_rows(report):
    rows = []
    for experiment in report["experiments"]:
        for run in experiment["runs"]:
            row = {"dataset": experiment["dataset"].get("name"), "model": experiment["model"]["name"],
                   "configuration_name": experiment["model"].get("display_name"),
                   "configuration": experiment.get("configuration"),
                   **{k: run.get(k) for k in ("run", "status", "started_at", "completed_at", "seconds", "error")}}
            row.update({f"param.{k}": v for k, v in experiment["parameters"].items()})
            row.update({f"param.{k}": v for k, v in run.get("parameters", {}).items()})
            row.update(run.get("metrics", {}))
            rows.append(json_safe(row))
    return rows


def report_csv(report):
    rows = report_rows(report)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def report_zip(report):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("results.json", report_json(report))
        archive.writestr("metrics.csv", report_csv(report))
    return output.getvalue()


class ExportableResult:
    """Results expose the report, saved path and explicit JSON/CSV/ZIP export."""

    report = None
    json_path = None

    def to_dict(self):
        if self.report is None:
            raise ValueError("Run the experiment before exporting")
        return json_safe(self.report)

    def save_json(self, output_dir=None):
        self.json_path = save_report(self.to_dict(), output_dir)
        return self.json_path

    def export(self, path):
        path = Path(path)
        formats = {".json": report_json, ".csv": report_csv, ".zip": report_zip}
        if path.suffix.lower() not in formats:
            raise ValueError("Use a .json, .csv or .zip filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(formats[path.suffix.lower()](self.to_dict()))
        return path
