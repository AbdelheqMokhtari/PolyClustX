"""Small declarative CLI plans, evaluated one dataset at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_plan(path):
    """Read data-only YAML/JSON; never execute imports, expressions or code."""
    import yaml

    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Experiment plans must be smaller than 1 MB")
    try:
        plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid experiment plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("A plan must be a mapping with datasets and models")
    return plan


def _positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def prepare_plan(plan, overrides=()):
    """Validate the plan before downloading data or beginning any fits."""
    from polyclustx.command import assignments
    from polyclustx.models import get_model_info

    unknown = set(plan) - {"name", "datasets", "models", "runs", "seed", "clusters", "metrics", "benchmark", "output"}
    if unknown:
        raise ValueError(f"Unknown plan fields: {sorted(unknown)}")
    plan = {"runs": 1, "seed": 0, "benchmark": True, "output": "experiments", **plan}
    _positive(plan["runs"], "runs")
    _positive(plan["seed"], "seed", 0)
    if plan.get("clusters") is not None:
        _positive(plan["clusters"], "clusters")
    if not isinstance(plan["benchmark"], bool):
        raise ValueError("benchmark must be true or false")
    if not isinstance(plan["output"], str) or not plan["output"].strip():
        raise ValueError("output must be a directory path")
    for key in ("datasets", "models"):
        if not isinstance(plan.get(key), list) or not plan[key]:
            raise ValueError(f"Provide a nonempty {key} list")
    datasets = []
    for entry in plan["datasets"]:
        if isinstance(entry, str):
            name, separator, variant = entry.partition(":")
            entry = {"name": name, **({"variant": variant} if separator else {})}
        if not isinstance(entry, dict) or set(entry) - {"name", "variant", "clusters"}:
            raise ValueError("Each dataset needs a name and optional variant/clusters")
        entry = dict(entry)
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("Dataset name must be a nonempty string")
        if "variant" in entry and (not isinstance(entry["variant"], str) or not entry["variant"].strip()):
            raise ValueError("Dataset variant must be a nonempty string")
        if entry.get("clusters") is not None:
            _positive(entry["clusters"], "dataset clusters")
        if entry in datasets:
            raise ValueError(f"Duplicate dataset entry: {entry['name']}")
        datasets.append(entry)
    models = []
    for entry in plan["models"]:
        entry = {"name": entry} if isinstance(entry, str) else entry
        if not isinstance(entry, dict) or set(entry) - {"name", "label", "params"}:
            raise ValueError("Each model needs a name and optional label/params")
        if not isinstance(entry.get("name"), str):
            raise ValueError("Model name must be a string")
        name = entry["name"].lower().replace("-", "_")
        get_model_info(name)
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Model params must be a mapping")
        label = entry.get("label", name)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Model label must be a nonempty string")
        if any(model["label"] == label for model in models):
            raise ValueError(f"Duplicate model label {label!r}; give each configuration a distinct label")
        models.append({"name": name, "label": label, "params": dict(params)})
    for qualified, value in assignments(overrides).items():
        name, separator, key = qualified.partition(".")
        matches = [model for model in models if model["name"] == name.replace("-", "_")]
        if not separator or not matches:
            raise ValueError("Use --param model.parameter=value for a selected model")
        for model in matches:
            model["params"][key] = value
    for model in models:
        allowed = set(get_model_info(model["name"])["defaults"])
        unknown = set(model["params"]) - allowed
        if unknown:
            raise ValueError(f"{model['name']}: unknown parameters {sorted(unknown)}; set K through clusters")
        # Reports and model constructors should receive finite data-only values.
        json.dumps(model["params"], allow_nan=False)
    metrics = plan.get("metrics") or (["acc", "nmi", "ari"] if plan["benchmark"] else ["silhouette"])
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise ValueError("metrics must be a list of metric names")
    return {**plan, "datasets": datasets, "models": models, "metrics": metrics}


def run_plan(plan):
    """Evaluate the Cartesian product, retaining a single exportable report.

    Only the current dataset is resident in memory. K is inferred separately
    for each labeled dataset, unless the plan explicitly supplies it.
    """
    from polyclustx.cli import _resolve_metrics
    from polyclustx.data import load_data
    from polyclustx.experiment import Experiment
    from polyclustx.experiment.exports import ExportableResult, new_report, timestamp
    from polyclustx.models import get_model_info

    plan = prepare_plan(plan)
    metrics = _resolve_metrics(plan["metrics"])
    combined = ExportableResult()
    combined.report = new_report(name=plan.get("name"))
    combined.report["plan"] = plan
    combined.report["status"] = "running"
    try:
        for spec in plan["datasets"]:
            data = load_data(spec["name"], variant=spec["variant"]) if "variant" in spec else load_data(spec["name"])
            if plan["benchmark"] and data.labels is None:
                raise ValueError(f"{spec['name']} has no labels; use --no-benchmark and choose clusters")
            clusters = spec.get("clusters", plan.get("clusters"))
            if clusters is None and plan["benchmark"]:
                clusters = data.n_clusters
            if clusters is None:
                raise ValueError("Non-benchmark plans require clusters, globally or per dataset")
            estimators = []
            for settings in plan["models"]:
                info = get_model_info(settings["name"])
                params = dict(settings["params"])
                if "random_state" in info["defaults"]:
                    params.setdefault("random_state", plan["seed"])
                model = info["class"](n_clusters=clusters, **params)
                model.name = settings["label"]
                estimators.append(model)
            label = spec["name"] + (":" + spec["variant"] if "variant" in spec else "")
            result = Experiment(datasets={label: data}, models=estimators, metrics=metrics,
                                benchmark=plan["benchmark"], runs=plan["runs"], seed=plan["seed"],
                                verbose=1, save_json=False).run()
            combined.report["experiments"].extend(result.report["experiments"])
            combined.save_json(plan["output"])
            del data, estimators, result, model
        combined.report["status"] = "failed" if any(
            run["status"] == "failed" for experiment in combined.report["experiments"]
            for run in experiment["runs"]
        ) else "complete"
    except Exception as exc:
        combined.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        combined.report["completed_at"] = timestamp()
        combined.save_json(plan["output"])
        print(f"Report: {combined.json_path}")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(prog="polyclustx experiment", description="Compare dataset × model combinations.")
    parser.add_argument("--config", help="Optional YAML/JSON plan; CLI settings override matching top-level fields")
    parser.add_argument("--datasets", nargs="+", help="Dataset names; use name:variant when needed")
    parser.add_argument("--models", nargs="+", help="Registered method names")
    parser.add_argument("--param", action="append", default=[], metavar="MODEL.PARAM=VALUE")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false", default=None)
    parser.add_argument("--output")
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    try:
        plan = read_plan(args.config) if args.config else {}
        plan.update({key: value for key, value in vars(args).items()
                     if key not in {"config", "param"} and value is not None})
        result = run_plan(prepare_plan(plan, args.param))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if result.report["status"] != "complete":
        parser.exit(1, "Some fits failed; inspect the report's error fields.\n")
