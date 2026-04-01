"""
Experiment tracking for pymv.

Automatically saves experiment artifacts to a structured folder::

    experiments/
      2026-03-31_14-30-12_acm_lmgec/
        config.yaml          Reproducible config (re-run with pymv-cli --config)
        results.json         Structured results (dataset, model, params, metrics)
        metrics.csv          Tabular metrics for programmatic analysis
        results.md           Well-organized Markdown report
        experiment.log       Full debug log
        figures/             Saved plots (loss curves, cluster visualizations)
        metadata/
          environment.json   Python, TF, GPU, OS info
          datasets.json      Dataset shapes and dimensions
          raw_scores.json    Per-run scores for re-analysis

Usage
-----
The tracker is used automatically when ``save=True``::

    pymv-cli --dataset acm --save
    pymv-cli --config experiment.yaml --save --name my_run
    pymv-cli --dataset acm --save --save-path results/

Or programmatically::

    from pymv.tracking import ExperimentTracker

    tracker = ExperimentTracker(name="my_experiment")
    tracker.save_config({"dataset": "acm", "model": {"name": "lmgec"}})
    tracker.save_results(result_dict)
    tracker.save_metrics_csv(metrics_data)
    tracker.save_figure(fig, "loss_curve")
    print(tracker.experiment_dir)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


# ── ExperimentTracker ─────────────────────────────────────────────

class ExperimentTracker:
    """Manages a single experiment's output folder and artifacts.

    Parameters
    ----------
    name : str or None
        Custom folder name.  If *None*, a timestamped name is
        generated automatically (``YYYY-MM-DD_HH-MM-SS``).
    save_path : str or None
        Base directory for experiments.  Defaults to
        ``./experiments`` relative to the current working directory.

    Attributes
    ----------
    experiment_dir : str
        Absolute path to the experiment folder (created lazily
        on first artifact save).
    """

    def __init__(
        self,
        name: str | None = None,
        save_path: str | None = None,
    ):
        self._base = save_path or os.path.join(os.getcwd(), "experiments")
        self._name = name
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._dir: str | None = None

    # ── Properties ────────────────────────────────────────────────

    @property
    def experiment_dir(self) -> str:
        """Absolute path to the experiment folder (created on access)."""
        if self._dir is None:
            self._dir = self._resolve_dir()
        return self._dir

    @property
    def figures_dir(self) -> str:
        """Absolute path to ``<experiment>/figures/``."""
        d = os.path.join(self.experiment_dir, "figures")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def metadata_dir(self) -> str:
        """Absolute path to ``<experiment>/metadata/``."""
        d = os.path.join(self.experiment_dir, "metadata")
        os.makedirs(d, exist_ok=True)
        return d

    # ── Save API ──────────────────────────────────────────────────

    def save_config(self, config: dict[str, Any]) -> str:
        """Save experiment configuration as ``config.yaml``.

        The saved config is designed to be **directly re-runnable**::

            pymv-cli --config experiments/<name>/config.yaml

        Parameters
        ----------
        config : dict
            Full experiment configuration (dataset, model, params,
            experiment settings, output settings).

        Returns
        -------
        str
            Path to the saved config file.
        """
        filepath = os.path.join(self.experiment_dir, "config.yaml")

        try:
            import yaml
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# pymv experiment config — {self._timestamp}\n")
                f.write(f"# Re-run: pymv-cli --config {filepath}\n\n")
                yaml.dump(
                    config, f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        except ImportError:
            # Fallback to JSON if pyyaml not available
            filepath = os.path.join(self.experiment_dir, "config.json")
            _save_json(config, filepath)

        return filepath

    def save_results(self, results: dict[str, Any]) -> str:
        """Save structured results as ``results.json``.

        Parameters
        ----------
        results : dict
            Structured results containing dataset, model, params,
            metrics, runs, timing, etc.

        Returns
        -------
        str
            Path to the saved file.
        """
        filepath = os.path.join(self.experiment_dir, "results.json")
        data = {
            "timestamp": self._timestamp,
            **results,
        }
        _save_json(data, filepath)
        return filepath

    def save_metrics_csv(
        self,
        rows: list[dict[str, Any]],
        filename: str = "metrics.csv",
    ) -> str:
        """Save metrics as a CSV table.

        Parameters
        ----------
        rows : list[dict]
            Each dict is one row.  Keys become column headers.
            Example::

                [
                    {"dataset": "acm", "model": "LMGEC", "acc": 0.93, ...},
                    {"dataset": "dblp", "model": "LMGEC", "acc": 0.71, ...},
                ]

        filename : str
            CSV filename (default: ``metrics.csv``).

        Returns
        -------
        str
            Path to the saved file.
        """
        if not rows:
            return ""

        filepath = os.path.join(self.experiment_dir, filename)
        fieldnames = list(rows[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return filepath

    def save_figure(
        self,
        figure: Any,
        name: str,
        fmt: str = "png",
        dpi: int = 150,
    ) -> str:
        """Save a matplotlib figure to ``figures/<name>.<fmt>``.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The figure to save.
        name : str
            Filename stem (without extension).
        fmt : str
            Image format (default: ``"png"``).
        dpi : int
            Resolution (default: 150).

        Returns
        -------
        str
            Path to the saved file.
        """
        filepath = os.path.join(self.figures_dir, f"{name}.{fmt}")
        figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
        return filepath

    def save_log(self, log_path: str | None) -> str | None:
        """Copy a log file into the experiment folder.

        Parameters
        ----------
        log_path : str or None
            Source log file.  If *None* or missing, nothing happens.

        Returns
        -------
        str or None
            Destination path, or *None* if nothing was copied.
        """
        if not log_path or not os.path.exists(log_path):
            return None
        dest = os.path.join(self.experiment_dir, "experiment.log")
        shutil.copy2(log_path, dest)
        return dest

    def save_environment(self) -> str:
        """Save runtime environment info to ``metadata/environment.json``."""
        info = _get_environment_info()
        info["export_time"] = self._timestamp
        filepath = os.path.join(self.metadata_dir, "environment.json")
        _save_json(info, filepath)
        return filepath

    def save_datasets_info(self, datasets_info: dict) -> str:
        """Save dataset shapes to ``metadata/datasets.json``."""
        filepath = os.path.join(self.metadata_dir, "datasets.json")
        _save_json(datasets_info, filepath)
        return filepath

    def save_raw_scores(self, raw_scores: dict) -> str:
        """Save per-run raw scores to ``metadata/raw_scores.json``."""
        filepath = os.path.join(self.metadata_dir, "raw_scores.json")
        _save_json(raw_scores, filepath)
        return filepath

    # ── Convenience: save everything at once ──────────────────────

    def save_all(
        self,
        *,
        config: dict | None = None,
        results: dict | None = None,
        metrics_rows: list[dict] | None = None,
        datasets_info: dict | None = None,
        raw_scores: dict | None = None,
        log_path: str | None = None,
    ) -> str:
        """Save all available artifacts in one call.

        Returns the experiment directory path.
        """
        if config:
            self.save_config(config)
        if results:
            self.save_results(results)
        if metrics_rows:
            self.save_metrics_csv(metrics_rows)
        if datasets_info:
            self.save_datasets_info(datasets_info)
        if raw_scores:
            self.save_raw_scores(raw_scores)

        self.save_environment()

        if log_path:
            self.save_log(log_path)

        return self.experiment_dir

    # ── Internal ──────────────────────────────────────────────────

    def _resolve_dir(self) -> str:
        """Resolve and create the experiment output directory."""
        os.makedirs(self._base, exist_ok=True)

        if self._name is not None:
            folder = os.path.join(self._base, self._name)
        else:
            folder = os.path.join(self._base, self._timestamp)

        os.makedirs(folder, exist_ok=True)
        return folder

    def __repr__(self) -> str:
        return f"ExperimentTracker(dir={self.experiment_dir!r})"


# ── Builders for experiment results ───────────────────────────────

def build_experiment_config(
    args: Any,
    model_params: dict | None = None,
) -> dict:
    """Build a reproducible YAML config from CLI args.

    The returned dict can be loaded back by ``pymv-cli --config``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    model_params : dict, optional
        Final resolved model parameters (after defaults & coercion).

    Returns
    -------
    dict
        Config suitable for ``config.yaml``.
    """
    config: dict[str, Any] = {}

    # Dataset
    config["dataset"] = getattr(args, "dataset", None)

    # Model
    model_name = getattr(args, "model", "lmgec")
    model_section: dict[str, Any] = {"name": model_name}
    if model_params:
        # Remove n_clusters (dataset-dependent, auto-detected)
        params = {k: v for k, v in model_params.items() if k != "n_clusters"}
        # Convert numpy types to plain Python
        params = {k: _to_python(v) for k, v in params.items()}
        model_section["params"] = params
    config["model"] = model_section

    # Experiment settings
    exp: dict[str, Any] = {}
    if getattr(args, "runs", 1) != 1:
        exp["runs"] = args.runs
    metrics = getattr(args, "metrics", None)
    if metrics:
        exp["metrics"] = metrics
    verbose = getattr(args, "verbose", 1)
    if verbose != 1:
        exp["verbose"] = verbose
    if getattr(args, "random_search", False):
        exp["random_search"] = True
        exp["n_trials"] = getattr(args, "n_trials", 20)
        bm = getattr(args, "best_metric", "acc")
        if bm != "acc":
            exp["best_metric"] = bm
    seed = getattr(args, "seed", None)
    if seed is not None:
        exp["seed"] = seed
    if exp:
        config["experiment"] = exp

    # Output
    output: dict[str, Any] = {"save": True}
    name = getattr(args, "name", None)
    if name:
        output["name"] = name
    if getattr(args, "log", False):
        output["log"] = True
    config["output"] = output

    return config


def build_results_dict(
    *,
    dataset_name: str,
    model_name: str,
    model_params: dict,
    metrics: dict[str, float],
    metrics_std: dict[str, float] | None = None,
    runs: int = 1,
    train_time: float = 0.0,
    extra: dict | None = None,
) -> dict:
    """Build a structured results dict for ``results.json``.

    Parameters
    ----------
    dataset_name : str
    model_name : str
    model_params : dict
    metrics : dict
        Metric name → mean value.
    metrics_std : dict, optional
        Metric name → std deviation.
    runs : int
    train_time : float
    extra : dict, optional
        Additional fields merged into the result.

    Returns
    -------
    dict
    """
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "params": {k: _to_python(v) for k, v in model_params.items()},
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "runs": runs,
        "train_time_seconds": round(train_time, 4),
    }
    if metrics_std:
        result["metrics_std"] = {
            k: round(float(v), 6) for k, v in metrics_std.items()
        }
    if extra:
        result.update(extra)
    return result


def build_metrics_csv_rows(
    results: Any,
    *,
    include_std: bool = True,
) -> list[dict]:
    """Convert experiment results to flat CSV-friendly rows.

    Parameters
    ----------
    results : ExperimentResult or RandomSearchResult
        Result object from an experiment run.
    include_std : bool
        Include standard deviation columns when available.

    Returns
    -------
    list[dict]
        One dict per (dataset, model) pair.
    """
    from pymv.experiment.result import ExperimentResult, RandomSearchResult

    rows: list[dict] = []

    if isinstance(results, ExperimentResult):
        for ds_name, result_list in results.all_results.items():
            for r in result_list:
                row: dict[str, Any] = {
                    "dataset": ds_name,
                    "model": r.model_name,
                }
                for metric_name, val in r.metric_avg.items():
                    row[metric_name] = round(val, 6)
                if include_std and r.metric_std:
                    for metric_name, val in r.metric_std.items():
                        row[f"{metric_name}_std"] = round(val, 6)
                row["train_time"] = round(r.train_time, 4)
                rows.append(row)

    elif isinstance(results, RandomSearchResult):
        for i, trial in enumerate(results.trials):
            row = {"trial": i + 1}
            row.update({k: _to_python(v) for k, v in trial["params"].items()})
            row.update({
                k: round(float(v), 6)
                for k, v in trial["metrics"].items()
            })
            row["train_time"] = round(trial["time"], 4)
            row["is_best"] = (i == results.best_idx)
            rows.append(row)

    return rows


# ── Helpers ───────────────────────────────────────────────────────

def _save_json(data: dict, filepath: str) -> None:
    """Save dict to JSON with nice formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def _to_python(value: Any) -> Any:
    """Convert numpy/TF scalars to plain Python types."""
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _get_environment_info() -> dict:
    """Collect runtime environment metadata."""
    import platform
    import sys

    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import pymv
        info["pymv_version"] = pymv.__version__
    except Exception:
        info["pymv_version"] = "unknown"

    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_devices"] = [g.name for g in gpus] if gpus else []
        if gpus:
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                info["gpu_name"] = details.get("device_name", "unknown")
            except Exception:
                info["gpu_name"] = "unknown"
    except Exception:
        info["tensorflow_version"] = "not installed"
        info["gpu_devices"] = []

    for pkg in ["numpy", "scipy", "scikit-learn"]:
        try:
            mod = __import__(
                pkg.replace("-", "_") if pkg == "scikit-learn" else pkg
            )
            info[f"{pkg}_version"] = mod.__version__
        except Exception:
            info[f"{pkg}_version"] = "not installed"

    return info
