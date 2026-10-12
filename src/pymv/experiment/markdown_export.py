"""
Experiment output — Markdown report + metadata + log file.

Each experiment is saved to its own folder inside ``experiments/``
(relative to the current working directory)::

    experiments/
      experiment#1/
        results.md           Well-organized Markdown report
        experiment.log       Full debug log file
        metadata/
          config.json        Models, datasets, metrics, runs, pymv version
          datasets.json      Per-dataset shape info (samples, views, clusters)
          environment.json   Python, TF, NumPy, SciPy versions, OS, GPU
          raw_scores.json    Every individual run score for re-analysis

The folder name is either user-chosen or auto-incremented as
``experiment#1``, ``experiment#2``, etc.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .result import ExperimentResult, RandomSearchResult

NUM_FMT = "{:.4f}"
MEAN_STD_FMT = "{:.4f} (±{:.4f})"
TIME_FMT = "{:.2f}s"


# Folder resolution

def _resolve_experiment_dir(name: str | None) -> str:
    """Return the absolute path to the experiment output folder.

    Parameters
    ----------
    name : str or None
        If given, use ``experiments/<name>``.
        If None, auto-increment: ``experiments/experiment#1``, ``#2``, …

    Returns
    -------
    str
        Absolute path to the created directory.
    """
    base = os.path.join(os.getcwd(), "experiments")
    os.makedirs(base, exist_ok=True)

    if name is not None:
        folder = os.path.join(base, name)
    else:
        k = 1
        while True:
            candidate = os.path.join(base, f"experiment#{k}")
            if not os.path.exists(candidate):
                folder = candidate
                break
            k += 1

    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "metadata"), exist_ok=True)
    return folder


# Markdown table helper

def _md_table(headers: list[str], rows: list[list[str]],
              bold_cells: set[tuple[int, int]] | None = None) -> str:
    """Render a Markdown table with optional bold highlighting."""
    if bold_cells is None:
        bold_cells = set()

    formatted = []
    for r_i, row in enumerate(rows):
        new_row = []
        for c_i, cell in enumerate(row):
            if (r_i, c_i) in bold_cells:
                new_row.append(f"**{cell}**")
            else:
                new_row.append(cell)
        formatted.append(new_row)

    all_rows = [headers] + formatted
    widths = [
        max(len(str(row[c])) for row in all_rows)
        for c in range(len(headers))
    ]

    def _fmt_row(cells):
        parts = [f" {str(c).ljust(w)} " for c, w in zip(cells, widths)]
        return "|" + "|".join(parts) + "|"

    lines = [
        _fmt_row(headers),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in formatted:
        lines.append(_fmt_row(row))

    return "\n".join(lines)


def _find_best_per_col(rows_data: list[list[float]]) -> dict[int, int]:
    """Return {col_idx: row_idx} for the highest value per column."""
    if not rows_data:
        return {}
    best = {}
    n_cols = len(rows_data[0])
    for c in range(n_cols):
        best_val = -np.inf
        best_row = 0
        for r, row in enumerate(rows_data):
            if not np.isnan(row[c]) and row[c] > best_val:
                best_val = row[c]
                best_row = r
        best[c] = best_row
    return best


# Metadata helpers

def _get_environment_info() -> dict:
    """Collect runtime environment metadata."""
    info = {
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
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["gpu_devices"] = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if info["gpu_devices"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info.setdefault("torch_version", "not installed")
        info["gpu_devices"] = []

    for pkg in ["numpy", "scipy", "scikit-learn"]:
        try:
            mod = __import__("sklearn" if pkg == "scikit-learn" else pkg)
            info[f"{pkg}_version"] = mod.__version__
        except Exception:
            info[f"{pkg}_version"] = "not installed"

    return info


def _save_json(data: dict, filepath: str):
    """Save dict to JSON with nice formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def _move_log_file(log_file_path: str | None, experiment_dir: str):
    """Move the log file into the experiment folder."""
    if log_file_path and os.path.exists(log_file_path):
        dest = os.path.join(experiment_dir, "experiment.log")
        shutil.copy2(log_file_path, dest)


# Experiment full export

def export_experiment(
    result: ExperimentResult,
    name: str | None = None,
    log_file: str | None = None,
    datasets_info: dict | None = None,
    experiment_config: dict | None = None,
    raw_scores: dict | None = None,
    experiment_dir: str | None = None,
) -> str:
    """Export an :class:`ExperimentResult` to a full experiment folder.

    Creates::

        experiments/<name>/
          results.md
          experiment.log
          metadata/
            config.json
            datasets.json
            environment.json
            raw_scores.json

    Parameters
    ----------
    result : ExperimentResult
        The result object from :meth:`Experiment.run`.
    name : str or None
        Folder name under ``experiments/``.
        ``None`` → auto-increment ``experiment#k``.
    log_file : str or None
        Path to the log file to copy into the experiment folder.
    datasets_info : dict or None
        Per-dataset shape info (auto-collected by Experiment).
    experiment_config : dict or None
        Experiment configuration (models, metrics, runs, etc.).
    raw_scores : dict or None
        Per-run raw metric scores for every (dataset, model) pair.
    experiment_dir : str or None
        If already resolved, pass it here to skip folder resolution.

    Returns
    -------
    str
        Path to the experiment folder.
    """
    folder = experiment_dir or _resolve_experiment_dir(name)
    os.makedirs(os.path.join(folder, "metadata"), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. results.md
    _write_experiment_markdown(result, folder, now)

    # 2. experiment.log
    _move_log_file(log_file, folder)

    # 3. metadata/environment.json
    env_info = _get_environment_info()
    env_info["export_time"] = now
    _save_json(env_info, os.path.join(folder, "metadata", "environment.json"))

    # 4. metadata/config.json
    if experiment_config:
        _save_json(experiment_config, os.path.join(folder, "metadata", "config.json"))

    # 5. metadata/datasets.json
    if datasets_info:
        _save_json(datasets_info, os.path.join(folder, "metadata", "datasets.json"))

    # 6. metadata/raw_scores.json
    if raw_scores:
        _save_json(raw_scores, os.path.join(folder, "metadata", "raw_scores.json"))

    return folder


def _write_experiment_markdown(result, folder: str, now: str):
    """Write results.md inside the experiment folder."""
    filepath = os.path.join(folder, "results.md")

    lines: list[str] = []
    lines.append("# Experiment Results")
    lines.append("")
    lines.append(f"> Generated by **pymv** on {now}")
    lines.append("")

    # Config table
    if result._models:
        lines.append("---")
        lines.append("")
        lines.append("## Model Configuration")
        lines.append("")

        _SKIP = {
            "labels_", "centroids_", "embedding_", "loss_history_",
            "_name",
        }
        _PREPROCESS_ONLY = {"beta", "tf_idf", "center", "scale"}
        all_preprocess_off = all(
            getattr(m, "preprocess", True) is False
            for m in result._models
        )

        all_keys: list[str] = []
        seen: set[str] = set()
        for m in result._models:
            for k in m.__dict__:
                if k not in seen and k not in _SKIP and not k.endswith("_"):
                    if all_preprocess_off and k in _PREPROCESS_ONLY:
                        continue
                    all_keys.append(k)
                    seen.add(k)

        headers = ["Model"] + all_keys
        rows = []
        for m, m_name in zip(result._models, result._model_names):
            row = [m_name]
            for k in all_keys:
                row.append(str(getattr(m, k, "—")))
            rows.append(row)

        lines.append(_md_table(headers, rows))
        lines.append("")

    # Per-dataset result tables
    show_std = any(
        any(v > 0 for v in r.metric_std.values())
        for rs in result._results.values()
        for r in rs
    )

    for ds_name, results in result._results.items():
        if not results:
            continue

        lines.append("---")
        lines.append("")
        lines.append(f"## Results — {ds_name}")
        lines.append("")

        metric_names = list(results[0].metric_avg.keys())
        headers = ["Model"] + metric_names + ["Time"]

        rows = []
        raw_data = []
        for r in results:
            row = [r.model_name]
            raw_row = []
            for m in metric_names:
                row.append(r.format_cell(m, show_std=show_std))
                raw_row.append(r.metric_avg[m])
            row.append(TIME_FMT.format(r.train_time))
            rows.append(row)
            raw_data.append(raw_row)

        bold_cells: set[tuple[int, int]] = set()
        if len(results) > 1:
            best_map = result._find_best(results)
            for col_i, row_i in best_map.items():
                bold_cells.add((row_i, col_i + 1))

        lines.append(_md_table(headers, rows, bold_cells))
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("### Folder contents")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.md` | This report |")
    lines.append("| `experiment.log` | Full debug log |")
    lines.append("| `metadata/config.json` | Experiment configuration |")
    lines.append("| `metadata/datasets.json` | Dataset dimensions & shapes |")
    lines.append("| `metadata/environment.json` | Python, TF, GPU, OS info |")
    lines.append("| `metadata/raw_scores.json` | Per-run scores for re-analysis |")
    lines.append("")

    content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


# RandomSearch full export

def export_random_search(
    result: RandomSearchResult,
    name: str | None = None,
    log_file: str | None = None,
    datasets_info: dict | None = None,
    experiment_config: dict | None = None,
    experiment_dir: str | None = None,
) -> str:
    """Export a :class:`RandomSearchResult` to a full experiment folder.

    Parameters
    ----------
    result : RandomSearchResult
    name : str or None
    log_file : str or None
    datasets_info : dict or None
    experiment_config : dict or None
    experiment_dir : str or None
        If already resolved, pass it here to skip folder resolution.

    Returns
    -------
    str
        Path to the experiment folder.
    """
    folder = experiment_dir or _resolve_experiment_dir(name)
    os.makedirs(os.path.join(folder, "metadata"), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. results.md
    _write_random_search_markdown(result, folder, now)

    # 2. experiment.log
    _move_log_file(log_file, folder)

    # 3. metadata/environment.json
    env_info = _get_environment_info()
    env_info["export_time"] = now
    _save_json(env_info, os.path.join(folder, "metadata", "environment.json"))

    # 4. metadata/config.json
    if experiment_config:
        _save_json(experiment_config, os.path.join(folder, "metadata", "config.json"))

    # 5. metadata/datasets.json
    if datasets_info:
        _save_json(datasets_info, os.path.join(folder, "metadata", "datasets.json"))

    # 6. metadata/raw_scores.json (all trials)
    if result.trials:
        raw = {
            "best_metric": result.best_metric_name,
            "best_trial_index": result.best_idx,
            "trials": result.trials,
        }
        _save_json(raw, os.path.join(folder, "metadata", "raw_scores.json"))

    return folder


def _write_random_search_markdown(result, folder: str, now: str):
    """Write results.md for a random search experiment."""
    filepath = os.path.join(folder, "results.md")

    lines: list[str] = []
    lines.append("# Random Search Results")
    lines.append("")
    lines.append(f"> Generated by **pymv** on {now}")
    lines.append("")
    lines.append(f"- **Model**: {result.model_name}")
    lines.append(f"- **Dataset**: {result.dataset_name}")
    lines.append(f"- **Trials**: {len(result.trials)}")
    direction = "↑ higher is better" if result.higher_is_better else "↓ lower is better"
    lines.append(f"- **Optimizing**: {result.best_metric_name} ({direction})")
    lines.append("")

    if not result.trials:
        lines.append("*No trials completed.*")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # All trials table
    lines.append("---")
    lines.append("")
    lines.append("## All Trials")
    lines.append("")

    sample = result.trials[0]
    param_keys = sorted(sample["params"].keys())
    metric_names = list(sample["metrics"].keys())

    headers = ["Trial"] + param_keys + metric_names + ["Time"]

    rows = []
    for i, t in enumerate(result.trials):
        tag = "★ best" if i == result.best_idx else f"#{i + 1}"
        row = [tag]
        row += [str(t["params"][k]) for k in param_keys]
        for m in metric_names:
            row.append(NUM_FMT.format(t["metrics"][m]))
        row.append(TIME_FMT.format(t["time"]))
        rows.append(row)

    bold_cells: set[tuple[int, int]] = set()
    bm_col = len(param_keys) + 1 + metric_names.index(result.best_metric_name)
    bold_cells.add((result.best_idx, bm_col))

    lines.append(_md_table(headers, rows, bold_cells))
    lines.append("")

    # Best configuration summary
    lines.append("---")
    lines.append("")
    lines.append("## Best Configuration")
    lines.append("")

    bp = result.best_params
    bm = result.best_metrics

    lines.append("### Parameters")
    lines.append("")
    for k, v in bp.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("### Scores")
    lines.append("")
    for k, v in bm.items():
        marker = " ← optimized" if k == result.best_metric_name else ""
        lines.append(f"- **{k}**: {v:.4f}{marker}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("### Folder contents")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.md` | This report |")
    lines.append("| `experiment.log` | Full debug log |")
    lines.append("| `metadata/config.json` | Search configuration |")
    lines.append("| `metadata/datasets.json` | Dataset dimensions & shapes |")
    lines.append("| `metadata/environment.json` | Python, TF, GPU, OS info |")
    lines.append("| `metadata/raw_scores.json` | All trial scores & params |")
    lines.append("")

    content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
