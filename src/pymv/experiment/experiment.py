"""
Experiment runner for pymv.

Inspired by Cornac's Experiment class, adapted for multi-view
clustering.  Accepts flexible input formats so you can pass
datasets / models / metrics in whichever way is most convenient.

Console output
--------------
Only **two clean tables** are printed:

1. **Model Configuration** — one row per model, columns = params.
2. **Results — <dataset>** — one table per dataset, rows = models,
   columns = metrics + Time (best values marked with ``*``).

Everything else (loading messages, per-run scores, timing, errors)
goes exclusively to the **log file** in ``log_dir`` (default:
``./logs``).

Usage
-----
>>> Experiment(
...     datasets=["acm", "dblp"],
...     models=[model_a, model_b],
...     metrics=[clustering_accuracy, nmi_score],
... ).run()
"""

from __future__ import annotations

import copy
import sys
import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from pymv.core import BaseModel
from .result import ExperimentResult, Result
from .logger import setup_experiment_logger


# Normalizers (accept many input shapes, return canonical form)

def _normalize_datasets(datasets) -> OrderedDict:
    """
    Accept any of the following and return
    ``OrderedDict[str, callable | tuple | MultiViewDataset]``:

    *   ``["acm", "dblp"]``          — names resolved via registry
    *   ``[load_acm, load_dblp]``    — callables (name = fn.__name__)
    *   ``{"acm": load_acm, ...}``   — dict of callables
    *   ``{"acm": (Xs, As, y)}``     — dict of pre-loaded tuples
    *   ``MultiViewDataset``         — single dataset object
    *   single string ``"acm"``      — treated as ``["acm"]``
    *   single callable              — treated as ``[callable]``
    *   single tuple ``(Xs, As, y)`` — treated as ``{"dataset": tuple}``
    """
    from pymv.data.registry import DATASETS, load_dataset

    if isinstance(datasets, str):
        datasets = [datasets]
    elif _is_dataset_obj(datasets):
        name = getattr(datasets, "name", None) or "dataset"
        datasets = {name: datasets}
    elif isinstance(datasets, tuple) and _is_data_tuple(datasets):
        datasets = {"dataset": datasets}
    elif callable(datasets):
        datasets = [datasets]

    out = OrderedDict()

    if isinstance(datasets, dict):
        for name, val in datasets.items():
            out[name] = val
    elif isinstance(datasets, (list, tuple)):
        for item in datasets:
            if isinstance(item, str):
                name = item
                out[name] = DATASETS[name.lower()]
            elif _is_dataset_obj(item):
                name = getattr(item, "name", None) or f"dataset_{len(out)}"
                out[name] = item
            elif callable(item):
                name = getattr(item, "__name__", type(item).__name__)
                if name.startswith("load_"):
                    name = name[5:]
                out[name] = item
            elif isinstance(item, tuple) and _is_data_tuple(item):
                out[f"dataset_{len(out)}"] = item
            else:
                raise TypeError(
                    f"Cannot interpret dataset entry: {type(item)}"
                )
    else:
        raise TypeError(
            f"datasets must be a str, list, dict, callable, or tuple, "
            f"got {type(datasets)}"
        )
    return out


def _is_data_tuple(t) -> bool:
    """Return True if *t* looks like ``(Xs, As, labels)``."""
    return isinstance(t, tuple) and len(t) == 3


def _is_dataset_obj(t) -> bool:
    """Return True if *t* is a MultiViewDataset."""
    from pymv.data.dataset import MultiViewDataset
    return isinstance(t, MultiViewDataset)


def _load_dataset(entry) -> tuple:
    """Call loader or return pre-loaded tuple / MultiViewDataset."""
    if _is_dataset_obj(entry):
        return (entry.Xs, entry.As, entry.labels)
    if callable(entry):
        result = entry()
        if _is_dataset_obj(result):
            return (result.Xs, result.As, result.labels)
        return result
    if isinstance(entry, tuple) and _is_data_tuple(entry):
        return entry
    raise TypeError(f"Cannot load dataset from {type(entry)}")


def _normalize_models(models) -> list:
    """
    Accept:
    *   a single ``BaseModel`` instance
    *   a list of instances
    """
    if isinstance(models, BaseModel):
        return [models]
    if isinstance(models, (list, tuple)):
        return list(models)
    raise TypeError(
        f"models must be a BaseModel or list of BaseModel, "
        f"got {type(models)}"
    )


def _normalize_metrics(metrics) -> OrderedDict:
    """
    Accept:
    *   ``[fn1, fn2]``           — name inferred from ``fn.__name__``
    *   ``{"ACC": fn1, ...}``    — explicit names
    *   a single callable        — treated as ``[callable]``
    """
    if callable(metrics) and not isinstance(metrics, dict):
        metrics = [metrics]

    if isinstance(metrics, dict):
        return OrderedDict(metrics)

    if isinstance(metrics, (list, tuple)):
        out = OrderedDict()
        for fn in metrics:
            name = getattr(fn, "__name__", type(fn).__name__)
            out[name] = fn
        return out

    raise TypeError(
        f"metrics must be a dict, list, or callable, got {type(metrics)}"
    )


# Experiment Class

class Experiment:
    """
    Run models × datasets × metrics and report formatted tables.

    Console output (controlled by *verbose*):

    *   **verbose=0** — nothing printed.
    *   **verbose=1** — config table + result tables only (default).
    *   **verbose=2** — one-liner progress per (dataset × model)
        + tables.
    *   **verbose=3** — full debug (every run, all params).

    All details are always written to the log file regardless of
    the verbose level.

    Parameters
    ----------
    datasets : str | list | dict | callable | tuple
        Flexible.  See module docstring examples.

    models : BaseModel | list[BaseModel]
        Model instances to evaluate.  Each is deep-copied per
        (dataset, run) so state is never shared.

    metrics : callable | list[callable] | dict[str, callable]
        Flexible.  If a list, display names are inferred from
        ``fn.__name__``.  If a dict, keys are display names.

    runs : int
        Number of repetitions.  When > 1 the table shows
        ``mean (±std)``.

    log_dir : str | None
        Directory for log files.  ``None`` → ``./logs`` (cwd).

    verbose : int
        Console verbosity level (0–3).  Default is **1**.

    save_markdown : bool
        If ``True``, save a well-organized Markdown report of
        the results to ``experiments/<experiment_name>/results.md``
        (relative to the current working directory).
        Default is ``False``.

    experiment_name : str or None
        Name of the folder under ``experiments/`` where the
        Markdown report is saved.  Only used when
        ``save_markdown=True``.

        * If a string is given, the folder is
          ``experiments/<experiment_name>``.
        * If ``None`` (default), a timestamped folder is created
          (e.g. ``2026-03-31_14-30-12``).

    log : bool
        If ``True``, a log file is created (either
        inside the experiment folder when ``save_markdown=True``,
        or in ``log_dir``).
        If ``False`` (default), **no log file is written** — only
        console output is produced.

    save_path : str or None
        Base directory for saving experiment results.  Only used
        when ``save_markdown=True``.  Defaults to ``./experiments``.
    """

    def __init__(
        self,
        datasets,
        models,
        metrics,
        runs: int = 1,
        log_dir: str | None = None,
        verbose: int = 1,
        save_markdown: bool = False,
        experiment_name: str | None = None,
        log: bool = False,
        save_path: str | None = None,
    ):
        self.datasets = _normalize_datasets(datasets)
        self.models = _normalize_models(models)
        self.metrics = _normalize_metrics(metrics)
        self.runs = max(1, int(runs))
        self.log_dir = log_dir
        self.verbose = verbose
        self.save_markdown = save_markdown
        self.experiment_name = experiment_name
        self.log = log
        self.save_path = save_path

        # Filled after run()
        self.result: ExperimentResult | None = None

    # helpers

    @staticmethod
    def _model_name(model) -> str:
        name = getattr(model, "name", None)
        if name:
            return str(name)
        return type(model).__name__

    def _evaluate(self, y_true, y_pred) -> OrderedDict:
        values = OrderedDict()
        for name, fn in self.metrics.items():
            values[name] = float(fn(y_true, y_pred))
        return values

    def _build_reproducible_config(self) -> dict:
        """Build a config dict that can re-run this experiment.

        The output is designed to be loadable by
        ``pymv-cli --config <path>/config.yaml``.
        """
        from pymv.tracking import _to_python

        _SKIP = {"labels_", "centroids_", "embedding_", "loss_history_", "_name"}
        datasets_list = list(self.datasets.keys())
        config: dict = {}

        # Dataset — single string or list
        config["dataset"] = datasets_list[0] if len(datasets_list) == 1 else datasets_list

        # Model(s) — use the first model for the config
        if self.models:
            m = self.models[0]
            model_section: dict = {"name": type(m).__name__.lower()}
            params = {
                k: _to_python(v)
                for k, v in m.__dict__.items()
                if k not in _SKIP and not k.startswith("_") and not k.endswith("_")
            }
            if params:
                model_section["params"] = params
            config["model"] = model_section

        # Experiment settings
        exp: dict = {}
        if self.runs != 1:
            exp["runs"] = self.runs
        metric_names = list(self.metrics.keys())
        if metric_names:
            exp["metrics"] = metric_names
        if self.verbose != 1:
            exp["verbose"] = self.verbose
        if exp:
            config["experiment"] = exp

        # Output
        config["output"] = {"save": True}

        return config

    # main loop

    def run(self) -> ExperimentResult:
        """Execute the experiment and return an :class:`ExperimentResult`."""

        # Resolve experiment folder first (if saving), so the log goes
        # inside it from the very beginning.
        experiment_dir: str | None = None
        tracker = None
        if self.save_markdown:
            from pymv.tracking import ExperimentTracker
            tracker = ExperimentTracker(
                name=self.experiment_name,
                save_path=self.save_path,
            )
            experiment_dir = tracker.experiment_dir

        # When save_markdown=True, log is always enabled inside the
        # experiment folder regardless of the log parameter.
        use_log = self.log or self.save_markdown

        logger = setup_experiment_logger(
            log_dir=self.log_dir,
            prefix="pymv_experiment",
            verbose=self.verbose,
            experiment_dir=experiment_dir,
            log=use_log,
        )

        # Log header (file + verbose>=3)
        logger.debug("=" * 60)
        logger.debug("  pymv Experiment")
        logger.debug("=" * 60)
        logger.debug("  Datasets : %s", list(self.datasets.keys()))
        logger.debug(
            "  Models   : %s",
            [self._model_name(m) for m in self.models],
        )
        logger.debug("  Metrics  : %s", list(self.metrics.keys()))
        logger.debug("  Runs     : %d", self.runs)
        if experiment_dir:
            logger.debug("  Output   : %s", experiment_dir)
        logger.debug("")

        # Build result container with model info ----
        model_names = [self._model_name(m) for m in self.models]
        self.result = ExperimentResult()
        self.result.set_models(self.models, model_names)

        total_combos = len(self.datasets) * len(self.models)
        total_work = total_combos * self.runs
        use_pbar = self.verbose >= 2

        # Metadata accumulators
        datasets_info: dict = {}
        raw_scores: dict = {}

        # ---- Progress bar (verbose>=2) ----
        pbar = None
        if use_pbar:
            pbar = tqdm(
                total=total_work,
                desc="Experiment",
                unit="run",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                file=sys.stdout,
            )

        for ds_name, ds_entry in self.datasets.items():

            logger.debug("Loading dataset: %s", ds_name)
            if pbar:
                pbar.set_postfix_str(f"loading {ds_name}…")

            Xs, As, labels = _load_dataset(ds_entry)
            labels = np.asarray(labels).reshape(-1)
            n_clusters = len(np.unique(labels))
            logger.debug(
                "  samples=%d, views=%d, k=%d",
                Xs[0].shape[0], len(Xs), n_clusters,
            )

            # Collect dataset info for metadata
            datasets_info[ds_name] = {
                "n_samples": int(Xs[0].shape[0]),
                "n_views": len(Xs),
                "n_clusters": n_clusters,
                "feature_dims": [
                    list(X.shape) for X in Xs
                ],
                "adjacency_shapes": [
                    list(A.shape) if A is not None else None
                    for A in As
                ] if As is not None else None,
            }

            for model_template in self.models:
                m_name = self._model_name(model_template)

                logger.debug("  %s × %s", ds_name, m_name)
                logger.debug("    params: %s", model_template.__dict__)

                run_metrics: dict[str, list] = {k: [] for k in self.metrics}
                run_times: list[float] = []

                for r in range(self.runs):
                    if pbar:
                        pbar.set_postfix_str(
                            f"{ds_name} × {m_name} [run {r+1}/{self.runs}]"
                        )

                    model = copy.deepcopy(model_template)

                    t0 = time.perf_counter()
                    model.fit(Xs, As)
                    elapsed = time.perf_counter() - t0

                    y_pred = model.labels_
                    scores = self._evaluate(labels, y_pred)

                    for k, v in scores.items():
                        run_metrics[k].append(v)
                    run_times.append(elapsed)

                    logger.debug(
                        "    run %d/%d: %s  (%.2fs)",
                        r + 1, self.runs,
                        ", ".join(f"{k}={v:.4f}" for k, v in scores.items()),
                        elapsed,
                    )

                    if pbar:
                        pbar.update(1)

                # Aggregate
                avg = OrderedDict()
                std = OrderedDict()
                for k in self.metrics:
                    vals = run_metrics[k]
                    avg[k] = float(np.mean(vals))
                    std[k] = float(np.std(vals))

                avg_time = float(np.mean(run_times))

                result = Result(
                    dataset_name=ds_name,
                    model_name=m_name,
                    metric_avg=avg,
                    metric_std=std if self.runs > 1 else None,
                    train_time=avg_time,
                )
                self.result.add(result)

                # Store raw per-run scores
                raw_key = f"{ds_name}/{m_name}"
                raw_scores[raw_key] = {
                    "runs": self.runs,
                    "metrics": {
                        k: [round(v, 6) for v in run_metrics[k]]
                        for k in self.metrics
                    },
                    "times": [round(t, 4) for t in run_times],
                }

                logger.debug(
                    "    → %s  (%.2fs)",
                    ", ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                    avg_time,
                )

        if pbar:
            pbar.close()

        # Print clean tables to console (WARNING level → verbose>=1)
        output = str(self.result)
        logger.warning("\n%s", output)

        # Also write full tables to log file
        logger.debug("\n[FULL RESULTS]\n%s", output)

        # Export full experiment folder

        if self.save_markdown and experiment_dir:
            from .markdown_export import export_experiment
            from pymv.tracking import (
                build_results_dict, build_metrics_csv_rows,
            )

            # Build config metadata
            _SKIP = {
                "labels_", "centroids_", "embedding_",
                "loss_history_", "_name",
            }
            _PREPROCESS_ONLY = {"beta", "tf_idf", "center", "scale"}
            all_preprocess_off = all(
                getattr(m, "preprocess", True) is False
                for m in self.models
            )
            config = {
                "datasets": list(self.datasets.keys()),
                "models": {
                    self._model_name(m): {
                        k: str(v) for k, v in m.__dict__.items()
                        if k not in _SKIP and not k.endswith("_")
                        and not (all_preprocess_off and k in _PREPROCESS_ONLY)
                    }
                    for m in self.models
                },
                "metrics": list(self.metrics.keys()),
                "runs": self.runs,
            }

            # 1. Markdown report (existing)
            folder = export_experiment(
                result=self.result,
                name=self.experiment_name,
                log_file=None,  # log is already in experiment_dir
                datasets_info=datasets_info,
                experiment_config=config,
                raw_scores=raw_scores,
                experiment_dir=experiment_dir,
            )

            # 2. Reproducible config.yaml
            reproducible_config = self._build_reproducible_config()
            tracker.save_config(reproducible_config)

            # 3. Structured results.json (per dataset × model)
            all_results_list = []
            for ds_name, result_list in self.result.all_results.items():
                for r in result_list:
                    model_template = None
                    for m in self.models:
                        if self._model_name(m) == r.model_name:
                            model_template = m
                            break
                    m_params = {}
                    if model_template:
                        m_params = {
                            k: v for k, v in model_template.__dict__.items()
                            if not k.startswith("_") and not k.endswith("_")
                        }
                    rd = build_results_dict(
                        dataset_name=ds_name,
                        model_name=r.model_name,
                        model_params=m_params,
                        metrics=dict(r.metric_avg),
                        metrics_std=dict(r.metric_std) if r.metric_std else None,
                        runs=self.runs,
                        train_time=r.train_time,
                    )
                    all_results_list.append(rd)

            tracker.save_results({"experiments": all_results_list})

            # 4. metrics.csv
            csv_rows = build_metrics_csv_rows(self.result)
            tracker.save_metrics_csv(csv_rows)

            logger.warning("Experiment saved to: %s", folder)

        return self.result
