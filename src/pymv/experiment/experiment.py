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
import os
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from pymv.core import BaseModel
from .result import ExperimentResult, Result
from .logger import setup_experiment_logger


# =====================================================================
# Normalizers — accept many input shapes, return canonical form
# =====================================================================

def _normalize_datasets(datasets) -> OrderedDict:
    """
    Accept any of the following and return
    ``OrderedDict[str, callable | tuple]``:

    *   ``["acm", "dblp"]``          — names resolved via registry
    *   ``[load_acm, load_dblp]``    — callables (name = fn.__name__)
    *   ``{"acm": load_acm, ...}``   — dict of callables
    *   ``{"acm": (Xs, As, y)}``     — dict of pre-loaded tuples
    *   single string ``"acm"``      — treated as ``["acm"]``
    *   single callable              — treated as ``[callable]``
    *   single tuple ``(Xs, As, y)`` — treated as ``{"dataset": tuple}``
    """
    from pymv.data.registry import DATASETS, load_dataset

    if isinstance(datasets, str):
        datasets = [datasets]
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


def _load_dataset(entry) -> tuple:
    """Call loader or return pre-loaded tuple."""
    if callable(entry):
        return entry()
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


# =====================================================================
# Experiment
# =====================================================================

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
    """

    def __init__(
        self,
        datasets,
        models,
        metrics,
        runs: int = 1,
        log_dir: str | None = None,
        verbose: int = 1,
    ):
        self.datasets = _normalize_datasets(datasets)
        self.models = _normalize_models(models)
        self.metrics = _normalize_metrics(metrics)
        self.runs = max(1, int(runs))
        self.log_dir = log_dir
        self.verbose = verbose

        # Filled after run()
        self.result: ExperimentResult | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """Execute the experiment and return an :class:`ExperimentResult`."""

        logger = setup_experiment_logger(
            log_dir=self.log_dir,
            prefix="pymv_experiment",
            verbose=self.verbose,
        )

        # ---- Log header (file + verbose>=3) ----
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
        logger.debug("")

        # ---- Build result container with model info ----
        model_names = [self._model_name(m) for m in self.models]
        self.result = ExperimentResult()
        self.result.set_models(self.models, model_names)

        total_combos = len(self.datasets) * len(self.models)
        combo_i = 0

        for ds_name, ds_entry in self.datasets.items():

            logger.debug("Loading dataset: %s", ds_name)
            Xs, As, labels = _load_dataset(ds_entry)
            labels = np.asarray(labels).reshape(-1)
            n_clusters = len(np.unique(labels))
            logger.debug(
                "  samples=%d, views=%d, k=%d",
                Xs[0].shape[0], len(Xs), n_clusters,
            )

            for model_template in self.models:
                combo_i += 1
                m_name = self._model_name(model_template)

                # verbose>=2: one-liner progress on console
                logger.info(
                    "  [%d/%d] %s × %s …",
                    combo_i, total_combos, ds_name, m_name,
                )
                logger.debug("    params: %s", model_template.__dict__)

                run_metrics: dict[str, list] = {k: [] for k in self.metrics}
                run_times: list[float] = []

                for r in range(self.runs):
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

                logger.debug(
                    "    → %s  (%.2fs)",
                    ", ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                    avg_time,
                )

        # ==============================================================
        # Print clean tables to console (WARNING level → verbose>=1)
        # ==============================================================
        output = str(self.result)
        logger.warning("\n%s", output)

        # Also write full tables to log file
        logger.debug("\n[FULL RESULTS]\n%s", output)

        return self.result
