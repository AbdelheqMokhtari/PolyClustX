"""
Random-search hyper-parameter tuning experiment.

Accepts flexible input formats for dataset and metrics, matching
the Experiment class API.

Console output: a single clean table (params + metrics + time)
with the best trial highlighted.  All noisy details (loading,
per-trial logs, errors) go exclusively to the log file.

Usage
-----
>>> from pymv.experiment import RandomSearchExperiment
>>> from pymv.models import LMGEC
>>> from pymv.metrics import clustering_accuracy, nmi_score
>>>
>>> rs = RandomSearchExperiment(
...     model_class=LMGEC,
...     param_distributions={
...         "n_clusters": [3],
...         "embedding_dim": [4, 8, 16],
...         "temperature": [1.0, 5.0, 10.0],
...         "beta": [0.1, 0.5, 1.0, 2.0],
...     },
...     dataset="acm",
...     metrics=[clustering_accuracy, nmi_score],
...     best_metric="clustering_accuracy",
...     n_iter=10,
... )
>>> rs.run()
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from typing import Any, Dict, Type

import numpy as np
from tqdm import tqdm

from pymv.core import BaseModel

from .logger import setup_experiment_logger
from .result import RandomSearchResult, _in_notebook


def _resolve_dataset(dataset) -> tuple[str, Any]:
    """
    Resolve flexible dataset input into ``(name, loader_or_tuple)``.

    Accepts:
    *   ``"acm"``          — name string (resolved via registry)
    *   ``load_acm``       — callable
    *   ``(Xs, As, y)``    — pre-loaded tuple
    *   ``("acm", load_acm)`` or ``("acm", (Xs, As, y))`` — named pair
    *   ``MultiViewDataset`` — dataset object
    """
    from pymv.data.dataset import MultiViewDataset
    from pymv.data.registry import DATASETS

    if isinstance(dataset, MultiViewDataset):
        name = getattr(dataset, "name", None) or "dataset"
        return name, dataset

    if isinstance(dataset, str):
        name = dataset
        return name, DATASETS[name.lower()]

    if isinstance(dataset, tuple) and len(dataset) == 2:
        name, entry = dataset
        if isinstance(name, str):
            return name, entry

    if callable(dataset):
        name = getattr(dataset, "__name__", type(dataset).__name__)
        if name.startswith("load_"):
            name = name[5:]
        return name, dataset

    if isinstance(dataset, tuple) and len(dataset) == 3:
        return "dataset", dataset

    raise TypeError(f"Cannot interpret dataset: {type(dataset)}")


def _load(entry) -> tuple:
    from pymv.data.dataset import MultiViewDataset

    if isinstance(entry, MultiViewDataset):
        return (entry.Xs, entry.As, entry.labels)
    if callable(entry):
        result = entry()
        if isinstance(result, MultiViewDataset):
            return (result.Xs, result.As, result.labels)
        return result
    if isinstance(entry, tuple) and len(entry) == 3:
        return entry
    raise TypeError(f"Cannot load dataset from {type(entry)}")


def _normalize_metrics(metrics) -> OrderedDict:
    """Same flexible parsing as Experiment."""
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

    raise TypeError(f"metrics must be dict, list, or callable, got {type(metrics)}")


class RandomSearchExperiment:
    """
    Random-search over hyper-parameters for a single model class.

    Console output: a single clean results table (best trial
    marked with ``★``).  Everything else goes to the log file.

    Parameters
    ----------
    model_class : type
        The model **class** (not an instance).

    param_distributions : dict[str, list | callable]
        For each hyper-parameter, either:
        - a list of discrete values to sample from, or
        - a callable() returning a single random value.

    dataset : str | callable | tuple
        Flexible — see :func:`_resolve_dataset`.

    metrics : list | dict | callable
        Flexible — same formats as :class:`Experiment`.

    best_metric : str
        The metric to optimise (highest wins).

    n_iter : int
        Number of random configurations to try.

    seed : int | None
        Seed for reproducibility.

    log_dir : str | None
        Directory for log files.  ``None`` → ``./logs`` (cwd).

    verbose : int
        Console verbosity (0–3).  Default is **1** (table only).

    save_markdown : bool
        If ``True``, save a well-organized Markdown report to
        ``experiments/<experiment_name>/results.md``.
        Default is ``False``.

    experiment_name : str or None
        Folder name under ``experiments/``.
        ``None`` → timestamped folder (e.g. ``2026-03-31_14-30-12``).

    log : bool
        If ``True``, a log file is created.
        If ``False`` (default), **no log file is written** — only
        console output is produced.

    save_path : str or None
        Base directory for saving experiment results.  Only used
        when ``save_markdown=True``.  Defaults to ``./experiments``.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        param_distributions: Dict[str, Any],
        dataset,
        metrics,
        best_metric: str,
        n_iter: int = 20,
        seed: int | None = None,
        log_dir: str | None = None,
        verbose: int = 1,
        save_markdown: bool = False,
        experiment_name: str | None = None,
        log: bool = False,
        save_path: str | None = None,
    ):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.dataset_name, self._dataset_entry = _resolve_dataset(dataset)
        self.metrics = _normalize_metrics(metrics)
        self.n_iter = n_iter
        self.seed = seed
        self.log_dir = log_dir
        self.verbose = verbose
        self.save_markdown = save_markdown
        self.experiment_name = experiment_name
        self.log = log
        self.save_path = save_path

        # Validate best_metric after normalisation
        if best_metric not in self.metrics:
            raise ValueError(
                f"best_metric '{best_metric}' not found in metrics "
                f"{list(self.metrics.keys())}"
            )
        self.best_metric = best_metric

        # Filled after run()
        self.result: RandomSearchResult | None = None

    # helpers

    def _sample_params(self, rng: np.random.RandomState) -> dict:
        params: dict = {}
        for key, dist in self.param_distributions.items():
            if callable(dist) and not isinstance(dist, (list, tuple)):
                params[key] = dist()
            else:
                idx = rng.randint(0, len(dist))
                params[key] = dist[idx]
        return params

    def _evaluate(self, y_true, y_pred) -> OrderedDict:
        values = OrderedDict()
        for name, fn in self.metrics.items():
            values[name] = float(fn(y_true, y_pred))
        return values

    def _build_reproducible_config(self) -> dict:
        """Build a config dict that can re-run this random search."""
        from pymv.tracking import _to_python

        config: dict = {
            "dataset": self.dataset_name,
            "model": {
                "name": self.model_class.__name__.lower(),
            },
            "search": {
                "n_iter": self.n_iter,
                "best_metric": self.best_metric,
                "seed": self.seed,
                "param_distributions": {
                    k: _to_python(v)
                    for k, v in self.param_distributions.items()
                },
            },
            "metrics": list(self.metrics.keys()),
        }

        # Add best trial params if available
        if self.result and self.result.best_trial:
            best = self.result.best_trial
            config["model"]["params"] = {
                k: _to_python(v) for k, v in best["params"].items()
            }
            config["best_trial"] = {
                "scores": {
                    k: _to_python(v) for k, v in best["metrics"].items()
                },
                "time": round(best.get("time", 0.0), 4),
            }

        return config

    # main loop

    def run(self) -> RandomSearchResult:
        rng = np.random.RandomState(self.seed)

        # Resolve experiment folder first so the log goes inside it
        experiment_dir: str | None = None
        tracker = None
        if self.save_markdown:
            from pymv.tracking import ExperimentTracker
            tracker = ExperimentTracker(
                name=self.experiment_name,
                save_path=self.save_path,
            )
            experiment_dir = tracker.experiment_dir

        use_log = self.log or self.save_markdown

        logger = setup_experiment_logger(
            log_dir=self.log_dir,
            prefix="pymv_random_search",
            verbose=self.verbose,
            experiment_dir=experiment_dir,
            log=use_log,
        )

        model_name = self.model_class.__name__

        # Log header (file only unless verbose>=3)
        logger.debug("=" * 60)
        logger.debug("  pymv Random Search")
        logger.debug("=" * 60)
        logger.debug("  Model     : %s", model_name)
        logger.debug("  Dataset   : %s", self.dataset_name)
        logger.debug("  Metrics   : %s", list(self.metrics.keys()))
        logger.debug("  Best by   : %s (↑ higher is better)", self.best_metric)
        logger.debug("  Trials    : %d", self.n_iter)
        logger.debug("  Seed      : %s", self.seed)
        if experiment_dir:
            logger.debug("  Output    : %s", experiment_dir)
        logger.debug("")

        logger.debug("Loading dataset: %s", self.dataset_name)
        Xs, As, labels = _load(self._dataset_entry)
        labels = np.asarray(labels).reshape(-1)
        n_clusters = len(np.unique(labels))

        # Collect dataset info for metadata
        datasets_info = {
            self.dataset_name: {
                "n_samples": int(Xs[0].shape[0]),
                "n_views": len(Xs),
                "n_clusters": n_clusters,
                "feature_dims": [list(X.shape) for X in Xs],
                "adjacency_shapes": [
                    list(A.shape) if A is not None else None
                    for A in As
                ] if As is not None else None,
            }
        }

        self.result = RandomSearchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            best_metric_name=self.best_metric,
        )

        use_pbar = self.verbose >= 2

        # Progress bar (verbose>=2)
        pbar = None
        if use_pbar:
            pbar = tqdm(
                total=self.n_iter,
                desc="Random Search",
                unit="trial",
                dynamic_ncols=True,
                file=sys.stdout,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )

        for i in range(self.n_iter):
            params = self._sample_params(rng)

            logger.debug("  [%d/%d] params=%s", i + 1, self.n_iter, params)

            if pbar:
                pbar.set_postfix_str(
                    f"trial {i+1}/{self.n_iter}"
                )

            try:
                model = self.model_class(**params)

                t0 = time.perf_counter()
                model.fit(Xs, As)
                elapsed = time.perf_counter() - t0

                scores = self._evaluate(labels, model.labels_)
                self.result.add_trial(params, scores, elapsed)

                # Show best so far in the progress bar
                if pbar:
                    best = self.result.best_score
                    pbar.set_postfix_str(
                        f"best {self.best_metric}={best:.4f}"
                    )

                logger.debug(
                    "    → %s  (%.2fs)",
                    ", ".join(f"{k}={v:.4f}" for k, v in scores.items()),
                    elapsed,
                )

            except Exception as exc:
                logger.error("    ✗ Trial %d FAILED: %s", i + 1, exc)
                nan_scores = OrderedDict(
                    (k, float("nan")) for k in self.metrics
                )
                self.result.add_trial(params, nan_scores, 0.0)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # Print clean table to console (WARNING → verbose>=1)
        output = str(self.result)

        if _in_notebook() and self.verbose >= 1 and output.strip():
            import html as _html_mod

            from IPython.display import HTML, display
            html_out = (
                '<pre style="white-space: pre; overflow-x: auto; '
                'font-family: monospace; font-size: 13px; '
                'line-height: 1.4; padding: 8px 0; margin: 0;">'
                f"{_html_mod.escape(output)}</pre>"
            )
            display(HTML(html_out))
        else:
            logger.warning("\n%s", output)

        logger.debug("\n[RANDOM SEARCH TABLE]\n%s", output)

        # Export full experiment folder
        if self.save_markdown and experiment_dir:
            from pymv.tracking import build_metrics_csv_rows, build_results_dict

            from .markdown_export import export_random_search

            config = {
                "model_class": model_name,
                "param_distributions": {
                    k: str(v) for k, v in self.param_distributions.items()
                },
                "best_metric": self.best_metric,
                "n_iter": self.n_iter,
                "seed": self.seed,
                "metrics": list(self.metrics.keys()),
            }

            # 1. Markdown report (existing)
            folder = export_random_search(
                result=self.result,
                name=self.experiment_name,
                log_file=None,  # log is already in experiment_dir
                datasets_info=datasets_info,
                experiment_config=config,
                experiment_dir=experiment_dir,
            )

            # 2. Reproducible config.yaml
            reproducible_config = self._build_reproducible_config()
            tracker.save_config(reproducible_config)

            # 3. Structured results.json
            best = self.result.best_trial
            best_params = best["params"] if best else {}
            rd = build_results_dict(
                dataset_name=self.dataset_name,
                model_name=model_name,
                model_params=best_params,
                metrics=dict(best["metrics"]) if best else {},
                runs=self.n_iter,
                train_time=best.get("time", 0.0) if best else 0.0,
            )
            rd["search"] = {
                "n_iter": self.n_iter,
                "best_metric": self.best_metric,
                "best_score": self.result.best_score,
                "n_trials": len(self.result.trials),
            }
            tracker.save_results(rd)

            # 4. metrics.csv
            csv_rows = build_metrics_csv_rows(self.result)
            tracker.save_metrics_csv(csv_rows)

            logger.warning("Experiment saved to: %s", folder)

        return self.result
