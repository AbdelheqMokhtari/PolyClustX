"""
Random-search hyper-parameter tuning experiment.

Accepts flexible input formats for dataset and metrics, matching
the Experiment class API.

Usage
-----
>>> from pymv.experiment import RandomSearchExperiment
>>> from pymv.models import LMGEC
>>> from pymv.metrics import clustering_accuracy, nmi_score
>>>
>>> # --- dict metrics, loader function ---
>>> rs = RandomSearchExperiment(
...     model_class=LMGEC,
...     param_distributions={
...         "n_clusters": [3],
...         "embedding_dim": [4, 8, 16],
...         "temperature": [1.0, 5.0, 10.0],
...         "beta": [0.1, 0.5, 1.0, 2.0],
...     },
...     dataset=load_acm,              # or "acm" or (Xs, As, y)
...     metrics=[clustering_accuracy, nmi_score],  # or dict
...     best_metric="clustering_accuracy",
...     n_iter=10,
... )
>>> rs.run()
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Type

import numpy as np

from pymv.core import BaseModel
from .result import RandomSearchResult
from .logger import setup_experiment_logger


def _resolve_dataset(dataset) -> tuple[str, Any]:
    """
    Resolve flexible dataset input into ``(name, loader_or_tuple)``.

    Accepts:
    *   ``"acm"``          — name string (resolved via registry)
    *   ``load_acm``       — callable
    *   ``(Xs, As, y)``    — pre-loaded tuple
    *   ``("acm", load_acm)`` or ``("acm", (Xs, As, y))`` — named pair
    """
    from pymv.data.registry import DATASETS

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
    if callable(entry):
        return entry()
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

    Parameters
    ----------
    model_class : type
        The model **class** (not an instance).  Must follow the
        ``BaseModel`` interface.

    param_distributions : dict[str, list | callable]
        For each hyper-parameter, either:
        - a list of discrete values to sample from, or
        - a callable() returning a single random value
          (e.g. ``lambda: np.random.uniform(0.1, 10)``).

    dataset : str | callable | tuple
        Flexible — see :func:`_resolve_dataset`.

    metrics : list | dict | callable
        Flexible — same formats as :class:`Experiment`.

    best_metric : str
        The metric to optimise (highest wins).
        Must be a key in *metrics* (after normalisation).

    n_iter : int
        Number of random configurations to try.

    seed : int | None
        Seed for reproducibility.

    log_dir : str | None
        Directory for log files.  ``None`` → ``~/pymv_logs``.

    verbose : int
        Console verbosity (0–3).
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
        verbose: int = 2,
    ):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.dataset_name, self._dataset_entry = _resolve_dataset(dataset)
        self.metrics = _normalize_metrics(metrics)
        self.n_iter = n_iter
        self.seed = seed
        self.log_dir = log_dir
        self.verbose = verbose

        # Validate best_metric after normalisation
        if best_metric not in self.metrics:
            raise ValueError(
                f"best_metric '{best_metric}' not found in metrics "
                f"{list(self.metrics.keys())}"
            )
        self.best_metric = best_metric

        # Filled after run()
        self.result: RandomSearchResult | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> RandomSearchResult:
        rng = np.random.RandomState(self.seed)

        logger = setup_experiment_logger(
            log_dir=self.log_dir,
            prefix="pymv_random_search",
            verbose=self.verbose,
        )

        model_name = self.model_class.__name__

        logger.info("=" * 60)
        logger.info("  pymv Random Search")
        logger.info("=" * 60)
        logger.info(f"  Model     : {model_name}")
        logger.info(f"  Dataset   : {self.dataset_name}")
        logger.info(f"  Metrics   : {list(self.metrics.keys())}")
        logger.info(f"  Best by   : {self.best_metric} (↑ higher is better)")
        logger.info(f"  Trials    : {self.n_iter}")
        logger.info(f"  Seed      : {self.seed}")
        logger.info("")

        logger.info(f"  Loading dataset: {self.dataset_name}")
        Xs, As, labels = _load(self._dataset_entry)
        labels = np.asarray(labels).reshape(-1)

        self.result = RandomSearchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            best_metric_name=self.best_metric,
        )

        for i in range(self.n_iter):
            params = self._sample_params(rng)
            logger.info(
                f"  [{i + 1}/{self.n_iter}] params={params}"
            )

            try:
                model = self.model_class(**params)

                t0 = time.perf_counter()
                model.fit(Xs, As)
                elapsed = time.perf_counter() - t0

                scores = self._evaluate(labels, model.labels_)
                self.result.add_trial(params, scores, elapsed)

                summary = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                logger.info(f"    → {summary}  ({elapsed:.2f}s)")

            except Exception as exc:
                logger.error(f"    ✗ Trial {i + 1} FAILED: {exc}")
                nan_scores = OrderedDict(
                    (k, float("nan")) for k in self.metrics
                )
                self.result.add_trial(params, nan_scores, 0.0)

        # Print results table — WARNING so it shows at verbose>=1
        output = str(self.result)
        logger.info("")
        logger.warning("\n" + output)
        logger.debug("\n[RANDOM SEARCH TABLE]\n" + output)

        return self.result
