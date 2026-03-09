"""
Random-search hyper-parameter tuning experiment.

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
...     dataset_loader=load_acm,
...     dataset_name="acm",
...     metrics={"ACC": clustering_accuracy, "NMI": nmi_score},
...     best_metric="ACC",
...     n_iter=10,
... )
>>> rs.run()
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Type

import numpy as np

from pymv.core import BaseModel
from .result import RandomSearchResult
from .logger import setup_experiment_logger


class RandomSearchExperiment:
    """
    Random-search over hyper-parameters for a single model class.

    Parameters
    ----------
    model_class : type
        The model **class** (not an instance). Must follow the
        ``BaseModel`` interface.

    param_distributions : dict[str, list | callable]
        For each hyper-parameter, either:
        - a list of discrete values to sample from, or
        - a callable() returning a single random value
          (e.g. ``lambda: np.random.uniform(0.1, 10)``).

    dataset_loader : callable
        Function returning ``(Xs, As, labels)``.

    dataset_name : str
        Human-readable name (used in table headers).

    metrics : dict[str, callable]
        Mapping name -> function(y_true, y_pred).

    best_metric : str
        The metric to optimise (must be a key in *metrics*).
        The trial with the **highest** value is selected as best.

    n_iter : int
        Number of random configurations to try.

    seed : int | None
        Seed for reproducibility.

    log_dir : str
        Directory for log files.

    verbose : bool
        Print progress to console.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        param_distributions: Dict[str, Any],
        dataset_loader: Callable,
        dataset_name: str,
        metrics: Dict[str, Callable],
        best_metric: str,
        n_iter: int = 20,
        seed: int | None = None,
        log_dir: str = "logs",
        verbose: bool = True,
    ):
        if best_metric not in metrics:
            raise ValueError(
                f"best_metric '{best_metric}' not found in metrics "
                f"{list(metrics.keys())}"
            )

        self.model_class = model_class
        self.param_distributions = param_distributions
        self.dataset_loader = dataset_loader
        self.dataset_name = dataset_name
        self.metrics = metrics
        self.best_metric = best_metric
        self.n_iter = n_iter
        self.seed = seed
        self.log_dir = log_dir
        self.verbose = verbose

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
        )

        model_name = self.model_class.__name__

        logger.info("=" * 60)
        logger.info("pymv Random Search")
        logger.info("=" * 60)
        logger.info(f"Model     : {model_name}")
        logger.info(f"Dataset   : {self.dataset_name}")
        logger.info(f"Metrics   : {list(self.metrics.keys())}")
        logger.info(f"Best by   : {self.best_metric} (max)")
        logger.info(f"Trials    : {self.n_iter}")
        logger.info(f"Seed      : {self.seed}")
        logger.info("")

        logger.info(f"Loading dataset: {self.dataset_name}")
        Xs, As, labels = self.dataset_loader()
        labels = np.asarray(labels).reshape(-1)

        self.result = RandomSearchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            best_metric_name=self.best_metric,
        )

        for i in range(self.n_iter):
            params = self._sample_params(rng)
            logger.info(f"Trial {i + 1}/{self.n_iter}  params={params}")

            try:
                model = self.model_class(**params)

                t0 = time.perf_counter()
                model.fit(Xs, As)
                elapsed = time.perf_counter() - t0

                scores = self._evaluate(labels, model.labels_)
                self.result.add_trial(params, scores, elapsed)

                summary = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                logger.info(f"  -> {summary}, time={elapsed:.2f}s")

            except Exception as exc:
                logger.error(f"  Trial {i + 1} FAILED: {exc}")
                # Store NaN metrics so the trial is visible
                nan_scores = OrderedDict(
                    (k, float("nan")) for k in self.metrics
                )
                self.result.add_trial(params, nan_scores, 0.0)

        # Summary
        best = self.result.best_params
        best_scores = self.result.best_metrics
        logger.info("")
        logger.info("=" * 60)
        logger.info("BEST CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Params  : {best}")
        logger.info(
            "Metrics : "
            + ", ".join(f"{k}={v:.4f}" for k, v in best_scores.items())
        )
        logger.info("")

        # Print table
        output = str(self.result)
        logger.info("\n" + output)

        return self.result
