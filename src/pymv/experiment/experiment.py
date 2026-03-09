"""
Experiment runner for pymv.

Inspired by Cornac's Experiment class, adapted for multi-view
clustering: pass lists of datasets, models, and metric functions,
and get a formatted results table plus a detailed log file.

Usage
-----
>>> from pymv.experiment import Experiment
>>> from pymv.data import load_acm, load_dblp
>>> from pymv.models import LMGEC
>>> from pymv.metrics import clustering_accuracy, nmi_score
>>>
>>> Experiment(
...     datasets={"acm": load_acm, "dblp": load_dblp},
...     models=[
...         LMGEC(n_clusters=3, embedding_dim=4, max_iter=10),
...     ],
...     metrics={"ACC": clustering_accuracy, "NMI": nmi_score},
... ).run()
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

from pymv.core import BaseModel
from .result import ExperimentResult, Result
from .logger import setup_experiment_logger


class Experiment:
    """
    Run models × datasets × metrics and report a formatted table.

    Parameters
    ----------
    datasets : dict[str, callable]
        Mapping of dataset name -> loader function.
        Each loader must return ``(Xs, As, labels)``.

    models : list of BaseModel
        Model instances to evaluate. Each is deep-copied per
        (dataset, run) so state is not shared.

    metrics : dict[str, callable]
        Mapping of metric display name -> function(y_true, y_pred).

    runs : int
        Number of repetitions (mean ± std are reported).

    log_dir : str
        Directory for log files.

    verbose : bool
        Print progress to console.
    """

    def __init__(
        self,
        datasets: Dict[str, Callable],
        models: List[BaseModel],
        metrics: Dict[str, Callable],
        runs: int = 1,
        log_dir: str = "logs",
        verbose: bool = True,
    ):
        self.datasets = datasets
        self.models = models
        self.metrics = metrics
        self.runs = runs
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
            return name
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
        logger = setup_experiment_logger(
            log_dir=self.log_dir,
            prefix="pymv_experiment",
        )

        logger.info("=" * 60)
        logger.info("pymv Experiment")
        logger.info("=" * 60)
        logger.info(f"Datasets : {list(self.datasets.keys())}")
        logger.info(f"Models   : {[self._model_name(m) for m in self.models]}")
        logger.info(f"Metrics  : {list(self.metrics.keys())}")
        logger.info(f"Runs     : {self.runs}")
        logger.info("")

        self.result = ExperimentResult()

        for ds_name, loader in self.datasets.items():

            logger.info(f"Loading dataset: {ds_name}")
            Xs, As, labels = loader()
            labels = np.asarray(labels).reshape(-1)
            n_clusters = len(np.unique(labels))
            logger.debug(
                f"  samples={Xs[0].shape[0]}, views={len(Xs)}, "
                f"k={n_clusters}"
            )

            for model_template in self.models:
                m_name = self._model_name(model_template)
                logger.info(f"  Model: {m_name}")
                logger.debug(f"    params: {model_template.__dict__}")

                run_metrics: dict[str, list] = {k: [] for k in self.metrics}
                run_times: list[float] = []

                for r in range(self.runs):
                    # Deep-copy so each run starts fresh
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
                        f"    run {r}: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                        + f", time={elapsed:.2f}s"
                    )

                # Aggregate across runs
                avg_metrics = OrderedDict()
                for k in self.metrics:
                    vals = run_metrics[k]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    avg_metrics[k] = mean
                    if self.runs > 1:
                        avg_metrics[f"{k}_std"] = std

                avg_time = float(np.mean(run_times))

                result = Result(
                    dataset_name=ds_name,
                    model_name=m_name,
                    metric_values=avg_metrics,
                    train_time=avg_time,
                )
                self.result.add(result)

                summary = ", ".join(
                    f"{k}={v:.4f}" for k, v in avg_metrics.items()
                )
                logger.info(f"    -> {summary}, time={avg_time:.2f}s")

        # Print final table
        output = str(self.result)
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info("\n" + output)

        return self.result
