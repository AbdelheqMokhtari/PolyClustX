"""
Logging utilities for pymv experiments.

Creates timestamped log files that record full experiment details:
datasets, models, parameters, metrics, timing, and results tables.

Verbose levels control console output:
    0 — silent   (nothing printed)
    1 — summary  (final table only)
    2 — progress (dataset/model names + summary)
    3 — debug    (every run, parameters, everything)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime


# Map verbose int -> console log level
_VERBOSE_MAP = {
    0: logging.CRITICAL + 1,   # effectively silent
    1: logging.WARNING,        # only results (we log them at WARNING)
    2: logging.INFO,           # progress + results
    3: logging.DEBUG,          # everything
}


def setup_experiment_logger(
    log_dir: str | None = None,
    prefix: str = "pymv_experiment",
    verbose: int = 2,
) -> logging.Logger:
    """
    Create and return a logger that writes to both console and a
    timestamped log file.

    Parameters
    ----------
    log_dir : str or None
        Directory where log files are saved.
        If ``None``, defaults to ``~/pymv_logs``.
    prefix : str
        Filename prefix.
    verbose : int
        Console verbosity level (0–3).

    Returns
    -------
    logging.Logger
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.expanduser("~"), "pymv_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger(f"pymv.experiment.{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — always captures everything
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler — controlled by verbose level
    console_level = _VERBOSE_MAP.get(verbose, logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.debug(f"Log file: {log_file}")
    if verbose >= 2:
        logger.info(f"Log file: {log_file}")
    return logger
