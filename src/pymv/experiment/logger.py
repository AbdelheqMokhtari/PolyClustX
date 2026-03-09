"""
Logging utilities for pymv experiments.

Creates timestamped log files that record full experiment details:
datasets, models, parameters, metrics, timing, and results tables.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime


def setup_experiment_logger(
    log_dir: str = "logs",
    prefix: str = "pymv_experiment",
) -> logging.Logger:
    """
    Create and return a logger that writes to both console and a
    timestamped log file.

    Parameters
    ----------
    log_dir : str
        Directory where log files are saved.
    prefix : str
        Filename prefix.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger(f"pymv.experiment.{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — all levels
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger
