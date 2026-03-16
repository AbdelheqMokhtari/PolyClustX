"""
Logging utilities for pymv experiments.

Design
------
*   **Log file** — captures *everything* at DEBUG level:
    dataset loading details, per-run scores, timing, errors,
    result tables, etc.
*   **Console** — shows only the final clean tables (config +
    results).  All noisy progress messages go exclusively to
    the log file.

When ``experiment_dir`` is provided the log file is written
directly inside that directory as ``experiment.log``.  Otherwise
it falls back to ``log_dir`` (default ``./logs``).

Verbose levels (console only):
    0 — silent   (nothing printed, not even tables)
    1 — tables   (model config table + result tables)
    2 — tables + one-liner progress per (dataset × model)
    3 — debug    (everything, same as the log file)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


# Map verbose int -> console log level
_VERBOSE_MAP = {
    0: logging.CRITICAL + 1,   # effectively silent
    1: logging.WARNING,        # tables only (we emit them at WARNING)
    2: logging.INFO,           # progress + tables
    3: logging.DEBUG,          # everything
}


def setup_experiment_logger(
    log_dir: str | None = None,
    prefix: str = "pymv_experiment",
    verbose: int = 1,
    experiment_dir: str | None = None,
    log: bool = True,
) -> logging.Logger:
    """
    Create and return a logger with a file handler (everything)
    and a console handler (controlled by *verbose*).

    Parameters
    ----------
    log_dir : str or None
        Directory where log files are saved when *experiment_dir*
        is ``None``.  Defaults to ``"./logs"``.
    prefix : str
        Filename prefix (used only when *experiment_dir* is None).
    verbose : int
        Console verbosity level (0–3).  Default is **1** (tables
        only).
    experiment_dir : str or None
        If provided, the log file is written as
        ``<experiment_dir>/experiment.log`` and *log_dir* / *prefix*
        are ignored.
    log : bool
        If ``False``, **no log file is created** — only the
        console handler is attached.  Default is ``True``.

    Returns
    -------
    logging.Logger
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = logging.getLogger(f"pymv.experiment.{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_file: str | None = None
    fh: logging.FileHandler | None = None

    # File handler (skipped when log=False)
    if log:
        if experiment_dir is not None:
            os.makedirs(experiment_dir, exist_ok=True)
            log_file = os.path.join(experiment_dir, "experiment.log")
        else:
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    # Console handler controlled by verbose 
    console_level = _VERBOSE_MAP.get(verbose, logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # Silence pymv.data loggers on console
    data_logger = logging.getLogger("pymv.data")
    data_logger.setLevel(logging.DEBUG)
    data_logger.handlers.clear()
    data_logger.propagate = False
    if fh is not None:
        data_logger.addHandler(fh)
    else:
        # No file handler — silence data loggers completely
        data_logger.addHandler(logging.NullHandler())

    if log_file:
        logger.debug("Log file: %s", log_file)

    # Store the log file path on the logger for later retrieval
    logger.log_file_path = log_file  # type: ignore[attr-defined]

    return logger
