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

The ``log_dir`` parameter defaults to ``"./logs"`` (relative to
the current working directory) so logs live next to your project.

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
) -> logging.Logger:
    """
    Create and return a logger with a file handler (everything)
    and a console handler (controlled by *verbose*).

    Parameters
    ----------
    log_dir : str or None
        Directory where log files are saved.
        ``None`` → ``"./logs"`` (current working directory).
    prefix : str
        Filename prefix for the log file.
    verbose : int
        Console verbosity level (0–3).  Default is **1** (tables
        only).

    Returns
    -------
    logging.Logger
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger(f"pymv.experiment.{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    # ---- File handler — always captures everything ----
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # ---- Console handler — controlled by verbose ----
    console_level = _VERBOSE_MAP.get(verbose, logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # ---- Silence pymv.data loggers on console ----
    # Their file output is still captured because we attach the
    # experiment file handler to the 'pymv.data' parent logger.
    data_logger = logging.getLogger("pymv.data")
    data_logger.setLevel(logging.DEBUG)
    data_logger.handlers.clear()
    data_logger.propagate = False
    # Only send to the file — never to the console
    data_logger.addHandler(fh)

    logger.debug("Log file: %s", log_file)

    return logger
