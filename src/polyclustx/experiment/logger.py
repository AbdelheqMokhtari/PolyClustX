"""
Logging utilities for polyclustx experiments.

Design
------
*   **Log file** — captures *everything* at DEBUG level:
    dataset loading details, per-run scores, timing, errors,
    result tables, etc.
*   **Console** — verbosity controls tables, live progress and debug output.
    The Python API defaults to tables only; the CLI defaults to progress.

When ``experiment_dir`` is provided the log file is written
directly inside that directory as ``experiment.log``.  Otherwise
it falls back to ``log_dir`` (default ``./logs``).

Verbose levels (console only):
    0 — silent   (nothing printed, not even tables)
    1 — tables   (model config table + result tables)
    2 — tables + run/trial and reported training-stage progress
    3 — debug    (everything, same as the log file)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

from tqdm import tqdm

# Map verbose int -> console log level
_VERBOSE_MAP = {
    0: logging.CRITICAL + 1,   # effectively silent
    1: logging.WARNING,        # tables only (we emit them at WARNING)
    2: logging.INFO,           # progress + tables
    3: logging.DEBUG,          # everything
}


class _TqdmHandler(logging.StreamHandler):
    """Logging handler that routes output through ``tqdm.write()``.

    This prevents log messages from breaking a tqdm progress bar
    that is updating in-place on the same stream.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_experiment_logger(
    log_dir: str | None = None,
    prefix: str = "polyclustx_experiment",
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

    logger = logging.getLogger(f"polyclustx.experiment.{timestamp}")
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
    # Uses _TqdmHandler so log messages don't break tqdm progress bars.
    console_level = _VERBOSE_MAP.get(verbose, logging.INFO)
    ch = _TqdmHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # Silence polyclustx.data loggers on console
    data_logger = logging.getLogger("polyclustx.data")
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
