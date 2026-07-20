"""Location of the on-disk dataset cache."""

import os
from pathlib import Path


def get_data_home() -> Path:
    """Root directory where downloaded datasets are cached.

    Defaults to ``~/.pymv_datasets``.  Override by setting the
    ``PYMV_DATA_DIR`` environment variable.
    """
    env = os.environ.get("PYMV_DATA_DIR")
    return Path(env) if env else Path.home() / ".pymv_datasets"


def get_dataset_path(name: str) -> Path:
    """Return (and create) the cache directory for one dataset."""
    path = get_data_home() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
