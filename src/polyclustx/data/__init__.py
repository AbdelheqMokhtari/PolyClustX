"""
polyclustx.data — multi-view benchmark datasets.

Every dataset downloads on first use and is cached under
``~/.polyclustx_datasets`` (override with the ``POLYCLUSTX_DATA_DIR`` environment
variable).  All loaders return a :class:`MultiViewDataset`.

Usage
-----
>>> from polyclustx.data import load_dataset, list_datasets
>>> list_datasets()                    # all names
>>> list_datasets(category="graph")    # filter by category
>>> data = load_dataset("acm")         # by name…
>>> from polyclustx.data import load_acm
>>> data = load_acm()                  # …or via its dedicated loader
>>> data.Xs, data.As, data.labels
>>> data.n_samples, data.n_views, data.n_clusters

The full catalog lives in :mod:`polyclustx.data.registry`; adding a dataset
is a single ``DatasetSpec`` entry there.
"""

from . import registry as _registry
from .base import get_data_home
from .dataset import MultiViewDataset
from .importers import dataset_from_arrays, load_files, read_arrays
from .missing import align_views, impute_missing, simulate_missing_views
from .registry import (
    CATALOG,
    DATASETS,
    get_dataset_info,
    list_datasets,
    list_variants,
    load_data,
    load_dataset,
)

# Re-export the per-dataset loaders generated from the catalog
# (load_acm, load_dblp, …) so `from polyclustx.data import load_acm` works.
# load_pascal / load_coco_all / load_coco_cross come through as their
# variant-aware wrappers.
_globals = globals()
for _name in DATASETS:
    _globals[f"load_{_name}"] = getattr(_registry, f"load_{_name}")

__all__ = [
    "MultiViewDataset",
    "align_views",
    "impute_missing",
    "simulate_missing_views",
    "dataset_from_arrays",
    "load_files",
    "read_arrays",
    "CATALOG",
    "DATASETS",
    "get_data_home",
    "get_dataset_info",
    "list_datasets",
    "list_variants",
    "load_data",
    "load_dataset",
    *sorted(f"load_{name}" for name in DATASETS),
]
