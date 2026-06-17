"""
Shared helpers for loading multi-view ``.mat`` datasets.

Two layouts are supported:

1. **Standard** – views stored in a cell array ``X`` and labels in ``y``.
2. **Multimodal** – views stored as ``X_image`` / ``X_text``, an adjacency
   matrix in ``A``, and labels in ``labels``.
"""

import logging

import numpy as np
import scipy.io as io
import scipy.sparse as sp

from .base import get_dataset_path
from .dataset import MultiViewDataset
from .downloader import download_file

logger = logging.getLogger(__name__)

_BASE_URL = "https://zenodo.org/records/20732535/files"


def _to_dense(arr):
    """Convert to a dense numpy array regardless of input type."""
    if sp.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def load_standard_mat(name: str, filename: str, n_views: int) -> MultiViewDataset:
    """Load a dataset with views in ``X{cell}`` and labels in ``y``.

    Parameters
    ----------
    name : str
        Human-readable dataset name used for caching and logging.
    filename : str
        Name of the ``.mat`` file (including extension).
    n_views : int
        Expected number of views inside the ``X`` cell array.
    """
    logger.info("Loading %s dataset...", name)

    url = f"{_BASE_URL}/{filename}?download=1"
    dataset_dir = get_dataset_path(name)
    file_path = dataset_dir / filename

    download_file(url, file_path)

    logger.debug("Reading dataset from: %s", file_path)
    data = io.loadmat(file_path)

    # X is stored as a (1, n_views) object array of matrices
    X_cell = data["X"]
    Xs = [_to_dense(X_cell[0, v]).astype(np.float64) for v in range(n_views)]

    labels = _to_dense(data["y"]).reshape(-1)

    As = [None] * n_views

    logger.info(
        "Dataset loaded: samples=%d, views=%d",
        Xs[0].shape[0],
        len(Xs),
    )
    return MultiViewDataset(Xs, As, labels, name=name)


def load_multimodal_mat(name: str, filename: str) -> MultiViewDataset:
    """Load a multimodal dataset (``X_image``, ``X_text``, ``A``, ``labels``).

    Parameters
    ----------
    name : str
        Human-readable dataset name.
    filename : str
        Name of the ``.mat`` file.
    """
    logger.info("Loading %s dataset...", name)

    url = f"{_BASE_URL}/{filename}?download=1"
    dataset_dir = get_dataset_path(name)
    file_path = dataset_dir / filename

    download_file(url, file_path)

    logger.debug("Reading dataset from: %s", file_path)
    data = io.loadmat(file_path)

    X_image = _to_dense(data["X_image"]).astype(np.float64)
    X_text = _to_dense(data["X_text"]).astype(np.float64)
    A = _to_dense(data["A"]).astype(np.float64)

    labels = _to_dense(data["labels"]).reshape(-1)

    Xs = [X_image, X_text]
    As = [A, A]

    logger.info(
        "Dataset loaded: samples=%d, views=%d",
        Xs[0].shape[0],
        len(Xs),
    )
    return MultiViewDataset(Xs, As, labels, name=name)
