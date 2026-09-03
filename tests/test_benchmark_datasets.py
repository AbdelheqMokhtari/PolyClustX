"""
Tests for benchmark multi-view dataset loaders.

For every dataset the loader is checked to return a ``MultiViewDataset``
with the expected number of views, 1-D labels matching the sample count,
and feature matrices that agree on that count.  Multimodal datasets must
additionally report ``has_graphs`` and carry square adjacency matrices.

Loading a dataset downloads it from Zenodo, so the loader tests are marked
``network`` and deselected by default.  Run them with ``pytest -m network``.
The registry check needs no network and always runs.
"""

from __future__ import annotations

import pytest

from pymv.data import list_datasets
from pymv.data.dataset import MultiViewDataset

from .helpers import load_or_skip

# name -> expected number of views
STANDARD_DATASETS = {
    "100leaves": 3,
    "3sources": 3,
    "acm_5view": 5,
    "aloi": 4,
    "bbc4view": 4,
    "bbcsport": 2,
    "caltech101_7": 6,
    "citeseer": 2,
    "cora": 4,
    "handwritten": 6,
    "mnist_10k": 3,
    "mnist_4": 3,
    "msrc_v5": 5,
    "outdoor_scene": 4,
    "protein_fold": 12,
    "reuters_1200": 5,
    "reuters_1500": 5,
    "uci": 3,
    "webkb": 2,
    "wikipedia": 2,
    "yale": 3,
}

MULTIMODAL_DATASETS = [
    "arts_multimodal",
    "grocery_multimodal",
    "grocerys_multimodal",
    "movies_multimodal",
    "photo_multimodal",
    "reddit_multimodal",
    "reddits_multimodal",
    "toys_multimodal",
]

# BBC4view.mat on Zenodo is corrupt (unreadable by scipy and by MATLAB/HDF5
# readers) — the file needs to be re-uploaded upstream.
BROKEN = {"bbc4view": "BBC4view.mat hosted on Zenodo is corrupt"}


def _param(name):
    marks = [pytest.mark.skip(reason=BROKEN[name])] if name in BROKEN else []
    return pytest.param(name, marks=marks, id=name)


def _check_common(ds, expected_views):
    assert isinstance(ds, MultiViewDataset)
    assert ds.n_views == expected_views
    assert ds.labels.ndim == 1
    assert ds.labels.shape[0] == ds.n_samples
    assert ds.n_clusters > 0
    for X in ds.Xs:
        assert X.shape[0] == ds.n_samples


@pytest.mark.network
@pytest.mark.parametrize("name", [_param(n) for n in STANDARD_DATASETS])
def test_standard_dataset(name):
    ds = load_or_skip(name)
    _check_common(ds, STANDARD_DATASETS[name])
    assert ds.has_graphs is False


@pytest.mark.network
@pytest.mark.parametrize("name", [_param(n) for n in MULTIMODAL_DATASETS])
def test_multimodal_dataset(name):
    ds = load_or_skip(name)
    _check_common(ds, expected_views=2)
    assert ds.has_graphs is True
    for A in ds.As:
        if A is not None:
            assert A.shape == (ds.n_samples, ds.n_samples)


@pytest.mark.parametrize(
    "name", sorted({*STANDARD_DATASETS, *MULTIMODAL_DATASETS})
)
def test_dataset_is_registered(name):
    assert name in list_datasets(), f"'{name}' missing from registry"
