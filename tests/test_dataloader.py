"""
Tests for pymv data loading and MultiViewDataset.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Registry & listing ────────────────────────────────────────────


class TestListDatasets:
    def test_returns_list(self):
        from pymv.data import list_datasets

        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_known_datasets_present(self):
        from pymv.data import list_datasets

        datasets = list_datasets()
        for name in ("acm", "dblp"):
            assert name in datasets, f"'{name}' missing from dataset list"


class TestLoadDatasetErrors:
    def test_unknown_dataset_raises(self):
        from pymv.data import load_dataset

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset_xyz")

    def test_empty_string_raises(self):
        from pymv.data import load_dataset

        with pytest.raises(ValueError):
            load_dataset("")


# ── MultiViewDataset container ────────────────────────────────────


class TestMultiViewDataset:
    def test_construction(self, mock_arrays):
        from pymv.data.dataset import MultiViewDataset

        Xs, As, labels = mock_arrays
        ds = MultiViewDataset(Xs, As, labels, name="test")

        assert ds.name == "test"
        assert ds.n_samples == 30
        assert ds.n_views == 2
        assert ds.n_clusters == 3

    def test_tuple_unpacking(self, mock_dataset):
        Xs, As, labels = mock_dataset

        assert isinstance(Xs, list)
        assert isinstance(As, list)
        assert isinstance(labels, np.ndarray)

    def test_len_is_three(self, mock_dataset):
        assert len(mock_dataset) == 3

    def test_getitem(self, mock_dataset):
        assert isinstance(mock_dataset[0], list)   # Xs
        assert isinstance(mock_dataset[1], list)   # As
        assert isinstance(mock_dataset[2], np.ndarray)  # labels

    def test_feature_dims(self, mock_dataset):
        dims = mock_dataset.feature_dims
        assert dims == [8, 6]

    def test_has_graphs_true(self, mock_dataset):
        assert mock_dataset.has_graphs is True

    def test_has_graphs_false(self, mock_arrays):
        from pymv.data.dataset import MultiViewDataset

        Xs, _, labels = mock_arrays
        ds = MultiViewDataset(Xs, [None, None], labels)
        assert ds.has_graphs is False

    def test_labels_are_flat(self, mock_dataset):
        assert mock_dataset.labels.ndim == 1

    def test_repr_contains_name(self, mock_dataset):
        r = repr(mock_dataset)
        assert "mock" in r

    def test_summary_contains_info(self, mock_dataset):
        s = mock_dataset.summary()
        assert "mock" in s
        assert "30" in s
        assert "2" in s


# ── Real dataset loading (ACM — smallest) ────────────────────────
# These tests actually download/load data.  Skip if data unavailable.


@pytest.fixture(scope="module")
def acm():
    from pymv.data import load_dataset

    return load_dataset("acm")


class TestLoadRealDataset:
    """Validate structure of a real loaded dataset (ACM)."""

    def test_returns_multiview_dataset(self, acm):
        from pymv.data.dataset import MultiViewDataset

        assert isinstance(acm, MultiViewDataset)

    def test_acm_views(self, acm):
        assert acm.n_views == 2

    def test_acm_has_graphs(self, acm):
        assert acm.has_graphs is True

    def test_acm_labels_shape(self, acm):
        assert acm.labels.ndim == 1
        assert acm.labels.shape[0] == acm.n_samples

    def test_acm_xs_shapes_consistent(self, acm):
        for X in acm.Xs:
            assert X.shape[0] == acm.n_samples

    def test_acm_as_shapes_consistent(self, acm):
        for A in acm.As:
            if A is not None:
                assert A.shape[0] == acm.n_samples
                assert A.shape[1] == acm.n_samples

    def test_acm_n_clusters_positive(self, acm):
        assert acm.n_clusters > 0
