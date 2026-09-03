"""
Tests for benchmark multi-view dataset loaders.

Each test class validates:
- The loader returns a ``MultiViewDataset``.
- The expected number of views is present.
- Labels are 1-D and match the sample count.
- All feature matrices share the same sample count.
- Multimodal datasets report ``has_graphs`` and have square adjacency.
"""

from __future__ import annotations

import pytest

from pymv.data.dataset import MultiViewDataset

# ── Helpers ───────────────────────────────────────────────────────────


def _check_standard(ds, expected_views: int):
    """Shared assertions for standard (no-graph) datasets."""
    assert isinstance(ds, MultiViewDataset)
    assert ds.n_views == expected_views
    assert ds.labels.ndim == 1
    assert ds.labels.shape[0] == ds.n_samples
    assert ds.n_clusters > 0
    for X in ds.Xs:
        assert X.shape[0] == ds.n_samples
    assert ds.has_graphs is False


def _check_multimodal(ds):
    """Shared assertions for multimodal (graph) datasets."""
    assert isinstance(ds, MultiViewDataset)
    assert ds.n_views == 2
    assert ds.labels.ndim == 1
    assert ds.labels.shape[0] == ds.n_samples
    assert ds.n_clusters > 0
    for X in ds.Xs:
        assert X.shape[0] == ds.n_samples
    assert ds.has_graphs is True
    for A in ds.As:
        if A is not None:
            assert A.shape[0] == ds.n_samples
            assert A.shape[1] == ds.n_samples


# =====================================================================
#  Standard multi-view datasets
# =====================================================================


class TestLoad100Leaves:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_100leaves
        return load_100leaves()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


class TestLoad3Sources:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_3sources
        return load_3sources()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


class TestLoadAcm5View:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_acm_5view
        return load_acm_5view()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=5)


class TestLoadAloi:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_aloi
        return load_aloi()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=4)


@pytest.mark.skip(
    reason="BBC4view.mat hosted on Zenodo is corrupt (unreadable by scipy "
    "and MATLAB/HDF5 readers) — the file needs to be re-uploaded."
)
class TestLoadBbc4View:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_bbc4view
        return load_bbc4view()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=4)


class TestLoadBbcSport:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_bbcsport
        return load_bbcsport()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=2)


class TestLoadCaltech101_7:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_caltech101_7
        return load_caltech101_7()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=6)


class TestLoadCiteSeer:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_citeseer
        return load_citeseer()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=2)


class TestLoadCora:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_cora
        return load_cora()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=4)


class TestLoadHandwritten:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_handwritten
        return load_handwritten()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=6)


class TestLoadMnist10k:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_mnist_10k
        return load_mnist_10k()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


class TestLoadMnist4:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_mnist_4
        return load_mnist_4()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


class TestLoadMsrcV5:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_msrc_v5
        return load_msrc_v5()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=5)


class TestLoadOutdoorScene:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_outdoor_scene
        return load_outdoor_scene()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=4)


class TestLoadProteinFold:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_protein_fold
        return load_protein_fold()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=12)


class TestLoadReuters1200:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_reuters_1200
        return load_reuters_1200()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=5)


class TestLoadReuters1500:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_reuters_1500
        return load_reuters_1500()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=5)


class TestLoadUci:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_uci
        return load_uci()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


class TestLoadWebKB:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_webkb
        return load_webkb()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=2)


class TestLoadWikipedia:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_wikipedia
        return load_wikipedia()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=2)


class TestLoadYale:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_yale
        return load_yale()

    def test_structure(self, ds):
        _check_standard(ds, expected_views=3)


# =====================================================================
#  Multimodal datasets
# =====================================================================


class TestLoadArtsMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_arts_multimodal
        return load_arts_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadGroceryMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_grocery_multimodal
        return load_grocery_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadGrocerySMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_grocerys_multimodal
        return load_grocerys_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadMoviesMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_movies_multimodal
        return load_movies_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadPhotoMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_photo_multimodal
        return load_photo_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadRedditMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_reddit_multimodal
        return load_reddit_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadRedditSMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_reddits_multimodal
        return load_reddits_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


class TestLoadToysMultimodal:
    @pytest.fixture(scope="class")
    def ds(self):
        from pymv.data import load_toys_multimodal
        return load_toys_multimodal()

    def test_structure(self, ds):
        _check_multimodal(ds)


# =====================================================================
#  Registry integration — all new datasets are discoverable
# =====================================================================


class TestNewDatasetsInRegistry:
    """Verify every new dataset name appears in ``list_datasets()``."""

    STANDARD_NAMES = [
        "100leaves", "3sources", "acm_5view", "aloi",
        "bbc4view", "bbcsport", "caltech101_7", "citeseer",
        "cora", "handwritten", "mnist_10k", "mnist_4",
        "msrc_v5", "outdoor_scene", "protein_fold",
        "reuters_1200", "reuters_1500", "uci", "webkb",
        "wikipedia", "yale",
    ]

    MULTIMODAL_NAMES = [
        "arts_multimodal", "grocery_multimodal",
        "grocerys_multimodal", "movies_multimodal",
        "photo_multimodal", "reddit_multimodal",
        "reddits_multimodal", "toys_multimodal",
    ]

    def test_standard_names_in_registry(self):
        from pymv.data import list_datasets

        available = list_datasets()
        for name in self.STANDARD_NAMES:
            assert name in available, f"'{name}' missing from registry"

    def test_multimodal_names_in_registry(self):
        from pymv.data import list_datasets

        available = list_datasets()
        for name in self.MULTIMODAL_NAMES:
            assert name in available, f"'{name}' missing from registry"
