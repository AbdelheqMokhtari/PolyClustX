import numpy as np
import pytest

from pymv.data import MultiViewDataset, get_dataset_info, list_datasets, list_variants, load_data, registry


def test_variants_are_grouped():
    assert "pascal" in list_datasets() and "pascal_clip" not in list_datasets()
    assert "clip" in list_variants("pascal")
    assert get_dataset_info("acm", variant="5view")["n_views"] == 5
    assert get_dataset_info("acm")["category"] == "attributed_graph_multiview"
    assert get_dataset_info("pascal")["category"] == "multimodal"
    assert get_dataset_info("arts_multimodal")["category"] == "attributed_graph_multimodal"


def test_loading_variant_and_missing_rate(monkeypatch):
    def fake_load(spec):
        assert spec.name == "pascal_clip"
        return MultiViewDataset([np.ones((10, 2)), np.ones((10, 3))])

    monkeypatch.setattr(registry, "_load", fake_load)
    data = load_data("pascal", .2, variant="clip", random_state=42)
    assert data.name == "pascal" and data.metadata["variant"] == "clip"
    assert (~data.observed_mask).sum() == 4
    with pytest.warns(DeprecationWarning):
        assert load_data("pascal_clip").metadata["variant"] == "clip"
    with pytest.raises(ValueError, match="Unknown pascal variant"):
        load_data("pascal", variant="unknown")
