import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from polyclustx.data import MultiViewDataset, get_dataset_info, list_datasets, list_variants, load_data, registry


def test_variants_are_grouped():
    assert "pascal" in list_datasets() and "pascal_clip" not in list_datasets()
    assert "clip" in list_variants("pascal")
    assert list_variants("acm") == ["default"]
    assert get_dataset_info("acm")["category"] == "attributed_graph_multiview"
    assert get_dataset_info("pascal")["category"] == "multimodal"
    with pytest.warns(DeprecationWarning):
        assert get_dataset_info("arts_multimodal")["category"] == "attributed_graph_multimodal"
    assert "arts_multimodal" not in list_datasets(category="multimodal")
    assert "pascal" in list_datasets(category="multimodal")
    assert "acm" not in list_datasets(category="multiview")
    assert "grocery_s" in list_datasets()
    assert "clip_minilm" in list_variants("grocery-S")
    assert get_dataset_info("Wiki")["n_views"] == 1
    assert get_dataset_info("Amazon_photos")["feature_dims"] == [745]
    assert get_dataset_info("wikipedia-test")["category"] == "multimodal"
    with pytest.warns(DeprecationWarning):
        assert get_dataset_info("grocery_multimodal", variant="small")["name"] == "grocery_s"


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
    with pytest.raises(ValueError, match="missing_rate"):
        load_data("acm", float("nan"))  # rejected before any download


@pytest.mark.parametrize("artifact", sorted(registry.CATALOG))
def test_every_catalog_artifact_uses_unified_mat_layout(artifact, monkeypatch, tmp_path):
    spec = registry.CATALOG[artifact]
    monkeypatch.setattr(registry, "get_data_home", lambda: tmp_path)
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"X{index + 1}": np.ones((12, 2 + index))
              for index in range(len(spec.feature_descriptions))}
    arrays.update({f"A{index + 1}": sparse.eye(12, format="csr")
                   for index in range(len(spec.graph_descriptions))})
    arrays["labels"] = np.arange(12) % 3
    savemat(spec.path, arrays)
    calls = []
    monkeypatch.setattr(registry, "download_file", lambda url, path: calls.append((url, path)))
    data = registry._load(spec)
    assert data.n_views == spec.n_views
    assert data.category == spec.category
    assert data.n_clusters == 3 and data.labels.shape == (12,)
    assert all(sparse.issparse(graph) for entry in data.As if entry is not None
               for graph in (entry if isinstance(entry, list) else [entry]))
    assert calls == [(spec.url, spec.path)]
    assert spec.url.startswith("https://huggingface.co/datasets/Abdelheq/PolyClustX/resolve/main/")
    assert spec.url.endswith(f"/{spec.filename}?download=true")


def test_catalog_selection_preserves_unused_array_descriptions(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "get_data_home", lambda: tmp_path)
    spec = registry.CATALOG["acm"]
    spec.path.parent.mkdir(parents=True)
    savemat(spec.path, {"X1_features": sparse.csr_matrix(np.ones((6, 2))),
                       "A1_PAP": sparse.eye(6), "A2_PLP": sparse.eye(6) * 2, "labels": np.arange(6) % 2})
    monkeypatch.setattr(registry, "download_file", lambda *args: None)
    selection = {"keywords only": "X1_features", "both relations": ("X1_features", ["A2_PLP", "A1_PAP"])}
    data = load_data("ACM", views=selection)
    assert data.As[0] is None and len(data.As[1]) == 2
    assert data.Xs[0] is data.Xs[1]
    assert data.metadata["available_graphs"] == ["A1_PAP", "A2_PLP"]
    assert "sharing authors" in data.metadata["graph_descriptions"]["A1_PAP"]
    assert data.metadata["view_selection"]["both relations"][1] == ["A2_PLP", "A1_PAP"]
    with pytest.raises(ValueError, match="Unknown feature"):
        load_data("acm", views={"bad": "X_missing"})
