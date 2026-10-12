import numpy as np
from examples.custom_datasets import create_demo_files, load_demo_variant

from polyclustx.data import get_dataset_info, list_variants, load_data, readers, registry


def test_local_dataset_variants(tmp_path):
    paths = create_demo_files(tmp_path)
    original = load_demo_variant(paths)
    compact = load_demo_variant(paths, "compact")
    assert original.feature_dims == [6, 4] and compact.feature_dims == [3, 2]
    assert original.n_samples == compact.n_samples == 24
    np.testing.assert_array_equal(original.labels, compact.labels)


def test_registered_variants_without_network(monkeypatch, tmp_path):
    paths = create_demo_files(tmp_path)
    monkeypatch.setattr(registry, "get_dataset_path", lambda _: tmp_path)
    calls = []
    monkeypatch.setattr(registry, "download_file", lambda url, path: calls.append((url, path)))
    for variant, path in paths.items():
        artifact = f"contribution_toy_{variant}"
        spec = registry.DatasetSpec(
            artifact, "test-record", path.name, readers.read_feature_views,
            kwargs={"keys": ("image", "text"), "label_key": "labels"},
            modalities=("image", "text"), reference="https://example.org/toy",
            description="Offline contribution fixture",
        )
        monkeypatch.setitem(registry.CATALOG, artifact, spec)
    monkeypatch.setitem(registry.VARIANTS, "contribution_toy", {
        variant: f"contribution_toy_{variant}" for variant in paths
    })
    assert list_variants("contribution_toy") == ["compact", "default"]
    info = get_dataset_info("contribution_toy", variant="compact")
    assert not calls and info["category"] == "multimodal"
    assert info["reference"] == "https://example.org/toy"
    result = load_data("contribution_toy", variant="compact")
    assert result.feature_dims == [3, 2] and result.metadata["variant"] == "compact"
    assert len(calls) == 1
