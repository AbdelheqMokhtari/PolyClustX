import numpy as np
from scipy.io import savemat

from polyclustx.data import get_dataset_info, list_variants, load_data, readers, registry


def test_registered_variants_without_network(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 8)
    paths = {}
    for variant, widths in {"default": (6, 4), "compact": (3, 2)}.items():
        path = tmp_path / f"toy_{variant}.mat"
        savemat(path, {"image": rng.normal(size=(24, widths[0])),
                       "text": rng.normal(size=(24, widths[1])), "labels": labels})
        paths[variant] = path
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
    np.testing.assert_array_equal(result.labels, labels)
    assert len(calls) == 1
