import numpy as np
from scipy.io import savemat

from polyclustx.data import get_dataset_info, list_variants, load_data, registry


def test_registered_variants_without_network(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 8)
    paths = {}
    directory = tmp_path / "huggingface" / "contribution_toy"
    directory.mkdir(parents=True)
    for variant, widths in {"default": (6, 4), "compact": (3, 2)}.items():
        path = directory / f"toy_{variant}.mat"
        savemat(path, {"X1_image": rng.normal(size=(24, widths[0])),
                       "X2_text": rng.normal(size=(24, widths[1])), "labels": labels})
        paths[variant] = path
    monkeypatch.setattr(registry, "get_data_home", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(registry, "download_file", lambda url, path: calls.append((url, path)))
    for variant, path in paths.items():
        artifact = f"contribution_toy_{variant}"
        spec = registry.DatasetSpec(
            artifact, f"https://example.org/{path.name}", path.name,
            category="multimodal", cache_dir="contribution_toy",
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
