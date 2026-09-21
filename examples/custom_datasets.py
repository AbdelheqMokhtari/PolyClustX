"""Offline dataset/variant example: python examples/custom_datasets.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import savemat

from pymv.data import dataset_from_arrays, read_arrays


def create_demo_files(directory):
    """Create two tiny representations of the same aligned samples."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 8)
    first = rng.normal(size=(24, 6)) + labels[:, None] * 2
    second = rng.normal(size=(24, 4)) + labels[:, None] * 2
    variants = {}
    for variant, views in {"default": [first, second], "compact": [first[:, :3], second[:, :2]]}.items():
        path = directory / f"toy_{variant}.mat"
        savemat(path, {"image": views[0], "text": views[1], "labels": labels,
                       "ids": np.arange(24)})
        variants[variant] = path
    return variants


def load_demo_variant(paths, variant="default"):
    """Local/private datasets need no global registration."""
    if variant not in paths:
        raise ValueError(f"Choose a variant from {sorted(paths)}")
    data = dataset_from_arrays(read_arrays(paths[variant]), view_keys=["image", "text"],
                               label_key="labels", modalities=["image", "text"])
    data.name = "toy"
    data.metadata["variant"] = variant
    return data


if __name__ == "__main__":
    with TemporaryDirectory(prefix="pymv-dataset-example-") as directory:
        paths = create_demo_files(directory)
        for variant in paths:
            data = load_demo_variant(paths, variant)
            print(f"{data.name}/{variant}: {data.n_samples} samples, dimensions {data.feature_dims}")
