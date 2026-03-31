"""
Shared fixtures for the pymv test suite.

All fixtures are lightweight and avoid heavy computation.
The ``mock_dataset`` fixture provides a tiny 2-view dataset that
every test can use without downloading real data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment: suppress TF/CUDA noise during tests
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_arrays():
    """Return raw numpy arrays for a tiny 2-view dataset.

    Returns
    -------
    tuple[list, list, np.ndarray]
        (Xs, As, labels) with 30 samples, 2 views, 3 clusters.
    """
    rng = np.random.RandomState(42)
    n, k = 30, 3
    labels = np.repeat(np.arange(k), n // k)  # [0,0,...,1,1,...,2,2,...]

    Xs = [rng.randn(n, 8).astype(np.float64),
          rng.randn(n, 6).astype(np.float64)]

    # Simple symmetric adjacency (identity + noise, kept sparse-like)
    def _adj(n):
        A = np.eye(n) + rng.rand(n, n) * 0.1
        return (A + A.T) / 2

    As = [_adj(n), _adj(n)]
    return Xs, As, labels


@pytest.fixture()
def mock_dataset(mock_arrays):
    """Return a ``MultiViewDataset`` wrapping the mock arrays."""
    from pymv.data.dataset import MultiViewDataset

    Xs, As, labels = mock_arrays
    return MultiViewDataset(Xs, As, labels, name="mock")


@pytest.fixture()
def mock_model(mock_dataset):
    """Return a fitted LMGEC model (very fast — tiny data, few iters)."""
    from pymv.models.lmgec import LMGEC

    model = LMGEC(
        n_clusters=mock_dataset.n_clusters,
        embedding_dim=4,
        max_iter=3,
        preprocess=False,  # skip graph propagation for speed
    )
    model.fit(mock_dataset.Xs)
    return model


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory (pytest built-in ``tmp_path``)."""
    return str(tmp_path)


@pytest.fixture()
def sample_config(tmp_path):
    """Write a minimal YAML config file and return its path."""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(
        "dataset: acm\n"
        "\n"
        "model:\n"
        "  name: lmgec\n"
        "  params:\n"
        "    embedding_dim: 8\n"
        "    temperature: 5.0\n"
        "    max_iter: 3\n"
        "\n"
        "experiment:\n"
        "  runs: 1\n"
        "  metrics: [acc, nmi]\n"
        "  verbose: 0\n"
        "\n"
        "output:\n"
        "  save: false\n",
        encoding="utf-8",
    )
    return str(cfg_path)


@pytest.fixture()
def sample_config_with_output_path(tmp_path):
    """YAML config that sets ``output.path``."""
    cfg_path = tmp_path / "cfg_with_path.yaml"
    out_dir = tmp_path / "custom_output"
    cfg_path.write_text(
        f"dataset: acm\n"
        f"\n"
        f"model:\n"
        f"  name: lmgec\n"
        f"  params:\n"
        f"    embedding_dim: 8\n"
        f"\n"
        f"output:\n"
        f"  save: true\n"
        f"  path: {out_dir}\n",
        encoding="utf-8",
    )
    return str(cfg_path), str(out_dir)


@pytest.fixture()
def metrics_fns():
    """Return a small dict of metric functions."""
    from pymv.metrics import clustering_accuracy, nmi_score

    return OrderedDict([
        ("clustering_accuracy", clustering_accuracy),
        ("nmi_score", nmi_score),
    ])
