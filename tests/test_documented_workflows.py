import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from polyclustx.data import MultiViewDataset
from polyclustx.models import LMGEC, SCM


def test_documented_examples(tmp_path, monkeypatch):
    """Execute the actual copy/paste examples, without a second tutorial script."""
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    source = Path(__file__).resolve().parents[1] / "docs/user_guide/examples.rst"
    blocks = re.findall(r"(?ms)^\.\. code-block:: python\n\n(.*?)(?=^\S|\Z)", source.read_text())
    assert len(blocks) == 10
    namespace = {}
    try:
        for block in blocks:
            exec(compile(textwrap.dedent(block), str(source), "exec"), namespace)
        for path in (tmp_path / "experiments").rglob("*.json"):
            report = json.loads(path.read_text())
            assert all(run["status"] == "complete" for experiment in report["experiments"]
                       for run in experiment["runs"])
        assert (tmp_path / "experiments/predicted-clusters.png").is_file()
        assert (tmp_path / "experiments/comparison.zip").is_file()
    finally:
        plt.close("all")


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("cls", [LMGEC, SCM])
def test_real_models_one_two_many_views(count, cls):
    rng = np.random.default_rng(6)
    data = MultiViewDataset([rng.normal(size=(15, 4 + j)) for j in range(count)])
    kwargs = {"n_clusters": 3, "embedding_dim": 3, "max_iter": 1, "random_state": 0}
    if cls is SCM:
        kwargs.update(projection_dim=3, hidden_dims=(5,), batch_size=15)
    model = data.fit(cls(**kwargs))
    assert model.labels_.shape == (15,)
    assert model.embedding_.shape[0] == 15
    assert model.inertia_ >= 0
    if cls is LMGEC:
        assert len(model.per_view_results_) == count
        assert all(item["stage"] == "initialization" for item in model.per_view_results_)


def test_direct_propagation_fuses_relations():
    from polyclustx.preprocessing import get_propagated_features
    x = np.arange(12).reshape(6, 2)
    a, b = np.eye(6), np.ones((6, 6))
    expected = get_propagated_features(x, adj=.2 * a + .8 * b)
    actual = get_propagated_features(x, adj=[a, b], relation_weights=[.2, .8])
    np.testing.assert_allclose(actual, expected)
