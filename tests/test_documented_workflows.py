import runpy
from pathlib import Path

import numpy as np
import pytest

from pymv.data import MultiViewDataset
from pymv.models import LMGEC, SCM


def test_offline_documented_workflows():
    script = Path(__file__).resolve().parents[1] / "examples/offline_workflows.py"
    runpy.run_path(str(script), run_name="__main__")


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


def test_direct_propagation_fuses_relations():
    from pymv.preprocessing import get_propagated_features
    x = np.arange(12).reshape(6, 2)
    a, b = np.eye(6), np.ones((6, 6))
    expected = get_propagated_features(x, adj=.2 * a + .8 * b)
    actual = get_propagated_features(x, adj=[a, b], relation_weights=[.2, .8])
    np.testing.assert_allclose(actual, expected)
