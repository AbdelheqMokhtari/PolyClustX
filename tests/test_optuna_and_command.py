import numpy as np
import pytest

from pymv.command import assignments, main
from pymv.data import MultiViewDataset
from pymv.experiment import OptunaSearchExperiment
from pymv.experiment.optuna_search import OptunaSampler, validate_space
from pymv.metrics import nmi_score
from pymv.models import MultiViewKMeans


def test_optuna_report(tmp_path):
    y = np.repeat([0, 1], 8)
    data = MultiViewDataset([y[:, None] * 3. + np.random.default_rng(1).normal(size=(16, 2))], labels=y)
    search = OptunaSearchExperiment(MultiViewKMeans,
                                    {"n_clusters": [2], "n_init": [1, 2],
                                     "tol": {"low": 1e-5, "high": .01, "log": True}},
                                    data, [nmi_score], "nmi_score", n_iter=3, runs=2,
                                    seed=1, save_path=tmp_path, verbose=0)
    result = search.run()
    assert len(search.study_.trials) == 3
    assert len(result.report["search"]["steps"]) == 3
    assert all(len(e["runs"]) == 2 for e in result.report["experiments"])
    assert result.report["search"]["strategy"] == "optuna"
    assert result.json_path


@pytest.mark.parametrize("values", [[], {"low": -1., "high": 1., "log": True},
                                   {"low": 2., "high": 1.}, {"low": 0., "high": float("inf")},
                                   {"type": "int", "low": 1.5, "high": 3}])
def test_bad_ranges(values):
    with pytest.raises(ValueError):
        validate_space({"x": values})


def test_failed_trial_and_integer_range():
    sampler = OptunaSampler({"x": {"type": "int", "low": 1, "high": 3}}, 0, False)
    assert 1 <= sampler.ask()["x"] <= 3
    sampler.tell(float("nan"))
    assert sampler.study.trials[0].state.name == "FAIL"


def test_simple_cli(capsys):
    assert assignments(["max_iter=10", "hidden_dims=[8,4]", "device=cpu"]) == {
        "max_iter": 10, "hidden_dims": [8, 4], "device": "cpu"}
    main(["models", "idec"])
    assert "pretrain_epochs" in capsys.readouterr().out
    with pytest.raises(ValueError):
        assignments(["oops"])
    with pytest.raises(ValueError):
        assignments(["x=1", "x=2"])
