import numpy as np
import pytest

from polyclustx.command import assignments, main
from polyclustx.data import MultiViewDataset
from polyclustx.experiment import OptunaSearchExperiment
from polyclustx.experiment.optuna_search import OptunaSampler, validate_space
from polyclustx.metrics import nmi_score
from polyclustx.models import MultiViewKMeans


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


def test_optuna_layer_width_candidates():
    sampler = OptunaSampler({"hidden_dims": [(16, 8), (8, 4)]}, 1, True)
    params = sampler.ask()
    assert params["hidden_dims"] in [(16, 8), (8, 4)]
    sampler.tell(.5)
    assert tuple(sampler.study.trials[0].user_attrs["resolved_parameters"]["hidden_dims"]) == params["hidden_dims"]


def test_simple_cli(capsys):
    assert assignments(["max_iter=10", "hidden_dims=[8,4]", "device=cpu"]) == {
        "max_iter": 10, "hidden_dims": [8, 4], "device": "cpu"}
    main(["models", "idec"])
    assert "pretrain_epochs" in capsys.readouterr().out
    with pytest.raises(ValueError):
        assignments(["oops"])
    with pytest.raises(ValueError):
        assignments(["x=1", "x=2"])


def test_cli_runs_and_searches_offline(monkeypatch, tmp_path, capsys):
    y = np.repeat([0, 1], 8)
    data = MultiViewDataset([np.column_stack([y, np.arange(16)])], labels=y)
    monkeypatch.setattr("polyclustx.data.load_data", lambda name: data)
    for command in ("run", "search"):
        args = [command, "offline", "--model", "multiview_kmeans", "--param", "n_init=1",
                "--output", str(tmp_path)]
        if command == "search":
            args += ["--strategy", "optuna", "--vary", "tol={\"low\":0.0001,\"high\":0.01}", "--trials", "2"]
        main(args)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert "Report:" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["run", "offline", "--clusters", "0"])


def test_workspace_optuna_and_ui_range(tmp_path):
    from polyclustx.display.workspace import ExperimentSpec, Workspace
    from tests.test_display_comparisons import app_test

    ws = Workspace()
    y = np.repeat([0, 1], 8)
    rng = np.random.default_rng(1)
    dataset_id = ws.add_dataset(MultiViewDataset([rng.normal(size=(16, 4)) + y[:, None]], labels=y))
    record = ws.run(ExperimentSpec(dataset_id, model="multiview_kmeans", kind="search",
                                    params={"n_init": 1}, search_strategy="optuna",
                                    search_space={"tol": {"low": .0001, "high": .01}},
                                    trials=2, repeats=2, clusters=[2, 3], benchmark=True,
                                    metrics=["NMI"], objective="NMI"), output_dir=tmp_path)
    assert len(record.rows) == 8
    assert all(row["status"] == "complete" for row in record.rows)
    assert record.report["search"]["best_configuration"] is not None
    at = app_test(ws, 7)
    next(s for s in at.selectbox if s.label == "Search strategy").select("Optuna (TPE)").run()
    at.checkbox(key="search_lmgec_temperature").check().run()
    at.toggle(key="range_lmgec_temperature").set_value(True).run()
    assert not at.exception and not at.error
    assert at.number_input(key="low_lmgec_temperature").value > 0
    at.number_input(key="parameter_lmgec_embedding_dim").set_value(2)
    at.number_input(key="parameter_lmgec_max_iter").set_value(1)
    next(b for b in at.button if b.label == "Run search").click().run()
    assert not at.exception and not at.error
    assert ws.records[0].report["search"]["strategy"] == "optuna"
    assert ws.records[0].rows[0]["status"] == "complete"
