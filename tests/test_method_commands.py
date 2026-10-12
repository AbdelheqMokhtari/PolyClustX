"""Direct estimator commands and multi-dataset experiment plans, offline."""

import json
from importlib.metadata import distribution

import numpy as np
import pytest

from polyclustx.command import main, model_main
from polyclustx.command_experiments import prepare_plan, read_plan, run_plan
from polyclustx.data import MultiViewDataset
from polyclustx.models import list_models


@pytest.fixture
def offline_loader(monkeypatch):
    def load(name, **kwargs):
        k = 2 if name == "two" else 3
        y = np.repeat(np.arange(k), 6)
        x = np.random.default_rng(4).normal(size=(len(y), 4)) + y[:, None] * 4
        return MultiViewDataset([x], labels=y, name=name)
    monkeypatch.setattr("polyclustx.data.load_data", load)
    return load


@pytest.mark.parametrize("name", list_models())
def test_every_method_has_command_and_parameter_help(name, capsys):
    scripts = {entry.name: entry for entry in distribution("polyclustx").entry_points}
    assert scripts[name.replace("_", "-")].value == f"polyclustx.models.{name}:main"
    with pytest.raises(SystemExit) as help_exit:
        model_main(name, ["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--max-iter" in help_text and "--search" in help_text
    assert "polyclustx-cli" not in scripts


def test_method_run_search_and_namespaced_alias(offline_loader, tmp_path):
    model_main("multiview_kmeans", ["two", "--n-init", "1", "--max-iter", "2", "--runs", "2",
                                     "--output", str(tmp_path)])
    model_main("multiview_kmeans", ["--dataset", "two", "--n-init", "1", "--search", "grid",
                                     "--vary", "tol=[0.001,0.01]", "--output", str(tmp_path)])
    main(["multiview-kmeans", "two", "--n-init", "1", "--output", str(tmp_path)])
    reports = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(reports) == 3
    assert all(run["status"] == "complete" for report in reports
               for experiment in report["experiments"] for run in experiment["runs"])
    assert any(report.get("search", {}).get("strategy") == "grid" for report in reports)


@pytest.mark.parametrize("args", [
    ["two", "--dataset", "two"], ["two", "--search", "grid"], ["two", "--vary", "tol=[1]"],
    ["two", "--n-init", "1.5"], ["two", "--objective", "nmi"],
    ["two", "--n-init", "1", "--param", "n_init=2"], ["two", "--made-up", "2"],
])
def test_bad_direct_options_fail_before_loading(args, monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("Invalid CLI arguments should not load a dataset")
    monkeypatch.setattr("polyclustx.data.load_data", never)
    with pytest.raises(SystemExit) as stopped:
        model_main("multiview_kmeans", args)
    assert stopped.value.code == 2


def test_typed_parameters_and_failed_fit_exit(offline_loader, tmp_path):
    with pytest.raises(SystemExit):
        model_main("idec", ["two", "--preprocess", "maybe"])
    with pytest.raises(SystemExit):
        model_main("idec", ["two", "--hidden-dims", "8,2.5"])
    with pytest.raises(SystemExit) as stopped:
        model_main("multiview_kmeans", ["two", "--n-init", "0", "--output", str(tmp_path)])
    assert stopped.value.code == 1
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["experiments"][0]["runs"][0]["status"] == "failed"


def test_batch_independent_k_and_one_report(offline_loader, tmp_path):
    plan = {
        "datasets": ["two", "three"], "runs": 2, "output": str(tmp_path),
        "models": [
            {"name": "multiview_kmeans", "label": "baseline", "params": {"n_init": 1}},
            {"name": "multiview_kmeans", "label": "more-restarts", "params": {"n_init": 2}},
        ],
    }
    result = run_plan(plan)
    assert result.report["status"] == "complete"
    assert len(result.report["experiments"]) == 4
    assert len(list(tmp_path.glob("*.json"))) == 1
    for experiment in result.report["experiments"]:
        expected = 2 if experiment["dataset"]["name"] == "two" else 3
        assert experiment["parameters"]["n_clusters"] == expected
        assert len(experiment["runs"]) == 2
        assert all(run["status"] == "complete" for run in experiment["runs"])
    assert result.export(tmp_path / "batch.zip").is_file()


def test_batch_plan_validation_and_overrides(tmp_path):
    base = {"datasets": ["acm", "pascal:clip"], "models": ["lmgec", "scm"]}
    plan = prepare_plan(base, ["lmgec.embedding_dim=8"])
    assert plan["models"][0]["params"]["embedding_dim"] == 8
    assert plan["datasets"][1]["variant"] == "clip"
    for changes in ({"runs": 0}, {"models": ["lmgec", "lmgec"]}, {"unknown": 1},
                    {"datasets": []}, {"benchmark": "false"}, {"seed": -1}):
        with pytest.raises(ValueError):
            prepare_plan({**base, **changes})
    for override in ("embedding_dim=8", "absent.embedding_dim=8", "lmgec.typo=1"):
        with pytest.raises(ValueError):
            prepare_plan(base, [override])
    malicious = tmp_path / "unsafe.yaml"
    malicious.write_text("!!python/object/apply:os.system ['echo unexpected']")
    with pytest.raises(ValueError):
        read_plan(malicious)


def test_batch_cli_overrides_config_and_keeps_failures(offline_loader, tmp_path):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps({"datasets": ["two"], "models": ["multiview_kmeans"], "runs": 9}))
    output = tmp_path / "results"
    main(["experiment", "--config", str(config), "--runs", "1", "--output", str(output),
          "--param", "multiview_kmeans.n_init=1"])
    report = json.loads(next(output.glob("*.json")).read_text())
    assert len(report["experiments"][0]["runs"]) == 1
    with pytest.raises(SystemExit) as stopped:
        main(["experiment", "--datasets", "two", "--models", "multiview_kmeans",
              "--param", "multiview_kmeans.n_init=0", "--output", str(output)])
    assert stopped.value.code == 1
