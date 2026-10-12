import json
import zipfile

import numpy as np
import pytest

from polyclustx.core import BaseModel
from polyclustx.data import MultiViewDataset
from polyclustx.experiment import Experiment, GridSearchExperiment, RandomSearchExperiment
from polyclustx.experiment.exports import normalized_loss
from polyclustx.metrics import davies_bouldin_score, nmi_score


class ReportModel(BaseModel):
    def __init__(self, n_clusters=2, random_state=0, fail=False):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.fail = fail

    def fit(self, Xs, As=None):
        if self.fail:
            raise ValueError("intentional failure")
        self.labels_ = np.arange(len(Xs[0])) % self.n_clusters
        self.embedding_ = Xs[0].copy()
        self.loss_history_ = [10.0, 5.0, 2.0]
        self.alphas_ = np.array([1.0])
        self.inertia_ = 12.0
        return self


def dataset():
    return MultiViewDataset([np.array([[0], [10], [1], [11], [2], [12]])],
                            labels=[0, 1, 0, 1, 0, 1], name="tiny")


def test_experiment_json_contains_every_configuration_and_run(tmp_path):
    result = Experiment({"first": dataset(), "second": dataset()},
                        [ReportModel(2), ReportModel(3)], {"NMI": nmi_score},
                        runs=2, verbose=0, save_path=tmp_path).run()
    report = json.loads(result.json_path.read_text())
    assert len(report["experiments"]) == 4
    for entry in report["experiments"]:
        assert len(entry["runs"]) == 2
        assert entry["aggregate"]["successful_runs"] == 2
        run = entry["runs"][0]
        assert run["view_weights"] == [1.0] and run["inertia"] == 12.0
        assert run["normalized_loss_history"] == [1.0, .375, 0.0]
        assert run["per_view_results"] is None
        assert entry["runs"][1]["parameters"]["random_state"] == 1
    assert report["started_at"] <= report["completed_at"]
    result.export(tmp_path / "output.zip")
    with zipfile.ZipFile(tmp_path / "output.zip") as archive:
        assert "param.n_clusters" in archive.read("metrics.csv").decode()
        assert json.loads(archive.read("results.json"))["id"] == report["id"]


@pytest.mark.parametrize("search", [RandomSearchExperiment, GridSearchExperiment])
def test_search_reporting_and_objective_direction(tmp_path, search):
    result = search(ReportModel, {"n_clusters": [2, 3]}, dataset(),
                    {"DB": davies_bouldin_score}, "DB", runs=2, n_iter=4,
                    seed=0, verbose=0, benchmark=False, save_path=tmp_path).run()
    report = result.to_dict()
    assert report["search"]["direction"] == "minimize"
    assert result.best_score == min(t["metrics"]["DB"] for t in result.trials)
    assert len(report["search"]["steps"]) == (2 if search is GridSearchExperiment else 4)
    assert all(len(entry["runs"]) == 2 for entry in report["experiments"])
    assert report["search"]["best_configuration"] is not None


def test_failed_search_has_no_winner(tmp_path):
    result = GridSearchExperiment(ReportModel, {"fail": [True]}, dataset(), {"NMI": nmi_score},
                                  "NMI", verbose=0, save_path=tmp_path).run()
    assert result.best_trial is None
    assert result.to_dict()["search"]["best_score"] is None
    assert result.to_dict()["experiments"][0]["runs"][0]["status"] == "failed"


def test_failed_experiment_preserves_other_runs(tmp_path):
    result = Experiment(dataset(), [ReportModel(fail=True), ReportModel()], {"NMI": nmi_score},
                        runs=2, verbose=0, save_path=tmp_path).run()
    first, second = result.to_dict()["experiments"]
    assert first["aggregate"]["failed_runs"] == 2
    assert second["aggregate"]["successful_runs"] == 2
    assert "intentional failure" in first["runs"][0]["error"]


def test_normalized_loss_handles_constant_and_nonfinite():
    assert normalized_loss([5, 5]) == [0, 0]
    assert normalized_loss([np.nan, np.inf]) == [None, None]
    assert normalized_loss([]) == []


def test_markdown_uses_same_detailed_json_and_no_duplicate_report(tmp_path):
    result = Experiment(dataset(), ReportModel(), {"NMI": nmi_score}, verbose=0,
                        save_markdown=True, save_path=tmp_path, experiment_name="demo").run()
    assert result.json_path == tmp_path / "demo" / "results.json"
    assert json.loads(result.json_path.read_text())["schema_version"] == "1.0"
    assert not list((tmp_path / "demo").glob("experiment-*.json"))


def test_explicit_zero_seed_is_not_overridden(tmp_path):
    result = GridSearchExperiment(ReportModel, {"random_state": [0]}, dataset(), {"NMI": nmi_score},
                                  "NMI", seed=42, runs=2, verbose=0, save_path=tmp_path).run()
    runs = result.to_dict()["experiments"][0]["runs"]
    assert [run["parameters"]["random_state"] for run in runs] == [0, 1]
