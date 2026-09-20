import json

import numpy as np
import pytest

from pymv.display.results import aggregate_results, compact_confusion, detailed_rows
from pymv.display.workspace import ExperimentSpec, Workspace
from tests.test_experiment_reports import ReportModel, dataset


@pytest.fixture
def workspace(monkeypatch):
    monkeypatch.setattr("pymv.display.workspace.get_model", lambda _: ReportModel)
    ws = Workspace()
    ws.add_dataset(dataset())
    ws.add_dataset(dataset())
    return ws


def test_batch_one_file_all_combinations(workspace, tmp_path):
    specs = [ExperimentSpec(key, params={"fail": False}, clusters=[2, 3], repeats=2,
                            benchmark=True, metrics=["NMI"]) for key in workspace.datasets]
    values = []
    report = workspace.run_batch(specs, name="Comparison", output_dir=tmp_path, progress=values.append)
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(report["experiments"]) == 4
    assert values[-1] == 1 and values == sorted(values)
    assert workspace.experiment_counter == 1
    assert all(entry["aggregate"]["successful_runs"] == 2 for entry in report["experiments"])
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["name"] == "Comparison"
    assert "parameters" not in aggregate_results(workspace.records[0])[0]
    assert "fail" in aggregate_results(workspace.records[0])[0]
    assert "param.n_clusters" in detailed_rows(workspace.records[0])[0]


def test_mixed_modes_and_bad_combination_fail_before_any_fit(workspace, tmp_path):
    keys = list(workspace.datasets)
    with pytest.raises(ValueError, match="cannot be combined"):
        workspace.run_batch([ExperimentSpec(keys[0], benchmark=True), ExperimentSpec(keys[1])], output_dir=tmp_path)
    assert not workspace.records
    with pytest.raises(ValueError, match="number of samples"):
        workspace.run_batch([ExperimentSpec(keys[0]), ExperimentSpec(keys[1], clusters=[999])], output_dir=tmp_path)
    assert not workspace.records


def test_grid_objective_and_tuning_steps(workspace, tmp_path):
    key = next(iter(workspace.datasets))
    record = workspace.run(ExperimentSpec(
        key, clusters=[2, 3], search_space={"fail": [False, True]}, search_strategy="grid",
        kind="search", objective="Davies–Bouldin", metrics=["Davies–Bouldin"], repeats=2,
    ), output_dir=tmp_path)
    search = record.report["search"]
    assert len(search["steps"]) == 4 and len(record.rows) == 8
    assert search["best_score"] == min(s["score"] for s in search["steps"] if s["score"] is not None)
    assert search["best_parameters"]["fail"] is False
    assert workspace.experiment_counter == 0
    assert all(row["diagnostics"]["normalized_loss_history"] == [1, .375, 0]
               for row in record.rows if row["status"] == "complete")


def test_repeat_limit_and_default_benchmark(workspace):
    key = next(iter(workspace.datasets))
    assert workspace.benchmark_mode(key)
    workspace.benchmark[key] = False
    assert not workspace.benchmark_mode(key)
    with pytest.raises(ValueError, match="between 1 and 20"):
        workspace.run(ExperimentSpec(key, repeats=21))


def test_confusion_aligns_permuted_clusters_and_preserves_counts():
    truth = np.repeat(["a", "b", "c"], [3, 4, 5])
    prediction = np.repeat([9, 1, 4], [3, 4, 5])
    matrix, _, _ = compact_confusion(truth, prediction)
    assert np.trace(matrix) == 12
    matrix, _, _ = compact_confusion(truth, prediction, max_labels=2)
    assert matrix.sum() == 12
    with pytest.raises(ValueError, match="aligned"):
        compact_confusion(truth, prediction[:2])
