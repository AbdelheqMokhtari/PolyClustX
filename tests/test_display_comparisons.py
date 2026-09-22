from pathlib import Path

import numpy as np
import pytest

from pymv.display.artifacts import StoredArtifact
from pymv.display.workspace import ExperimentSpec, Workspace
from tests.test_experiment_reports import ReportModel, dataset

st = pytest.importorskip("streamlit")

def app_test(workspace, page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/pymv/display/app.py", default_timeout=60)
    at.session_state["workspace"] = workspace
    at.session_state["workflow_step"] = page
    return at.run()

@pytest.mark.skipif(st.__version__ == "1.51.0", reason="AppTest unmount bug in older Streamlit")
def test_grid_search_copy_to_experiment_and_visualize():
    ws = Workspace()
    data = dataset()
    data.Xs[0] = np.column_stack([data.Xs[0], np.arange(6), [1, 2, 4, 0, 3, 2]])
    ws.add_dataset(data)
    at = app_test(ws, 7)
    at.number_input(key="parameter_lmgec_embedding_dim").set_value(2)
    at.number_input(key="parameter_lmgec_max_iter").set_value(1)
    next(s for s in at.selectbox if s.label == "Search strategy").select("Grid search").run()
    at.checkbox(key="search_lmgec_temperature").check().run()
    at.multiselect(key="candidates_lmgec_temperature").set_value([".5", "2"]).run()
    next(s for s in at.selectbox if s.label == "Optimize metric").select("NMI").run()
    next(b for b in at.button if b.label == "Run search").click().run()
    assert not at.exception and len(ws.records) == 1
    assert ws.records[0].report["search"]["strategy"] == "grid"
    assert ws.records[0].report["search"]["best_configuration"] is not None
    best_name = next(t for t in at.text_input if t.label == "Best model name")
    assert best_name.value == "LMGEC_grid_best_#1"
    best_name.set_value("Selected graph baseline")
    at.session_state["search_lmgec_embedding_dim"] = False
    at.session_state["search_lmgec_max_iter"] = False
    next(b for b in at.button if b.label == "Copy best to experiment").click().run()
    assert not at.exception and len(ws.models) == 1
    assert next(iter(ws.models.values()))["name"] == "Selected graph baseline"
    slider = next(s for s in at.slider if s.label == "Independent runs per configuration")
    assert slider.min == 1 and slider.max == 20
    slider.set_value(2).run()
    next(b for b in at.button if b.label == "Run experiment").click().run()
    assert not at.exception and len(ws.records) == 2
    assert len(ws.records[0].rows) == 2
    assert ws.records[0].report["experiments"][0]["model"]["display_name"] == "Selected graph baseline"
    at.button(key="nav_Visualizations").click().run()
    assert not at.exception
    comparisons = next(m for m in at.multiselect if m.label == "Models and runs from any experiment")
    assert len(comparisons.value) == 3  # two search configurations + experiment run 1
    next(b for b in at.button if b.label == "Compute projection").click().run()
    assert not at.exception and not at.error
    at.button(key="nav_Export").click().run()
    assert not at.exception


def test_batch_ui_rejects_mixed_modes():
    ws = Workspace()
    first, second = ws.add_dataset(dataset()), ws.add_dataset(dataset())
    ws.benchmark[first], ws.benchmark[second] = True, False
    ws.add_model("test", "lmgec", {"embedding_dim": 1, "max_iter": 1}, [2])
    at = app_test(ws, 3)
    next(m for m in at.multiselect if m.label == "Datasets").set_value([first, second]).run()
    assert not at.exception and at.error
    assert next(b for b in at.button if b.label == "Run experiment").disabled
    assert not ws.records


def test_artifacts_are_lazy_and_removed_with_record(monkeypatch, tmp_path):
    monkeypatch.setattr("pymv.display.workspace.get_model", lambda _: ReportModel)
    ws = Workspace()
    key = ws.add_dataset(dataset())
    record = ws.run(ExperimentSpec(key, clusters=[2], repeats=2), output_dir=tmp_path)
    artifact = record.artifacts[0]
    assert isinstance(artifact, StoredArtifact)
    assert artifact.path.exists() and artifact["embedding"].shape == (6, 1)
    assert not any(hasattr(value, "shape") for value in vars(artifact).values())
    ws.delete_record(record.id)
    assert not artifact.path.exists()
    assert Path(record.json_path).exists()  # Session cleanup never deletes persistent reports.


def test_unload_stays_unloaded_until_explicitly_loaded():
    ws = Workspace()
    key = ws.add_lazy_dataset("tiny", dataset)
    ws.load_dataset(key)
    at = app_test(ws, 0)
    next(b for b in at.button if b.label == "Unload").click().run()
    assert not at.exception and not ws.datasets[key].loaded
    next(b for b in at.button if b.label == "Load active dataset").click().run()
    assert not at.exception and ws.datasets[key].loaded
