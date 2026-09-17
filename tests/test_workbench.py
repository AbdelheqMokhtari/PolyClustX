import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pymv.data import MultiViewDataset
from pymv.display.preprocessing import PreprocessingSpec, ViewPreprocessing, prepare_dataset
from pymv.display.profiles import dataset_profile
from pymv.display.synthetic import SyntheticSpec, generate_synthetic
from pymv.display.workspace import ExperimentSpec, Workspace, export_record
from tests.test_unsupervised_experiments import DummyModel


@pytest.fixture
def workspace(monkeypatch):
    import pymv.display.workspace as module

    monkeypatch.setattr(module, "get_model", lambda name: DummyModel)
    result = Workspace()
    result.add_dataset(MultiViewDataset([np.array([[0], [10], [1], [11], [2], [12]], dtype=float)]))
    return result


def test_independent_history_and_export(workspace):
    key = next(iter(workspace.datasets))
    spec = ExperimentSpec(key, clusters=[2], metrics=["Silhouette"])
    first = workspace.run(spec)
    spec.clusters = [3]
    second = workspace.run(spec)
    assert first.spec.clusters == [2] and workspace.records == [second, first]
    with zipfile.ZipFile(io.BytesIO(export_record(first))) as archive:
        payload = json.loads(archive.read("experiment.json"))
        assert payload["spec"]["benchmark"] is False
        assert payload["dataset"]["evaluation_space"] == "learned embedding (Euclidean)"
        assert "trial_1.npz" in archive.namelist()
        with np.load(io.BytesIO(archive.read("trial_1.npz")), allow_pickle=False) as output:
            assert output["labels"].shape == (6,)
    workspace.delete_record(first.id)
    assert workspace.records == [second]


def test_no_label_leakage(workspace):
    key = next(iter(workspace.datasets))
    workspace.datasets[key].labels = np.arange(6)
    with pytest.raises(ValueError, match="External metrics"):
        workspace.run(ExperimentSpec(key, metrics=["ACC"], benchmark=False))
    assert not workspace.records
    record = workspace.run(ExperimentSpec(key, clusters=[2], metrics=["ACC"], benchmark=True))
    assert record.rows[0]["K"] == 2  # Never replace manual K with six ground-truth classes.


def test_single_double_many_and_missing_imputation(monkeypatch):
    import pymv.display.workspace as module

    monkeypatch.setattr(module, "get_model", lambda name: DummyModel)
    for views in [1, 2, 5]:
        workspace = Workspace()
        key = workspace.add_dataset(MultiViewDataset([np.arange(12).reshape(6, 2)] * views))
        if views > 1:
            with pytest.raises(ValueError, match="missing views"):
                workspace.run(ExperimentSpec(key, missing_rate=0.2))
        spec = ExperimentSpec(key, clusters=[2], missing_rate=0.2 if views > 1 else 0, imputation="mean")
        assert workspace.run(spec).rows[0]["status"] == "complete"


def test_lazy_resources_unload_other_datasets():
    workspace = Workspace()
    calls = []

    def loader(name):
        calls.append(name)
        return MultiViewDataset([np.ones((4, 2))], name=name)

    first = workspace.add_lazy_dataset("first", lambda: loader("first"))
    second = workspace.add_lazy_dataset("second", lambda: loader("second"))
    assert not workspace.datasets[first].loaded
    workspace.load_dataset(first)
    workspace.load_dataset(second)
    assert calls == ["first", "second"]
    assert not workspace.datasets[first].loaded and workspace.datasets[second].loaded
    workspace.remove_dataset(first)
    assert first not in workspace.datasets


def test_selected_view_preprocessing_and_missing_imputation():
    data = MultiViewDataset(
        [np.array([[3.0, 4.0], [6.0, 8.0]]), np.array([[1.0], [2.0]])],
        name="example",
    )
    recipe = PreprocessingSpec(
        mode="selected",
        selected_views=[0],
        shared=ViewPreprocessing(normalize=True),
    )
    prepared = prepare_dataset(data, recipe)
    np.testing.assert_allclose(np.linalg.norm(prepared.Xs[0], axis=1), 1)
    np.testing.assert_array_equal(prepared.Xs[1], data.Xs[1])
    assert prepared is not data


def test_dataset_profile_and_configurable_synthetic_data():
    spec = SyntheticSpec(
        n_samples=80,
        n_classes=4,
        dimensions=[5, 7, 3],
        distributions=["Gaussian", "Heavy-tailed", "Nonlinear"],
        class_balance="Strongly imbalanced",
        graph_relations=2,
    )
    data = generate_synthetic(spec)
    profile = dataset_profile(data)
    assert data.n_views == 3 and data.feature_dims == [5, 7, 3]
    assert data.n_clusters == 4 and all(len(graphs) == 2 for graphs in data.As)
    assert profile["labels"]["imbalance_ratio"] > 1
    assert len(profile["views"]) == 3 and len(profile["graphs"]) == 6
    assert profile["memory_bytes"] > 0


def test_streamlit_create_run_duplicate_delete():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src/pymv/display/app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    at.radio[0].set_value("Synthetic example").run()
    next(b for b in at.button if b.label == "Create example").click().run()
    assert not at.exception
    at.text_input(key="clusters").set_value("2, 3")
    at.text_area(key="params").set_value('{"embedding_dim": 3, "max_iter": 1}')
    next(b for b in at.button if b.label == "Run experiment").click().run()
    assert not at.exception
    ws = at.session_state["workspace"]
    assert len(ws.records) == 1
    assert all(r["status"] == "complete" for r in ws.records[0].rows)
    next(b for b in at.button if b.label == "Edit / duplicate settings").click().run()
    assert not at.exception
    assert at.text_input(key="clusters").value == "2, 3"
    next(b for b in at.button if b.label == "Run experiment").click().run()
    assert len(at.session_state["workspace"].records) == 2
    next(b for b in at.button if b.label == "Delete result").click().run()
    assert len(at.session_state["workspace"].records) == 1


def test_projection_sampling():
    from pymv.display.projection import project_embedding

    x = np.random.default_rng(1).normal(size=(20, 4))
    projected, rows = project_embedding(x, max_samples=10)
    assert projected.shape == (10, 2) and len(set(rows)) == 10
    np.testing.assert_array_equal(rows, project_embedding(x, max_samples=10)[1])
    with pytest.raises(ValueError, match="Choose"):
        project_embedding(x, "invalid")


def test_streamlit_benchmark_toggle_removes_external_metrics():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "src/pymv/display/app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    at.radio[0].set_value("Synthetic example").run()
    next(c for c in at.checkbox if c.label == "Include evaluation labels").check()
    next(b for b in at.button if b.label == "Create example").click().run()
    at.checkbox(key="benchmark").check().run()
    at.multiselect(key="metrics").set_value(["ACC", "NMI", "Silhouette"]).run()
    at.checkbox(key="benchmark").uncheck().run()
    assert not at.exception
    assert at.multiselect(key="metrics").value == ["Silhouette"]
    assert "ACC" not in at.multiselect(key="metrics").options
