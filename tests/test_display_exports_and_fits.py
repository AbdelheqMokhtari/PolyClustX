import io
import json
import zipfile

import numpy as np
import pytest
from scipy import sparse
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

from pymv.data import MultiViewDataset
from pymv.display.exporting import dataset_mat
from pymv.display.results import aggregate_results, compact_confusion
from pymv.display.workspace import ExperimentSpec, Workspace


def test_effective_dimensions_in_results_and_search(tmp_path):
    ws = Workspace()
    rng = np.random.default_rng(1)
    key = ws.add_dataset(MultiViewDataset([rng.normal(size=(20, 3)), rng.normal(size=(20, 6))]))
    spec = ExperimentSpec(key, params={"embedding_dim": 16, "max_iter": 1}, clusters=[2],
                          kind="search", objective="Silhouette", metrics=["Silhouette"],
                          search_space={"embedding_dim": [16, 32]}, trials=2, search_strategy="grid")
    record = ws.run(spec, output_dir=tmp_path)
    assert spec.params["embedding_dim"] == 16
    for row in record.rows:
        assert row["status"] == "complete"
        assert row["params"]["embedding_dim"] == 2
        assert row["requested_params"]["embedding_dim"] in [16, 32]
    assert aggregate_results(record)[0]["embedding_dim"] == 2
    assert aggregate_results(record)[0]["requested embedding_dim"] == 16
    assert record.report["search"]["best_parameters"]["embedding_dim"] == 2
    exported = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert exported["experiments"][0]["runs"][0]["parameter_adjustments"]["embedding_dim"]["effective"] == 2


@pytest.mark.parametrize("n_true,n_pred", [(3, 3), (5, 3), (3, 5), (40, 40)])
def test_full_confusion_matching_and_compaction(n_true, n_pred):
    rng = np.random.default_rng(7)
    truth = rng.integers(n_true, size=500)
    prediction = rng.integers(n_pred, size=500) * 7 - 2
    original = np.zeros((n_true, n_pred), dtype=int)
    np.add.at(original, (truth, (prediction + 2) // 7), 1)
    rows, cols = linear_sum_assignment(-original)
    matrix, _, column_names = compact_confusion(truth, prediction, max_labels=100)
    assert matrix.trace() == original[rows, cols].sum()
    assert sorted(map(int, column_names)) == list(range(n_pred))
    compact, _, _ = compact_confusion(truth, prediction, max_labels=2)
    assert compact.sum() == matrix.sum() == len(truth)


def test_mat_export_preserves_sparse_relations_masks_and_ids():
    data = MultiViewDataset([sparse.eye(3, format="csr"), np.arange(6).reshape(3, 2)],
                            [[sparse.eye(3), sparse.eye(3) * 2], None], labels=[1, 2, 1],
                            sample_ids=["a", "b", "c"],
                            observed_mask=np.array([[1, 1], [1, 0], [1, 1]], dtype=bool))
    loaded = loadmat(io.BytesIO(dataset_mat(data)))
    assert sparse.issparse(loaded["view_1"]) and sparse.issparse(loaded["graph_1_2"])
    np.testing.assert_array_equal(loaded["graph_mapping"], [[1, 1], [1, 2]])
    np.testing.assert_array_equal(loaded["observed_mask"], data.observed_mask)
    assert loaded["labels"].ravel().tolist() == [1, 2, 1]
    assert "labels" not in loadmat(io.BytesIO(dataset_mat(data, include_labels=False)))


def test_plot_snapshot_export_and_cleanup():
    pd = pytest.importorskip("pandas")
    go = pytest.importorskip("plotly.graph_objects")
    from pymv.display.exporting import PlotSnapshot

    plot = PlotSnapshot("My plot", go.Figure(go.Scatter(y=[1, 2])), pd.DataFrame({"loss": [1, 2]}))
    assert "data" in json.loads(plot.export("json"))
    assert b"loss" in plot.export("csv")
    with zipfile.ZipFile(io.BytesIO(plot.export("zip"))) as archive:
        assert set(archive.namelist()) == {"figure.html", "figure.json", "data.csv"}
    path = plot.path
    plot.cleanup()
    assert not path.exists()


def test_export_center_prepares_applied_data_and_optional_labels():
    from pymv.display.preprocessing import PreprocessingSpec, ViewPreprocessing
    from tests.test_display_comparisons import app_test

    ws = Workspace()
    original = np.arange(30, dtype=float).reshape(10, 3)
    key = ws.add_dataset(MultiViewDataset([original], labels=np.arange(10) % 2, name="Prepared example"))
    recipe = PreprocessingSpec(mode="all", shared=ViewPreprocessing(center=True))
    ws.apply_dataset(key, recipe)
    at = app_test(ws, 6)
    next(r for r in at.radio if r.label == "Export type").set_value("Prepared datasets").run()
    next(b for b in at.button if b.label == "Prepare MATLAB file").click().run()
    assert not at.exception and not at.error
    prepared = loadmat(io.BytesIO(at.session_state._prepared_mat["data"]))
    np.testing.assert_allclose(prepared["view_1"], original - original.mean(axis=0))
    assert "labels" in prepared
    next(c for c in at.checkbox if c.label == "Include ground-truth labels").uncheck().run()
    next(b for b in at.button if b.label == "Prepare MATLAB file").click().run()
    assert "labels" not in loadmat(io.BytesIO(at.session_state._prepared_mat["data"]))


def test_benchmark_k_defaults_are_editable_without_changing_models():
    from tests.test_display_comparisons import app_test

    ws = Workspace()
    key = ws.add_dataset(MultiViewDataset([np.arange(72).reshape(24, 3)],
                                         labels=np.arange(24) % 4, name="Four classes"))
    ws.add_model("Manual K", "lmgec", {"embedding_dim": 2, "max_iter": 1}, [2])
    at = app_test(ws, 3)
    next(m for m in at.multiselect if m.label == "Datasets").set_value([key]).run()
    field = next(n for n in at.number_input if n.label == "K for Four classes")
    assert field.value == 4
    field.set_value(3).run()
    assert next(n for n in at.number_input if n.label == "K for Four classes").value == 3
    assert next(iter(ws.models.values()))["clusters"] == [2]
    at.checkbox(key="dataset_class_counts").uncheck().run()
    assert not at.exception
    assert not any(n.label == "K for Four classes" for n in at.number_input)
