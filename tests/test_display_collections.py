import numpy as np
import pytest

from polyclustx.data import MultiViewDataset
from polyclustx.display.preprocessing import PreprocessingSpec
from polyclustx.display.workspace import Workspace
from tests.test_display_comparisons import app_test


def data():
    return MultiViewDataset([np.ones((8, 3)), np.ones((8, 4))], name="example")


def test_friendly_unique_names_and_rename_survive_reload():
    ws = Workspace()
    first = ws.add_lazy_dataset("example", data)
    second = ws.add_lazy_dataset("example", data)
    assert ws.datasets[second].name == "example#2"
    ws.rename_dataset(first, "My experiment data")
    assert ws.load_dataset(first).name == "My experiment data"
    ws.unload_dataset(first)
    assert ws.load_dataset(first).name == "My experiment data"
    ws.advance_name("synthetic-dataset")
    ws.remove_dataset(first)
    assert ws.suggested_name("synthetic-dataset") == "synthetic-dataset#2"


def test_pending_dataset_must_be_applied_and_pairs_are_filtered():
    ws = Workspace()
    key = ws.add_dataset(data())
    ws.pending_datasets.add(key)
    model = ws.add_model("LMGEC#1", "lmgec", {"embedding_dim": 2}, [2])
    assert not ws.compatible_pairs([key], [model])
    with pytest.raises(ValueError):
        ws.apply_dataset(key, PreprocessingSpec(missing_rate=2))
    assert key in ws.pending_datasets
    ws.apply_dataset(key, PreprocessingSpec(missing_rate=.2))
    assert not ws.compatible_pairs([key], [model])
    ws.apply_dataset(key, PreprocessingSpec(missing_rate=.2, imputation="mean"))
    assert ws.compatible_pairs([key], [model]) == [(key, model)]


def test_models_can_be_named_and_prepared_without_datasets():
    ws = Workspace()
    at = app_test(ws, 2)
    assert not at.exception
    assert not any(s.label == "Active dataset" for s in at.selectbox)
    assert next(t for t in at.text_input if t.label == "Configuration name").value == "LMGEC#1"
    next(b for b in at.button if b.label == "Save model settings").click().run()
    assert not at.exception and next(iter(ws.models.values()))["name"] == "LMGEC#1"
    assert next(t for t in at.text_input if t.label == "Configuration name").value == "LMGEC#2"
    next(t for t in at.text_input if t.label == "Configuration name").set_value("My baseline")
    next(b for b in at.button if b.label == "Save model settings").click().run()
    assert not at.exception
    assert [item["name"] for item in ws.models.values()] == ["LMGEC#1", "My baseline"]


def test_synthetic_names_preparation_and_dataset_browsing():
    ws = Workspace()
    at = app_test(ws, 0)
    next(r for r in at.radio if r.label == "Data source").set_value("Synthetic").run()
    assert next(t for t in at.text_input if t.label == "Dataset name").value == "synthetic-dataset#1"
    next(n for n in at.number_input if n.label == "Samples").set_value(20)
    next(b for b in at.button if b.label == "Create synthetic dataset").click().run()
    assert not at.exception
    first = next(iter(ws.datasets))
    assert ws.datasets[first].name == "synthetic-dataset#1" and first in ws.pending_datasets
    assert next(t for t in at.text_input if t.label == "Dataset name").value == "synthetic-dataset#2"
    next(s for s in at.slider if s.label.startswith("Simulate")).set_value(.2).run()
    next(b for b in at.button if b.label == "Apply changes").click().run()
    assert not at.exception and first not in ws.pending_datasets
    assert ws.preprocessing[first].missing_rate == .2
    assert ws.datasets[first].hint["prepared_profile"]["complete"] is False
    next(t for t in at.text_input if t.label == "Dataset name").set_value("My second dataset")
    next(b for b in at.button if b.label == "Create synthetic dataset").click().run()
    assert not at.exception
    second = list(ws.datasets)[1]
    assert ws.datasets[second].name == "My second dataset"
    assert next(s for s in at.slider if s.label.startswith("Simulate")).value == 0
    at.selectbox(key="dataset_browser").select(first).run()
    assert not at.exception
    assert next(s for s in at.slider if s.label.startswith("Simulate")).value == .2


def test_large_pair_table_uses_names_without_internal_identifiers():
    ws = Workspace()
    for i in range(12):
        ws.add_dataset(data())
        ws.add_model(f"Baseline {i + 1}", "lmgec", {"embedding_dim": 2}, [2])
    at = app_test(ws, 3)
    next(m for m in at.multiselect if m.label == "Datasets").set_value(list(ws.datasets)).run()
    next(m for m in at.multiselect if m.label == "Model configurations").set_value(list(ws.models)).run()
    next(r for r in at.radio if r.label == "Combination selection").set_value("Choose in table").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert table.shape == (12, 13)  # Dataset name + one selection column per model.
    assert set(table["dataset"]) == {r.name for r in ws.datasets.values()}
    assert all(key not in str(at.multiselect[0].options) for key in ws.datasets)
    matrix_key = "pair_matrix_" + "_".join([*ws.datasets, *ws.models])
    at.session_state[matrix_key] = {"edited_rows": {0: {"model_0": False}}, "added_rows": [], "deleted_rows": []}
    at.run()
    assert not at.exception
    assert any("143 dataset/model combinations" in caption.value for caption in at.caption)
