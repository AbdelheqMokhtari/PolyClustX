"""Regressions for mixed label uploads and user-defined search candidates."""

from pathlib import Path

import numpy as np
import pytest

from pymv.data import dataset_from_arrays, load_files
from pymv.display.model_controls import COMMON, parse_candidates


def test_upload_batch_labels_are_not_features(tmp_path):
    first = np.arange(12).reshape(6, 2)
    second = np.arange(18).reshape(6, 3)
    labels = np.repeat([0, 1], 3)
    for name, array in [("view1", first), ("view2", second), ("Y", labels)]:
        np.save(tmp_path / f"{name}.npy", array)
    data = load_files(
        [tmp_path / "view1.npy", tmp_path / "Y.npy", tmp_path / "view2.npy"],
        view_keys=["view1.npy:view1", "view2.npy:view2"], label_key="Y.npy:Y",
    )
    assert data.n_views == 2
    np.testing.assert_array_equal(data.labels, labels)
    np.testing.assert_array_equal(data.Xs[0], first)
    np.testing.assert_array_equal(data.Xs[1], second)


def test_embedded_column_survives_file_reload_without_label_leakage(tmp_path):
    features = np.arange(12).reshape(6, 2)
    labels = np.repeat([0, 1], 3)
    combined = np.column_stack([features, labels])
    path = tmp_path / "view.npy"
    np.save(path, combined)
    mapping = {"view_keys": ["view.npy:view"], "label_column": ("view.npy:view", 2)}
    imported = dataset_from_arrays({"view.npy:view": combined}, **mapping)
    reloaded = load_files([path], **mapping)
    for data in [imported, reloaded]:
        np.testing.assert_array_equal(data.labels, labels)
        np.testing.assert_array_equal(data.Xs[0], features)
    assert combined.shape == (6, 3)


@pytest.mark.parametrize("mapping", [
    {"view_keys": ["X", "Y"], "label_key": "Y"},
    {"view_keys": ["X"], "label_key": "X"},
    {"view_keys": ["X"], "label_column": ("X", -1)},
    {"view_keys": ["X"], "label_column": ("X", 0.5)},
    {"view_keys": ["X"], "label_column": ("X", 9)},
    {"view_keys": ["X"], "label_column": ("X", 1), "label_key": "Y"},
])
def test_invalid_label_mapping_is_rejected(mapping):
    with pytest.raises(ValueError):
        dataset_from_arrays({"X": np.ones((4, 2)), "Y": np.arange(4)}, **mapping)


@pytest.mark.parametrize("parameter, values, expected", [
    ("embedding_dim", ["17", "17.0", "3e1"], [17, 30]),
    ("temperature", ["0.00001", "250", "1e-4"], [0.00001, 250.0, 0.0001]),
    ("learning_rate", ["2e-5", "0.0007"], [0.00002, 0.0007]),
    ("beta", ["0", "125.5"], [0.0, 125.5]),
])
def test_custom_candidates(parameter, values, expected):
    assert parse_candidates(values, COMMON[parameter]) == expected


@pytest.mark.parametrize("parameter, value", [
    ("embedding_dim", "3.5"), ("embedding_dim", "0"), ("embedding_dim", "-3"),
    ("temperature", "0"), ("temperature", "-0.1"), ("temperature", "nan"),
    ("temperature", "inf"), ("temperature", "1e9999"), ("temperature", "1e-9999"),
    ("temperature", "hello"), ("miss_rate", "1.1"), ("noise_rate", "-1"),
])
def test_invalid_candidates(parameter, value):
    with pytest.raises(ValueError):
        parse_candidates([value], COMMON[parameter])


def test_custom_search_persists_across_navigation_and_rejects_bad_values():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from pymv.display.workspace import Workspace

    workspace = Workspace()
    features = np.random.default_rng(7).normal(size=(12, 4))
    workspace.add_dataset(dataset_from_arrays({"X": features}, view_keys=["X"]))
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/pymv/display/app.py", default_timeout=60)
    at.session_state["workspace"] = workspace
    at.session_state["workflow_step"] = 7
    at.run()
    at.number_input(key="parameter_lmgec_max_iter").set_value(1)
    at.number_input(key="parameter_lmgec_embedding_dim").set_value(2)
    next(s for s in at.selectbox if s.label == "Search strategy").select("Random search").run()
    at.checkbox(key="search_lmgec_temperature").check().run()
    at.multiselect(key="candidates_lmgec_temperature").set_value(["0.0007", "250"]).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Run search").click().run()
    assert not at.exception
    record = at.session_state["workspace"].records[0]
    assert record.rows[0]["status"] == "complete"
    assert record.rows[0]["params"]["temperature"] in [0.0007, 250.0]
    at.button(key="nav_Model").click().run()
    at.button(key="nav_Search").click().run()
    assert not at.exception
    assert at.multiselect(key="candidates_lmgec_temperature").value == ["0.0007", "250.0"]
    at.multiselect(key="candidates_lmgec_temperature").set_value(["-1", "nonsense"]).run()
    assert not at.exception and at.error
    assert next(b for b in at.button if b.label == "Run search").disabled
    at.multiselect(key="candidates_lmgec_temperature").set_value([]).run()
    assert not at.exception and at.error
    assert next(b for b in at.button if b.label == "Run search").disabled


@pytest.mark.parametrize("embedded", [False, True])
def test_simple_upload_ui_excludes_labels_and_reloads(embedded):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string('''
import io
from unittest.mock import patch
import numpy as np
import streamlit as st
from pymv.display.app import _simple_upload
from pymv.display.workspace import Workspace

class Upload(io.BytesIO):
    def __init__(self, name, data):
        super().__init__()
        self.name = name
        np.save(self, data)

labels = np.repeat([0, 1], 3)
if st.session_state.embedded:
    files = [Upload("view.npy", np.column_stack([np.ones((6, 3)), labels]))]
else:
    files = [Upload("view.npy", np.ones((6, 3))), Upload("Y.npy", labels)]
if "workspace" not in st.session_state:
    st.session_state.workspace = Workspace()
with patch.object(st, "file_uploader", return_value=files):
    _simple_upload(st.session_state.workspace)
''', default_timeout=60)
    at.session_state["embedded"] = embedded
    at.run()
    if embedded:
        at.radio(key="simple_label_location").set_value("A column in a view").run()
    else:
        at.radio(key="simple_label_location").set_value("A file in this upload").run()
        at.selectbox(key="simple_label_key").select("Y.npy:Y").run()
    next(b for b in at.button if b.label == "Add uploaded dataset").click().run()
    assert not at.exception and not at.error
    workspace = at.session_state["workspace"]
    key = next(iter(workspace.datasets))
    workspace.unload_dataset(key)
    data = workspace.load_dataset(key)
    assert data.n_views == 1 and data.n_clusters == 2
    np.testing.assert_array_equal(data.Xs[0], np.ones((6, 3)))
    workspace.remove_dataset(key)
