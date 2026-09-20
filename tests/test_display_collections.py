import numpy as np
import pytest

from pymv.data import MultiViewDataset
from pymv.display.preprocessing import PreprocessingSpec
from pymv.display.workspace import Workspace


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
