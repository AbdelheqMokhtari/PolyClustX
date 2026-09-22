from pathlib import Path

import pytest

from pymv.display.results import download_stem, record_download_stem, record_label, run_label
from pymv.display.workspace import ExperimentSpec, Workspace
from tests.test_display_comparisons import app_test
from tests.test_experiment_reports import ReportModel, dataset


def test_result_labels_and_download_names(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.delta_generator import DeltaGenerator

    monkeypatch.setattr("pymv.display.workspace.get_model", lambda _: ReportModel)
    ws = Workspace()
    data = dataset()
    data.name = "Demo dataset"
    key = ws.add_dataset(data)
    spec = ExperimentSpec(key, name="Comparison 1", model_name="My baseline", clusters=[2], repeats=2,
                          benchmark=True, metrics=["NMI"])
    ws.run_batch([spec], name=spec.name, output_dir=tmp_path)
    record = ws.records[0]
    ws.run(spec, output_dir=tmp_path)  # Identical names still have distinguishable loss labels.
    assert record_label(record) == "Comparison 1 · Demo dataset · My baseline"
    assert run_label(record, 1) == "Configuration 1 · K=2 · Run 2"
    assert record.id not in record_download_stem(record)

    filenames = []
    original = DeltaGenerator.download_button

    def download(self, label, data, file_name=None, *args, **kwargs):
        filenames.append(file_name)
        return original(self, label, data, file_name, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", download)
    import streamlit as st
    original_download = st.download_button

    def direct_download(label, data, file_name=None, *args, **kwargs):
        filenames.append(file_name)
        return original_download(label, data, file_name, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", direct_download)
    at = app_test(ws, 4)
    assert not at.exception
    assert all("My baseline" in e.label for e in at.expander if "Comparison 1" in e.label)
    at.button(key="nav_Visualizations").click().run()
    assert not at.exception
    comparisons = next(m for m in at.multiselect if m.label == "Models and runs from any experiment")
    assert len(set(comparisons.options)) == 4
    assert all("My baseline" in label and "Run " in label for label in comparisons.options)
    assert all(r.id[:6] not in label for r in ws.records for label in comparisons.options)
    experiments = next(s for s in at.selectbox if s.label == "Experiment")
    assert all("Demo dataset" in label for label in experiments.options)
    next(b for b in at.button if b.key and b.key.startswith("save_plot_confusion-")).click().run()
    assert len(ws.plots) == 1 and not at.exception
    at.button(key="nav_Export").click().run()
    next(r for r in at.radio if r.label == "Export type").set_value("Visualizations").run()
    assert not at.exception
    assert filenames and any(name.endswith("_confusion.zip") for name in filenames)
    for filename in filenames:
        assert Path(filename).name == filename
        assert all(r.id not in filename for r in ws.records)
        if not filename.startswith("loss-comparison"):
            assert filename.startswith("Comparison-1")


@pytest.mark.parametrize("name", ["../../\\unsafe: name", "", "你好" * 200, "x" * 500])
def test_download_names_are_safe_and_bounded(name):
    stem = download_stem(name)
    assert stem and "/" not in stem and "\\" not in stem and ":" not in stem
    assert len(stem.encode("utf-8")) <= 160
