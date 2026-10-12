import numpy as np
import pytest

from polyclustx.cli import _coerce, build_parser
from polyclustx.data import MultiViewDataset
from polyclustx.display.model_controls import compatible_models, parameter_controls, parse_candidates
from polyclustx.display.workspace import ExperimentSpec, Workspace
from polyclustx.experiment import Experiment
from polyclustx.metrics import nmi_score
from polyclustx.models import COMPLETER, DCP, IDEC, MFLVC, MultiViewKMeans, MultiViewSpectralClustering, get_model
from tests.test_display_comparisons import app_test
from tests.test_new_deep_models import configuration

KEYS = ["multiview_kmeans", "multiview_spectral", "idec", "completer", "dcp", "mflvc"]


def options(key):
    cls = get_model(key)
    if cls in (IDEC, COMPLETER, DCP, MFLVC):
        params = configuration(cls)
        params.pop("n_clusters")
        return params
    return dict(max_iter=2, n_init=2, random_state=3)


def dataset(mask=None):
    rng = np.random.default_rng(14)
    return MultiViewDataset([rng.normal(size=(13, 5)), rng.normal(size=(13, 6))],
                            labels=np.arange(13) % 3, observed_mask=mask, name="sample")


@pytest.mark.parametrize("key", KEYS)
def test_python_and_display_experiments_export_new_models(key, tmp_path):
    model = get_model(key)(3, **options(key))
    result = Experiment(dataset(), model, [nmi_score], verbose=0, save_path=tmp_path).run()
    assert result.report["experiments"][0]["runs"][0]["status"] == "complete"
    ws = Workspace()
    ds = ws.add_dataset(dataset())
    record = ws.run(ExperimentSpec(dataset_id=ds, model=key, params=options(key),
                                   clusters=[3], metrics=["NMI"], benchmark=True), output_dir=tmp_path)
    assert record.rows[0]["status"] == "complete"
    assert record.rows[0]["diagnostics"]["device"] == "cpu"


def test_optional_and_choice_controls_and_cli_parameters():
    _, controls = parameter_controls("multiview_spectral")
    assert controls["gamma"].nullable and controls["info_view"].nullable
    assert parse_candidates(["rbf", "poly"], controls["affinity"]) == ["rbf", "poly"]
    with pytest.raises(ValueError):
        parse_candidates(["invalid"], controls["affinity"])
    with pytest.raises(ValueError):
        parse_candidates(["0"], controls["gamma"])
    args = build_parser().parse_args(["--model", "idec", "--pretrain-epochs", "2", "--hidden-dims", "8,4"])
    assert args.pretrain_epochs == "2"
    assert _coerce(args.hidden_dims, (500, 500, 2000)) == (8, 4)
    assert _coerce("0.1", None) == .1
    assert _coerce("0", None) == 0
    assert _coerce("None", None) is None


def test_missing_model_compatibility_checks_overlap():
    mask = np.ones((13, 2), dtype=bool)
    mask[6:, 0] = False
    available, unavailable = compatible_models(dataset(mask))
    assert {"completer", "dcp"} <= set(available)
    assert {"idec", "mflvc", "multiview_kmeans", "multiview_spectral"} <= set(unavailable)
    mask[:6, 1] = False
    available, _ = compatible_models(dataset(mask))
    assert "completer" not in available and "dcp" not in available
    ws = Workspace()
    ds = ws.add_dataset(dataset(mask))
    model = ws.add_model("Missing", "completer", options("completer"), [3])
    assert ws.compatible_pairs([ds], [model]) == []
    with pytest.raises(ValueError, match="fully observed"):
        ws.run(ExperimentSpec(dataset_id=ds, model="completer"), validate_only=True)


@pytest.mark.parametrize("key", KEYS)
def test_display_renders_and_saves_each_new_model(key):
    pytest.importorskip("streamlit")
    ws = Workspace()
    at = app_test(ws, 2)
    next(s for s in at.selectbox if s.label == "Model").select(key).run()
    assert not at.exception and not at.error
    if key == "multiview_spectral":
        assert at.checkbox(key="auto_parameter_multiview_spectral_gamma").value
        at.checkbox(key="auto_parameter_multiview_spectral_gamma").uncheck().run()
        at.number_input(key="parameter_multiview_spectral_gamma").set_value(.2).run()
        assert not at.exception
    next(b for b in at.button if b.label == "Save model settings").click().run()
    assert not at.exception
    assert next(iter(ws.models.values()))["model"] == key


def test_reproducible_config_uses_registry_key():
    for cls, key in [(MultiViewKMeans, "multiview_kmeans"),
                     (MultiViewSpectralClustering, "multiview_spectral")]:
        experiment = Experiment(dataset(), cls(3), [nmi_score], verbose=0)
        assert experiment._build_reproducible_config()["model"]["name"] == key


def test_spectral_categorical_grid_search(tmp_path):
    ws = Workspace()
    ds = ws.add_dataset(dataset())
    spec = ExperimentSpec(dataset_id=ds, model="multiview_spectral", clusters=[3],
                          params={"max_iter": 1, "n_init": 2}, benchmark=True, metrics=["NMI"],
                          search_space={"affinity": ["rbf", "poly"]}, search_strategy="grid")
    record = ws.run(spec, output_dir=tmp_path)
    assert len(record.rows) == 2
    assert all(row["status"] == "complete" for row in record.rows)


def test_offline_new_model_example(tmp_path):
    from examples.new_models import run

    complete, missing = run(tmp_path)
    assert len(complete.report["experiments"]) == 6
    assert len(missing.report["experiments"]) == 2
    for result in (complete, missing):
        assert all(run["status"] == "complete" for experiment in result.report["experiments"]
                   for run in experiment["runs"])
