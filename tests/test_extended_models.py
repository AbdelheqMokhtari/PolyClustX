import numpy as np
import pytest
import torch
from scipy import sparse

from polyclustx.data import MultiViewDataset
from polyclustx.display.model_controls import parameter_controls
from polyclustx.display.workspace import ExperimentSpec, Workspace
from polyclustx.models import CONAN, DGF, SGLA, UMC, GCoMVKM, get_model
from polyclustx.models.conan.conan import divergence_loss
from polyclustx.models.dgf.dgf import walk_contexts
from polyclustx.models.sgla.sgla import spectral_objective
from polyclustx.models.umc.umc import high_quality_indices, supervised_contrastive

MODELS = ["conan", "dgf", "g_comvkm", "sgla", "umc"]


def parameters(name):
    result = dict(max_iter=2, random_state=4, device="cpu")
    if name in {"conan", "umc"}:
        result.update(hidden_dims=(8,), embedding_dim=4, batch_size=8)
    if name == "conan":
        result["projection_dim"] = 4
    if name == "umc":
        result.update(pretrain_epochs=1, n_init=1)
    if name == "dgf":
        result.update(embedding_dim=4, num_layers=2, batch_size=8, n_init=1, walks_per_node=1)
    if name == "sgla":
        result.update(max_iter=5, n_init=1, n_neighbors=3)
    return result


def data(views=2):
    rng = np.random.default_rng(5)
    labels = np.arange(15) % 3
    return MultiViewDataset([rng.normal(size=(15, v + 4)) + labels[:, None] for v in range(views)], labels=labels)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("views", [1, 2, 3])
def test_fit_api_determinism_and_inputs(name, views):
    dataset = data(views)
    originals = [x.copy() for x in dataset.Xs]
    model = get_model(name)(3, **parameters(name))
    state = torch.random.get_rng_state().clone()
    dataset.fit(model)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.labels_.shape == (15,)
    assert model.labels_.min() >= 0 and model.labels_.max() < 3
    assert np.isfinite(model.embedding_).all() and np.isfinite(model.loss_history_).all()
    for before, after in zip(originals, dataset.Xs):
        np.testing.assert_array_equal(before, after)
    other = get_model(name)(3, **parameters(name))
    dataset.fit(other)
    np.testing.assert_array_equal(model.labels_, other.labels_)
    np.testing.assert_allclose(model.embedding_, other.embedding_, atol=1e-5)
    if hasattr(model, "predict"):
        np.testing.assert_array_equal(model.predict(dataset.Xs), model.labels_)
    if name == "g_comvkm":
        np.testing.assert_allclose(model.view_weights_.sum(), 1)
        for w, indices in zip(model.feature_weights_, model.selected_features_):
            assert len(w) == len(indices) >= 1
            np.testing.assert_allclose(w.sum(), 1)


@pytest.mark.parametrize("name", MODELS)
def test_workspace_and_controls(name, tmp_path):
    ws = Workspace()
    dataset_id = ws.add_dataset(data())
    events = []
    record = ws.run(ExperimentSpec(dataset_id, model=name, params=parameters(name), clusters=[3],
                                    benchmark=True, metrics=["NMI"]), output_dir=tmp_path, fit_progress=events.append)
    assert record.rows[0]["status"] == "complete", record.rows[0].get("error")
    assert events and events[-1]["model"] == name
    defaults, controls = parameter_controls(name)
    assert set(defaults) - {"random_state"} <= set(controls)


@pytest.mark.parametrize("cls", [CONAN, DGF, SGLA, UMC, GCoMVKM])
def test_missing_rejected(cls):
    dataset = data()
    mask = np.ones((15, 2), dtype=bool)
    mask[0, 0] = False
    with pytest.raises(ValueError, match="complete"):
        cls(3).fit(dataset.Xs, observed_mask=mask)


def test_formulations_and_degenerate_cases():
    eigenvalues = torch.tensor([0., .2, .5])
    weights = torch.tensor([.25, .75])
    assert float(spectral_objective(eigenvalues, weights, 2, .5)) == pytest.approx(.5125)
    q = torch.softmax(torch.randn(8, 3), 1).requires_grad_()
    value = divergence_loss(q, torch.ones(8, 4))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(q.grad).all()
    h = torch.randn(4, 5, requires_grad=True)
    loss = supervised_contrastive([h, h + .1], torch.arange(4), .1)
    loss.backward()
    assert torch.isfinite(h.grad).all()
    chosen = high_quality_indices(np.zeros((8, 3)), np.repeat([0, 1], 4), .5)
    assert len(chosen) == 4
    paths = walk_contexts(sparse.csr_matrix((4, 4)), 2, 3, 2, np.random.default_rng(0))
    assert paths.min() == 0 and paths.max() == 3
    np.testing.assert_array_equal(paths[:, 0], paths[:, 1])


def test_graph_inputs_and_memory_guards():
    dataset = data()
    graphs = [sparse.eye(15), None]
    for name in ("dgf", "sgla"):
        model = get_model(name)(3, **parameters(name)).fit(dataset.Xs, graphs)
        assert np.isfinite(model.embedding_).all()
        with pytest.raises(ValueError, match="Graphs"):
            model.fit(dataset.Xs, [np.ones((2, 2)), None])
        with pytest.raises(ValueError, match="allocation"):
            get_model(name)(3, max_memory_mb=.001).fit(dataset.Xs)
    with pytest.raises(ValueError, match="gamma"):
        GCoMVKM(3, gamma=1).fit(dataset.Xs)


@pytest.mark.parametrize("name", MODELS)
def test_new_model_controls_render(name):
    from tests.test_display_comparisons import app_test
    at = app_test(Workspace(), 2)
    next(s for s in at.selectbox if s.label == "Model").select(name).run()
    assert not at.exception and not at.error


def test_umc_density_memory_guard():
    with pytest.raises(ValueError, match="allocation"):
        high_quality_indices(np.zeros((100, 3)), np.zeros(100), .5, max_memory_mb=.001)
