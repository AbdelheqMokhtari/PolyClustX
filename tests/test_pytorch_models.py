"""Backend-independent equation checks plus PyTorch resource/RNG contracts."""

import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest
import torch
from scipy import sparse

from pymv.models import LMGEC, SCM
from pymv.models._torch_utils import resolve_device
from pymv.models.lmgec.lmgec import _train_loop, _update_rule_F, _update_rule_G, _update_rule_W


def numpy_loop(views, centers, labels, weights, k, iterations):
    losses = []
    for _ in range(iterations):
        consensus = np.zeros((len(labels), centers.shape[1]))
        loss = 0.0
        for view, weight in zip(views, weights):
            left, _, right = np.linalg.svd(view.T @ centers[labels], full_matrices=False)
            projection = left @ right
            consensus += weight * (view @ projection)
            loss += np.linalg.norm(view - (centers @ projection.T)[labels])
        labels = ((consensus[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        centers = np.zeros_like(centers)
        np.add.at(centers, labels, consensus)
        centers /= np.maximum(np.bincount(labels, minlength=k), 1)[:, None]
        losses.append(loss)
    return labels, centers, consensus, np.asarray(losses)


@pytest.mark.parametrize("n_views", [1, 2, 4])
def test_lmgec_loop_preserves_update_equations(n_views):
    rng = np.random.default_rng(81)
    views = [rng.normal(size=(30, d + 4)) for d in range(n_views)]
    centers = rng.normal(size=(3, 2))
    labels = np.arange(30) % 3
    weights = np.ones(n_views) / n_views
    expected = numpy_loop(views, centers.copy(), labels.copy(), weights, 3, 4)
    actual = _train_loop([torch.from_numpy(v) for v in views], torch.from_numpy(centers),
                         torch.from_numpy(labels), torch.from_numpy(weights), 3, 4)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value.numpy(), reference, rtol=1e-10, atol=1e-10)


def test_empty_centroid_and_assignment_ties():
    x = torch.tensor([[2., 4.], [4., 8.]], dtype=torch.float64)
    centers = _update_rule_F(x, torch.tensor([0, 0]), 3)
    torch.testing.assert_close(centers, torch.tensor([[3., 6.], [0., 0.], [0., 0.]], dtype=torch.float64))
    assert _update_rule_G(torch.zeros((2, 2)), torch.zeros((3, 2))).tolist() == [0, 0]
    projection = _update_rule_W(x, centers, torch.tensor([0, 0]))
    torch.testing.assert_close(projection.T @ projection, torch.eye(2, dtype=torch.float64))


def test_lmgec_rank_deficient_svd_reaches_procrustes_optimum():
    # m > k makes X.T @ F[G] rank deficient. Different SVD backends may
    # select different null-space bases; compare the objective, not the basis.
    rng = np.random.default_rng(72)
    x = torch.from_numpy(rng.normal(size=(25, 7)))
    centers = torch.from_numpy(rng.normal(size=(3, 5)))
    labels = torch.from_numpy(np.arange(25) % 3)
    target = centers[labels]
    projection = _update_rule_W(x, centers, labels)
    torch.testing.assert_close(projection.T @ projection, torch.eye(5, dtype=torch.float64))
    optimum = x.square().sum() + target.square().sum() - 2 * torch.linalg.svdvals(x.T @ target).sum()
    actual = (x - target @ projection.T).square().sum()
    torch.testing.assert_close(actual, optimum, rtol=1e-12, atol=1e-12)


def test_scm_loss_and_gradients_match_explicit_positive_negative_equation():
    generator = torch.Generator().manual_seed(4)
    left = torch.randn((5, 3), generator=generator, dtype=torch.float64, requires_grad=True)
    right = torch.randn((5, 3), generator=generator, dtype=torch.float64, requires_grad=True)
    h = torch.cat((left, right))
    similarity = h @ h.T / 0.7
    expected = torch.stack([
        -similarity[i, (i + 5) % 10] + torch.logsumexp(similarity[i, torch.arange(10) != i], dim=0)
        for i in range(10)
    ]).mean()
    actual = SCM._contrastive_loss(left, right, .7)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    reference_gradients = torch.autograd.grad(expected, (left, right), retain_graph=True)
    gradients = torch.autograd.grad(actual, (left, right))
    for value, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("views", [1, 2, 3])
@pytest.mark.parametrize("reconstruction_weight", [0., 1.])
def test_scm_seeded_fit_sparse_inputs_and_cpu_outputs(views, reconstruction_weight):
    rng = np.random.default_rng(9)
    xs = [sparse.csr_matrix(rng.normal(size=(13, 4 + i))) for i in range(views)]
    snapshots = [x.copy() for x in xs]
    options = dict(n_clusters=3, hidden_dims=(8,), embedding_dim=6, projection_dim=3,
                   max_iter=2, batch_size=6, random_state=9, device="cpu",
                   reconstruction_weight=reconstruction_weight, weight_decay=.001)
    before = torch.random.get_rng_state().clone()
    first = SCM(**options).fit(xs)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    second = SCM(**options).fit(xs)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.embedding_, second.embedding_, atol=0, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(first.embedding_, axis=1), 1, atol=1e-6)
    assert np.isfinite(first.loss_history_).all() and len(first.loss_history_) == 2
    assert first.device_ == "cpu" and not first.model_.training
    assert all(p.device.type == "cpu" for p in first.model_.parameters())
    assert all((a != b).nnz == 0 for a, b in zip(xs, snapshots))


def test_scm_masks_never_erase_all_views_and_noise_is_independent():
    model = SCM(2, miss_rate=1., noise_rate=.5)
    x = np.ones((1000, 6), dtype=np.float32)
    masked, noisy = model._augment(x, [2, 2, 2], np.random.default_rng(3))
    kept = masked[:, ::2].sum(axis=1)
    assert (kept > 0).all() and (kept < 3).all()
    assert np.any((noisy[:, 0] == 1) & (noisy[:, 2] != 1))
    np.testing.assert_array_equal(x, np.ones_like(x))


def test_scm_float32_is_independent_of_caller_default_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = SCM(2, hidden_dims=(4,), max_iter=1, device="cpu", random_state=2).fit(
            [np.random.default_rng(2).normal(size=(8, 4))])
        assert model.embedding_.dtype == np.float32
        assert all(p.dtype == torch.float32 for p in model.model_.parameters())
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("cls,kwargs", [(LMGEC, {"embedding_dim": 2}), (SCM, {"hidden_dims": (4,)})])
def test_explicit_unavailable_cuda_is_not_silently_changed(monkeypatch, cls, kwargs):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        cls(2, device="cuda", max_iter=1, **kwargs).fit([np.ones((6, 3))])


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"learning_rate": -1}, {"gaussian_noise": -1},
                                    {"weight_decay": float("nan")}, {"batch_size": 1}, {"hidden_dims": (0,)}])
def test_scm_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SCM(2, **kwargs).fit([np.ones((6, 3))])


@pytest.mark.parametrize("value", ["bad-device", "mps", "meta", None, "cuda:-1"])
def test_invalid_devices(value):
    with pytest.raises(ValueError):
        resolve_device(value)


@pytest.mark.parametrize("cls,kwargs", [(LMGEC, {"embedding_dim": 2}),
                                      (SCM, {"hidden_dims": (4,), "projection_dim": 2})])
@pytest.mark.parametrize("automatic", [True, False])
def test_cuda_oom_only_retries_for_automatic_selection(monkeypatch, cls, kwargs, automatic):
    module = import_module(cls.__module__)
    monkeypatch.setattr(module, "resolve_device", lambda value: torch.device("cuda"))
    calls = []
    if cls is SCM:
        original = SCM._fit_on_device

        def fit_on_device(self, fused, widths, device):
            calls.append(device.type)
            if device.type == "cuda":
                raise torch.cuda.OutOfMemoryError("simulated")
            return original(self, fused, widths, device)

        monkeypatch.setattr(SCM, "_fit_on_device", fit_on_device)
    else:
        original = torch.as_tensor

        def as_tensor(*args, **options):
            device = options.get("device")
            if device is not None:
                calls.append(device.type)
                if device.type == "cuda":
                    raise torch.cuda.OutOfMemoryError("simulated")
            return original(*args, **options)

        monkeypatch.setattr(torch, "as_tensor", as_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = cls(2, device="auto" if automatic else "cuda", max_iter=1, random_state=3, **kwargs)
    xs = [np.random.default_rng(3).normal(size=(8, 4))]
    if automatic:
        model.fit(xs)
        assert model.device_ == "cpu" and np.isfinite(model.embedding_).all()
        assert calls[0] == "cuda" and "cpu" in calls
    else:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
            model.fit(xs)
        assert calls == ["cuda"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is not available")
@pytest.mark.parametrize("cls,kwargs", [(LMGEC, {"embedding_dim": 2}),
                                      (SCM, {"hidden_dims": (8,), "projection_dim": 3})])
def test_cuda_fit_returns_numpy_results_and_cpu_network(cls, kwargs):
    xs = [np.random.default_rng(6).normal(size=(20, 5))]
    model = cls(3, device="cuda", max_iter=2, random_state=6, **kwargs).fit(xs)
    assert model.device_.startswith("cuda")
    assert isinstance(model.embedding_, np.ndarray) and np.isfinite(model.embedding_).all()
    assert np.isfinite(model.loss_history_).all()
    if cls is SCM:
        assert all(p.device.type == "cpu" for p in model.model_.parameters())


def test_pytorch_backend_and_actual_device_in_portable_reports():
    from pymv.experiment.exports import diagnostics, model_details

    for cls, kwargs in [(LMGEC, {"embedding_dim": 2}), (SCM, {"hidden_dims": (4,)})]:
        model = cls(2, max_iter=1, device="cpu", random_state=2, **kwargs).fit(
            [np.random.default_rng(2).normal(size=(8, 4))])
        assert model_details(model)["backend"] == "pytorch"
        assert diagnostics(model)["device"] == "cpu"


def test_package_and_environment_reports_do_not_import_tensorflow():
    script = '''
import builtins, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError("TensorFlow must not be imported")
    return original_import(name, *args, **kwargs)
# PyTorch may probe optional module specs; only actual imports are prohibited.
builtins.__import__ = guarded_import
import pymv
import numpy as np
from pymv.models import LMGEC, SCM
from pymv.tracking import _get_environment_info
from pymv.experiment.markdown_export import _get_environment_info as markdown_environment
views = [np.random.default_rng(2).normal(size=(8, 4))]
LMGEC(2, 2, max_iter=1, device="cpu").fit(views)
SCM(2, hidden_dims=(4,), max_iter=1, device="cpu").fit(views)
assert "torch_version" in _get_environment_info()
assert "torch_version" in markdown_environment()
assert "tensorflow" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
