import numpy as np
import pytest
import torch
from scipy import sparse
from sklearn.exceptions import NotFittedError

from pymv.data import MultiViewDataset
from pymv.models import COMPLETER, DCP, IDEC, MFLVC
from pymv.models.completer import consistency_loss
from pymv.models.idec import student_assignments, target_distribution
from pymv.models.mflvc import aligned_pseudo_labels, label_contrastive_loss


def configuration(cls):
    options = dict(n_clusters=3, embedding_dim=4, hidden_dims=(8,), max_iter=2,
                   batch_size=6, n_init=2, random_state=12, device="cpu")
    if cls is IDEC:
        options.update(pretrain_epochs=1, update_interval=1, tol=0)
    elif cls is MFLVC:
        options.update(projection_dim=3, pretrain_epochs=1, finetune_epochs=1)
    else:
        options.update(prediction_dims=(6, 4), prediction_start=0)
    return options


def arrays(n_views):
    rng = np.random.default_rng(13)
    return [rng.normal(size=(13, 4 + v)).astype(np.float32) for v in range(n_views)]


@pytest.mark.parametrize("cls", [IDEC, COMPLETER, DCP, MFLVC])
@pytest.mark.parametrize("n_views", [1, 2, 3])
def test_models_fit_predict_seed_and_finite_arrays(cls, n_views):
    xs = arrays(n_views)
    options = configuration(cls)
    state = torch.random.get_rng_state().clone()
    a = cls(**options).fit([sparse.csr_matrix(x) for x in xs])
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    b = cls(**options).fit(xs)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_allclose(a.embedding_, b.embedding_, rtol=0, atol=0)
    np.testing.assert_array_equal(a.labels_, a.predict(xs))
    np.testing.assert_allclose(a.embedding_, a.transform(xs), rtol=1e-6, atol=1e-6)
    assert a.labels_.shape == (13,)
    assert np.isfinite(a.embedding_).all() and np.isfinite(a.loss_history_).all()
    assert a.centroids_.shape[0] == 3
    assert all(p.device.type == "cpu" for p in a.model_.parameters())
    assert not a.model_.training


@pytest.mark.parametrize("cls", [COMPLETER, DCP])
@pytest.mark.parametrize("n_views", [2, 3])
def test_mask_completion_and_hidden_values_do_not_affect_fit(cls, n_views):
    xs = arrays(n_views)
    mask = np.ones((13, n_views), dtype=bool)
    mask[6:9, 0] = False
    mask[9:, 1:] = False
    hidden = [x.copy() for x in xs]
    for v, x in enumerate(hidden):
        x[~mask[:, v]] = np.nan
    a = cls(**configuration(cls)).fit(hidden, observed_mask=mask)
    b = cls(**configuration(cls)).fit(xs, observed_mask=mask)
    np.testing.assert_allclose(a.embedding_, b.embedding_, atol=0, rtol=0)
    np.testing.assert_array_equal(a.labels_, a.predict(hidden, observed_mask=mask))
    assert np.isfinite(a.transform([x[:1] for x in hidden], observed_mask=mask[:1])).all()
    data = MultiViewDataset(hidden, observed_mask=mask)
    c = data.fit(cls(**configuration(cls)))
    np.testing.assert_allclose(c.embedding_, a.embedding_)
    assert a.n_complete_samples_ == 6


@pytest.mark.parametrize("cls", [COMPLETER, DCP])
def test_reject_unlearnable_missing_patterns(cls):
    xs = arrays(2)
    mask = np.zeros((13, 2), dtype=bool)
    mask[:6, 0], mask[6:, 1] = True, True
    with pytest.raises(ValueError, match="fully observed"):
        cls(**configuration(cls)).fit(xs, observed_mask=mask)
    mask[:3] = True
    options = configuration(cls)
    options["prediction_start"] = 10
    with pytest.raises(ValueError, match="prediction_start"):
        cls(**options).fit(xs, observed_mask=mask)


@pytest.mark.parametrize("cls", [IDEC, COMPLETER, DCP, MFLVC])
def test_neural_invalid_prediction_and_budget(cls):
    model = cls(**configuration(cls))
    with pytest.raises(NotFittedError):
        model.predict(arrays(2))
    options = configuration(cls)
    options["max_memory_mb"] = .001
    with pytest.raises(ValueError, match="allocation"):
        cls(**options).fit(arrays(2))
    options = configuration(cls)
    options["learning_rate"] = -1
    with pytest.raises(ValueError, match="learning_rate"):
        cls(**options).fit(arrays(2))


def test_idec_student_target_and_kl_equations():
    z = torch.tensor([[0., 0.], [2., 1.]], dtype=torch.float64)
    c = torch.tensor([[1., 0.], [3., 1.]], dtype=torch.float64)
    weights = 1 / (1 + ((z[:, None] - c[None])**2).sum(2))
    expected = weights / weights.sum(1, keepdim=True)
    q = student_assignments(z, c)
    torch.testing.assert_close(q, expected)
    w = q.square() / q.sum(0)
    torch.testing.assert_close(target_distribution(q), w / w.sum(1, keepdim=True))


def test_consistency_loss_explicit_joint_and_gradient():
    left = torch.tensor([[.2, .8], [.7, .3]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[.4, .6], [.9, .1]], dtype=torch.float64, requires_grad=True)
    joint = (left[:, :, None] * right[:, None, :]).sum(0)
    joint = (joint + joint.T) / 2
    joint = joint / joint.sum()
    expected = -(joint * (joint.log() - 10 * joint.sum(0)[None].log()
                         - 10 * joint.sum(1)[:, None].log())).sum()
    actual = consistency_loss(left, right, 10.)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (left, right), retain_graph=True),
                    torch.autograd.grad(expected, (left, right))):
        torch.testing.assert_close(a, b)


def test_mflvc_label_contrast_and_pseudo_alignment():
    q = torch.tensor([[.9, .1], [.2, .8]], dtype=torch.float64, requires_grad=True)
    loss = label_contrastive_loss(q, q, 1.)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(q.grad).all()
    np.testing.assert_array_equal(aligned_pseudo_labels(np.array([0, 1, 2]), np.array([2, 0, 1]), 3), [2, 0, 1])


@pytest.mark.parametrize("cls", [IDEC, COMPLETER, DCP, MFLVC])
def test_neural_models_preserve_inputs_and_handle_nondefault_dtype(cls):
    xs = arrays(2)
    copies = [x.copy() for x in xs]
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = cls(**configuration(cls)).fit(xs)
        assert model.embedding_.dtype == np.float32
        assert torch.get_default_dtype() == torch.float64
        for x, reference in zip(xs, copies):
            np.testing.assert_array_equal(x, reference)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("cls", [IDEC, COMPLETER, DCP, MFLVC])
def test_neural_cuda_training_returns_cpu_state(cls):
    options = configuration(cls)
    options["device"] = "cuda"
    model = cls(**options).fit(arrays(2))
    assert model.device_.startswith("cuda")
    assert all(p.device.type == "cpu" for p in model.model_.parameters())
    assert np.isfinite(model.embedding_).all()
