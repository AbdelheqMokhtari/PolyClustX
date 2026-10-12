import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

from polyclustx.models import MultiViewKMeans, MultiViewSpectralClustering
from polyclustx.models.multiview_kmeans.multiview_kmeans import em_step
from polyclustx.models.multiview_spectral.multiview_spectral import refine_similarities, spectral_components


def views(count=2):
    rng = np.random.default_rng(12)
    labels = np.arange(24) % 3
    return [labels[:, None] * 5 + rng.normal(scale=.15, size=(24, v + 3)) for v in range(count)]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_coem_view_counts_and_predictions(count):
    xs = views(count)
    model = MultiViewKMeans(3, n_init=2, random_state=12).fit(xs)
    np.testing.assert_array_equal(model.labels_, model.predict(xs))
    assert len(model.centroids_) == count
    assert all(c.shape == (3, x.shape[1]) for c, x in zip(model.centroids_, xs))
    if count == 1:
        km = KMeans(3, n_init=2, max_iter=100, tol=1e-4, random_state=12).fit(xs[0])
        np.testing.assert_array_equal(model.labels_, km.labels_)
        np.testing.assert_allclose(model.centroids_[0], km.cluster_centers_)
    else:
        assert model.coem_objective_ == min(model.restart_objectives_)
        for k in range(3):
            consensus = np.all(np.asarray(model.view_labels_) == k, axis=0)
            if consensus.any():
                for v, x in enumerate(xs):
                    np.testing.assert_allclose(model.centroids_[v][k], x[consensus].mean(0))


def test_em_objective_and_empty_cluster_equations():
    x = torch.tensor([[0.], [2.], [10.]], dtype=torch.float64)
    labels, centers, objective = em_step(x, torch.tensor([0, 0, 0]), torch.tensor([[0.], [9.]]).double())
    torch.testing.assert_close(centers, torch.tensor([[4.], [9.]]).double())
    assert labels.tolist() == [0, 0, 1]
    assert float(objective) == 7.


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("affinity", ["rbf", "nearest_neighbors", "poly"])
def test_spectral_views_affinities(count, affinity):
    xs = views(count)
    model = MultiViewSpectralClustering(3, affinity=affinity, n_neighbors=4,
                                       max_iter=2, n_init=2, random_state=1).fit(xs)
    assert model.embedding_.shape == (24, 3 * count)
    assert np.isfinite(model.embedding_).all()
    assert len(np.unique(model.labels_)) == 3
    assert model.n_iter_ == (0 if count == 1 else 2)
    if affinity == "rbf":
        for x, gamma in zip(xs, model.gammas_):
            median = np.median(np.linalg.norm(x[:, None] - x[None], axis=2))
            assert gamma == pytest.approx(1 / (2 * median**2))


def test_spectral_exchanges_other_view_projectors_synchronously():
    generator = torch.Generator().manual_seed(4)
    ks = [torch.randn(8, 8, generator=generator).double() for _ in range(3)]
    us = [torch.linalg.qr(torch.randn(8, 2, generator=generator).double())[0] for _ in ks]
    actual = refine_similarities(ks, us)
    for i, k in enumerate(ks):
        expected = sum(u @ u.T @ k for j, u in enumerate(us) if i != j)
        torch.testing.assert_close(actual[i], (expected + expected.T) / 2)
    u = spectral_components(torch.zeros((8, 8), dtype=torch.float64), 2)
    assert torch.isfinite(u).all()


def test_spectral_selected_view_and_memory_guard():
    model = MultiViewSpectralClustering(3, info_view=1, max_iter=1).fit(views())
    np.testing.assert_array_equal(model.embedding_, model.view_embeddings_[1])
    with pytest.raises(ValueError, match="allocation"):
        MultiViewSpectralClustering(3, max_memory_mb=.001).fit(views())
    with pytest.raises(ValueError, match="info_view"):
        MultiViewSpectralClustering(3, info_view=2).fit(views())


def test_shared_executor_only_falls_back_on_automatic_cuda_oom(monkeypatch):
    from polyclustx.models import _common

    monkeypatch.setattr(_common, "resolve_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def operation(device):
        calls.append(device.type)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated")
        return {"labels_": np.array([0, 1])}

    model = MultiViewKMeans(2)
    model._execute(operation)
    assert model.device_ == "cpu" and calls == ["cuda", "cpu"]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        MultiViewKMeans(2, device="cuda")._execute(operation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("cls", [MultiViewKMeans, MultiViewSpectralClustering])
def test_classical_cuda_returns_cpu_arrays(cls):
    model = cls(3, n_init=2, max_iter=2, random_state=12, device="cuda").fit(views())
    assert model.device_.startswith("cuda") and np.isfinite(model.embedding_).all()
