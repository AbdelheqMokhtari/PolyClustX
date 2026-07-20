"""
Tests for LMGECplus — the modular heterogeneous extension of LMGEC.

Covers:
* component-level unit tests (each extension in isolation),
* backward-compatibility parity with LMGEC (all extensions off),
* that every extension toggle runs and yields well-formed output,
* registry / framework integration.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Component unit tests ─────────────────────────────────────────


class TestComponents:
    def test_procrustes_orthonormal(self):
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(0)
        H = rng.randn(20, 8)
        T = rng.randn(20, 4)
        W = C.procrustes_W(H, T)
        assert W.shape == (8, 4)
        # columns orthonormal
        assert np.allclose(W.T @ W, np.eye(4), atol=1e-8)

    def test_procrustes_node_weight_constant_is_noop(self):
        """A constant node weight must not change the Procrustes solution."""
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(1)
        H, T = rng.randn(15, 6), rng.randn(15, 3)
        W0 = C.procrustes_W(H, T)
        W1 = C.procrustes_W(H, T, node_w=np.full(15, 3.0))
        # same subspace up to sign — compare |WᵀW0|
        assert np.allclose(np.abs(W1.T @ W0), np.eye(3), atol=1e-6)

    def test_attention_sums_to_one(self):
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(2)
        errs = [rng.rand(30), rng.rand(30), rng.rand(30)]
        for reg in (None, "entropy", "l2"):
            B = C.view_attention(errs, lam=1.0, reg=reg)
            stacked = np.stack(B, axis=1)
            assert stacked.shape == (30, 3)
            assert np.allclose(stacked.sum(1), 1.0, atol=1e-6)
            assert stacked.min() >= -1e-9

    def test_attention_downweights_bad_view(self):
        """A view with larger errors should receive less weight."""
        from pymv.models.lmgecplus import components as C

        good = np.full(50, 0.1)
        bad = np.full(50, 10.0)
        B = C.view_attention([good, bad], lam=1.0)
        assert B[0].mean() > B[1].mean()

    def test_simplex_projection(self):
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(3)
        V = rng.randn(10, 4)
        P = C.simplex_project(V)
        assert np.allclose(P.sum(1), 1.0, atol=1e-6)
        assert P.min() >= -1e-9

    def test_irls_weights_downweight_outliers(self):
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(4)
        H = rng.randn(20, 5)
        T = rng.randn(20, 3)
        W = C.procrustes_W(H, T)
        # inject an outlier row
        H[0] *= 50.0
        u = C.irls_weights(H, T, W)
        assert u[0] < u[1:].mean()

    def test_controlled_propagation_endpoints(self):
        from pymv.models.lmgecplus import components as C
        import scipy.sparse as sp

        rng = np.random.RandomState(5)
        Hn = rng.randn(12, 4)
        S = sp.eye(12).tocsr()  # identity propagation
        # gamma=0 → raw features
        assert np.allclose(C.controlled_propagation(Hn, S, 0.0), Hn)
        # with S=I, any gamma returns Hn
        assert np.allclose(C.controlled_propagation(Hn, S, 1.0), Hn)

    def test_soft_assign_simplex(self):
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(6)
        Z = rng.randn(40, 4)
        F = rng.randn(3, 4)
        R = C.soft_assign(Z, F, beta=10.0)
        assert R.shape == (40, 3)
        assert np.allclose(R.sum(1), 1.0, atol=1e-6)


# ── Backward compatibility: all extensions off == LMGEC ──────────


class TestBackwardCompat:
    def test_all_off_matches_lmgec(self, mock_dataset):
        """LMGECplus with every extension disabled reproduces LMGEC."""
        from pymv.models.lmgec import LMGEC
        from pymv.models.lmgecplus import LMGECplus
        from pymv.metrics import clustering_accuracy, nmi_score

        k = mock_dataset.n_clusters
        base = LMGEC(n_clusters=k, embedding_dim=4, max_iter=20)
        base.fit(mock_dataset.Xs, mock_dataset.As)

        plus = LMGECplus(n_clusters=k, embedding_dim=4, max_iter=20,
                         random_state=0)
        plus.fit(mock_dataset.Xs, mock_dataset.As)

        # Same algorithm → metrics should match within KMeans-init noise.
        a_base = clustering_accuracy(mock_dataset.labels, base.labels_)
        a_plus = clustering_accuracy(mock_dataset.labels, plus.labels_)
        n_base = nmi_score(mock_dataset.labels, base.labels_)
        n_plus = nmi_score(mock_dataset.labels, plus.labels_)
        assert abs(a_base - a_plus) < 0.15
        assert abs(n_base - n_plus) < 0.15

    def test_defaults_are_all_off(self):
        from pymv.models.lmgecplus import LMGECplus

        m = LMGECplus(n_clusters=3, embedding_dim=4)
        assert m.robust_loss is False
        assert m.node_attention is False
        assert m.soft_clustering is False
        assert m.cross_view_alignment is False
        assert m.controlled_propagation is False


# ── Each extension toggle runs and is well-formed ────────────────


def _fit(mock_dataset, **kw):
    from pymv.models.lmgecplus import LMGECplus

    m = LMGECplus(
        n_clusters=mock_dataset.n_clusters,
        embedding_dim=4,
        max_iter=4,
        random_state=0,
        **kw,
    )
    m.fit(mock_dataset.Xs, mock_dataset.As)
    return m


class TestExtensionsRun:
    @pytest.mark.parametrize("kw", [
        {"robust_loss": True},
        {"node_attention": True},
        {"node_attention": True, "dim_normalized_attention": True},
        {"node_attention": True, "dynamic_attention": False},
        {"node_attention": True, "attention_regularization": "entropy"},
        {"node_attention": True, "attention_regularization": "l2"},
        {"controlled_propagation": True, "gamma": 0.5},
        {"controlled_propagation": True, "gamma": "auto"},
        {"soft_clustering": True, "soft_beta": 50.0},
        {"cross_view_alignment": True},
    ])
    def test_single_extension_runs(self, mock_dataset, kw):
        m = _fit(mock_dataset, **kw)
        n = mock_dataset.n_samples
        assert m.labels_.shape == (n,)
        assert m.labels_.min() >= 0
        assert m.labels_.max() < mock_dataset.n_clusters

    def test_combined_extensions_run(self, mock_dataset):
        m = _fit(mock_dataset, robust_loss=True, node_attention=True,
                 dim_normalized_attention=True, cross_view_alignment=True)
        assert m.labels_.shape == (mock_dataset.n_samples,)

    def test_attention_attribute_shape(self, mock_dataset):
        m = _fit(mock_dataset, node_attention=True)
        assert m.attention_.shape == (
            mock_dataset.n_views, mock_dataset.n_samples)
        # per-node attention sums to 1 across views
        assert np.allclose(m.attention_.sum(0), 1.0, atol=1e-6)

    def test_soft_responsibilities_shape(self, mock_dataset):
        m = _fit(mock_dataset, soft_clustering=True)
        assert m.responsibilities_.shape == (
            mock_dataset.n_samples, mock_dataset.n_clusters)
        assert np.allclose(m.responsibilities_.sum(1), 1.0, atol=1e-6)

    def test_gammas_recorded(self, mock_dataset):
        m = _fit(mock_dataset, controlled_propagation=True, gamma=0.5)
        assert m.gammas_ is not None
        assert len(m.gammas_) == mock_dataset.n_views

    def test_loss_history_length(self, mock_dataset):
        m = _fit(mock_dataset, node_attention=True)
        assert len(m.loss_history_) == 4

    def test_fit_predict(self, mock_dataset):
        from pymv.models.lmgecplus import LMGECplus

        m = LMGECplus(n_clusters=mock_dataset.n_clusters, embedding_dim=4,
                      max_iter=3, random_state=0, node_attention=True)
        labels = m.fit_predict(mock_dataset.Xs, mock_dataset.As)
        assert labels.shape == (mock_dataset.n_samples,)


# ── Registry / framework integration ─────────────────────────────


class TestRegistry:
    def test_listed(self):
        from pymv.models import list_models

        assert "lmgecplus" in list_models()

    def test_get_model(self):
        from pymv.models import get_model
        from pymv.models.lmgecplus import LMGECplus

        assert get_model("lmgecplus") is LMGECplus

    def test_does_not_break_lmgec(self):
        from pymv.models import get_model
        from pymv.models.lmgec import LMGEC

        assert get_model("lmgec") is LMGEC

    def test_defaults_have_extension_flags(self):
        from pymv.models import get_model_info

        d = get_model_info("lmgecplus")["defaults"]
        for key in ("robust_loss", "node_attention", "dim_normalized_attention",
                    "dynamic_attention", "attention_regularization",
                    "controlled_propagation", "soft_clustering",
                    "cross_view_alignment"):
            assert key in d

    def test_is_base_model(self):
        from pymv.core import BaseModel
        from pymv.models.lmgecplus import LMGECplus

        assert isinstance(LMGECplus(n_clusters=3, embedding_dim=4), BaseModel)

    def test_name(self):
        from pymv.models.lmgecplus import LMGECplus

        assert LMGECplus(n_clusters=3, embedding_dim=4).name == "LMGECplus"
