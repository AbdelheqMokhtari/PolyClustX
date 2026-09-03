"""
Tests for the modular LMGEC framework (component architecture).

Covers, per module:
* the component factory / registry (specs: str, dict, instance, class),
* every reconstruction loss, normalization, propagation, attention,
  weighting, alignment and clustering-head strategy in isolation,
* the optimizer (early stopping, convergence reporting),
* the LMGECResult object,
* facade equivalence: component API ≡ legacy flag API (exact),
* the ablation-suite builder,
* one-class extensibility (custom components via ``register_component``).

The flag-level behaviour (and LMGEC parity) is covered by
``tests/test_lmgecplus.py``; this file focuses on the strategy layer.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────


def _tiny_state(n=24, dims=(6, 4), k=3, m=3, seed=0):
    """A minimal FitState with projections/errors filled in."""
    from pymv.models.lmgecplus import FitState

    rng = np.random.RandomState(seed)
    Hs = [rng.randn(n, d) for d in dims]
    state = FitState(Hs=Hs, n_clusters=k, embedding_dim=m, random_state=0)
    state.Ws = [rng.randn(d, m) for d in dims]
    state.Zs = [H @ W for H, W in zip(state.Hs, state.Ws)]
    state.consensus = sum(state.Zs) / len(state.Zs)
    state.Z_prev = state.consensus.copy()
    state.F = rng.randn(k, m)
    state.G = rng.randint(0, k, size=n).astype(np.int32)
    state.errors = [rng.rand(n) for _ in dims]
    state.residual_norms = [np.sqrt(e) for e in state.errors]
    state.us = [np.ones(n) for _ in dims]
    state.view_node_weights = [np.full(n, 1.0 / len(dims)) for _ in dims]
    return state


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


# ── factory / registry ────────────────────────────────────────────


class TestFactory:
    def test_list_components_all_kinds(self):
        from pymv.models.lmgecplus import list_components

        listing = list_components()
        for kind in ("loss", "normalization", "propagation", "attention",
                     "weighting", "alignment", "clustering", "optimizer"):
            assert listing[kind], f"no components registered for {kind}"

    def test_resolve_string(self):
        from pymv.models.lmgecplus import L21Loss, resolve_component

        assert isinstance(resolve_component("loss", "l21"), L21Loss)

    def test_resolve_alias(self):
        from pymv.models.lmgecplus import L21Loss, resolve_component

        assert isinstance(resolve_component("loss", "robust"), L21Loss)

    def test_resolve_instance_passthrough(self):
        from pymv.models.lmgecplus import HuberLoss, resolve_component

        inst = HuberLoss(delta=2.0)
        assert resolve_component("loss", inst) is inst

    def test_resolve_dict_with_params(self):
        from pymv.models.lmgecplus import SoftKMeansHead, resolve_component

        head = resolve_component("clustering", {"method": "soft", "beta": 7.0})
        assert isinstance(head, SoftKMeansHead)
        assert head.beta == 7.0

    def test_resolve_dict_enabled_false(self):
        from pymv.models.lmgecplus import NoPropagation, resolve_component

        prop = resolve_component("propagation", {"enabled": False})
        assert isinstance(prop, NoPropagation)

    def test_resolve_dict_enabled_true_defaults_to_controlled(self):
        from pymv.models.lmgecplus import ControlledPropagation, resolve_component

        prop = resolve_component("propagation", {"enabled": True, "gamma": 0.4})
        assert isinstance(prop, ControlledPropagation)
        assert prop.gamma == 0.4

    def test_resolve_none_uses_default(self):
        from pymv.models.lmgecplus import L2Loss, resolve_component

        assert isinstance(resolve_component("loss", None, default="l2"), L2Loss)

    def test_unknown_name_raises_with_available(self):
        from pymv.models.lmgecplus import resolve_component

        with pytest.raises(ValueError, match="Available"):
            resolve_component("loss", "nope")

    def test_duplicate_registration_raises(self):
        from pymv.models.lmgecplus import register_component

        with pytest.raises(ValueError, match="already registered"):
            @register_component("loss", "l2")
            class Dup:  # pragma: no cover
                pass


# ── reconstruction losses ─────────────────────────────────────────


class TestLosses:
    def test_l2_is_trivial(self):
        from pymv.models.lmgecplus import L2Loss

        r = np.array([0.5, 2.0, 4.0])
        loss = L2Loss()
        assert not loss.reweights_nodes
        assert np.allclose(loss.penalty(r), r ** 2)
        assert np.allclose(loss.irls_node_weights(r), 1.0)

    def test_l21_matches_components_irls(self):
        """L21Loss must reproduce the historical C.irls_weights exactly."""
        from pymv.models.lmgecplus import L21Loss
        from pymv.models.lmgecplus import components as C

        rng = np.random.RandomState(0)
        H, T = rng.randn(20, 6), rng.randn(20, 3)
        W = C.procrustes_W(H, T)
        rown = np.sqrt(((H - T @ W.T) ** 2).sum(1))
        assert np.array_equal(L21Loss().irls_node_weights(rown),
                              C.irls_weights(H, T, W))

    def test_l21_downweights_outliers(self):
        from pymv.models.lmgecplus import L21Loss

        r = np.array([0.1, 0.1, 10.0])
        u = L21Loss().irls_node_weights(r)
        assert u[2] < u[0]

    def test_huber_regions(self):
        from pymv.models.lmgecplus import HuberLoss

        loss = HuberLoss(delta=1.0, normalize=False)
        r = np.array([0.5, 1.0, 3.0])
        rho = loss.penalty(r)
        assert rho[0] == pytest.approx(0.125)          # quadratic: r²/2
        assert rho[2] == pytest.approx(1.0 * (3.0 - 0.5))  # linear
        u = loss.irls_node_weights(r)
        assert u[0] == pytest.approx(1.0)              # inlier: full weight
        assert u[2] == pytest.approx(1.0 / 3.0)        # outlier: δ/r

    def test_huber_auto_delta(self):
        from pymv.models.lmgecplus import HuberLoss

        r = np.array([1.0, 2.0, 3.0, 100.0])
        u = HuberLoss(delta="auto", normalize=False).irls_node_weights(r)
        assert u[3] < u[0]  # outlier is downweighted vs the median scale


# ── normalization ─────────────────────────────────────────────────


class TestNormalization:
    def test_none_passthrough(self):
        from pymv.models.lmgecplus import NoNormalization

        X = np.random.RandomState(0).randn(10, 4)
        assert NoNormalization().transform(X) is X

    def test_l2_unit_rows(self):
        from pymv.models.lmgecplus import L2Normalization

        X = np.random.RandomState(1).randn(15, 5)
        Xn = L2Normalization().transform(X)
        assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0)

    def test_zscore(self):
        from pymv.models.lmgecplus import ZScoreNormalization

        X = np.random.RandomState(2).randn(50, 4) * 3 + 5
        Xn = ZScoreNormalization().transform(X)
        assert np.allclose(Xn.mean(0), 0.0, atol=1e-10)
        assert np.allclose(Xn.std(0), 1.0, atol=1e-10)

    def test_minmax_range_and_constant_column(self):
        from pymv.models.lmgecplus import MinMaxNormalization

        X = np.random.RandomState(3).randn(20, 3)
        X[:, 2] = 7.0  # constant column must not divide by zero
        Xn = MinMaxNormalization().transform(X)
        assert Xn.min() >= 0.0 and Xn.max() <= 1.0
        assert np.all(np.isfinite(Xn))

    def test_whitening_identity_covariance(self):
        from pymv.models.lmgecplus import WhiteningNormalization

        X = np.random.RandomState(4).randn(200, 5) @ np.diag([5, 3, 1, 1, 0.5])
        Xw = WhiteningNormalization().transform(X)
        cov = (Xw - Xw.mean(0)).T @ (Xw - Xw.mean(0)) / (len(Xw) - 1)
        assert np.allclose(cov, np.eye(Xw.shape[1]), atol=1e-6)

    def test_pipeline_matches_legacy_preprocessing(self):
        """zscore → l2 pipeline ≡ get_propagated_features (no graph)."""
        from pymv.models.lmgecplus import (
            L2Normalization,
            NormalizationPipeline,
            ZScoreNormalization,
        )
        from pymv.preprocessing import get_propagated_features

        X = np.random.RandomState(5).rand(30, 8)
        pipe = NormalizationPipeline(
            [ZScoreNormalization(center=True, scale=True), L2Normalization()])
        expected = get_propagated_features(X, adj=None, center=True, scale=True)
        assert np.array_equal(pipe.transform(X), expected)

    def test_sparse_input(self):
        import scipy.sparse as sp

        from pymv.models.lmgecplus import L2Normalization, MinMaxNormalization

        X = sp.random(20, 6, density=0.3, random_state=6, format="csr")
        assert L2Normalization().transform(X).shape == (20, 6)
        assert MinMaxNormalization().transform(X).shape == (20, 6)


# ── graph propagation ─────────────────────────────────────────────


class TestPropagation:
    def _data(self):
        rng = np.random.RandomState(7)
        X = rng.rand(25, 5)
        A = (rng.rand(25, 25) > 0.7).astype(float)
        A = (A + A.T) / 2
        return X, A

    def test_none_ignores_adjacency(self):
        from pymv.models.lmgecplus import NoPropagation

        X, A = self._data()
        H, g = NoPropagation().propagate(X, A)
        assert np.array_equal(H, X)
        assert g == 0.0

    def test_standard_matches_legacy(self):
        from pymv.models.lmgecplus import StandardPropagation
        from pymv.preprocessing import preprocess_graph

        X, A = self._data()
        H, g = StandardPropagation(beta=2.0).propagate(X, A)
        S = preprocess_graph(A, beta=2.0)
        assert np.allclose(H, np.asarray((S @ X)))
        assert g == 1.0

    def test_standard_without_graph(self):
        from pymv.models.lmgecplus import StandardPropagation

        X, _ = self._data()
        H, g = StandardPropagation().propagate(X, None)
        assert np.array_equal(H, X)
        assert g == 0.0

    def test_controlled_endpoints(self):
        from pymv.models.lmgecplus import ControlledPropagation, StandardPropagation

        X, A = self._data()
        H0, g0 = ControlledPropagation(gamma=0.0).propagate(X, A)
        assert np.allclose(H0, X) and g0 == 0.0
        H1, g1 = ControlledPropagation(gamma=1.0).propagate(X, A)
        Hs, _ = StandardPropagation().propagate(X, A)
        assert np.allclose(H1, Hs) and g1 == 1.0

    def test_controlled_auto_gamma_in_range(self):
        from pymv.models.lmgecplus import ControlledPropagation

        X, A = self._data()
        _, g = ControlledPropagation(gamma="auto").propagate(X, A)
        assert 0.0 <= g <= 1.0


# ── attention ─────────────────────────────────────────────────────


class TestAttention:
    def test_node_attention_simplex(self):
        from pymv.models.lmgecplus import NodeLevelAttention

        state = _tiny_state()
        B = NodeLevelAttention().compute(state.errors, state)
        stacked = np.stack(B, axis=0)
        assert stacked.shape == (state.n_views, state.n_samples)
        assert np.allclose(stacked.sum(0), 1.0)

    def test_node_attention_downweights_bad_view(self):
        from pymv.models.lmgecplus import NodeLevelAttention

        state = _tiny_state()
        errs = [np.full(state.n_samples, 0.1), np.full(state.n_samples, 5.0)]
        B = NodeLevelAttention().compute(errs, state)
        assert B[0].mean() > B[1].mean()

    def test_dim_normalization_changes_allocation(self):
        from pymv.models.lmgecplus import NodeLevelAttention

        state = _tiny_state(dims=(50, 2))
        errs = [np.full(state.n_samples, 1.0), np.full(state.n_samples, 0.5)]
        plain = NodeLevelAttention(dim_normalized=False).compute(errs, state)
        dimn = NodeLevelAttention(dim_normalized=True).compute(errs, state)
        # normalizing by d=50 vs d=2 must flip the preference
        assert plain[1].mean() > plain[0].mean()
        assert dimn[0].mean() > dimn[1].mean()

    def test_static_preset_is_not_dynamic(self):
        from pymv.models.lmgecplus import StaticAttention

        assert StaticAttention().dynamic is False

    def test_view_level_is_constant_per_view(self):
        from pymv.models.lmgecplus import ViewLevelAttention

        state = _tiny_state()
        B = ViewLevelAttention().compute(state.errors, state)
        for b in B:
            assert np.ptp(b) == 0.0  # one scalar broadcast per view
        assert np.allclose(np.stack(B, 0).sum(0), 1.0)


# ── view weighting ────────────────────────────────────────────────


class TestWeighting:
    def test_uniform(self):
        from pymv.models.lmgecplus import UniformWeighting

        state = _tiny_state()
        w = UniformWeighting().initial_weights(state)
        assert all(np.allclose(b, 0.5) for b in w)

    def test_inertia_start_weights_normalized(self):
        from pymv.models.lmgecplus import InertiaWeighting

        state = _tiny_state()
        weighting = InertiaWeighting(temperature=1.0)
        state.view_node_weights = weighting.initial_weights(state)
        w = weighting.start_weights(state)
        total = sum(b[0] for b in w)
        assert total == pytest.approx(1.0)
        # fixed strategy: update() keeps them unchanged
        state.view_node_weights = w
        assert weighting.update(state) is w

    def test_attention_weighting_static_frozen(self):
        from pymv.models.lmgecplus import AttentionWeighting, StaticAttention

        state = _tiny_state()
        weighting = AttentionWeighting(attention=StaticAttention())
        state.view_node_weights = weighting.initial_weights(state)
        frozen = weighting.start_weights(state)
        state.errors = [e * 10 for e in state.errors]  # new errors…
        assert weighting.update(state) is frozen        # …ignored

    def test_attention_weighting_dynamic_reacts(self):
        from pymv.models.lmgecplus import AttentionWeighting

        state = _tiny_state()
        weighting = AttentionWeighting(attention="node")
        state.view_node_weights = weighting.initial_weights(state)
        state.view_node_weights = weighting.start_weights(state)
        state.errors = [np.full(state.n_samples, 0.1),
                        np.full(state.n_samples, 9.0)]
        w = weighting.update(state)
        assert w[0].mean() > w[1].mean()

    def test_entropy_weighting_scalar_per_view(self):
        from pymv.models.lmgecplus import EntropyWeighting

        state = _tiny_state()
        weighting = EntropyWeighting(lam=1.0)
        state.view_node_weights = weighting.initial_weights(state)
        state.view_node_weights = weighting.start_weights(state)
        state.errors = [np.full(state.n_samples, 0.1),
                        np.full(state.n_samples, 5.0)]
        w = weighting.update(state)
        assert all(np.ptp(b) == 0.0 for b in w)
        assert w[0][0] > w[1][0]


# ── alignment ─────────────────────────────────────────────────────


class TestAlignment:
    def test_none(self):
        from pymv.models.lmgecplus import NoAlignment

        state = _tiny_state()
        T = state.F[state.G]
        assert NoAlignment().augment_target(T, state, 0) is T

    def test_consensus_matches_legacy_formula(self):
        from pymv.models.lmgecplus import ConsensusAlignment

        state = _tiny_state()
        T = state.F[state.G]
        out = ConsensusAlignment(mu=0.3).augment_target(T, state, 0)
        assert np.array_equal(out, T + 0.3 * state.Z_prev)

    def test_cosine_uses_directions(self):
        from pymv.models.lmgecplus import CosineAlignment

        state = _tiny_state()
        T = state.F[state.G]
        out = CosineAlignment(mu=0.5).augment_target(T, state, 0)
        norms = np.linalg.norm(state.Z_prev, axis=1, keepdims=True)
        expected = T + 0.5 * float(norms.mean()) * (state.Z_prev / norms)
        assert np.allclose(out, expected)
        assert np.isfinite(CosineAlignment().penalty(state))

    def test_contrastive_centers_consensus(self):
        from pymv.models.lmgecplus import ContrastiveAlignment

        state = _tiny_state()
        T = state.F[state.G]
        out = ContrastiveAlignment(mu=1.0).augment_target(T, state, 0)
        expected = T + (state.Z_prev - state.Z_prev.mean(0, keepdims=True))
        assert np.allclose(out, expected)

    def test_no_zprev_is_noop(self):
        from pymv.models.lmgecplus import ConsensusAlignment, CosineAlignment

        state = _tiny_state()
        state.Z_prev = None
        T = state.F[state.G]
        assert ConsensusAlignment().augment_target(T, state, 0) is T
        assert CosineAlignment().augment_target(T, state, 0) is T


# ── clustering heads ──────────────────────────────────────────────


class TestClusteringHeads:
    def test_hard_head(self):
        from pymv.models.lmgecplus import HardKMeansHead

        state = _tiny_state()
        head = HardKMeansHead()
        assert not head.provides_soft
        assert np.array_equal(head.shared_target(state), state.F[state.G])
        G, F, R = head.refit(state, None)
        assert G.shape == (state.n_samples,)
        assert F.shape == (state.n_clusters, state.embedding_dim)
        assert R is None

    def test_soft_head(self):
        from pymv.models.lmgecplus import SoftKMeansHead

        state = _tiny_state()
        head = SoftKMeansHead(beta=10.0)
        G, F, R = head.refit(state, None)
        assert R.shape == (state.n_samples, state.n_clusters)
        assert np.allclose(R.sum(1), 1.0)
        assert np.array_equal(G, R.argmax(1))
        # soft target after R exists
        state.R = R
        assert np.allclose(head.shared_target(state), R @ state.F)

    def test_gmm_head(self):
        from pymv.models.lmgecplus import GMMHead

        state = _tiny_state(n=45)
        head = GMMHead(covariance_type="diag")
        assert head.provides_soft
        G, F, R = head.refit(state, None)
        assert R.shape == (state.n_samples, state.n_clusters)
        assert np.allclose(R.sum(1), 1.0, atol=1e-6)
        assert F.shape == (state.n_clusters, state.embedding_dim)

    def test_shared_kmeans_initialize(self):
        from pymv.models.lmgecplus import GMMHead, HardKMeansHead

        state = _tiny_state()
        G1, F1, R1 = HardKMeansHead().initialize(state.consensus, state)
        G2, F2, R2 = GMMHead().initialize(state.consensus, state)
        # all heads share the same (comparable) K-Means initialisation
        assert np.array_equal(G1, G2)
        assert np.array_equal(F1, F2)
        assert R1 is None and R2 is None


# ── optimizer ─────────────────────────────────────────────────────


class TestOptimizer:
    def test_runs_max_iter_by_default(self, mock_dataset):
        m = _fit(mock_dataset)
        assert m.results_.convergence["n_iter"] == 4
        assert len(m.loss_history_) == 4

    def test_tol_early_stopping(self, mock_dataset):
        from pymv.models.lmgecplus import LMGECplus

        m = LMGECplus(n_clusters=mock_dataset.n_clusters, embedding_dim=4,
                      max_iter=50, tol=1e-3, random_state=0)
        m.fit(mock_dataset.Xs, mock_dataset.As)
        conv = m.results_.convergence
        assert conv["converged"] is True
        assert conv["n_iter"] < 50
        assert len(m.loss_history_) == conv["n_iter"]

    def test_convergence_dict_fields(self, mock_dataset):
        conv = _fit(mock_dataset).results_.convergence
        assert set(conv) == {"converged", "n_iter", "final_loss"}
        assert np.isfinite(conv["final_loss"])


# ── result object ─────────────────────────────────────────────────


class TestResultObject:
    def test_fields(self, mock_dataset):
        m = _fit(mock_dataset, attention="dynamic", clustering="soft")
        r = m.results_
        n, k = mock_dataset.n_samples, mock_dataset.n_clusters
        assert r.labels.shape == (n,)
        assert r.embedding.shape == (n, 4)
        assert r.centroids.shape == (k, 4)
        assert r.view_weights.shape == (mock_dataset.n_views,)
        assert r.attention_weights.shape == (mock_dataset.n_views, n)
        assert r.cluster_probabilities.shape == (n, k)
        assert len(r.reconstruction_errors) == mock_dataset.n_views
        assert len(r.view_projections) == mock_dataset.n_views
        assert r.training_time > 0
        assert r.components["clustering"].startswith("SoftKMeansHead")

    def test_matches_legacy_attributes(self, mock_dataset):
        m = _fit(mock_dataset)
        r = m.results_
        assert r.labels is m.labels_
        assert r.embedding is m.embedding_
        assert np.array_equal(r.view_weights, m.alphas_)
        assert np.array_equal(r.loss_history, m.loss_history_)

    def test_memory_tracking(self, mock_dataset):
        m = _fit(mock_dataset, track_memory=True)
        assert m.results_.memory_usage > 0
        m2 = _fit(mock_dataset)
        assert m2.results_.memory_usage is None

    def test_to_dict_json_serializable(self, mock_dataset):
        import json

        payload = _fit(mock_dataset).results_.to_dict()
        assert json.dumps(payload)

    def test_summary_is_string(self, mock_dataset):
        s = _fit(mock_dataset).results_.summary()
        assert "LMGECResult" in s and "view weights" in s


# ── facade: component API ≡ flag API ──────────────────────────────


class TestFacadeEquivalence:
    @pytest.mark.parametrize("flags,specs", [
        ({"robust_loss": True}, {"reconstruction_loss": "l21"}),
        ({"soft_clustering": True, "soft_beta": 50.0}, {"clustering": "soft"}),
        ({"cross_view_alignment": True, "align_mu": 0.1},
         {"alignment": "consensus"}),
        ({"controlled_propagation": True, "gamma": 0.5},
         {"graph_propagation": {"enabled": True, "gamma": 0.5}}),
        ({"node_attention": True},
         {"attention": "dynamic"}),
        ({"node_attention": True, "dynamic_attention": False},
         {"attention": "static"}),
    ])
    def test_flag_equals_component_spec(self, mock_dataset, flags, specs):
        a = _fit(mock_dataset, **flags)
        b = _fit(mock_dataset, **specs)
        assert np.array_equal(a.labels_, b.labels_)
        assert np.array_equal(a.loss_history_, b.loss_history_)
        assert np.array_equal(a.embedding_, b.embedding_)

    def test_component_override_beats_flag(self, mock_dataset):
        """An explicit component spec wins over a conflicting flag."""
        from pymv.models.lmgecplus import L2Loss

        m = _fit(mock_dataset, robust_loss=True, reconstruction_loss="l2")
        assert isinstance(m._build_loss(), L2Loss)

    def test_per_view_normalization(self, mock_dataset):
        m = _fit(mock_dataset, normalization=["zscore", "l2"])
        assert m.labels_.shape == (mock_dataset.n_samples,)

    def test_gammas_reported_for_component_spec(self, mock_dataset):
        m = _fit(mock_dataset,
                 graph_propagation={"enabled": True, "gamma": 0.3})
        assert m.gammas_ == [0.3] * mock_dataset.n_views

    def test_defaults_reproduce_original_lmgec_path(self, mock_dataset):
        """No flags, no specs → classic LMGEC components."""
        from pymv.models.lmgecplus import (
            AlternatingOptimizer,
            HardKMeansHead,
            InertiaWeighting,
            L2Loss,
            NoAlignment,
            StandardPropagation,
        )

        m = _fit(mock_dataset)
        bundle = m._build_components(mock_dataset.n_views)
        assert isinstance(bundle.loss, L2Loss)
        assert isinstance(bundle.weighting, InertiaWeighting)
        assert isinstance(bundle.alignment, NoAlignment)
        assert isinstance(bundle.clustering, HardKMeansHead)
        assert isinstance(bundle.optimizer, AlternatingOptimizer)
        assert all(isinstance(p, StandardPropagation)
                   for p in bundle.propagations)


# ── ablation suite ────────────────────────────────────────────────


class TestAblationSuite:
    def test_default_ladder(self):
        from pymv.models.lmgecplus import DEFAULT_ABLATIONS, build_ablation_suite

        suite = build_ablation_suite(n_clusters=3, embedding_dim=4)
        assert list(suite) == list(DEFAULT_ABLATIONS)
        assert all(m.name == name for name, m in suite.items())

    def test_include_filter(self):
        from pymv.models.lmgecplus import build_ablation_suite

        suite = build_ablation_suite(
            n_clusters=3, embedding_dim=4,
            include=["baseline", "E7 soft-clustering"])
        assert list(suite) == ["baseline", "E7 soft-clustering"]

    def test_ladder_covers_all_eight_extensions(self):
        from pymv.models.lmgecplus import DEFAULT_ABLATIONS

        assert list(DEFAULT_ABLATIONS)[0] == "baseline"
        assert list(DEFAULT_ABLATIONS)[-1] == "full"
        numbers = [n.split()[0] for n in DEFAULT_ABLATIONS
                   if n.startswith("E")]
        assert numbers == [f"E{i}" for i in range(1, 9)]
        # the baseline really is the original LMGEC (no overrides)
        assert DEFAULT_ABLATIONS["baseline"] == {}

    def test_unknown_variant_raises(self):
        from pymv.models.lmgecplus import build_ablation_suite

        with pytest.raises(ValueError, match="Unknown ablation"):
            build_ablation_suite(n_clusters=3, include=["nope"])

    def test_base_kwargs_shared_and_overridable(self):
        from pymv.models.lmgecplus import build_ablation_suite

        suite = build_ablation_suite(
            n_clusters=3, embedding_dim=4, max_iter=7, random_state=1)
        assert all(m.max_iter == 7 for m in suite.values())
        assert suite["E1 robust-l21"].reconstruction_loss == "l21"

    def test_suite_runs(self, mock_dataset):
        from pymv.models.lmgecplus import build_ablation_suite

        suite = build_ablation_suite(
            n_clusters=mock_dataset.n_clusters, embedding_dim=4,
            max_iter=2, random_state=0,
            include=["baseline", "full"])
        for m in suite.values():
            m.fit(mock_dataset.Xs, mock_dataset.As)
            assert m.labels_.shape == (mock_dataset.n_samples,)


# ── extensibility: one class, no library changes ──────────────────


class TestExtensibility:
    def test_custom_loss(self, mock_dataset):
        from pymv.models.lmgecplus import ReconstructionLoss, register_component

        @register_component("loss", "_test_cauchy")
        class CauchyLoss(ReconstructionLoss):
            def __init__(self, c=1.0):
                self.c = c

            def penalty(self, r):
                return 0.5 * self.c ** 2 * np.log1p((r / self.c) ** 2)

            def irls_node_weights(self, r):
                return 1.0 / (1.0 + (r / self.c) ** 2)

        m = _fit(mock_dataset, reconstruction_loss="_test_cauchy")
        assert m.labels_.shape == (mock_dataset.n_samples,)

    def test_custom_attention_instance(self, mock_dataset):
        from pymv.models.lmgecplus import Attention

        class FlatAttention(Attention):
            dynamic = True

            def compute(self, errors, state):
                n, V = state.n_samples, state.n_views
                return [np.full(n, 1.0 / V) for _ in range(V)]

        m = _fit(mock_dataset, attention=FlatAttention())
        assert np.allclose(m.attention_, 0.5)

    def test_custom_alignment_instance(self, mock_dataset):
        from pymv.models.lmgecplus import Alignment

        class ShrinkAlignment(Alignment):
            def augment_target(self, T_shared, state, view):
                return 0.9 * T_shared

        m = _fit(mock_dataset, alignment=ShrinkAlignment())
        assert m.labels_.shape == (mock_dataset.n_samples,)


# ── framework integration ─────────────────────────────────────────


class TestFrameworkIntegration:
    def test_registry_defaults_expose_component_params(self):
        from pymv.models import get_model_info

        d = get_model_info("lmgecplus")["defaults"]
        for key in ("reconstruction_loss", "normalization",
                    "graph_propagation", "attention", "view_weighting",
                    "alignment", "clustering", "optimizer", "tol"):
            assert key in d

    def test_experiment_runner_compat(self, mock_dataset):
        """Deep-copied models with component specs run in Experiment."""
        import copy

        m = _fit(mock_dataset, clustering="soft")
        clone = copy.deepcopy(m)
        clone.fit(mock_dataset.Xs, mock_dataset.As)
        assert clone.labels_.shape == (mock_dataset.n_samples,)
