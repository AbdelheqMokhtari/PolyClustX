"""
Tests for pymv models: LMGEC, registry, BaseModel.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Model initialisation ─────────────────────────────────────────


class TestLMGECInit:
    def test_default_params(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=8)
        assert m.n_clusters == 3
        assert m.embedding_dim == 8
        assert m.temperature == 1.0
        assert m.beta == 1.0
        assert m.tf_idf is False
        assert m.center is False
        assert m.scale is False
        assert m.preprocess is True
        assert m.max_iter == 30

    def test_custom_params(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=5,
            embedding_dim=32,
            temperature=10.0,
            beta=2.0,
            tf_idf=True,
            center=True,
            scale=True,
            preprocess=False,
            max_iter=50,
        )
        assert m.n_clusters == 5
        assert m.embedding_dim == 32
        assert m.temperature == 10.0
        assert m.preprocess is False
        assert m.max_iter == 50

    def test_per_view_beta(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=4, beta=[1.0, 2.0])
        assert m.beta == [1.0, 2.0]

    def test_per_view_tf_idf(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=4, tf_idf=[True, False])
        assert m.tf_idf == [True, False]

    def test_is_base_model(self):
        from pymv.core import BaseModel
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=4)
        assert isinstance(m, BaseModel)


# ── Fit & predict ─────────────────────────────────────────────────


class TestLMGECFit:
    def test_fit_returns_self(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = m.fit(mock_dataset.Xs)
        assert result is m

    def test_labels_shape(self, mock_model, mock_dataset):
        assert mock_model.labels_.shape == (mock_dataset.n_samples,)

    def test_labels_dtype_int(self, mock_model):
        assert np.issubdtype(mock_model.labels_.dtype, np.integer)

    def test_labels_in_range(self, mock_model, mock_dataset):
        assert mock_model.labels_.min() >= 0
        assert mock_model.labels_.max() < mock_dataset.n_clusters

    def test_centroids_shape(self, mock_model, mock_dataset):
        assert mock_model.centroids_.shape == (
            mock_dataset.n_clusters,
            4,  # embedding_dim
        )

    def test_embedding_shape(self, mock_model, mock_dataset):
        assert mock_model.embedding_.shape == (mock_dataset.n_samples, 4)

    def test_loss_history_shape(self, mock_model):
        assert mock_model.loss_history_.ndim == 1
        assert len(mock_model.loss_history_) == 3  # max_iter=3

    def test_fit_predict(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        labels = m.fit_predict(mock_dataset.Xs)
        assert labels.shape == (mock_dataset.n_samples,)

    def test_fit_with_preprocess(self, mock_dataset):
        """Fit with preprocess=True and real adjacency matrices."""
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=True,
        )
        m.fit(mock_dataset.Xs, mock_dataset.As)
        assert m.labels_ is not None
        assert m.labels_.shape[0] == mock_dataset.n_samples

    def test_fit_without_adjacency(self, mock_dataset):
        """preprocess=True but As=None → normalise without propagation."""
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=True,
        )
        m.fit(mock_dataset.Xs, As=None)
        assert m.labels_ is not None

    def test_per_view_beta_fit(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            beta=[1.0, 2.0],
        )
        m.fit(mock_dataset.Xs, mock_dataset.As)
        assert m.labels_ is not None

    def test_wrong_beta_length_raises(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            beta=[1.0, 2.0, 3.0],  # 3 betas but 2 views
        )
        with pytest.raises(ValueError, match="beta"):
            m.fit(mock_dataset.Xs, mock_dataset.As)


# ── Model name property ──────────────────────────────────────────


class TestLMGECName:
    def test_default_name(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=4)
        assert m.name == "LMGEC"

    def test_custom_name(self):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(n_clusters=3, embedding_dim=4)
        m.name = "my_model"
        assert m.name == "my_model"


# ── Model registry ───────────────────────────────────────────────


class TestModelRegistry:
    def test_list_models_contains_lmgec(self):
        from pymv.models import list_models

        models = list_models()
        assert "lmgec" in models

    def test_get_model_returns_class(self):
        from pymv.models import get_model
        from pymv.models.lmgec import LMGEC

        cls = get_model("lmgec")
        assert cls is LMGEC

    def test_get_model_case_insensitive(self):
        from pymv.models import get_model

        cls1 = get_model("lmgec")
        cls2 = get_model("LMGEC")
        assert cls1 is cls2

    def test_get_model_unknown_raises(self):
        from pymv.models import get_model

        with pytest.raises(ValueError, match="not registered"):
            get_model("nonexistent_model")

    def test_get_model_info(self):
        from pymv.models import get_model_info

        info = get_model_info("lmgec")
        assert "class" in info
        assert "defaults" in info
        assert isinstance(info["defaults"], dict)

    def test_defaults_contain_expected_keys(self):
        from pymv.models import get_model_info

        info = get_model_info("lmgec")
        expected = {
            "embedding_dim", "temperature", "beta", "tf_idf",
            "center", "scale", "preprocess", "max_iter",
        }
        assert expected.issubset(info["defaults"].keys())

    def test_list_models_info(self):
        from pymv.models import list_models_info

        registry = list_models_info()
        assert isinstance(registry, dict)
        assert "lmgec" in registry

    def test_register_duplicate_raises(self):
        from pymv.models.registry import register_model

        with pytest.raises(ValueError, match="already registered"):
            @register_model("lmgec")
            class Dummy:
                pass


class TestLMGECReproducibility:
    """LMGEC must produce identical labels for a fixed ``random_state``."""

    def test_same_seed_gives_same_labels(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        def _run():
            return LMGEC(
                n_clusters=mock_dataset.n_clusters,
                embedding_dim=4,
                max_iter=3,
                preprocess=False,
                random_state=42,
            ).fit_predict(mock_dataset.Xs)

        assert np.array_equal(_run(), _run())

    def test_random_state_is_a_registered_default(self):
        from pymv.models import get_model_info

        assert "random_state" in get_model_info("lmgec")["defaults"]


class TestLMGECEmbeddingDimClamp:
    """``embedding_dim`` above a view's rank used to allocate a
    consensus buffer wider than the per-view projections, raising a
    broadcast ValueError."""

    def test_oversized_embedding_dim_fits(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=99,          # views are 8- and 6-dimensional
            max_iter=3,
            preprocess=False,
            random_state=0,
        )
        m.fit(mock_dataset.Xs)

        # Capped by the lowest-rank view (6 features → 5 components).
        assert m.embedding_dim_ == 5
        assert m.embedding_.shape == (mock_dataset.n_samples, 5)
        assert m.labels_.shape == (mock_dataset.n_samples,)

    def test_effective_dim_reported_when_not_clamped(self, mock_dataset):
        from pymv.models.lmgec import LMGEC

        m = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
            random_state=0,
        )
        m.fit(mock_dataset.Xs)
        assert m.embedding_dim_ == 4
