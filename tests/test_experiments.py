"""
Tests for pymv Experiment and RandomSearchExperiment.
"""

from __future__ import annotations

import os

# ── Experiment ────────────────────────────────────────────────────


class TestExperiment:
    """Test Experiment with mock data (no real dataset loading)."""

    def test_run_returns_experiment_result(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.experiment.result import ExperimentResult
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()

        assert isinstance(result, ExperimentResult)

    def test_result_has_correct_dataset(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()

        assert "mock" in result.all_results

    def test_result_has_metrics(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()

        for r in result.all_results["mock"]:
            assert "clustering_accuracy" in r.metric_avg
            assert "nmi_score" in r.metric_avg

    def test_multiple_runs_produce_std(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=2,
            verbose=0,
        ).run()

        for r in result.all_results["mock"]:
            assert r.metric_std is not None

    def test_str_output(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()

        text = str(result)
        assert "LMGEC" in text
        assert "mock" in text

    def test_save_creates_files(self, mock_dataset, metrics_fns, tmp_dir):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=mock_dataset.n_clusters,
            embedding_dim=4,
            max_iter=3,
            preprocess=False,
        )
        Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
            save_markdown=True,
            experiment_name="test_exp",
            save_path=tmp_dir,
        ).run()

        exp_dir = os.path.join(tmp_dir, "test_exp")
        assert os.path.isdir(exp_dir)
        assert os.path.isfile(os.path.join(exp_dir, "results.md"))
        assert os.path.isfile(os.path.join(exp_dir, "config.yaml"))
        assert os.path.isfile(os.path.join(exp_dir, "results.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metrics.csv"))


# ── Experiment with flexible input formats ────────────────────────


class TestExperimentInputFormats:
    def test_dataset_as_tuple(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        Xs, As, labels = mock_dataset
        model = LMGEC(
            n_clusters=3, embedding_dim=4, max_iter=3, preprocess=False,
        )
        result = Experiment(
            datasets={"mock": (Xs, As, labels)},
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()
        assert result.all_results

    def test_dataset_as_multiview_object(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=3, embedding_dim=4, max_iter=3, preprocess=False,
        )
        result = Experiment(
            datasets=mock_dataset,
            models=[model],
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()
        assert result.all_results

    def test_single_model(self, mock_dataset, metrics_fns):
        from pymv.experiment import Experiment
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=3, embedding_dim=4, max_iter=3, preprocess=False,
        )
        # single model, not wrapped in list
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=model,
            metrics=metrics_fns,
            runs=1,
            verbose=0,
        ).run()
        assert result.all_results

    def test_metrics_as_list(self, mock_dataset):
        from pymv.experiment import Experiment
        from pymv.metrics import clustering_accuracy, nmi_score
        from pymv.models.lmgec import LMGEC

        model = LMGEC(
            n_clusters=3, embedding_dim=4, max_iter=3, preprocess=False,
        )
        result = Experiment(
            datasets={"mock": mock_dataset},
            models=[model],
            metrics=[clustering_accuracy, nmi_score],
            runs=1,
            verbose=0,
        ).run()
        assert result.all_results


# ── Result objects ────────────────────────────────────────────────


class TestResult:
    def test_result_creation(self):
        from pymv.experiment.result import Result

        r = Result(
            dataset_name="acm",
            model_name="LMGEC",
            metric_avg={"acc": 0.9, "nmi": 0.8},
            train_time=1.5,
        )
        assert r.dataset_name == "acm"
        assert r.model_name == "LMGEC"
        assert r.metric_avg["acc"] == 0.9
        assert r.train_time == 1.5

    def test_result_format_cell(self):
        from pymv.experiment.result import Result

        r = Result(
            dataset_name="acm",
            model_name="LMGEC",
            metric_avg={"acc": 0.912345},
        )
        cell = r.format_cell("acc")
        assert "0.9123" in cell

    def test_result_repr(self):
        from pymv.experiment.result import Result

        r = Result(
            dataset_name="acm",
            model_name="LMGEC",
            metric_avg={"acc": 0.9},
        )
        assert "acm" in repr(r)
        assert "LMGEC" in repr(r)


class TestRandomSearchResult:
    def test_add_trial(self):
        from pymv.experiment.result import RandomSearchResult

        rs = RandomSearchResult("acm", "LMGEC", "acc")
        rs.add_trial({"beta": 1.0}, {"acc": 0.8, "nmi": 0.7}, 1.0)
        rs.add_trial({"beta": 2.0}, {"acc": 0.9, "nmi": 0.6}, 1.5)

        assert len(rs.trials) == 2
        assert rs.best_score == 0.9
        assert rs.best_params == {"beta": 2.0}

    def test_best_idx(self):
        from pymv.experiment.result import RandomSearchResult

        rs = RandomSearchResult("acm", "LMGEC", "nmi")
        rs.add_trial({"beta": 1.0}, {"acc": 0.5, "nmi": 0.9}, 1.0)
        rs.add_trial({"beta": 2.0}, {"acc": 0.9, "nmi": 0.3}, 1.5)

        assert rs.best_idx == 0  # nmi=0.9 is best
        assert rs.best_score == 0.9

    def test_str_output(self):
        from pymv.experiment.result import RandomSearchResult

        rs = RandomSearchResult("acm", "LMGEC", "acc")
        rs.add_trial({"beta": 1.0}, {"acc": 0.8}, 1.0)
        rs.add_trial({"beta": 2.0}, {"acc": 0.9}, 1.5)

        text = str(rs)
        assert "best" in text.lower() or "★" in text

    def test_empty_result(self):
        from pymv.experiment.result import RandomSearchResult

        rs = RandomSearchResult("acm", "LMGEC", "acc")
        assert str(rs) == "(no trials)"

    def test_best_trial(self):
        from pymv.experiment.result import RandomSearchResult

        rs = RandomSearchResult("acm", "LMGEC", "acc")
        assert rs.best_trial is None  # no trials yet
        rs.add_trial({"beta": 1.0}, {"acc": 0.5}, 1.0)
        rs.add_trial({"beta": 2.0}, {"acc": 0.9}, 1.5)

        best = rs.best_trial
        assert best["params"] == {"beta": 2.0}
        assert best["metrics"]["acc"] == 0.9
        assert best["time"] == 1.5


class TestRandomSearchExperiment:
    def test_run_and_best_trial(self, mock_dataset, metrics_fns):
        from pymv.experiment import RandomSearchExperiment
        from pymv.models.lmgec import LMGEC

        rs = RandomSearchExperiment(
            model_class=LMGEC,
            param_distributions={
                "n_clusters": [mock_dataset.n_clusters],
                "embedding_dim": [4],
                "max_iter": [3],
                "preprocess": [False],
            },
            dataset=("mock", mock_dataset),
            metrics=metrics_fns,
            best_metric=next(iter(metrics_fns)),
            n_iter=2,
            seed=0,
            verbose=0,
        )
        result = rs.run()
        assert result.best_trial is not None
        assert "params" in result.best_trial

    def test_save_creates_files(self, mock_dataset, metrics_fns, tmp_dir):
        # Regression: the save path referenced a non-existent ``best_trial``
        # attribute and the wrong metrics key, crashing save_markdown runs.
        from pymv.experiment import RandomSearchExperiment
        from pymv.models.lmgec import LMGEC

        rs = RandomSearchExperiment(
            model_class=LMGEC,
            param_distributions={
                "n_clusters": [mock_dataset.n_clusters],
                "embedding_dim": [4],
                "max_iter": [3],
                "preprocess": [False],
            },
            dataset=("mock", mock_dataset),
            metrics=metrics_fns,
            best_metric=next(iter(metrics_fns)),
            n_iter=2,
            seed=0,
            verbose=0,
            save_markdown=True,
            experiment_name="test_rs",
            save_path=tmp_dir,
        )
        rs.run()

        exp_dir = os.path.join(tmp_dir, "test_rs")
        assert os.path.isfile(os.path.join(exp_dir, "results.md"))
        assert os.path.isfile(os.path.join(exp_dir, "config.yaml"))
        assert os.path.isfile(os.path.join(exp_dir, "results.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metrics.csv"))
