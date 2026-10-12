"""
Tests for polyclustx experiment tracking (ExperimentTracker + helpers).
"""

from __future__ import annotations

import csv
import json
import os

# ── ExperimentTracker basic ───────────────────────────────────────


class TestExperimentTrackerInit:
    def test_default_base_dir(self):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="test")
        assert "experiments" in tracker.experiment_dir

    def test_custom_save_path(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="run1", save_path=tmp_dir)
        assert tracker.experiment_dir == os.path.join(tmp_dir, "run1")

    def test_timestamp_name_when_none(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(save_path=tmp_dir)
        # Timestamp format: YYYY-MM-DD_HH-MM-SS
        name = os.path.basename(tracker.experiment_dir)
        parts = name.split("_")
        assert len(parts) >= 2  # date_time at minimum

    def test_dir_created_on_access(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="lazy_test", save_path=tmp_dir)
        path = tracker.experiment_dir
        assert os.path.isdir(path)

    def test_repr(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="repr_test", save_path=tmp_dir)
        r = repr(tracker)
        assert "ExperimentTracker" in r


# ── save_config ───────────────────────────────────────────────────


class TestSaveConfig:
    def test_saves_yaml_file(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="cfg_test", save_path=tmp_dir)
        config = {"dataset": "acm", "model": {"name": "lmgec"}}
        path = tracker.save_config(config)

        assert path.endswith(".yaml") or path.endswith(".json")
        assert os.path.isfile(path)

    def test_yaml_content_readable(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="cfg_read", save_path=tmp_dir)
        config = {"dataset": "dblp", "model": {"name": "lmgec", "params": {"beta": 2}}}
        path = tracker.save_config(config)

        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "dblp" in content
        assert "lmgec" in content

    def test_yaml_has_header_comment(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="cfg_header", save_path=tmp_dir)
        path = tracker.save_config({"dataset": "acm"})

        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        assert first_line.startswith("#")


# ── save_results ──────────────────────────────────────────────────


class TestSaveResults:
    def test_saves_json(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="res_test", save_path=tmp_dir)
        results = {"experiments": [{"dataset": "acm", "model": "LMGEC", "acc": 0.9}]}
        path = tracker.save_results(results)

        assert path.endswith("results.json")
        assert os.path.isfile(path)

    def test_json_valid_and_has_timestamp(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="res_ts", save_path=tmp_dir)
        path = tracker.save_results({"acc": 0.9})

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "timestamp" in data

    def test_json_preserves_data(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="res_data", save_path=tmp_dir)
        original = {"experiments": [{"acc": 0.912345}]}
        path = tracker.save_results(original)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiments"][0]["acc"] == 0.912345


# ── save_metrics_csv ──────────────────────────────────────────────


class TestSaveMetricsCsv:
    def test_saves_csv(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="csv_test", save_path=tmp_dir)
        rows = [
            {"dataset": "acm", "model": "LMGEC", "acc": 0.9, "nmi": 0.8},
            {"dataset": "dblp", "model": "LMGEC", "acc": 0.7, "nmi": 0.6},
        ]
        path = tracker.save_metrics_csv(rows)

        assert path.endswith("metrics.csv")
        assert os.path.isfile(path)

    def test_csv_has_correct_rows(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="csv_rows", save_path=tmp_dir)
        rows = [
            {"dataset": "acm", "acc": 0.9},
            {"dataset": "dblp", "acc": 0.7},
        ]
        path = tracker.save_metrics_csv(rows)

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 2
        assert read_rows[0]["dataset"] == "acm"
        assert float(read_rows[1]["acc"]) == 0.7

    def test_empty_rows_returns_empty_string(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="csv_empty", save_path=tmp_dir)
        result = tracker.save_metrics_csv([])
        assert result == ""

    def test_custom_filename(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="csv_custom", save_path=tmp_dir)
        path = tracker.save_metrics_csv(
            [{"a": 1}], filename="custom_metrics.csv"
        )
        assert "custom_metrics.csv" in path


# ── save_environment ──────────────────────────────────────────────


class TestSaveEnvironment:
    def test_saves_environment_json(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="env_test", save_path=tmp_dir)
        path = tracker.save_environment()

        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "python_version" in data
        assert "polyclustx_version" in data

    def test_environment_has_platform(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="env_plat", save_path=tmp_dir)
        path = tracker.save_environment()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "platform" in data


# ── save_datasets_info ────────────────────────────────────────────


class TestSaveDatasetsInfo:
    def test_saves_datasets_json(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="ds_test", save_path=tmp_dir)
        info = {"acm": {"n_samples": 3025, "n_views": 2, "n_clusters": 3}}
        path = tracker.save_datasets_info(info)

        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["acm"]["n_samples"] == 3025


# ── save_raw_scores ───────────────────────────────────────────────


class TestSaveRawScores:
    def test_saves_raw_scores(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="raw_test", save_path=tmp_dir)
        scores = {"acm": {"LMGEC": {"run_0": {"acc": 0.9}}}}
        path = tracker.save_raw_scores(scores)

        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "acm" in data


# ── save_log ──────────────────────────────────────────────────────


class TestSaveLog:
    def test_copies_log_file(self, tmp_dir, tmp_path):
        from polyclustx.tracking import ExperimentTracker

        # Create a fake log file
        log_file = tmp_path / "test.log"
        log_file.write_text("test log content")

        tracker = ExperimentTracker(name="log_test", save_path=tmp_dir)
        path = tracker.save_log(str(log_file))

        assert path is not None
        assert os.path.isfile(path)
        with open(path) as f:
            assert "test log content" in f.read()

    def test_none_log_returns_none(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="log_none", save_path=tmp_dir)
        assert tracker.save_log(None) is None

    def test_missing_log_returns_none(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="log_miss", save_path=tmp_dir)
        assert tracker.save_log("/nonexistent/path.log") is None


# ── save_all ──────────────────────────────────────────────────────


class TestSaveAll:
    def test_save_all_creates_all_files(self, tmp_dir):
        from polyclustx.tracking import ExperimentTracker

        tracker = ExperimentTracker(name="all_test", save_path=tmp_dir)

        path = tracker.save_all(
            config={"dataset": "acm"},
            results={"acc": 0.9},
            metrics_rows=[{"dataset": "acm", "acc": 0.9}],
            datasets_info={"acm": {"n_samples": 100}},
            raw_scores={"acm": {"LMGEC": {}}},
        )

        assert os.path.isdir(path)

        # Check individual files
        exp_dir = path
        assert os.path.isfile(os.path.join(exp_dir, "config.yaml")) or \
               os.path.isfile(os.path.join(exp_dir, "config.json"))
        assert os.path.isfile(os.path.join(exp_dir, "results.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metrics.csv"))
        assert os.path.isfile(os.path.join(exp_dir, "metadata", "environment.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metadata", "datasets.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metadata", "raw_scores.json"))


# ── build_experiment_config ───────────────────────────────────────


class TestBuildExperimentConfig:
    def test_builds_config_from_args(self):
        import argparse

        from polyclustx.tracking import build_experiment_config

        args = argparse.Namespace(
            dataset="acm",
            model="lmgec",
            runs=3,
            metrics=["acc", "nmi"],
            verbose=1,
            random_search=False,
            seed=None,
            name=None,
            log=False,
        )
        config = build_experiment_config(args, model_params={"beta": 2.0})

        assert config["dataset"] == "acm"
        assert config["model"]["name"] == "lmgec"
        assert config["model"]["params"]["beta"] == 2.0
        assert config["experiment"]["runs"] == 3
        assert config["output"]["save"] is True


# ── build_results_dict ────────────────────────────────────────────


class TestBuildResultsDict:
    def test_basic(self):
        from polyclustx.tracking import build_results_dict

        rd = build_results_dict(
            dataset_name="acm",
            model_name="LMGEC",
            model_params={"beta": 1.0, "embedding_dim": 16},
            metrics={"acc": 0.912345, "nmi": 0.812345},
            runs=3,
            train_time=2.5,
        )
        assert rd["dataset"] == "acm"
        assert rd["model"] == "LMGEC"
        assert rd["params"]["beta"] == 1.0
        assert rd["metrics"]["acc"] == 0.912345
        assert rd["runs"] == 3
        assert rd["train_time_seconds"] == 2.5

    def test_with_std(self):
        from polyclustx.tracking import build_results_dict

        rd = build_results_dict(
            dataset_name="acm",
            model_name="LMGEC",
            model_params={},
            metrics={"acc": 0.9},
            metrics_std={"acc": 0.01},
        )
        assert "metrics_std" in rd
        assert rd["metrics_std"]["acc"] == 0.01


# ── build_metrics_csv_rows ────────────────────────────────────────


class TestBuildMetricsCsvRows:
    def test_from_experiment_result(self):
        from polyclustx.experiment.result import ExperimentResult, Result
        from polyclustx.tracking import build_metrics_csv_rows

        er = ExperimentResult()
        er.add(Result("acm", "LMGEC", {"acc": 0.9, "nmi": 0.8}, train_time=1.0))

        rows = build_metrics_csv_rows(er)
        assert len(rows) == 1
        assert rows[0]["dataset"] == "acm"
        assert rows[0]["model"] == "LMGEC"
        assert rows[0]["acc"] == 0.9

    def test_from_random_search_result(self):
        from polyclustx.experiment.result import RandomSearchResult
        from polyclustx.tracking import build_metrics_csv_rows

        rs = RandomSearchResult("acm", "LMGEC", "acc")
        rs.add_trial({"beta": 1.0}, {"acc": 0.8, "nmi": 0.7}, 1.0)
        rs.add_trial({"beta": 2.0}, {"acc": 0.9, "nmi": 0.6}, 1.5)

        rows = build_metrics_csv_rows(rs)
        assert len(rows) == 2
        assert rows[0]["trial"] == 1
        assert rows[1]["trial"] == 2
        assert rows[1]["is_best"] is True  # second trial has higher acc


# ── _to_python helper ─────────────────────────────────────────────


class TestToPython:
    def test_numpy_int(self):
        import numpy as np

        from polyclustx.tracking import _to_python

        assert isinstance(_to_python(np.int64(42)), int)
        assert _to_python(np.int64(42)) == 42

    def test_numpy_float(self):
        import numpy as np

        from polyclustx.tracking import _to_python

        assert isinstance(_to_python(np.float64(3.14)), float)

    def test_numpy_bool(self):
        import numpy as np

        from polyclustx.tracking import _to_python

        assert isinstance(_to_python(np.bool_(True)), bool)

    def test_numpy_array(self):
        import numpy as np

        from polyclustx.tracking import _to_python

        arr = np.array([1, 2, 3])
        result = _to_python(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_plain_python_passthrough(self):
        from polyclustx.tracking import _to_python

        assert _to_python(42) == 42
        assert _to_python("hello") == "hello"
        assert _to_python([1, 2]) == [1, 2]
