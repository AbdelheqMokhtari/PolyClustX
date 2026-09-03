"""
Tests for pymv config loading, flattening, merging, and validation.
"""

from __future__ import annotations

import argparse
import json

import pytest

# ── load_config ───────────────────────────────────────────────────


class TestLoadConfig:
    def test_load_yaml(self, sample_config):
        from pymv.config import load_config

        cfg = load_config(sample_config)
        assert isinstance(cfg, dict)
        assert cfg["dataset"] == "acm"
        assert cfg["model"]["name"] == "lmgec"

    def test_load_json(self, tmp_path):
        from pymv.config import load_config

        cfg_path = tmp_path / "test.json"
        cfg_path.write_text(json.dumps({
            "dataset": "dblp",
            "model": {"name": "lmgec"},
        }))

        cfg = load_config(str(cfg_path))
        assert cfg["dataset"] == "dblp"

    def test_file_not_found(self):
        from pymv.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_unsupported_format(self, tmp_path):
        from pymv.config import load_config

        cfg_path = tmp_path / "config.txt"
        cfg_path.write_text("hello")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(str(cfg_path))


# ── config_to_args ────────────────────────────────────────────────


class TestConfigToArgs:
    def test_basic_flattening(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "model": {
                "name": "lmgec",
                "params": {
                    "embedding_dim": 16,
                    "temperature": 5.0,
                },
            },
            "experiment": {
                "runs": 3,
                "metrics": ["acc", "nmi"],
            },
            "output": {
                "save": True,
                "name": "my_exp",
            },
        }
        flat = config_to_args(cfg)

        assert flat["dataset"] == "acm"
        assert flat["model"] == "lmgec"
        assert flat["embedding_dim"] == 16
        assert flat["temperature"] == 5.0
        assert flat["runs"] == 3
        assert flat["metrics"] == ["acc", "nmi"]
        assert flat["save"] is True
        assert flat["name"] == "my_exp"

    def test_preprocess_false_becomes_no_preprocess(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "model": {
                "name": "lmgec",
                "params": {"preprocess": False},
            },
        }
        flat = config_to_args(cfg)
        assert flat["no_preprocess"] is True

    def test_preprocess_true_no_flag(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "model": {
                "name": "lmgec",
                "params": {"preprocess": True},
            },
        }
        flat = config_to_args(cfg)
        assert flat.get("no_preprocess") is False

    def test_output_path_mapped(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "output": {
                "save": True,
                "path": "/custom/output",
            },
        }
        flat = config_to_args(cfg)
        assert flat["save_path"] == "/custom/output"

    def test_output_log(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "output": {"log": True},
        }
        flat = config_to_args(cfg)
        assert flat["log"] is True

    def test_search_space_preserved(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "search_space": {
                "embedding_dim": [4, 8, 16],
                "temperature": [0.1, 1.0, 10.0],
            },
        }
        flat = config_to_args(cfg)
        assert "_search_space" in flat
        assert flat["_search_space"]["embedding_dim"] == [4, 8, 16]

    def test_random_search_fields(self):
        from pymv.config import config_to_args

        cfg = {
            "dataset": "acm",
            "experiment": {
                "random_search": True,
                "n_trials": 10,
                "best_metric": "nmi",
                "seed": 42,
            },
        }
        flat = config_to_args(cfg)
        assert flat["random_search"] is True
        assert flat["n_trials"] == 10
        assert flat["best_metric"] == "nmi"
        assert flat["seed"] == 42

    def test_model_as_string(self):
        from pymv.config import config_to_args

        cfg = {"dataset": "acm", "model": "lmgec"}
        flat = config_to_args(cfg)
        assert flat["model"] == "lmgec"

    def test_empty_config(self):
        from pymv.config import config_to_args

        flat = config_to_args({})
        assert flat == {}


# ── validate_config ───────────────────────────────────────────────


class TestValidateConfig:
    def test_valid_config(self):
        from pymv.config import validate_config

        errors = validate_config({"dataset": "acm"})
        assert errors == []

    def test_missing_dataset(self):
        from pymv.config import validate_config

        errors = validate_config({"model": {"name": "lmgec"}})
        assert any("dataset" in e.lower() for e in errors)

    def test_model_params_without_name(self):
        from pymv.config import validate_config

        errors = validate_config({
            "dataset": "acm",
            "model": {"params": {"beta": 2}},
        })
        assert any("model.name" in e for e in errors)


# ── merge_config_with_cli ─────────────────────────────────────────


class TestMergeConfigWithCLI:
    def test_config_fills_missing_cli_args(self):
        from pymv.config import merge_config_with_cli

        config_args = {"dataset": "dblp", "runs": 5}
        ns = argparse.Namespace(dataset=None, runs=1, verbose=1)
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", default=None)
        parser.add_argument("--runs", type=int, default=1)
        parser.add_argument("--verbose", type=int, default=1)

        merged = merge_config_with_cli(config_args, ns, parser)
        assert merged.dataset == "dblp"
        assert merged.runs == 5
        assert merged.verbose == 1  # untouched
