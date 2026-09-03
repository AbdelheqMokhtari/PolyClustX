"""
Tests for pymv CLI (pymv-cli).

Uses subprocess to test the real entry point, ensuring the CLI
behaves as a user would experience it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Build the command prefix to run pymv-cli in the test environment
_PYTHON = sys.executable


def _run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run ``pymv-cli`` via the Python entry point."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [_PYTHON, "-m", "pymv.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


# ── Discovery commands (fast, no model loading) ──────────────────


class TestCLIDiscovery:
    def test_version(self):
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "pymv" in result.stdout.lower() or "pymv" in result.stderr.lower()

    def test_list_datasets(self):
        result = _run_cli("--list-datasets")
        assert result.returncode == 0
        assert "acm" in result.stdout
        assert "dblp" in result.stdout

    def test_list_models(self):
        result = _run_cli("--list-models")
        assert result.returncode == 0
        assert "lmgec" in result.stdout

    def test_no_args_shows_quick_start(self):
        result = _run_cli()
        assert result.returncode == 0
        assert "pymv-cli" in result.stdout

    def test_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--dataset" in result.stdout


# ── Error handling ────────────────────────────────────────────────


class TestCLIErrors:
    def test_unknown_dataset(self):
        result = _run_cli("--dataset", "nonexistent_xyz")
        assert result.returncode != 0
        assert "Unknown dataset" in result.stdout or "Error" in result.stdout

    def test_unknown_model(self):
        result = _run_cli("--dataset", "acm", "--model", "nonexistent_model")
        assert result.returncode != 0
        assert "Unknown model" in result.stdout or "Error" in result.stdout


# ── Experiment run (requires real data — uses ACM) ────────────────


class TestCLIExperiment:
    def test_basic_run(self):
        """Run a minimal experiment via CLI."""
        result = _run_cli(
            "--dataset", "acm",
            "--verbose", "0",
            "--max-iter", "3",
            "--embedding-dim", "4",
        )
        assert result.returncode == 0

    def test_run_with_save(self, tmp_path):
        """Run experiment with --save and verify output files."""
        out_dir = str(tmp_path / "cli_experiments")
        result = _run_cli(
            "--dataset", "acm",
            "--verbose", "0",
            "--max-iter", "3",
            "--embedding-dim", "4",
            "--save",
            "--name", "cli_test",
            "--save-path", out_dir,
        )
        assert result.returncode == 0

        exp_dir = os.path.join(out_dir, "cli_test")
        assert os.path.isdir(exp_dir)
        assert os.path.isfile(os.path.join(exp_dir, "results.md"))
        assert os.path.isfile(os.path.join(exp_dir, "config.yaml"))
        assert os.path.isfile(os.path.join(exp_dir, "results.json"))
        assert os.path.isfile(os.path.join(exp_dir, "metrics.csv"))

        # Verify results.json is valid JSON
        with open(os.path.join(exp_dir, "results.json")) as f:
            data = json.load(f)
        assert "experiments" in data

    def test_run_with_metrics(self):
        """CLI with explicit metrics."""
        result = _run_cli(
            "--dataset", "acm",
            "--verbose", "0",
            "--max-iter", "3",
            "--embedding-dim", "4",
            "--metrics", "acc", "nmi",
        )
        assert result.returncode == 0

    def test_run_with_config_file(self, sample_config):
        """Run from a YAML config file."""
        result = _run_cli("--config", sample_config)
        assert result.returncode == 0

    def test_run_no_preprocess(self):
        result = _run_cli(
            "--dataset", "acm",
            "--verbose", "0",
            "--max-iter", "3",
            "--embedding-dim", "4",
            "--no-preprocess",
        )
        assert result.returncode == 0
