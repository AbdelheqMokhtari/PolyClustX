"""
Config file support for pymv-cli.

Load experiment configuration from YAML or JSON files with
validation, defaults, and CLI override merging.

Supported formats
-----------------
* **YAML** (preferred) — requires ``pyyaml``
* **JSON** — built-in, zero dependencies

Example YAML
-------------
::

    dataset: acm

    model:
      name: lmgec
      params:
        beta: 2
        temperature: 10

    experiment:
      runs: 3
      metrics: [acc, nmi, ari]
      verbose: 1

    output:
      save: true
      name: acm_experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Built-in config directory ─────────────────────────────────────

_BUILTIN_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


# ── Loader ────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    """
    Load a config file (YAML or JSON) and return a flat dict.

    Parameters
    ----------
    path : str
        Path to a ``.yaml`` / ``.yml`` / ``.json`` file, **or** a
        shorthand name that maps to ``configs/<name>.yaml``.

    Returns
    -------
    dict
        Raw parsed config (nested dict).

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the format is unsupported.
    """
    resolved = _resolve_path(path)

    ext = resolved.suffix.lower()
    text = resolved.read_text(encoding="utf-8")

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required to load YAML configs.\n"
                "Install it with:  pip install pyyaml"
            )
        data = yaml.safe_load(text)
    elif ext == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported config format '{ext}'.\n"
            f"Use .yaml, .yml, or .json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML/JSON mapping, got {type(data).__name__}.")

    return data


def _resolve_path(path: str) -> Path:
    """
    Resolve a config path.

    Checks in order:
    1. Exact path (absolute or relative to cwd)
    2. ``configs/<path>.yaml``  (shorthand)
    3. Built-in configs shipped with pymv
    """
    p = Path(path)

    # 1. Exact path
    if p.is_file():
        return p

    # 2. Shorthand — try configs/<name>.yaml
    for ext in (".yaml", ".yml", ".json"):
        candidate = Path("configs") / f"{path}{ext}"
        if candidate.is_file():
            return candidate

    # 3. Built-in configs
    for ext in (".yaml", ".yml", ".json"):
        candidate = _BUILTIN_CONFIGS_DIR / f"{path}{ext}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Config file not found: '{path}'\n"
        f"Searched:\n"
        f"  • {p}\n"
        f"  • configs/{path}.yaml\n"
        f"  • {_BUILTIN_CONFIGS_DIR / path}.yaml"
    )


# ── Flatten to argparse-compatible namespace ──────────────────────

def config_to_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a nested config dict into a flat dict whose keys match
    the argparse ``dest`` names used by ``cli.build_parser()``.

    Mapping
    -------
    ::

        dataset              → dataset
        model.name           → model
        model.params.*       → embedding_dim, temperature, beta, …
        model.params.preprocess: false  → no_preprocess = True
        experiment.runs      → runs
        experiment.metrics   → metrics
        experiment.verbose   → verbose
        experiment.random_search → random_search
        experiment.n_trials  → n_trials
        experiment.best_metric → best_metric
        experiment.seed      → seed
        output.save          → save
        output.name          → name
        output.path          → save_path
        output.log           → log
        search_space.*       → (used directly by random-search builder)

    Returns
    -------
    dict
        Flat mapping ready to be merged with CLI args.
    """
    flat: dict[str, Any] = {}

    # ── Top-level ──
    if "dataset" in cfg:
        flat["dataset"] = cfg["dataset"]

    # ── Model ──
    model = cfg.get("model", {})
    if isinstance(model, str):
        flat["model"] = model
    elif isinstance(model, dict):
        if "name" in model:
            flat["model"] = model["name"]
        for k, v in model.get("params", {}).items():
            # preprocess: false → no_preprocess = True
            if k == "preprocess":
                flat["no_preprocess"] = not v
            else:
                flat[k] = v

    # ── Experiment ──
    exp = cfg.get("experiment", {})
    if isinstance(exp, dict):
        _copy_keys(exp, flat, [
            "runs", "metrics", "verbose",
            "random_search", "n_trials", "best_metric", "seed",
        ])

    # ── Output ──
    out = cfg.get("output", {})
    if isinstance(out, dict):
        if "save" in out or "save_results" in out:
            flat["save"] = out.get("save", out.get("save_results", False))
        if "path" in out:
            flat["save_path"] = out["path"]
        _copy_keys(out, flat, ["name", "log"])

    # ── Search space (kept nested — consumed by random-search) ──
    if "search_space" in cfg:
        flat["_search_space"] = cfg["search_space"]

    return flat


def _copy_keys(src: dict, dst: dict, keys: list[str]) -> None:
    for k in keys:
        if k in src:
            dst[k] = src[k]


# ── Merge: config + CLI overrides ─────────────────────────────────

def merge_config_with_cli(
    config_args: dict[str, Any],
    cli_namespace: Any,
    parser: Any,
) -> Any:
    """
    Merge config values into the argparse Namespace, letting
    explicit CLI flags **override** config values.

    An arg is "explicitly provided on CLI" if it differs from the
    parser's default *and* is not None, **or** if it appears in
    ``sys.argv``.

    Parameters
    ----------
    config_args : dict
        Output of :func:`config_to_args`.
    cli_namespace : argparse.Namespace
        Parsed CLI args.
    parser : argparse.ArgumentParser
        The parser (used to read defaults).

    Returns
    -------
    argparse.Namespace
        Merged namespace (mutated in place).
    """
    import sys

    raw_argv = sys.argv[1:] if hasattr(sys, "argv") else []

    # Build set of flags that the user typed on the command line
    cli_explicit = set()
    for tok in raw_argv:
        if tok.startswith("-"):
            # Normalise: --tf-idf → tf_idf, --no-preprocess → no_preprocess
            key = tok.lstrip("-").replace("-", "_")
            cli_explicit.add(key)

    ns = vars(cli_namespace)

    for key, cfg_val in config_args.items():
        # Skip internal keys
        if key.startswith("_"):
            setattr(cli_namespace, key, cfg_val)
            continue

        # CLI explicitly provided → CLI wins
        if key in cli_explicit:
            continue

        # Otherwise config fills in what CLI didn't set
        ns[key] = cfg_val

    return cli_namespace


# ── Validation ────────────────────────────────────────────────────

def validate_config(cfg: dict[str, Any]) -> list[str]:
    """
    Return a list of warning/error strings for the config.
    An empty list means the config is valid.
    """
    errors: list[str] = []

    if "dataset" not in cfg:
        errors.append("Missing required field: 'dataset'.")

    model = cfg.get("model", {})
    if isinstance(model, dict) and "name" not in model and "params" in model:
        errors.append(
            "Config has 'model.params' but no 'model.name'.  "
            "Add 'model.name: lmgec' (or the model you want)."
        )

    exp = cfg.get("experiment", {})
    if isinstance(exp, dict):
        if exp.get("random_search") and "search_space" not in cfg:
            # Not an error — we'll use default search ranges
            pass

    return errors
