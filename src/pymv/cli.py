"""
pymv-cli — Command-line interface for the pymv library.

Run multi-view clustering experiments directly from the terminal
without writing any Python code.

Examples
--------
::

    # Basic experiment
    pymv-cli --dataset acm

    # With model parameters
    pymv-cli --dataset dblp --beta 2 --temperature 10 --runs 3

    # Random search
    pymv-cli --dataset acm --random-search --n-trials 20

    # Save results to Markdown
    pymv-cli --dataset acm --save --name my_experiment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

# ── Model registry ────────────────────────────────────────────────
# Query model metadata when constructing commands.


def _get_model_registry() -> dict:
    """Return the global model registry (lazy import)."""
    from pymv.models.registry import list_models_info
    return list_models_info()


# ── Metric registry ──────────────────────────────────────────────

_METRIC_ALIASES = {
    "acc": "clustering_accuracy",
    "f1": "clustering_f1_score",
    "nmi": "nmi_score",
    "ari": "ari_score",
    "silhouette": "silhouette_score",
    "db": "davies_bouldin_score",
    "ch": "calinski_harabasz_score",
}

_DEFAULT_METRICS = ["acc", "nmi", "ari"]


def _resolve_metrics(names: list[str]) -> list:
    """Turn short metric names into callables."""
    import pymv.metrics as M

    out = []
    for name in names:
        full = _METRIC_ALIASES.get(name.lower(), name.lower())
        fn = getattr(M, full, None)
        if fn is None:
            _die(
                f"Unknown metric '{name}'.\n"
                f"Available: {', '.join(sorted(_METRIC_ALIASES))} "
                f"(or full names like clustering_accuracy)"
            )
        out.append(fn)
    return out


# ── Helpers ───────────────────────────────────────────────────────

def _die(msg: str, code: int = 1):
    # Keep errors visible alongside the CLI's result output.
    print(f"Error: {msg}")
    sys.exit(code)


def _configure_verbosity(level: int) -> None:
    """Configure logging: silent, tables, progress, or debug."""
    if level == 0:
        logging.basicConfig(level=logging.CRITICAL + 1)
    elif level <= 2:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.DEBUG)


def _parse_json_or_csv(value: str):
    """Parse a CLI value that may be JSON list or comma-separated."""
    value = value.strip()
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            _die(f"Invalid JSON list: {value}")
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts
    return value


def _coerce(value: str, default):
    """Coerce a CLI string to the same type as *default*, allowing lists."""
    value = value.strip()
    
    # Always parse lists first, regardless of the default type
    if value.startswith("[") or "," in value:
        parsed = _parse_json_or_csv(value)
        if isinstance(default, tuple):
            numbers = [float(item) for item in parsed]
            if any(not item.is_integer() or item < 1 for item in numbers):
                raise ValueError("Layer widths must be positive integers")
            return tuple(int(item) for item in numbers)
        return parsed

    if default is None:
        if value.lower() in {"none", "null"}:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(default, tuple):
        return (int(value),)

    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    
    return value


def _ensure_list(value):
    """Wrap a scalar in a list; leave lists as-is."""
    if isinstance(value, list):
        return value
    return [value]


# ── Argument parser ───────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    from pymv import __version__

    p = argparse.ArgumentParser(
        prog="pymv-cli",
        description="Run multi-view clustering experiments from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              pymv-cli --dataset acm
              pymv-cli --dataset dblp --beta 2 --temperature 10 --runs 3
              pymv-cli --dataset acm --random-search --n-trials 20
              pymv-cli --config configs/experiment.yaml
              pymv-cli --config configs/experiment.yaml --dataset acm  # override
              pymv-cli --list-datasets
              pymv-cli --list-models
        """),
    )

    p.add_argument(
        "--version", "-V", action="version",
        version=f"%(prog)s {__version__} (pymv {__version__})",
    )

    # ── Config file ──
    p.add_argument(
        "--config", "-c", default=None, metavar="PATH",
        help="Path to a YAML/JSON config file.  "
             "Shorthand names (e.g. 'default') resolve to configs/<name>.yaml.  "
             "CLI flags override config values.",
    )

    # ── Discovery ──
    info = p.add_argument_group("discovery")
    info.add_argument(
        "--list-datasets", action="store_true",
        help="List available datasets and exit.",
    )
    info.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit.",
    )

    # ── Required ──
    core = p.add_argument_group("core")
    core.add_argument("--variant", default=None, help="Dataset variant; see --list-datasets.")
    core.add_argument("--missing-rate", type=float, default=0.0, help="Simulated missing-view fraction.")
    core.add_argument("--data-seed", type=int, default=None, help="Separate missingness seed; otherwise use --seed.")
    core.add_argument("--impute", choices=["mean", "zero"], default=None, help="Explicit missing-view baseline.")
    core.add_argument("--no-benchmark", action="store_true", help="Disable all use of ground-truth labels.")
    core.add_argument(
        "--dataset", "-d",
        help="Dataset name (e.g. acm, dblp, prokaryotic).",
    )
    core.add_argument(
        "--model", "-m", default="lmgec",
        help="Model name (default: lmgec).",
    )

    # ── Model parameters ──
    params = p.add_argument_group(
        "model parameters",
        "Override any model parameter.  "
        "Lists can be passed as JSON or comma-separated.",
    )
    params.add_argument("--n-clusters", "-k", type=int, default=None,
                        help="Number of clusters (auto-detected from labels if omitted).")
    params.add_argument("--embedding-dim", type=int, default=None,
                        help="Embedding dimension (default: 16).")
    params.add_argument("--temperature", type=float, default=None,
                        help="Softmax temperature (default: 1.0).")
    params.add_argument("--beta", default=None,
                        help="Self-loop weight.  Scalar or list (e.g. 2 or '1.0,0.5').")
    params.add_argument("--tf-idf", default=None,
                        help="TF-IDF mode.  true/false or list (e.g. 'true,false').")
    params.add_argument("--center", default=None,
                        help="Center features (true/false, default: false).")
    params.add_argument("--scale", default=None,
                        help="Scale features (true/false, default: false).")
    params.add_argument("--no-preprocess", action="store_true",
                        help="Disable internal preprocessing (preprocess=False).")
    params.add_argument("--max-iter", type=int, default=None,
                        help="Max optimisation iterations (default: 30).")
    # Make registered model parameters available without maintaining a second
    # hand-written model list. Each selected model coerces its own defaults.
    reserved = {action.dest for action in p._actions} | {"random_state", "preprocess"}
    for info in _get_model_registry().values():
        for name in info["defaults"]:
            if name not in reserved:
                params.add_argument("--" + name.replace("_", "-"), default=None,
                                    help=f"Model-specific {name.replace('_', ' ')}.")
                reserved.add(name)

    # ── Experiment ──
    exp = p.add_argument_group("experiment")
    exp.add_argument("--runs", "-r", type=int, default=1,
                     help="Number of independent runs (default: 1).")
    exp.add_argument("--metrics", nargs="+", default=None,
                     help="Metrics to report (default: acc nmi ari).  "
                          "Short names: acc, nmi, ari, f1.")
    exp.add_argument("--verbose", "-v", type=int, default=1,
                     choices=[0, 1, 2, 3],
                     help="Verbosity level: "
                          "0=silent, "
                          "1=results tables only (default), "
                          "2=progress + tables, "
                          "3=debug (includes library logs).")

    # ── Random search ──
    rs = p.add_argument_group("random search")
    rs.add_argument("--random-search", action="store_true",
                    help="Run random hyper-parameter search instead of a fixed experiment.")
    rs.add_argument("--n-trials", type=int, default=20,
                    help="Number of random-search trials (default: 20).")
    rs.add_argument("--best-metric", default=None,
                    help="Metric to optimise (default: acc in benchmark mode, silhouette otherwise).")
    rs.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility.")

    # ── Output ──
    out = p.add_argument_group("output")
    out.add_argument("--save", action="store_true",
                     help="Save a Markdown report to experiments/ folder.")
    out.add_argument("--name", default=None,
                     help="Experiment folder name (used with --save).")
    out.add_argument("--save-path", default=None,
                     help="Base directory for experiment output (default: ./experiments).")
    out.add_argument("--log", action="store_true",
                     help="Write a log file.")

    return p


# ── Commands ──────────────────────────────────────────────────────

def cmd_list_datasets():
    from pymv.data import get_dataset_info, list_datasets
    print("Available datasets:")
    for category in ("attributed_graph_multiview", "attributed_graph_multimodal", "multiview", "multimodal"):
        names = list_datasets(category=category)
        if not names:
            continue
        print(f"\n  {category}:")
        for name in names:
            info = get_dataset_info(name)
            cached = "  [cached]" if info["cached"] else ""
            print(f"    {name:<26} {info['description']}{cached} | variants: {', '.join(info['variants'])}")


def _load_cli_data(args):
    from pymv.data import impute_missing, load_dataset
    data_seed = getattr(args, "data_seed", None)
    data = load_dataset(args.dataset, getattr(args, "missing_rate", 0.0),
                        variant=getattr(args, "variant", None),
                        random_state=args.seed if data_seed is None else data_seed)
    if getattr(args, "impute", None):
        data = impute_missing(data, args.impute)
    if getattr(args, "no_benchmark", False) and args.n_clusters is None:
        _die("--no-benchmark requires a manually chosen --n-clusters")
    return data


def cmd_list_models():
    registry = _get_model_registry()
    print("Available models:")
    for name, info in registry.items():
        defaults = info["defaults"]
        params = ", ".join(f"{k}={v}" for k, v in defaults.items())
        print(f"  • {name}  ({params})")


def cmd_run_experiment(args):
    """Standard experiment: fixed model config → results table."""
    from pymv.experiment import Experiment

    registry = _get_model_registry()
    model_name = args.model.lower()
    if model_name not in registry:
        _die(
            f"Unknown model '{args.model}'.\n"
            f"Available: {', '.join(registry)}"
        )

    # Load dataset to auto-detect n_clusters
    data = _load_cli_data(args)
    n_clusters = args.n_clusters or data.n_clusters

    # Build model kwargs
    model_info = registry[model_name]
    kwargs = {"n_clusters": n_clusters}
    defaults = model_info["defaults"]

    param_map = {
        "embedding_dim": args.embedding_dim,
        "temperature": args.temperature,
        "beta": args.beta,
        "tf_idf": args.tf_idf,
        "center": args.center,
        "scale": args.scale,
        "max_iter": args.max_iter,
    }

    for key, cli_val in param_map.items():
        # Model registries are heterogeneous: only pass parameters declared
        # by the selected implementation.
        if key not in defaults:
            continue
        if cli_val is not None:
            kwargs[key] = _coerce(str(cli_val), defaults[key])
        else:
            kwargs[key] = defaults[key]

    if "preprocess" in defaults:
        if args.no_preprocess:
            kwargs["preprocess"] = False
        else:
            kwargs["preprocess"] = defaults["preprocess"]

    # Pass through every other registered model parameter.  Config
    # files can set params that have no dedicated CLI flag;
    # merge_config_with_cli attaches them to `args`, so pick them up
    # here and fall back to the registry default otherwise.
    for key, default in defaults.items():
        if key in kwargs or key == "preprocess":
            continue
        val = getattr(args, key, None)
        kwargs[key] = default if val is None else val

    if "random_state" in defaults and args.seed is not None:
        kwargs["random_state"] = args.seed

    model = model_info["class"](**kwargs)
    benchmark = not getattr(args, "no_benchmark", False)
    metrics = _resolve_metrics(args.metrics or (_DEFAULT_METRICS if benchmark else ["silhouette", "db", "ch"]))

    result = Experiment(
        datasets={args.dataset: data},
        models=[model],
        benchmark=benchmark,
        metrics=metrics,
        runs=args.runs,
        verbose=args.verbose,
        save_markdown=args.save,
        experiment_name=args.name,
        log=args.log,
        save_path=args.save_path,
    ).run()

    return result


def cmd_random_search(args):
    """Random hyper-parameter search."""
    from pymv.experiment import RandomSearchExperiment

    registry = _get_model_registry()
    model_name = args.model.lower()
    if model_name not in registry:
        _die(
            f"Unknown model '{args.model}'.\n"
            f"Available: {', '.join(registry)}"
        )

    data = _load_cli_data(args)
    n_clusters = args.n_clusters or data.n_clusters

    model_info = registry[model_name]
    defaults = model_info["defaults"]

    # Build search distributions — any CLI-provided value is fixed,
    # others get sensible search ranges.
    # If a config file provided a search_space, use it as the base.
    config_space = getattr(args, "_search_space", None) or {}

    distributions: dict = {"n_clusters": [n_clusters]}

    # Embedding dim
    if args.embedding_dim is not None:
        distributions["embedding_dim"] = [args.embedding_dim]
    elif "embedding_dim" in config_space:
        distributions["embedding_dim"] = _ensure_list(config_space["embedding_dim"])
    else:
        distributions["embedding_dim"] = [4, 8, 16, 32, 64]

    # Temperature
    if args.temperature is not None:
        distributions["temperature"] = [args.temperature]
    elif "temperature" in config_space:
        distributions["temperature"] = _ensure_list(config_space["temperature"])
    else:
        distributions["temperature"] = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    # Beta
    if "beta" in defaults:
        if args.beta is not None:
            distributions["beta"] = [_coerce(str(args.beta), defaults["beta"])]
        elif "beta" in config_space:
            distributions["beta"] = _ensure_list(config_space["beta"])
        else:
            distributions["beta"] = [0.1, 0.5, 1.0, 2.0, 5.0]

    # TF-IDF
    if "tf_idf" in defaults:
        if args.tf_idf is not None:
            distributions["tf_idf"] = [_coerce(str(args.tf_idf), defaults["tf_idf"])]
        elif "tf_idf" in config_space:
            distributions["tf_idf"] = _ensure_list(config_space["tf_idf"])
        else:
            distributions["tf_idf"] = [True, False]

    # Max iter
    if args.max_iter is not None:
        distributions["max_iter"] = [args.max_iter]
    elif "max_iter" in config_space:
        distributions["max_iter"] = _ensure_list(config_space["max_iter"])
    else:
        distributions["max_iter"] = [defaults["max_iter"]]

    # Preprocess
    if "preprocess" in defaults:
        if args.no_preprocess:
            distributions["preprocess"] = [False]
        else:
            distributions["preprocess"] = [defaults["preprocess"]]

    # Model-specific parameters can be supplied through config search_space;
    # otherwise keep their registered defaults fixed during the search.
    for key, default in defaults.items():
        if key in distributions:
            continue
        distributions[key] = (
            _ensure_list(config_space[key]) if key in config_space else [default]
        )

    benchmark = not getattr(args, "no_benchmark", False)
    metrics = _resolve_metrics(args.metrics or (_DEFAULT_METRICS if benchmark else ["silhouette", "db", "ch"]))

    # Resolve best_metric full name
    chosen_metric = args.best_metric or ("acc" if benchmark else "silhouette")
    best_full = _METRIC_ALIASES.get(chosen_metric.lower(), chosen_metric.lower())

    result = RandomSearchExperiment(
        model_class=model_info["class"],
        benchmark=benchmark,
        param_distributions=distributions,
        dataset=(args.dataset, data),
        metrics=metrics,
        best_metric=best_full,
        n_iter=args.n_trials,
        seed=args.seed,
        verbose=args.verbose,
        save_markdown=args.save,
        experiment_name=args.name,
        log=args.log,
        save_path=args.save_path,
    ).run()

    return result


# ── Quick-start guide ─────────────────────────────────────────────

QUICK_START = """\
\033[1mpymv-cli\033[0m — Run multi-view clustering experiments from the terminal.

\033[1;33m⚡ Quick start:\033[0m

  pymv-cli --dataset acm                          # run with defaults
  pymv-cli --dataset dblp --beta 2 --runs 3       # custom params
  pymv-cli --dataset acm --random-search           # hyper-parameter search
  pymv-cli --config configs/experiment.yaml        # run from config file

\033[1;33m🔍 Discover:\033[0m

  pymv-cli --version                               # show library version
  pymv-cli --list-datasets                         # show available datasets
  pymv-cli --list-models                           # show available models
  pymv-cli --help                                  # full usage reference

\033[1;33m📖 Common examples:\033[0m

  pymv-cli -d acm --beta 2 --temperature 100 --tf-idf true
  pymv-cli -d acm --random-search --n-trials 30 --seed 42
  pymv-cli -d dblp --runs 5 --metrics acc nmi f1 --save
  pymv-cli --config experiment.yaml --dataset acm   # config + override

\033[1;33m💡 Tips:\033[0m

  • n_clusters is auto-detected from the dataset if not provided
  • Use --no-preprocess if you want to skip internal preprocessing
  • Use --save to export a Markdown report to experiments/
  • Use -v 0 for silent, -v 1 for tables (default), -v 2 for progress, -v 3 for full debug
  • Use --config with a YAML/JSON file for reproducible experiments
  • CLI flags always override config file values
  • Run pymv-cli --help for the full list of parameters
"""


# ── Entry point ───────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    parser = build_parser()

    # No arguments at all → show quick-start guide
    if argv is None and len(sys.argv) == 1:
        print(QUICK_START)
        return

    args = parser.parse_args(argv)

    # ── Config file loading ──
    if args.config:
        from pymv.config import config_to_args, load_config, merge_config_with_cli, validate_config
        try:
            raw_cfg = load_config(args.config)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            _die(str(exc))

        errors = validate_config(raw_cfg)
        if errors:
            _die("Config validation failed:\n  • " + "\n  • ".join(errors))

        cfg_args = config_to_args(raw_cfg)
        args = merge_config_with_cli(cfg_args, args, parser)

    # Configure Python-side verbosity (env vars already set at module load)
    _configure_verbosity(args.verbose)

    # Discovery commands (no dataset needed)
    if args.list_datasets:
        cmd_list_datasets()
        sys.exit(0)
    if args.list_models:
        cmd_list_models()
        sys.exit(0)

    # Validate required --dataset
    if not args.dataset:
        print(QUICK_START)
        print("\033[1;31mError:\033[0m --dataset is required.")
        print("       Use --list-datasets to see available options.\n")
        sys.exit(1)

    # Validate dataset name early (before heavy imports)
    from pymv.data import list_datasets
    available = list_datasets(include_aliases=True)
    if args.dataset.lower() not in available:
        _die(
            f"Unknown dataset '{args.dataset}'.\n"
            f"Available: {', '.join(available)}"
        )
    args.dataset = args.dataset.lower()

    # Validate model name early
    registry = _get_model_registry()
    if args.model.lower() not in registry:
        _die(
            f"Unknown model '{args.model}'.\n"
            f"Available: {', '.join(registry)}"
        )

    # Dispatch
    if args.random_search:
        cmd_random_search(args)
    else:
        cmd_run_experiment(args)


if __name__ == "__main__":
    main()
