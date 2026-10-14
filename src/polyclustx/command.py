"""Discovery, method commands, search and multi-model experiment plans."""
from __future__ import annotations

import argparse
import json
import sys


def assignments(items):
    """Read repeatable name=value options without evaluating Python code."""
    result = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise ValueError("Use --param name=value (for example --param max_iter=50)")
        if name in result:
            raise ValueError(f"Duplicate parameter: {name}")
        try:
            result[name] = json.loads(value)
        except json.JSONDecodeError:
            result[name] = value
    return result


def add_verbosity_argument(parser, *, default=2):
    """Shared CLI levels; progress is on unless the user chooses quieter output."""
    parser.add_argument("--verbose", "-v", type=int, choices=range(4), default=default,
                        help="0=quiet, 1=tables, 2=progress + tables (default), 3=debug + progress")


def main(argv=None):
    from polyclustx import __version__
    from polyclustx.models import list_models

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].replace("-", "_") in list_models():
        return model_main(argv[0].replace("-", "_"), argv[1:])
    if argv and argv[0] == "experiment":
        from polyclustx.command_experiments import main as experiment_main
        return experiment_main(argv[1:])
    if argv and argv[0] == "display":
        from polyclustx.display import launch
        return launch(argv[1:])  # Compatibility only; the public launcher is polyclustx-display.
    parser = argparse.ArgumentParser(prog="polyclustx", description="Prepare, compare and explore clustering models.",
                                     epilog="Launch the graphical interface with polyclustx-display.")
    parser.add_argument("--version", action="version", version=f"PolyClustX {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("experiment", help="Compare methods and datasets, optionally from a YAML plan")
    for name in list_models():
        commands.add_parser(name.replace("_", "-"), help=f"Run {name}; use --help for its parameters")
    commands.add_parser("datasets", help="List dataset names")
    models = commands.add_parser("models", help="List methods or inspect a model's parameters")
    models.add_argument("model", nargs="?")
    for action in ("run", "search"):
        p = commands.add_parser(action, help="Run repeated fits" if action == "run" else "Tune hyperparameters")
        p.add_argument("dataset")
        p.add_argument("--variant")
        p.add_argument("--missing-rate", type=float, default=0.)
        p.add_argument("--data-seed", type=int)
        p.add_argument("--impute", choices=["mean", "zero"])
        p.add_argument("--model", default="multiview_kmeans")
        p.add_argument("--clusters", type=int, help="Defaults to number of labels; required without labels")
        p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
        p.add_argument("--runs", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--metrics", nargs="+", help="For example: acc nmi ari, or silhouette db")
        p.add_argument("--no-benchmark", action="store_true", help="Ignore ground-truth labels")
        p.add_argument("--output", default="experiments", help="Directory for JSON reports")
        add_verbosity_argument(p)
        if action == "search":
            p.add_argument("--strategy", choices=["random", "grid", "optuna"], default="random")
            p.add_argument("--vary", action="append", required=True, metavar="NAME=VALUES",
                           help='Choices: n_init=[1,5,10]; Optuna range: \'tol={"low":0.0001,"high":0.01}\'')
            p.add_argument("--trials", type=int, default=20)
            p.add_argument("--objective", help="Metric to optimize; defaults to first selected metric")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    from polyclustx.data import list_datasets, load_data
    from polyclustx.models.registry import get_model_info, list_models_info
    try:
        if args.command == "datasets":
            print("\n".join(list_datasets()))
            return
        if args.command == "models":
            if args.model:
                info = get_model_info(args.model)
                print(info["description"])
                for key, value in info["defaults"].items():
                    print(f"  {key} = {value!r}")
            else:
                for name, info in list_models_info().items():
                    print(f"{name}: {info['description']}")
            return
        from polyclustx.cli import _METRIC_ALIASES, _resolve_metrics
        from polyclustx.experiment import Experiment, RandomSearchExperiment
        info = get_model_info(args.model)
        params = assignments(args.param)
        unknown = set(params) - set(info["defaults"]) - {"n_clusters"}
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}; run polyclustx models {args.model}")
        if args.runs < 1 or args.seed < 0:
            raise ValueError("runs must be positive and seed nonnegative")
        load_options = {"variant": args.variant} if args.variant else {}
        if args.missing_rate:
            load_options.update(missing_rate=args.missing_rate,
                                random_state=args.seed if args.data_seed is None else args.data_seed)
        if args.verbose >= 2:
            variant = f" ({args.variant})" if args.variant else ""
            print(f"Loading dataset: {args.dataset}{variant}", file=sys.stderr, flush=True)
        data = load_data(args.dataset, **load_options)
        if args.impute:
            from polyclustx.data import impute_missing
            data = impute_missing(data, args.impute)
        benchmark = data.labels is not None and not args.no_benchmark
        supplied_clusters = params.pop("n_clusters", None)
        clusters = args.clusters if args.clusters is not None else supplied_clusters
        if clusters is None and benchmark:
            clusters = data.n_clusters
        if clusters is None or clusters < 1:
            raise ValueError("Provide --clusters for unlabeled data")
        params["n_clusters"] = clusters
        if "random_state" in info["defaults"]:
            params.setdefault("random_state", args.seed)
        metrics = _resolve_metrics(args.metrics or (["acc", "nmi", "ari"] if benchmark else ["silhouette"]))
        if args.command == "run":
            result = Experiment(datasets=[data], models=[info["class"](**params)], metrics=metrics,
                                runs=args.runs, benchmark=benchmark, save_path=args.output,
                                verbose=args.verbose).run()
        else:
            space = {key: [value] for key, value in params.items()}
            space.update(assignments(args.vary))
            if set(space) - set(info["defaults"]) - {"n_clusters"}:
                raise ValueError(f"Unknown search parameter; run polyclustx models {args.model}")
            objective = _METRIC_ALIASES.get(args.objective, args.objective) if args.objective else metrics[0].__name__
            result = RandomSearchExperiment(info["class"], space, data, metrics, objective,
                                            strategy=args.strategy, n_iter=args.trials, seed=args.seed,
                                            runs=args.runs, benchmark=benchmark, save_path=args.output,
                                            verbose=args.verbose).run()
        print(f"Report: {result.json_path}")
        if args.command == "run" and any(run["status"] == "failed"
                                          for experiment in result.report["experiments"]
                                          for run in experiment["runs"]):
            print("Some fits failed; inspect the report's error fields.", file=sys.stderr)
            raise SystemExit(1)
        return
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


def _parameter_value(text, default):
    """Typed values without eval; booleans must be explicit and valid."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if isinstance(default, tuple):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, (list, tuple)):
            value = [value]
        try:
            if any(isinstance(v, bool) or float(v) != int(v) or int(v) < 1 for v in value):
                raise ValueError
            return tuple(int(v) for v in value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise argparse.ArgumentTypeError("Use positive layer widths, for example '64,32'") from exc
    # LMGEC permits per-view lists for several scalar-default parameters.
    if isinstance(value, list):
        return value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise argparse.ArgumentTypeError("Use true or false")
    if isinstance(default, int) and (isinstance(value, bool) or not isinstance(value, int)):
        raise argparse.ArgumentTypeError("Use an integer")
    if isinstance(default, float) and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise argparse.ArgumentTypeError("Use a number")
    return value


def model_main(model_name, argv=None):
    """Run a method by its installed command, with model-specific flags."""
    from polyclustx import __version__
    from polyclustx.models import get_model_info

    info = get_model_info(model_name)
    parser = argparse.ArgumentParser(prog=model_name.replace("_", "-"), description=info["description"])
    parser.add_argument("dataset", nargs="?", help="Catalog dataset name")
    parser.add_argument("--dataset", "-d", dest="named_dataset", help="Alternative to the positional dataset")
    parser.add_argument("--version", action="version", version=f"PolyClustX {__version__}")
    parser.add_argument("--variant")
    parser.add_argument("--missing-rate", type=float, default=0.)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--impute", choices=["mean", "zero"])
    parser.add_argument("--clusters", "--n-clusters", "-k", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--output", default="experiments")
    parser.add_argument("--search", choices=["grid", "random", "optuna"])
    parser.add_argument("--vary", action="append", default=[], metavar="NAME=VALUES")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--objective")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    add_verbosity_argument(parser)
    group = parser.add_argument_group("model parameters", "Defaults match the Python estimator.")
    for name, default in info["defaults"].items():
        if name in {"n_clusters", "random_state"}:
            continue
        flag = "--" + name.replace("_", "-")
        if flag in parser._option_string_actions:
            continue  # Plugin parameters colliding with CLI options remain available through --param.
        group.add_argument(flag, dest="model_" + name, default=argparse.SUPPRESS,
                           type=lambda value, d=default: _parameter_value(value, d),
                           help=f"{name}; default: {default!r}")
    if "preprocess" in info["defaults"]:
        group.add_argument("--no-preprocess", dest="model_preprocess", action="store_const", const=False,
                           default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dataset and args.named_dataset:
        parser.error("Supply the dataset once: positionally or with --dataset")
    dataset = args.dataset or args.named_dataset
    if not dataset:
        parser.error("Provide a dataset; use polyclustx datasets to list available names")
    if args.vary and not args.search:
        parser.error("--vary requires --search grid, random or optuna")
    if args.search and not args.vary:
        parser.error("--search requires at least one --vary name=values")
    try:
        params = assignments(args.param)
    except ValueError as exc:
        parser.error(str(exc))
    for key, value in vars(args).items():
        if key.startswith("model_"):
            name = key[6:]
            if name in params:
                parser.error(f"Supply {name} only once (dedicated flag or --param)")
            params[name] = value
    forwarded = ["search" if args.search else "run", dataset, "--model", model_name,
                 "--runs", str(args.runs), "--seed", str(args.seed), "--output", args.output,
                 "--verbose", str(args.verbose)]
    for name in ("variant", "clusters", "objective", "missing_rate", "data_seed", "impute"):
        value = getattr(args, name)
        if value is not None:
            if name == "objective" and not args.search:
                parser.error("--objective requires --search")
            forwarded.extend(["--" + name.replace("_", "-"), str(value)])
    if args.metrics:
        forwarded += ["--metrics", *args.metrics]
    if args.no_benchmark:
        forwarded.append("--no-benchmark")
    for name, value in params.items():
        forwarded += ["--param", f"{name}={json.dumps(value)}"]
    if args.search:
        forwarded += ["--strategy", args.search, "--trials", str(args.trials)]
        for value in args.vary:
            forwarded += ["--vary", value]
    return main(forwarded)


if __name__ == "__main__":
    main()
