"""Small, model-independent command line; legacy pymv-cli remains available."""
from __future__ import annotations

import argparse
import json


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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pymv", description="Prepare, compare and explore clustering models.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("display", help="Open the graphical interface")
    commands.add_parser("datasets", help="List dataset names")
    models = commands.add_parser("models", help="List methods or inspect a model's parameters")
    models.add_argument("model", nargs="?")
    for action in ("run", "search"):
        p = commands.add_parser(action, help="Run repeated fits" if action == "run" else "Tune hyperparameters")
        p.add_argument("dataset")
        p.add_argument("--variant")
        p.add_argument("--model", default="multiview_kmeans")
        p.add_argument("--clusters", type=int, help="Defaults to number of labels; required without labels")
        p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
        p.add_argument("--runs", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--metrics", nargs="+", help="For example: acc nmi ari, or silhouette db")
        p.add_argument("--no-benchmark", action="store_true", help="Ignore ground-truth labels")
        p.add_argument("--output", default="experiments", help="Directory for JSON reports")
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
    if args.command == "display":
        from pymv.display import launch
        return launch([])
    from pymv.data import list_datasets, load_data
    from pymv.models.registry import get_model_info, list_models_info
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
        from pymv.cli import _METRIC_ALIASES, _resolve_metrics
        from pymv.experiment import Experiment, RandomSearchExperiment
        info = get_model_info(args.model)
        params = assignments(args.param)
        unknown = set(params) - set(info["defaults"]) - {"n_clusters"}
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}; run pymv models {args.model}")
        if args.runs < 1 or args.seed < 0:
            raise ValueError("runs must be positive and seed nonnegative")
        data = load_data(args.dataset, variant=args.variant) if args.variant else load_data(args.dataset)
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
                                runs=args.runs, benchmark=benchmark, save_path=args.output).run()
        else:
            space = {key: [value] for key, value in params.items()}
            space.update(assignments(args.vary))
            if set(space) - set(info["defaults"]) - {"n_clusters"}:
                raise ValueError(f"Unknown search parameter; run pymv models {args.model}")
            objective = _METRIC_ALIASES.get(args.objective, args.objective) if args.objective else metrics[0].__name__
            result = RandomSearchExperiment(info["class"], space, data, metrics, objective,
                                            strategy=args.strategy, n_iter=args.trials, seed=args.seed,
                                            runs=args.runs, benchmark=benchmark, save_path=args.output).run()
        print(f"Report: {result.json_path}")
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
