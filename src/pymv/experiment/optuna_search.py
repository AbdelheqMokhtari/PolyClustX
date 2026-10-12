"""Adaptive TPE search using the same evaluation and reporting as grid/random."""
from __future__ import annotations

import math

from .random_search import RandomSearchExperiment


def validate_space(space):
    """Accept explicit choices or bounded float/int ranges; fail before fitting."""
    from optuna.distributions import FloatDistribution, IntDistribution

    for name, values in space.items():
        if isinstance(values, (list, tuple)) and values:
            if any(v is not None and not isinstance(v, (str, bool, int, float)) for v in values):
                raise ValueError(f"{name}: choices must be scalar values")
            if any(isinstance(v, float) and not math.isfinite(v) for v in values):
                raise ValueError(f"{name}: choices must be finite")
        elif isinstance(values, dict):
            options = dict(values)
            kind = options.pop("type", "float")
            if kind not in {"float", "int"} or not {"low", "high"} <= options.keys():
                raise ValueError(f"{name}: range needs type float/int, low and high")
            if kind == "int" and any(isinstance(options[k], bool) or not isinstance(options[k], int)
                                     for k in ("low", "high", "step") if k in options):
                raise ValueError(f"{name}: integer bounds and steps must be integers")
            if any(not math.isfinite(options[k]) for k in ("low", "high", "step") if k in options):
                raise ValueError(f"{name}: range bounds must be finite")
            try:
                (IntDistribution if kind == "int" else FloatDistribution)(**options)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: invalid range: {exc}") from exc
        else:
            raise ValueError(f"{name}: provide a nonempty list or a bounded range")


class OptunaSampler:
    """One ask/tell trial at a time: failed configurations never become winners."""

    def __init__(self, space, seed, maximize):
        import optuna

        validate_space(space)
        self.space = space
        self.study = optuna.create_study(direction="maximize" if maximize else "minimize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
        self.trial = None

    def ask(self):
        self.trial = self.study.ask()
        result = {}
        for name, values in self.space.items():
            if isinstance(values, dict):
                options = dict(values)
                kind = options.pop("type", "float")
                suggest = self.trial.suggest_int if kind == "int" else self.trial.suggest_float
                result[name] = suggest(name, **options)
            else:
                result[name] = self.trial.suggest_categorical(name, values)
        return result

    def tell(self, score):
        from optuna.trial import TrialState

        if score is None or not math.isfinite(score):
            self.study.tell(self.trial, state=TrialState.FAIL)
        else:
            self.study.tell(self.trial, float(score))


class OptunaSearchExperiment(RandomSearchExperiment):
    """TPE search over choices or ranges, selected by mean repeated-run score.

    ``param_distributions`` maps names to lists or dictionaries, for example
    ``{"learning_rate": {"low": 1e-4, "high": 1e-2, "log": True}}``.
    Add ``"type": "int"`` for integer ranges. Other options match
    :class:`RandomSearchExperiment`; ``n_iter`` is the trial budget.
    """

    def __init__(self, model_class, param_distributions, dataset, metrics, best_metric, **kwargs):
        kwargs.pop("strategy", None)
        super().__init__(model_class, param_distributions, dataset, metrics, best_metric,
                         strategy="optuna", **kwargs)
