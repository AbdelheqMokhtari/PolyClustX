"""Exhaustive discrete search with the same reporting API as random search."""

from .random_search import RandomSearchExperiment


class GridSearchExperiment(RandomSearchExperiment):
    """Evaluate every combination in param_grid; optimize the mean over runs.

    Parameters other than param_grid match RandomSearchExperiment. The grid
    must contain finite, nonempty candidate sequences; it is iterated lazily.
    """

    def __init__(self, model_class, param_grid, dataset, metrics, best_metric, **kwargs):
        kwargs.pop("strategy", None)
        super().__init__(model_class, param_grid, dataset, metrics, best_metric, strategy="grid", **kwargs)
