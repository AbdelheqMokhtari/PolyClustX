"""
Result formatting utilities for pymv experiments.

Provides table formatting and result containers inspired by
Cornac's result module, adapted for multi-view clustering.
"""

from __future__ import annotations

from collections import OrderedDict
import numpy as np


NUM_FMT = "{:.4f}"
TIME_FMT = "{:.2f}"


def _table_format(data, headers, index, extra_spaces=0, h_bars=None):
    """
    Format a 2D list into a pretty ASCII table.

    Parameters
    ----------
    data : list of list of str
        Cell values (already formatted).
    headers : list of str
        Column headers.
    index : list of str
        Row labels.
    extra_spaces : int
        Padding per column.
    h_bars : list of int or None
        Row indices after which to draw a horizontal bar.

    Returns
    -------
    str
    """
    if h_bars is None:
        h_bars = []

    # Compute column widths
    col_widths = [max(len(str(h)), *(len(str(row[i])) for row in data))
                  for i, h in enumerate(headers)]
    idx_width = max(len(str(idx)) for idx in index) if index else 0

    # Header line
    header_line = " " * (idx_width + extra_spaces) + " | ".join(
        str(h).rjust(w) for h, w in zip(headers, col_widths)
    )
    sep_line = "-" * (idx_width + extra_spaces) + " + ".join(
        "-" * w for w in col_widths
    )

    lines = [header_line, sep_line]

    for i, (idx, row) in enumerate(zip(index, data)):
        row_str = str(idx).ljust(idx_width + extra_spaces) + " | ".join(
            str(v).rjust(w) for v, w in zip(row, col_widths)
        )
        lines.append(row_str)
        if (i + 1) in h_bars:
            lines.append(sep_line)

    return "\n".join(lines)


class Result:
    """
    Stores metric results for a single (dataset, model) combination.

    Attributes
    ----------
    dataset_name : str
    model_name : str
    metric_values : OrderedDict
        metric_name -> float
    train_time : float
        Seconds spent in fit().
    """

    def __init__(self, dataset_name, model_name, metric_values, train_time):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.metric_values = OrderedDict(metric_values)
        self.train_time = train_time

    def __str__(self):
        headers = list(self.metric_values.keys()) + ["Time (s)"]
        data = [[NUM_FMT.format(v) for v in self.metric_values.values()]
                + [TIME_FMT.format(self.train_time)]]
        label = f"{self.dataset_name}/{self.model_name}"
        return _table_format(data, headers, index=[label], h_bars=[1])


class ExperimentResult:
    """
    Collects results across multiple datasets and models.

    Organized as results[dataset_name] = list of Result.
    """

    def __init__(self):
        self._results: dict[str, list[Result]] = OrderedDict()

    def add(self, result: Result):
        ds = result.dataset_name
        if ds not in self._results:
            self._results[ds] = []
        self._results[ds].append(result)

    def __str__(self):
        parts = []
        for ds_name, results in self._results.items():
            parts.append(f"\n--- {ds_name} ---")
            if not results:
                continue
            headers = list(results[0].metric_values.keys()) + ["Time (s)"]
            data = []
            index = []
            for r in results:
                row = [NUM_FMT.format(v) for v in r.metric_values.values()]
                row.append(TIME_FMT.format(r.train_time))
                data.append(row)
                index.append(r.model_name)
            parts.append(_table_format(data, headers, index, h_bars=[1]))
        return "\n".join(parts)

    @property
    def all_results(self):
        return self._results


class RandomSearchResult:
    """
    Stores results of a random hyperparameter search.

    Attributes
    ----------
    dataset_name : str
    model_name : str
    trials : list of dict
        Each dict has 'params', 'metrics' (OrderedDict), 'time'.
    best_idx : int
        Index of the best trial.
    best_metric_name : str
        The metric used to select the best trial.
    """

    def __init__(self, dataset_name, model_name, best_metric_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.best_metric_name = best_metric_name
        self.trials = []
        self.best_idx = 0

    def add_trial(self, params, metric_values, train_time):
        self.trials.append({
            "params": dict(params),
            "metrics": OrderedDict(metric_values),
            "time": train_time,
        })
        # Update best (higher is better)
        best_val = self.trials[self.best_idx]["metrics"][self.best_metric_name]
        curr_val = metric_values[self.best_metric_name]
        if curr_val > best_val:
            self.best_idx = len(self.trials) - 1

    @property
    def best_params(self):
        return self.trials[self.best_idx]["params"]

    @property
    def best_metrics(self):
        return self.trials[self.best_idx]["metrics"]

    def __str__(self):
        if not self.trials:
            return "(no trials)"

        sample_metrics = self.trials[0]["metrics"]
        param_keys = sorted(self.trials[0]["params"].keys())

        headers = param_keys + list(sample_metrics.keys()) + ["Time (s)"]
        data = []
        index = []
        for i, t in enumerate(self.trials):
            row = [str(t["params"][k]) for k in param_keys]
            row += [NUM_FMT.format(v) for v in t["metrics"].values()]
            row.append(TIME_FMT.format(t["time"]))
            data.append(row)
            marker = " *" if i == self.best_idx else ""
            index.append(f"trial-{i}{marker}")

        title = f"\n--- Random Search: {self.dataset_name} / {self.model_name} ---"
        table = _table_format(data, headers, index, h_bars=[1])
        best_line = f"\nBest trial: {self.best_idx}  params={self.best_params}"
        return title + "\n" + table + best_line
