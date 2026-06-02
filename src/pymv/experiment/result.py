"""
Result formatting utilities for pymv experiments.

Provides clean, well-aligned ASCII tables with box-drawing
characters for experiment results.  Two table types:

1. **Config table** — one row per model, shows hyper-parameters.
2. **Results table** — one per dataset, shows metric scores with
   best-value highlighting.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

NUM_FMT = "{:.4f}"
TIME_FMT = "{:.2f}s"
MEAN_STD_FMT = "{:.4f} (±{:.4f})"


def _in_notebook() -> bool:
    """Return ``True`` when running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _terminal_width() -> int:
    """Best-effort terminal width; fall back to 120."""
    import shutil
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 120


# Table renderer

def _render_horizontal(
    rows: list[list[str]],
    headers: list[str],
    index: list[str],
    title: str | None,
    index_header: str,
    pad: int,
) -> str:
    """Standard horizontal table (columns = params)."""
    n_cols = len(headers)
    sp = " " * pad

    idx_w = max((len(s) for s in index), default=0)
    idx_w = max(idx_w, len(index_header))

    col_ws = []
    for c in range(n_cols):
        w = len(headers[c])
        for r in rows:
            w = max(w, len(r[c]))
        col_ws.append(w)

    def _sep(left, mid, right, fill="─"):
        parts = [fill * (idx_w + 2 * pad)]
        for w in col_ws:
            parts.append(fill * (w + 2 * pad))
        return left + mid.join(parts) + right

    top = _sep("┌", "┬", "┐")
    mid = _sep("├", "┼", "┤")
    bottom = _sep("└", "┴", "┘")

    def _row(label, cells):
        parts = [f"{sp}{label.ljust(idx_w)}{sp}"]
        for c, w in zip(cells, col_ws):
            parts.append(f"{sp}{c.rjust(w)}{sp}")
        return "│" + "│".join(parts) + "│"

    lines = []
    if title:
        lines.append("")
        lines.append(f"  {title}")

    lines.append(top)
    lines.append(_row(index_header, headers))
    lines.append(mid)
    for idx_label, row in zip(index, rows):
        lines.append(_row(idx_label, row))
    lines.append(bottom)

    return "\n".join(lines)


def _render_transposed(
    rows: list[list[str]],
    headers: list[str],
    index: list[str],
    title: str | None,
    index_header: str,
    pad: int,
) -> str:
    """Transposed table: rows = params, columns = models.

    Used automatically when the horizontal layout is too wide for the
    terminal.  Each parameter becomes a row, each model a column.
    """
    sp = " " * pad

    # First column is the parameter name, then one column per model
    t_index = headers                          # param names
    t_headers = index                          # model names
    t_index_header = "Parameter"

    t_rows: list[list[str]] = []
    for c in range(len(headers)):
        t_rows.append([rows[r][c] for r in range(len(rows))])

    idx_w = max((len(s) for s in t_index), default=0)
    idx_w = max(idx_w, len(t_index_header))

    col_ws = []
    for c in range(len(t_headers)):
        w = len(t_headers[c])
        for r in t_rows:
            w = max(w, len(r[c]))
        col_ws.append(w)

    def _sep(left, mid, right, fill="─"):
        parts = [fill * (idx_w + 2 * pad)]
        for w in col_ws:
            parts.append(fill * (w + 2 * pad))
        return left + mid.join(parts) + right

    top = _sep("┌", "┬", "┐")
    mid = _sep("├", "┼", "┤")
    bottom = _sep("└", "┴", "┘")

    def _row(label, cells):
        parts = [f"{sp}{label.ljust(idx_w)}{sp}"]
        for c, w in zip(cells, col_ws):
            parts.append(f"{sp}{c.rjust(w)}{sp}")
        return "│" + "│".join(parts) + "│"

    lines = []
    if title:
        lines.append("")
        lines.append(f"  {title}")

    lines.append(top)
    lines.append(_row(t_index_header, t_headers))
    lines.append(mid)
    for param_name, row in zip(t_index, t_rows):
        lines.append(_row(param_name, row))
    lines.append(bottom)

    return "\n".join(lines)


def _table_width(
    index: list[str],
    headers: list[str],
    rows: list[list[str]],
    index_header: str,
    pad: int,
) -> int:
    """Calculate the width of the horizontal table without building it."""
    idx_w = max((len(s) for s in index), default=0)
    idx_w = max(idx_w, len(index_header))
    total = idx_w + 2 * pad + 1  # index col + left border

    for c in range(len(headers)):
        w = len(headers[c])
        for r in rows:
            w = max(w, len(r[c]))
        total += w + 2 * pad + 1  # col + separator

    return total


def _build_table(
    rows: list[list[str]],
    headers: list[str],
    index: list[str],
    title: str | None = None,
    index_header: str = "Model",
    best_cols: dict[int, int] | None = None,
    pad: int = 1,
) -> str:
    """
    Render an ASCII table with box-drawing characters.

    In **Jupyter notebooks** the table always uses the horizontal
    layout (models = rows) and is wrapped in a scrollable HTML
    ``<pre>`` block so that wide tables never wrap or overlap.

    In a **terminal** the table automatically **transposes**
    (rows ↔ columns) when the horizontal layout would be wider
    than the terminal, so the output stays readable.

    Parameters
    ----------
    rows : list[list[str]]
        Pre-formatted cell values.
    headers : list[str]
        Column header labels.
    index : list[str]
        Row labels (first column).
    title : str, optional
        Title line printed above the table.
    index_header : str
        Label for the first (index) column.
    best_cols : dict[col_idx, row_idx], optional
        Map of column index -> row index of the best value.
        The best cell gets a ``*`` suffix.
    pad : int
        Extra spaces on each side of a cell.
    """
    if best_cols is None:
        best_cols = {}

    n_cols = len(headers)

    # Mark best values with *
    for col_i, row_i in best_cols.items():
        if 0 <= row_i < len(rows) and 0 <= col_i < n_cols:
            rows[row_i][col_i] = rows[row_i][col_i] + " *"

    notebook = _in_notebook()

    if notebook:
        # In notebooks, always use horizontal layout (models = rows).
        # Jupyter renders output in a <pre> block; wide tables get a
        # horizontal scrollbar automatically, so line-wrapping is not
        # an issue.  The transposed layout (models = columns) breaks
        # when there are many models because columns become too narrow.
        return _render_horizontal(
            rows, headers, index, title, index_header, pad,
        )
    else:
        # Terminal: transpose when the table is wider than the terminal
        # to avoid line-wrapping that breaks box-drawing alignment.
        width = _table_width(index, headers, rows, index_header, pad)
        term_w = _terminal_width()
        if width > term_w:
            return _render_transposed(
                rows, headers, index, title, index_header, pad,
            )
        return _render_horizontal(
            rows, headers, index, title, index_header, pad,
        )


# Config table builder

def _build_config_table(
    models: list,
    model_names: list[str],
) -> str:
    """
    Build a table showing model hyper-parameters.

    One row per model.  Columns = union of all parameter names
    across models (excluding private attrs like ``labels_``).

    Preprocessing-only parameters (``beta``, ``tf_idf``, ``center``,
    ``scale``) are **hidden** when every model in the table has
    ``preprocess=False``, since those values have no effect in that
    mode.
    """
    # Collect all parameter keys across models
    _SKIP = {
        "labels_", "centroids_", "embedding_", "loss_history_",
        "_name",
    }

    # Parameters that only matter when preprocess=True
    _PREPROCESS_ONLY = {"beta", "tf_idf", "center", "scale"}

    # Determine whether preprocessing params should be shown:
    # hide them only when *every* model has preprocess=False.
    all_preprocess_off = all(
        getattr(m, "preprocess", True) is False for m in models
    )

    all_keys: list[str] = []
    seen: set[str] = set()
    for m in models:
        for k in m.__dict__:
            if k not in seen and k not in _SKIP and not k.endswith("_"):
                if all_preprocess_off and k in _PREPROCESS_ONLY:
                    continue
                all_keys.append(k)
                seen.add(k)

    headers = all_keys
    index = model_names

    rows = []
    for m in models:
        row = []
        for k in all_keys:
            val = getattr(m, k, "—")
            row.append(str(val))
        rows.append(row)

    return _build_table(
        rows, headers, index,
        title="Model Configuration",
        index_header="Model",
    )


# Result containers

class Result:
    """
    Stores metric results for a single (dataset, model) run.

    Attributes
    ----------
    dataset_name : str
    model_name : str
    metric_avg : OrderedDict[str, float]
        Metric name -> mean value.
    metric_std : OrderedDict[str, float]
        Metric name -> standard deviation (0.0 if single run).
    train_time : float
        Mean seconds spent in ``fit()``.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        metric_avg: dict,
        metric_std: dict | None = None,
        train_time: float = 0.0,
        alpha_str: str = "",
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.metric_avg = OrderedDict(metric_avg)
        self.metric_std = OrderedDict(metric_std) if metric_std else OrderedDict(
            {k: 0.0 for k in metric_avg}
        )
        self.train_time = train_time
        self.alpha_str = alpha_str

    def format_cell(self, metric_name: str, show_std: bool = False) -> str:
        mean = self.metric_avg[metric_name]
        if show_std and self.metric_std.get(metric_name, 0.0) > 0:
            return MEAN_STD_FMT.format(mean, self.metric_std[metric_name])
        return NUM_FMT.format(mean)

    def __repr__(self):
        return (
            f"Result({self.dataset_name!r}, {self.model_name!r}, "
            f"metrics={dict(self.metric_avg)}, time={self.train_time:.2f}s)"
        )


class ExperimentResult:
    """
    Collects results across multiple datasets and models.

    Provides a formatted multi-table string grouped by dataset,
    with the best value per metric highlighted.
    """

    def __init__(self):
        self._results: dict[str, list[Result]] = OrderedDict()
        self._models: list = []
        self._model_names: list[str] = []

    def set_models(self, models: list, model_names: list[str]):
        """Store the model list for the config table."""
        self._models = models
        self._model_names = model_names

    def add(self, result: Result):
        ds = result.dataset_name
        if ds not in self._results:
            self._results[ds] = []
        self._results[ds].append(result)

    @property
    def all_results(self) -> dict[str, list[Result]]:
        return self._results

    def _find_best(self, results: list[Result]) -> dict[int, int]:
        """Return {col_idx: row_idx} for highest mean per metric."""
        if not results:
            return {}
        metric_names = list(results[0].metric_avg.keys())
        best: dict[int, int] = {}
        for col_i, m in enumerate(metric_names):
            best_val = -np.inf
            best_row = 0
            for row_i, r in enumerate(results):
                v = r.metric_avg.get(m, -np.inf)
                if not np.isnan(v) and v > best_val:
                    best_val = v
                    best_row = row_i
            best[col_i] = best_row
        return best

    def config_table(self) -> str:
        """Return the model-configuration table string."""
        if not self._models:
            return ""
        return _build_config_table(self._models, self._model_names)

    def results_tables(self) -> str:
        """Return all per-dataset result tables as a single string."""
        show_std = any(
            any(v > 0 for v in r.metric_std.values())
            for rs in self._results.values()
            for r in rs
        )

        parts: list[str] = []
        for ds_name, results in self._results.items():
            if not results:
                continue

            metric_names = list(results[0].metric_avg.keys())
            has_alphas = any(getattr(r, "alpha_str", "") for r in results)
            
            headers = metric_names.copy()
            if has_alphas:
                headers.append("Alphas")
            headers.append("Time")

            rows = []
            index = []
            for r in results:
                row = [r.format_cell(m, show_std=show_std) for m in metric_names]
                if has_alphas:
                    row.append(getattr(r, "alpha_str", ""))
                row.append(TIME_FMT.format(r.train_time))
                rows.append(row)
                index.append(r.model_name)

            best = self._find_best(results) if len(results) > 1 else {}
            table = _build_table(
                rows, headers, index,
                title=f"Results — {ds_name}",
                index_header="Model",
                best_cols=best,
            )
            parts.append(table)

        return "\n".join(parts)

    def __str__(self):
        sections = []
        cfg = self.config_table()
        if cfg:
            sections.append(cfg)
        res = self.results_tables()
        if res:
            sections.append(res)
        return "\n".join(sections)


class RandomSearchResult:
    """
    Stores results of a random hyper-parameter search.

    Tracks all trials and identifies the best configuration
    by the chosen metric (higher is better).
    """

    def __init__(self, dataset_name: str, model_name: str, best_metric_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.best_metric_name = best_metric_name
        self.trials: list[dict] = []
        self.best_idx: int = 0

    def add_trial(self, params: dict, metric_values: dict, train_time: float):
        self.trials.append({
            "params": dict(params),
            "metrics": OrderedDict(metric_values),
            "time": train_time,
        })
        # Update best (higher is better)
        curr_val = metric_values.get(self.best_metric_name, -np.inf)
        best_val = self.trials[self.best_idx]["metrics"].get(
            self.best_metric_name, -np.inf
        )
        if not np.isnan(curr_val) and curr_val > best_val:
            self.best_idx = len(self.trials) - 1

    @property
    def best_params(self) -> dict:
        return self.trials[self.best_idx]["params"]

    @property
    def best_metrics(self) -> OrderedDict:
        return self.trials[self.best_idx]["metrics"]

    @property
    def best_score(self) -> float:
        """Best value of the target metric across all trials."""
        return float(self.best_metrics.get(self.best_metric_name, float("nan")))

    def __str__(self):
        if not self.trials:
            return "(no trials)"

        sample = self.trials[0]
        param_keys = sorted(sample["params"].keys())
        metric_names = list(sample["metrics"].keys())

        headers = param_keys + metric_names + ["Time"]

        rows = []
        index = []
        for i, t in enumerate(self.trials):
            row = [str(t["params"][k]) for k in param_keys]
            row += [NUM_FMT.format(t["metrics"][m]) for m in metric_names]
            row.append(TIME_FMT.format(t["time"]))
            rows.append(row)
            tag = "★ best" if i == self.best_idx else f"#{i + 1}"
            index.append(tag)

        # Best column = the one for best_metric_name
        bm_col = len(param_keys) + metric_names.index(self.best_metric_name)
        best_cols = {bm_col: self.best_idx}

        title = (
            f"Random Search — {self.model_name} on {self.dataset_name}  "
            f"({len(self.trials)} trials, best by {self.best_metric_name})"
        )

        table = _build_table(
            rows, headers, index,
            title=title,
            index_header="Trial",
            best_cols=best_cols,
        )

        bp = self.best_params
        bm = self.best_metrics
        summary = (
            f"\n  Best params  : {bp}\n"
            f"  Best metrics : "
            + ", ".join(f"{k}={v:.4f}" for k, v in bm.items())
        )
        return table + summary
