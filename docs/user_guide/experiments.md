# Experiments and evaluation

Training is unsupervised in both modes. **Benchmark evaluation** only enables
comparison with ground-truth labels; it does not pass labels to a model.

| Mode | Labels | Metrics | Cluster counts |
| --- | --- | --- | --- |
| Unlabeled / real-world | Not needed | Internal | Manually chosen |
| Benchmark evaluation | Required for external scores | Internal and/or external | Interface defaults to the number of true classes; editable |

## Metrics and model selection

External scores include ACC, NMI, ARI and F1. Internal scores include Silhouette
(higher is better), Calinski–Harabasz (higher), and Davies–Bouldin (lower).
Runners dispatch the correct inputs: true labels for external scores, and the
model's learned embedding for internal scores.

Internal indices require between two and n−1 distinct predicted clusters.
Undefined scores are NaN, not fabricated zeros. Internal indices describe
geometry, not semantic correctness. Comparisons across different embedding
spaces need care. Silhouette can be expensive for large datasets.

```python
from pymv.experiment import Experiment, RandomSearchExperiment, GridSearchExperiment
from pymv.models import LMGEC
from pymv.metrics import silhouette_score, davies_bouldin_score

# data is a MultiViewDataset created or loaded earlier.
comparison = Experiment(
    data,
    [LMGEC(n_clusters=k, embedding_dim=4, random_state=42) for k in [2, 3, 7]],
    {"Silhouette": silhouette_score, "DB": davies_bouldin_score},
    benchmark=False,
    runs=5,
    save_path="experiments/comparison",
).run()

search = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={"n_clusters": [2, 3, 7], "embedding_dim": [4, 8]},
    dataset=data, metrics={"DB": davies_bouldin_score}, best_metric="DB",
    n_iter=6, seed=42, benchmark=False,
)
result = search.run()

grid = GridSearchExperiment(
    model_class=LMGEC,
    param_grid={"n_clusters": [2, 3, 7], "embedding_dim": [4, 8]},
    dataset=data, metrics={"DB": davies_bouldin_score}, best_metric="DB",
    runs=3, seed=42, benchmark=False, save_path="experiments/search",
).run()
print(grid.best_params)
```

Both searches optimize the selected metric's mean over repeated runs. For NMI,
pass `metrics={"NMI": nmi_score}`, `best_metric="NMI"`, and labeled data with
`benchmark=True`. Grid search evaluates every Cartesian combination; random
search samples candidate lists with replacement. Random search also accepts
callables, but their random state is the callable's responsibility. Grid search
requires finite candidate sequences. A configuration with any failed run cannot
win; if no configuration has a finite objective, `best_trial` is `None`.

For models exposing `random_state`, run 1 uses the model's supplied seed (or
the runner's `seed`, default 0), run 2 adds one, and so on. Search uses the same
sequence for each configuration. This makes repetitions reproducible without
repeating an identical initialization. Exact numerical reproducibility can still
depend on the model, hardware and numerical libraries.

Custom functions default to external, higher-is-better scores. Set
`fn.requires_features = True` for an internal function and
`fn.higher_is_better = False` for minimization.

## Optuna: adaptive discrete or continuous search

```python
from pymv.experiment import OptunaSearchExperiment
from pymv.models import LMGEC
from pymv.metrics import nmi_score

search = OptunaSearchExperiment(
    LMGEC,
    {"n_clusters": [data.n_clusters], "embedding_dim": [4, 8],
     "temperature": {"low": .1, "high": 5., "log": True}},
    data, metrics={"NMI": nmi_score}, best_metric="NMI",
    n_iter=10, runs=3, seed=42, save_path="experiments/optuna",
)
result = search.run()
print(result.best_params)
print(search.study_.trials)  # full Optuna trial objects
```

Lists restrict choices to your selected values. A range uses `low`, `high`,
optional `log`, and optional `step`; add `"type": "int"` for integer ranges.
Float ranges are continuous when `step` is omitted. Logarithmic bounds must
be positive; a float range cannot combine logarithmic sampling and a step.
Use grid search when every candidate combination must be evaluated. Optuna's
TPE sampler uses prior trial scores to suggest subsequent configurations.

The selected metric is optimized using the mean of repeated fits, with the
same seeds per configuration. Failed/nonfinite trials cannot win. All fits
finish normally; pruning is not enabled. The standard JSON report records
strategy, search space, each trial's parameters, scores, diagnostics and
best-so-far steps. Optuna is included in the normal installation.
Layer-width choices can be nested lists, for example `hidden_dims=[[64,32],[128,64]]`.
Optuna stores such choices as indices internally; `result.best_params` and
exported reports contain the actual widths. A trial's `resolved_parameters`
user attribute also preserves them.
[Optuna trial API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).

## Prepare a comparison batch

Datasets and model configurations can be prepared separately. `Experiment`
evaluates their Cartesian product, one fit at a time. Use dataset callables to
load one dataset at a time instead of retaining a collection of matrices.

```python
from pymv.data import load_data

datasets = {
    "acm": lambda: load_data("acm"),
    "dblp": lambda: load_data("dblp"),
}
models = [
    LMGEC(n_clusters=3, embedding_dim=8, max_iter=30),
    LMGEC(n_clusters=3, embedding_dim=16, max_iter=30),
]
results = Experiment(
    datasets, models, {"Silhouette": silhouette_score},
    benchmark=False, runs=10, seed=42,
    experiment_name="Representation comparison", save_path="experiments/comparison",
).run()
print(results.json_path)
```

In Python, K is the explicit value on each model, not inferred during fitting.
To use the benchmark default, construct a model with `n_clusters=data.n_clusters`.
Prepare separate jobs if datasets need different K values or evaluation modes.
Failed fits are recorded with errors while subsequent fits continue. Structural
errors such as unsupported missing views are rejected before fitting that pair.

## Saved results

Every `Experiment`, `RandomSearchExperiment`, and `GridSearchExperiment` writes
one uniquely named JSON report by default. `save_path=` selects its directory;
the default is `experiments/` under the **current working directory**, not inside
Python's installed package directory, which may be read-only. Set
`save_json=False` to disable automatic saving. Optional Markdown reports continue
to work; their detailed JSON is placed in the same report folder.

```python
report = results.to_dict()
results.export("reports/comparison.json")
results.export("reports/comparison.csv")
results.export("reports/comparison.zip")
```

The versioned JSON format contains:

- Library/schema versions, report ID, name, start and completion timestamps.
- Dataset details, model class/module and publication metadata where available,
  and each configuration's hyperparameters.
- Every run's parameters, timestamps, duration, status, metrics and errors.
- Mean, population standard deviation (`ddof=0`) and finite-score counts.
- Raw and normalized loss histories, view weights, inertia and per-view
  diagnostics when the model exposes them. Unavailable diagnostics and nonfinite
  scores are `null`, never invented zeros.
- For searches: strategy, objective, direction, search space, seed, tuning steps,
  best score so far and winning configuration.

Normalization is `(loss − minimum) / (maximum − minimum)` within each run;
constant histories become zero. It describes optimization shape, not a common
objective scale across unrelated methods. LMGEC's per-view diagnostics describe
**initialization** inertia, not final per-view scores. Final inertia is the sum
of squared distances to the assigned centroid in the learned representation.

CSV has one row per run and one column per hyperparameter. Python ZIP exports
contain the JSON report and CSV table. Feature matrices and fitted models are
not serialized. The interface's per-record ZIP additionally includes compressed
prediction, embedding and visualization-label arrays.

## Compare losses across experiments

```python
from pymv.visualization import plot_loss_comparison

ax = plot_loss_comparison([comparison, grid], run=1, show=False)
ax.figure.savefig("loss-comparison.svg", bbox_inches="tight")
```

Run 1 represents every configuration by default. Set `run=2` to choose another
repeat or `run=None` to include all repeats. Loaded JSON report dictionaries are
accepted too. For a single fitted model, use
`plot_loss(model.loss_history_, normalize=True)`.

## Command-line use

```bash
pymv-cli --dataset pascal --variant clip --n-clusters 5
pymv-cli --dataset acm --no-benchmark --n-clusters 3 --metrics silhouette db ch
pymv-cli --dataset acm --missing-rate 0.2 --impute mean --seed 42 --n-clusters 3
pymv-cli --dataset acm --no-benchmark --n-clusters 3 \
  --random-search --metrics db --best-metric db --seed 42
```

For compatibility, CLI benchmark mode can infer K from labels when omitted.
`--no-benchmark` requires a manual `--n-clusters`. The workbench always uses
an editable dataset class-count default for benchmarks, or explicitly selected
model K values. Real-world runs use manual K. YAML top-level keys `variant`, `missing_rate`,
`impute`, and `no_benchmark` correspond to these switches. Use `--data-seed`
or YAML `data_seed` when missingness and model initialization need separate seeds.
