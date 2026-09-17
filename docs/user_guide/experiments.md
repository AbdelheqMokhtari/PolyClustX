# Experiments and evaluation

Training is unsupervised in both modes. **Benchmark evaluation** only enables
comparison with ground-truth labels; it does not pass labels to a model.

| Mode | Labels | Metrics | Cluster counts |
| --- | --- | --- | --- |
| Unlabeled / real-world | Not needed | Internal | Manually chosen |
| Benchmark evaluation | Required for external scores | Internal and/or external | Manually chosen |

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
from pymv.experiment import Experiment, RandomSearchExperiment
from pymv.models import LMGEC
from pymv.metrics import silhouette_score, davies_bouldin_score

# data is a MultiViewDataset created or loaded earlier.
Experiment(
    data,
    [LMGEC(n_clusters=k, embedding_dim=4, random_state=42) for k in [2, 3, 7]],
    {"Silhouette": silhouette_score, "DB": davies_bouldin_score},
    benchmark=False,
).run()

search = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={"n_clusters": [2, 3, 7], "embedding_dim": [4, 8]},
    dataset=data, metrics={"DB": davies_bouldin_score}, best_metric="DB",
    n_iter=6, seed=42, benchmark=False,
)
result = search.run()
```

Random search honors metric direction, including DB minimization. A fixed seed
in `Experiment(runs=...)` repeats that model seed. To measure initialization
variability, supply separate seeded models or use the workbench, which records
`base_seed + repeat_index`.

Custom functions default to external, higher-is-better scores. Set
`fn.requires_features = True` for an internal function and
`fn.higher_is_better = False` for minimization.

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
the manual K field in both modes. YAML top-level keys `variant`, `missing_rate`,
`impute`, and `no_benchmark` correspond to these switches. Use `--data-seed`
or YAML `data_seed` when missingness and model initialization need separate seeds.
