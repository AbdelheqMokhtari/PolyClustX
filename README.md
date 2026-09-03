<h1 align="center">pymv</h1>

<p align="center">
  <strong>A comparative framework for multi-view and multi-modal clustering in Python.</strong>
</p>

<p align="center">
  <a href="https://github.com/AbdelheqMokhtari/pymv/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AbdelheqMokhtari/pymv/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Version" src="https://img.shields.io/badge/version-1.1.1-orange">
</p>

`pymv` makes it easy to **benchmark**, **compare** and **develop** multi-view
clustering algorithms. It ships a reference model, **55 dataset loaders** that
auto-download from Zenodo, an experiment and hyper-parameter-search harness,
and the standard clustering metrics — so you can go from an idea to a results
table in a few lines.

**Implemented model:** `LMGEC` — Linear Multi-view Graph Embedding Clustering.
New models drop in through a one-decorator registry; see
[Adding a model](#adding-a-model).

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [The LMGEC model](#the-lmgec-model)
- [Preprocessing](#preprocessing)
- [Metrics](#metrics)
- [Experiments](#experiments)
- [Random search](#random-search)
- [Visualization](#visualization)
- [Command-line interface](#command-line-interface)
- [Config files](#config-files)
- [Adding a model](#adding-a-model)
- [Notebooks](#notebooks)
- [Development](#development)
- [License](#license)

---

## Installation

Requires **Python ≥ 3.9**.

```bash
git clone https://github.com/AbdelheqMokhtari/pymv.git
cd pymv
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `scikit-learn`, `matplotlib`,
`tensorflow>=2.10`, `pyyaml`, `tqdm`, `requests`) install automatically.

```bash
pip install -e ".[dev]"     # + pytest, ruff, build
pip install -e ".[docs]"    # + sphinx and the docs theme
pip install -e ".[umap]"    # + UMAP projection for plot_clusters
```

---

## Quick start

```python
from pymv.data import load_acm
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score

Xs, As, y = load_acm()                       # auto-downloads on first call

labels = LMGEC(n_clusters=3, embedding_dim=16).fit_predict(Xs, As)

print("ACC:", clustering_accuracy(y, labels))
print("NMI:", nmi_score(y, labels))
```

Compare configurations across datasets in one call:

```python
from pymv.experiment import Experiment

Experiment(
    datasets=["acm", "dblp", "imdb"],
    models=[
        LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0),
        LMGEC(n_clusters=3, embedding_dim=16, temperature=10.0),
    ],
    metrics=[clustering_accuracy, nmi_score],
    runs=5,
).run()
```

The console prints a model-configuration table and a per-dataset results
table with mean ± std; per-run detail goes to a log file.

> **Views are always lists.** To cluster a single view, wrap it:
> `([Xs[0]], [As[0]], y)` — not `(Xs[0], As[0], y)`.

---

## Datasets

Every loader returns a `MultiViewDataset`, which also unpacks as a
`(Xs, As, labels)` tuple.

```python
from pymv.data import load_dataset, list_datasets, get_dataset_info

print(list_datasets())            # 55 names
print(get_dataset_info("acm"))    # views, samples, clusters, source

data = load_dataset("acm")
data.Xs            # list of feature matrices, one per view
data.As            # list of adjacency matrices (entries may be None)
data.labels        # ground-truth labels
data.n_samples, data.n_views, data.n_clusters
data.feature_dims  # [dim_v0, dim_v1, ...]
data.has_graphs    # True if any adjacency is present
print(data.summary())
```

Datasets are downloaded on first use and cached under `~/.pymv_datasets`
(override with the `PYMV_DATA_DIR` environment variable).

| Kind | Examples |
|---|---|
| Attributed graphs | `acm`, `dblp`, `imdb`, `cora`, `citeseer`, `wiki`, `webkb`, `amazon_photos` |
| Multi-view features | `100leaves`, `3sources`, `bbcsport`, `bdgp`, `caltech101_7`, `handwritten`, `msrc_v5`, `prokaryotic`, `uci`, `yale` |
| Multi-modal | `coco_all`, `coco_cross`, `pascal`, `flickr30k`, `nus_wide`, `esp_game`, `iaprtc12`, `wikipedia` |

Several multi-modal datasets have **encoder variants** — the same samples
with different feature extractors:

```python
from pymv.data import load_pascal, load_dataset

load_pascal(variant="clip")           # explicit variant
load_dataset("pascal_clip")           # or as a registry name
load_dataset("coco_all_vit_sbert")
```

---

## The LMGEC model

```python
LMGEC(
    n_clusters,                     # number of clusters k
    embedding_dim,                  # latent dimension m
    temperature=1.0,                # softmax temperature for view weights
    beta=1.0,                       # self-loop weight (float or per-view list)
    tf_idf=False,                   # TF-IDF + L2 vs plain L2 (bool or list)
    center=False,                   # center features
    scale=False,                    # scale features to unit variance
    preprocess=True,                # master switch for internal preprocessing
    max_iter=30,                    # alternating-optimisation iterations
    random_state=None,              # seed the SVD/KMeans initialisation
)
```

**How it works.** Each view is normalized and propagated through its graph
(`H = S·X`). LMGEC then alternates three closed-form updates: an orthogonal
projection `W` per view (via SVD), a consensus embedding weighted by per-view
coefficients `α`, and cluster assignments `G` with centroids `F`. The `α` come
from a softmax over each view's initial KMeans inertia — noisier views get
down-weighted automatically. `temperature` controls how sharp that weighting
is.

**`preprocess=False`** turns off all internal preprocessing: views are fed
straight into the optimiser, `As` is ignored, and `beta` / `tf_idf` /
`center` / `scale` have no effect. Use it when you prepare views yourself.

```python
from pymv.preprocessing import get_propagated_features

H0 = get_propagated_features(X0, A0, beta=1.0, tf_idf=True)
H1 = get_propagated_features(X1, A1, beta=0.5, tf_idf=False)

LMGEC(n_clusters=3, embedding_dim=16, preprocess=False).fit([H0, H1])
```

**Attributes after `fit`:**

| Attribute | Shape | Description |
|---|---|---|
| `labels_` | `(n,)` | Cluster assignments |
| `centroids_` | `(k, m)` | Centroids in consensus space |
| `embedding_` | `(n, m)` | Consensus embedding |
| `loss_history_` | `(max_iter,)` | Reconstruction loss per iteration |
| `alphas_` | `(n_views,)` | Normalized per-view weights |
| `embedding_dim_` | `int` | Dimension actually used (capped by the lowest-rank view) |

`model.name` is settable and is what appears in experiment tables.

---

## Preprocessing

```python
from pymv.preprocessing import (
    get_propagated_features,   # normalize -> propagate: H = S·X
    normalize_features,        # TF-IDF + L2, or plain L2
    preprocess_graph,          # S = D⁻¹(A + beta·I)
    preprocess_features,       # center and/or scale
)
```

The order inside `get_propagated_features` is center/scale (column-wise) →
L2 normalize (row-wise) → propagate, so feature distributions are aligned
before being projected onto the unit sphere.

---

## Metrics

```python
from pymv.metrics import (
    clustering_accuracy, clustering_f1_score, nmi_score, ari_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)
```

| Function | Range | Needs labels | Better |
|---|---|---|---|
| `clustering_accuracy(y_true, y_pred)` | [0, 1] | yes | higher |
| `clustering_f1_score(y_true, y_pred, average="macro")` | [0, 1] | yes | higher |
| `nmi_score(y_true, y_pred)` | [0, 1] | yes | higher |
| `ari_score(y_true, y_pred)` | [-1, 1] | yes | higher |
| `silhouette_score(X, labels)` | [-1, 1] | no | higher |
| `calinski_harabasz_score(X, labels)` | [0, ∞) | no | higher |
| `davies_bouldin_score(X, labels)` | [0, ∞) | no | lower |

ACC and F1 align predicted to true labels with the Hungarian algorithm.

---

## Experiments

```python
Experiment(
    datasets,               # names, tuples, MultiViewDataset, or a {name: data} dict
    models,                 # a model or a list of them
    metrics,                # metric callables
    runs=1,                 # independent re-fits per model
    verbose=1,              # 0 silent · 1 tables · 2 progress · 3 debug
    save_markdown=False,    # write a Markdown report
    experiment_name=None,
    save_path=None,         # default ./experiments
    log=False,
    log_dir=None,
).run()
```

`datasets` is flexible — all of these work:

```python
datasets=["acm", "dblp"]                        # registry names
datasets=(Xs, As, y)                            # one anonymous tuple
datasets={"my_data": (Xs, As, y)}               # named
datasets=[load_acm(), load_dblp()]              # MultiViewDataset objects
```

Each run deep-copies the model, so runs are independent. `run()` returns an
`ExperimentResult`:

```python
result = Experiment(...).run()

for dataset_name, rows in result.all_results.items():
    for r in rows:
        print(dataset_name, r.model_name,
              r.metric_avg["clustering_accuracy"],
              r.metric_std["clustering_accuracy"])

print(result.config_table())      # the printed tables, as strings
print(result.results_tables())
```

---

## Random search

```python
from pymv.experiment import RandomSearchExperiment

search = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={
        "n_clusters": [3],
        "embedding_dim": [8, 16, 32, 64],
        "temperature": [0.1, 1.0, 10.0, 100.0],
        "beta": [0.5, 1.0, 2.0],
        "tf_idf": [True, False],
    },
    dataset="acm",
    metrics=[clustering_accuracy, nmi_score],
    best_metric="clustering_accuracy",
    n_iter=20,
    seed=42,
).run()

print(search.best_params, search.best_score)
best_model = LMGEC(**search.best_params).fit(Xs, As)
```

A list is sampled uniformly; a `(low, high)` tuple samples from that range;
a callable is called for each trial.

---

## Visualization

```python
from pymv.visualization import plot_clusters, plot_loss

plot_clusters(model.embedding_, model.labels_, method="tsne", title="LMGEC")
plot_clusters(model.embedding_, model.labels_, method="pca")   # faster, deterministic
plot_loss(model.loss_history_)
```

---

## Command-line interface

```bash
pymv-cli --list-datasets
pymv-cli --list-models

pymv-cli --dataset acm                                   # defaults, 1 run
pymv-cli --dataset dblp --beta 2 --temperature 100 --runs 3
pymv-cli --dataset acm --metrics acc nmi f1 -k 3 --embedding-dim 32
pymv-cli --dataset acm --no-preprocess

pymv-cli --dataset acm --random-search --n-trials 50 --seed 42
pymv-cli --dataset acm --random-search --best-metric nmi

pymv-cli --dataset acm --save --name acm_baseline        # Markdown report
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--dataset` | `-d` | Dataset name | *required* |
| `--model` | `-m` | Model name | `lmgec` |
| `--n-clusters` | `-k` | Number of clusters | auto |
| `--embedding-dim` | | Latent dimension | `16` |
| `--temperature` | | Softmax temperature | `1.0` |
| `--beta` | | Self-loop weight | `1.0` |
| `--tf-idf` / `--center` / `--scale` | | Preprocessing switches | `false` |
| `--no-preprocess` | | Disable internal preprocessing | off |
| `--max-iter` | | Iterations | `30` |
| `--runs` | `-r` | Independent runs | `1` |
| `--metrics` | | `acc` `nmi` `ari` `f1` | `acc nmi ari` |
| `--verbose` | `-v` | 0–3 | `1` |
| `--random-search` | | Enable random search | off |
| `--n-trials` | | Search trials | `20` |
| `--best-metric` | | Metric to optimise | `acc` |
| `--seed` | | Random seed | none |
| `--save` / `--name` / `--save-path` | | Markdown report output | off |
| `--log` | | Write a log file | off |
| `--config` | `-c` | YAML/JSON config file | none |

---

## Config files

Put the whole experiment in YAML and run it with one flag. Four configs ship
with the package and are addressable by shorthand name:

```bash
pymv-cli --config default
pymv-cli --config experiment
pymv-cli --config random_search
pymv-cli --config minimal

pymv-cli --config experiment --dataset dblp --runs 5   # CLI flags override
```

```yaml
dataset: acm

model:
  name: lmgec
  params:
    embedding_dim: 16
    temperature: 100
    beta: 2
    tf_idf: true
    preprocess: true
    max_iter: 30

experiment:
  runs: 3
  metrics: [acc, nmi, ari, f1]
  verbose: 1

output:
  save: true
  name: acm_tuned
```

For a search, add `random_search: true` under `experiment` and a
`search_space:` block mapping each parameter to its candidate values.
Only `dataset:` is required; everything else falls back to defaults.

---

## Adding a model

Subclass `BaseModel`, decorate it, and add one import — nothing else in the
framework needs to change. The CLI, config system and `Experiment` pick it up
automatically.

```python
# src/pymv/models/mymodel/mymodel.py
from pymv.core import BaseModel
from pymv.models.registry import register_model


@register_model("mymodel")
class MyModel(BaseModel):
    def __init__(self, n_clusters, embedding_dim=16):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim

    def fit(self, Xs, As=None):
        ...
        self.labels_ = ...
        return self
```

Then add `from .mymodel import MyModel` to `src/pymv/models/__init__.py`.
Constructor defaults become the CLI/config defaults automatically.

---

## Notebooks

Four runnable notebooks in [`notebooks/`](notebooks/) — each executes offline
in seconds on synthetic data, with a commented line to swap in a real
benchmark.

| # | Notebook | Stage |
|---|---|---|
| 00 | [`00_quickstart.ipynb`](notebooks/00_quickstart.ipynb) | the whole pipeline at a glance |
| 01 | [`01_data_and_preprocessing.ipynb`](notebooks/01_data_and_preprocessing.ipynb) | the dataset catalog, normalize → propagate |
| 02 | [`02_lmgec_baseline.ipynb`](notebooks/02_lmgec_baseline.ipynb) | LMGEC step by step |
| 03 | [`03_experiments_and_tuning.ipynb`](notebooks/03_experiments_and_tuning.ipynb) | repeated comparisons + random search |

---

## Development

```bash
pip install -e ".[dev]"

pytest -q                    # test suite
ruff check src/ tests/       # lint
python -m build              # wheel + sdist
```

`make help` lists the shortcuts. CI runs lint, the test suite on Python
3.9/3.10/3.11, and a packaging check on every push to `main` and `develop`.

Docs are built with Sphinx:

```bash
pip install -e ".[docs]"
cd docs && make html
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TypeError` about array shapes on one view | Views must be lists: `[Xs[0]]`, not `Xs[0]`. |
| `embedding_dim exceeds the rank limit` warning | The dimension is capped by the lowest-rank view; lower `embedding_dim`. |
| One `alpha` near 1, the rest near 0 | Raise `temperature` to soften the view weighting. |
| Noisy TensorFlow / CUDA logs | `export TF_CPP_MIN_LOG_LEVEL=3`, or use `--verbose 1`. |
| Download failure on first load | Check the network; delete the partial file in `~/.pymv_datasets` and retry. |

---

## License

MIT — see [LICENSE](LICENSE).
