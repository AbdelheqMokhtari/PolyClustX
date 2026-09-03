# pymv

**A comparative framework for multi-view and multi-modal clustering in Python.**

`pymv` makes it easy to **benchmark**, **compare**, and **develop** multi-view clustering algorithms.
It provides a growing collection of models, **55 built-in dataset loaders** (auto-downloaded from Zenodo), a full **experiment / hyper-parameter-search harness**, and standard **clustering metrics** — all designed so you can go from idea to publishable results table in a few lines of code.

> **Currently implemented:**
> - **LMGEC** — Linear Multi-view Graph Embedding Clustering (the reference baseline).
> - **LMGECplus** — a modular research framework where every stage (loss,
>   normalization, propagation, attention, view weighting, alignment,
>   clustering head, optimizer) is an independently replaceable component,
>   with built-in ablation support.
>
> More algorithms are on the roadmap — the architecture is ready for drop-in additions.

---

## Quick start

### 30-second example

```python
from pymv.data import load_acm
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score

Xs, As, y = load_acm()                          # auto-downloads on first call

labels = LMGEC(n_clusters=3, embedding_dim=16).fit_predict(Xs, As)

print("ACC:", clustering_accuracy(y, labels))    # e.g. 0.91
print("NMI:", nmi_score(y, labels))              # e.g. 0.68
```

### Compare configurations with one call

```python
from pymv.experiment import Experiment
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score

Experiment(
    datasets=["acm", "dblp", "imdb"],
    models=[
        LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0),
        LMGEC(n_clusters=3, embedding_dim=16, temperature=10.0),
    ],
    metrics=[clustering_accuracy, nmi_score],
).run()
```

The console prints **two clean tables** — model configuration + per-dataset results.
Everything else (loading, per-run scores, errors) is saved to a log file in `./logs/`.

### Run on a single view

pymv always expects **lists** of views. To run the algorithm on just one
view, wrap the single feature matrix (and adjacency) in a list:

```python
from pymv.data import load_flickr30k
from pymv.models import LMGEC
from pymv.experiment import Experiment
from pymv.metrics import clustering_accuracy, nmi_score, ari_score

data = load_flickr30k()
Xs, As, labels = data

# ⚠ Wrap the single view in a list — [Xs[0]], not Xs[0]
model = LMGEC(
    n_clusters=data.n_clusters,
    embedding_dim=data.n_clusters + 1,
    preprocess=True,
    tf_idf=False,
    center=True,
    scale=False,
)
model.name = "LMGEC base"

result = Experiment(
    datasets={"flickr30k_view0": ([Xs[0]], [As[0]], labels)},
    models=model,
    metrics=[clustering_accuracy, nmi_score, ari_score],
    runs=5,
    verbose=2,
).run()
```

> **Key point:** `([Xs[0]], [As[0]], labels)` — square brackets turn
> the single array into a one-element list, which is what the model and
> the Experiment expect. Writing `(Xs[0], As[0], labels)` without the
> wrapping lists will raise an error.

---

## Table of contents

1. [Installation](#installation)
2. [Package structure](#package-structure)
3. [Quickstart — in depth](#quickstart--in-depth)
4. [Tutorial — full workflow with per-view preprocessing](#tutorial--full-workflow-with-per-view-preprocessing)
5. [Datasets](#datasets)
6. [Models](#models)
7. [Preprocessing](#preprocessing)
8. [Metrics](#metrics)
9. [Experiment framework](#experiment-framework)
10. [Random search](#random-search)
11. [Visualization](#visualization)
12. [Command-line interface](#command-line-interface)
13. [Config-based experiments](#config-based-experiments)
14. [Algorithm — LMGEC intuition](#algorithm--lmgec-intuition)
15. [Tips & troubleshooting](#tips--troubleshooting)
16. [License](#license)

---

## Installation

Requires **Python >= 3.8**.

```bash
pip install -e .            # editable / development install
```

Core dependencies (`numpy`, `scipy`, `scikit-learn`, `networkx`, `pandas`, `tensorflow >= 2.10`) are declared in `pyproject.toml`.

```bash
# CPU-only TensorFlow (if not already installed)
pip install "tensorflow>=2.10"

# Optional — for visualization
pip install matplotlib seaborn
```

---

## Package structure

```
pyproject.toml
src/pymv/
    __init__.py                     # top-level exports, __version__
    cli.py                          # pymv-cli command-line interface
    core/
        base_model.py               # BaseModel ABC (all models inherit this)
    models/
        lmgec/lmgec.py              # LMGEC (first algorithm — more coming)
    data/
        dataset.py                  # MultiViewDataset container class
        registry.py                 # dataset catalog: one DatasetSpec per dataset,
                                    # load_dataset() / list_datasets() / load_<name>()
        readers.py                  # generic .mat readers (one per file layout)
        downloader.py               # Zenodo auto-download helper (atomic, cached)
        base.py                     # cache dir (~/.pymv_datasets, or $PYMV_DATA_DIR)
    preprocessing/
        features.py                 # normalize, propagate, TF-IDF, center/scale
    metrics/
        clustering.py               # ACC, F1, NMI, ARI, Silhouette, CH, DB
    experiment/
        experiment.py               # Experiment runner
        random_search.py            # RandomSearchExperiment
        result.py                   # Result containers + table formatting
        logger.py                   # dual file + console logging
    visualization/
        visualizer.py               # t-SNE / PCA cluster plots
```

---

## Quickstart — in depth

### Example 1 — Load, fit, evaluate (3 lines)

The simplest workflow: load a dataset, fit a model, evaluate.

```python
from pymv.data import load_acm
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score, ari_score

# 1. Load data - returns a MultiViewDataset (tuple unpacking still works)
Xs, As, y = load_acm()

# 2. Fit and predict
model = LMGEC(n_clusters=3, embedding_dim=16)
labels = model.fit_predict(Xs, As)

# 3. Evaluate
print("ACC:", clustering_accuracy(y, labels))
print("NMI:", nmi_score(y, labels))
print("ARI:", ari_score(y, labels))
```

You can also use named attribute access:

```python
data = load_acm()
print(data)  # MultiViewDataset('acm', n_samples=3025, n_views=2, ...)

model = LMGEC(n_clusters=data.n_clusters, embedding_dim=16)
labels = model.fit_predict(data.Xs, data.As)
print("ACC:", clustering_accuracy(data.labels, labels))
```

### Example 2 — Per-view preprocessing options

LMGEC can apply different preprocessing to each view. Pass lists to `beta` and `tf_idf`:

```python
model = LMGEC(
    n_clusters=3,
    embedding_dim=16,
    tf_idf=[True, False],      # TF-IDF on view 0, plain L2 on view 1
    beta=[1.0, 0.5],           # different self-loop weights per view
    center=True,               # center features after propagation
)
labels = model.fit_predict(Xs, As)
```

### Example 3 — Bring your own preprocessed views

Set `preprocess=False` to skip all internal preprocessing and pass your own pre-computed feature matrices:

```python
from pymv.preprocessing import get_propagated_features, preprocess_features

# Propagate manually
H0 = get_propagated_features(X0, A0, beta=1.0, tf_idf=True)
H1 = get_propagated_features(X1, A1, beta=0.5, tf_idf=False)

# Optionally center / scale
Hs = preprocess_features([H0, H1], center=True, scale=False)

# Fit — no adjacency needed
model = LMGEC(n_clusters=3, embedding_dim=16, preprocess=False)
labels = model.fit_predict(Hs)
```

### Example 4 — Feature-only clustering (no graphs)

If your data has no natural graph structure, just pass the raw feature matrices:

```python
model = LMGEC(n_clusters=5, embedding_dim=32, preprocess=False)
labels = model.fit_predict([X_view1, X_view2, X_view3])
```

### Example 5 — Full comparative experiment

Compare multiple model configurations across multiple datasets with automatic logging:

```python
from pymv.experiment import Experiment
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score, ari_score

# Name your configs for clearer tables
m1 = LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0)
m1.name = "LMGEC-T1"

m2 = LMGEC(n_clusters=3, embedding_dim=16, temperature=10.0)
m2.name = "LMGEC-T10"

m3 = LMGEC(n_clusters=3, embedding_dim=32, temperature=5.0, tf_idf=True)
m3.name = "LMGEC-d32-tfidf"

result = Experiment(
    datasets=["acm", "dblp", "imdb"],
    models=[m1, m2, m3],
    metrics=[clustering_accuracy, nmi_score, ari_score],
    n_runs=10,        # 10 independent runs -> mean +/- std
    verbose=1,        # tables only (default)
).run()

# Access results programmatically
print(result.config_table())     # model hyper-parameters
print(result.results_tables())   # per-dataset results
```

**Console output** (verbose=1):

| Table | Content |
|---|---|
| **Model Configuration** | One row per model, columns = hyper-parameters |
| **Results — acm** | Rows = models, columns = metrics + Time, best marked with `*` |
| **Results — dblp** | ... |
| **Results — imdb** | ... |

> **Smart config table:** When all models have `preprocess=False`, preprocessing-only
> parameters (`beta`, `tf_idf`, `center`, `scale`) are automatically hidden from the
> configuration table since they have no effect in that mode.

### Example 6 — Hyper-parameter search

Find the best configuration for a single dataset:

```python
from pymv.experiment import RandomSearchExperiment
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy, nmi_score

rs = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={
        "n_clusters": [3],
        "embedding_dim": [4, 8, 16, 32, 64],
        "temperature": [0.1, 0.5, 1.0, 5.0, 10.0],
        "beta": [0.1, 0.5, 1.0, 2.0],
        "max_iter": [30],
    },
    dataset="acm",
    metrics=[clustering_accuracy, nmi_score],
    best_metric="clustering_accuracy",
    n_iter=30,
)
result = rs.run()

print("Best params:", result.best_params)
print("Best ACC:   ", result.best_score)
print("All metrics:", result.best_metrics)
```

### Example 7 — Using custom / pre-loaded data with the Experiment

You don't have to use the built-in loaders. Pass your own data:

```python
from pymv.experiment import Experiment
from pymv.models import LMGEC
from pymv.metrics import clustering_accuracy

# Option A: pre-loaded tuple (still works)
my_data = (Xs, As, y)

# Option B: pass a MultiViewDataset directly
data = load_acm()

result = Experiment(
    datasets={"my_dataset": my_data},      # dict with custom name
    models=[LMGEC(n_clusters=5, embedding_dim=16)],
    metrics=[clustering_accuracy],
).run()

# Or pass the dataset object directly
result = Experiment(
    datasets=data,                         # uses data.name automatically
    models=[LMGEC(n_clusters=data.n_clusters, embedding_dim=16)],
    metrics=[clustering_accuracy],
).run()
```

### Example 8 — Visualize the consensus embedding

```python
from pymv.data import load_acm
from pymv.models import LMGEC
from pymv.visualization.visualizer import plot_clusters

Xs, As, y = load_acm()
model = LMGEC(n_clusters=3, embedding_dim=16)
model.fit(Xs, As)

# t-SNE plot colored by predicted clusters
plot_clusters(model.embedding_, model.labels_, method="tsne", title="LMGEC on ACM")

# PCA (faster, deterministic)
plot_clusters(model.embedding_, y, method="pca", title="Ground truth on ACM")
```

---

## Tutorial — full workflow with per-view preprocessing

This tutorial walks through the **complete pymv workflow** — from loading
raw data to running a full comparative experiment — using **only pymv
functions** (no external libraries needed).

We use **Flickr30k** as an example. It has two views (image embeddings
and text embeddings) with no graph structure.

### Step 1 — Load the dataset

```python
from pymv.data import load_flickr30k

# Auto-downloads from Zenodo on first call
data = load_flickr30k()
print(data)
# MultiViewDataset('flickr30k', n_samples=12154, n_views=2, ...)

# Unpack into views, adjacencies, and labels
Xs, As, labels = data
# Xs = [img_features (12154 × 100), txt_features (12154 × 100)]
# As = [None, None]  (no graph structure)
```

### Step 2 — Per-view preprocessing

`pymv.preprocessing` gives you full control over each view independently.
Here we build **new** preprocessed arrays without modifying the originals.

**Goal:**
| View | Normalization | Centering | Scaling |
|---|---|---|---|
| View 0 (image) | L2 | ✓ | ✓ |
| View 1 (text) | L2 | ✓ | ✗ |

```python
from pymv.preprocessing import normalize_features, preprocess_features

# ---- View 0 (image): L2 normalization + centering + scaling ----
H0 = normalize_features(Xs[0], tf_idf=False)          # L2 row-normalization
H0 = preprocess_features(H0, center=True, scale=True)  # center + scale

# ---- View 1 (text): L2 normalization + centering only ----
H1 = normalize_features(Xs[1], tf_idf=False)           # L2 row-normalization
H1 = preprocess_features(H1, center=True, scale=False)  # center only

# Group preprocessed views into a new list
Hs = [H0, H1]
```

> **Tip:** `normalize_features` applies L2 normalization (each row has
> unit norm). Set `tf_idf=True` for TF-IDF + L2 if your features are
> raw term-frequency counts.

### Step 3 — Declare the model

Since we already preprocessed manually, set `preprocess=False` so LMGEC
skips its internal normalization / propagation:

```python
from pymv.models import LMGEC

model = LMGEC(
    n_clusters=data.n_clusters,   # auto-detected from labels
    embedding_dim=16,
    temperature=1.0,
    max_iter=30,
    preprocess=False,              # we already preprocessed
)
```

### Step 4 — Fit and evaluate

```python
from pymv.metrics import clustering_accuracy, nmi_score, ari_score

# Fit with the preprocessed views
pred_labels = model.fit_predict(Hs)

# Evaluate
print("ACC:", clustering_accuracy(labels, pred_labels))
print("NMI:", nmi_score(labels, pred_labels))
print("ARI:", ari_score(labels, pred_labels))
```

### Step 5 — Compare configurations with Experiment

Wrap everything into an `Experiment` to compare multiple configurations
and get clean result tables automatically.

We pass the preprocessed `Hs` as a custom dataset. There are **several
ways** to do this:

```python
from pymv.experiment import Experiment

# Define model variants — all with preprocess=False
m1 = LMGEC(n_clusters=data.n_clusters, embedding_dim=16, temperature=1.0, preprocess=False)
m1.name = "LMGEC-T1"

m2 = LMGEC(n_clusters=data.n_clusters, embedding_dim=16, temperature=10.0, preprocess=False)
m2.name = "LMGEC-T10"

m3 = LMGEC(n_clusters=data.n_clusters, embedding_dim=32, temperature=5.0, preprocess=False)
m3.name = "LMGEC-d32"

metrics = [clustering_accuracy, nmi_score, ari_score]
models  = [m1, m2, m3]

# ── Option A: named dict with a (Xs, As, labels) tuple ──
result = Experiment(
    datasets={"flickr30k_preprocessed": (Hs, As, labels)},
    models=models,
    metrics=metrics,
    runs=5,
    verbose=2,
).run()

# ── Option B: anonymous tuple (auto-named "dataset") ──
result = Experiment(
    datasets=(Hs, As, labels),
    models=models,
    metrics=metrics,
    runs=5,
).run()

# ── Option C: build a MultiViewDataset from the preprocessed views ──
from pymv.data.dataset import MultiViewDataset

preprocessed_data = MultiViewDataset(Hs, As, labels, name="flickr30k_preprocessed")
result = Experiment(
    datasets=preprocessed_data,       # uses .name automatically
    models=models,
    metrics=metrics,
    runs=5,
).run()
```

This prints a **model configuration table** and a **results table** with
mean ± std over 5 runs, with the best values highlighted.

### Step 6 — Hyper-parameter search

Find the best configuration automatically using the preprocessed views:

```python
from pymv.experiment import RandomSearchExperiment

rs = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={
        "n_clusters": [data.n_clusters],
        "embedding_dim": [4, 8, 16, 32, 64],
        "temperature": [0.1, 0.5, 1.0, 5.0, 10.0],
        "max_iter": [30],
        "preprocess": [False],
    },
    dataset={"flickr30k_preprocessed": (Hs, As, labels)},
    metrics=[clustering_accuracy, nmi_score, ari_score],
    best_metric="ACC",
    n_iter=20,
    verbose=2,
)
result = rs.run()

print("Best params:", result.best_params)
print("Best ACC:   ", result.best_score)
```

### Putting it all together

Here is the complete script in one block:

```python
# ── pymv: Full workflow with per-view preprocessing ──────────────

# 1. Load
from pymv.data import load_flickr30k

data = load_flickr30k()
Xs, As, labels = data

# 2. Per-view preprocessing (into new variables)
from pymv.preprocessing import normalize_features, preprocess_features

# View 0 (image): L2 + center + scale
H0 = normalize_features(Xs[0], tf_idf=False)
H0 = preprocess_features(H0, center=True, scale=True)

# View 1 (text): L2 + center only
H1 = normalize_features(Xs[1], tf_idf=False)
H1 = preprocess_features(H1, center=True, scale=False)

Hs = [H0, H1]

# 3. Declare model
from pymv.models import LMGEC

model = LMGEC(
    n_clusters=data.n_clusters,
    embedding_dim=16,
    temperature=1.0,
    preprocess=False,
)

# 4. Fit & evaluate
from pymv.metrics import clustering_accuracy, nmi_score, ari_score

pred = model.fit_predict(Hs)
print("ACC:", clustering_accuracy(labels, pred))
print("NMI:", nmi_score(labels, pred))
print("ARI:", ari_score(labels, pred))

# 5. Full experiment (compare configurations)
from pymv.experiment import Experiment

m1 = LMGEC(n_clusters=data.n_clusters, embedding_dim=16, temperature=1.0, preprocess=False)
m1.name = "LMGEC-T1"

m2 = LMGEC(n_clusters=data.n_clusters, embedding_dim=16, temperature=10.0, preprocess=False)
m2.name = "LMGEC-T10"

m3 = LMGEC(n_clusters=data.n_clusters, embedding_dim=32, temperature=5.0, preprocess=False)
m3.name = "LMGEC-d32"

Experiment(
    datasets={"flickr30k_preprocessed": (Hs, As, labels)},
    models=[m1, m2, m3],
    metrics=[clustering_accuracy, nmi_score, ari_score],
    n_runs=5,
    verbose=2,
).run()
```

---

## Datasets

All loaders return a `MultiViewDataset` object containing feature matrices, adjacency matrices, and ground-truth labels. Datasets are downloaded automatically from Zenodo on first use.

**Tuple unpacking still works** (backward compatible):

```python
Xs, As, y = load_acm()          # classic 3-tuple unpacking
```

**Named attribute access** (new):

```python
data = load_acm()
data.Xs                          # list of feature matrices
data.As                          # list of adjacency matrices (or Nones)
data.labels                      # ground-truth label vector
data.n_samples                   # number of samples
data.n_views                     # number of views
data.n_clusters                  # number of unique labels
data.feature_dims                # [dim_v0, dim_v1, ...]
data.has_graphs                  # True if any adjacency is not None
print(data)                      # one-line summary
print(data.summary())            # multi-line summary
```

| Function | Dataset | Type |
|---|---|---|
| `load_acm()` | ACM | Attributed graph (2 meta-paths) |
| `load_amazon_photos()` | Amazon Photos | Attributed graph |
| `load_bdgp()` | BDGP | Multi-view features |
| `load_coco_all()` | COCO (all views) | Multi-view features |
| `load_coco_cross()` | COCO (cross-modal) | Multi-view features |
| `load_dblp()` | DBLP | Attributed graph (3 meta-paths) |
| `load_imdb()` | IMDB | Attributed graph (2 meta-paths) |
| `load_pascal()` | Pascal | Multi-view features |
| `load_prokaryotic()` | Prokaryotic | Multi-view features |
| `load_wiki()` | Wikipedia | Attributed graph |

**Unified access:**

```python
from pymv.data import load_dataset, list_datasets

print(list_datasets())
# ['acm', 'amazon_photos', 'bdgp', 'coco_all', 'coco_cross',
#  'dblp', 'imdb', 'pascal', 'prokaryotic', 'wiki']

data = load_dataset("acm")     # returns MultiViewDataset
Xs, As, y = load_dataset("acm")  # tuple unpacking also works
```

---

## Models

All models inherit from `pymv.core.BaseModel` and expose `fit()` / `fit_predict()`.

> The framework is designed to host many algorithms. Currently **LMGEC** is the first — contributions and new models are welcome.

### LMGEC

```python
from pymv.models import LMGEC
```

**Constructor:**

```python
LMGEC(
    n_clusters: int,
    embedding_dim: int,
    temperature: float = 1.0,
    beta: float | list[float] = 1.0,
    tf_idf: bool | list[bool] = False,
    center: bool = False,
    scale: bool = False,
    preprocess: bool = True,
    max_iter: int = 30,
)
```

| Parameter | Description |
|---|---|
| `n_clusters` | Number of clusters *k*. |
| `embedding_dim` | Latent dimension *m* for per-view projections. |
| `temperature` | Softmax temperature for view-weight alpha. Lower = sharper weighting. |
| `beta` | Self-loop weight for graph propagation. `float` (all views) or `list[float]` (per-view). |
| `tf_idf` | Normalization mode. `True` = TF-IDF + L2; `False` = plain L2. `bool` or `list[bool]` (per-view). |
| `center` / `scale` | Center / scale features after propagation. |
| `preprocess` | `True` (default): internal preprocessing. `False`: user provides pre-processed views; `As`, `beta`, `tf_idf`, `center`, `scale` are all ignored. |
| `max_iter` | Alternating-optimisation iterations. |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `fit(Xs, As=None)` | `self` | Fit the model. |
| `fit_predict(Xs, As=None)` | `np.ndarray` | Fit and return cluster labels. |

**Attributes (after fit):**

| Attribute | Shape | Description |
|---|---|---|
| `labels_` | `(n,)` | Cluster assignments. |
| `centroids_` | `(k, m)` | Cluster centroids in consensus space. |
| `embedding_` | `(n, m)` | Consensus embedding. |
| `loss_history_` | `(max_iter,)` | Loss per iteration. |
| `name` | `str` | Display name (get/set), used in experiment tables. |

---

## Preprocessing

```python
from pymv.preprocessing import (
    get_propagated_features,   # full pipeline: normalize -> propagate
    preprocess_features,       # center / scale one or more views
    preprocess_graph,          # build S = D^-1(A + beta*I)
    normalize_features,        # TF-IDF or L2 normalization
)
```

| Function | Description |
|---|---|
| `get_propagated_features(X, A, beta, tf_idf)` | Normalize features, build propagation matrix S, return H = S*X. |
| `preprocess_features(Xs, center, scale)` | Center and/or scale features. Accepts a single matrix or a list. |
| `preprocess_graph(A, beta)` | Row-normalized propagation matrix S = D^-1(A + beta*I). |
| `normalize_features(X, tf_idf)` | TF-IDF + L2 or plain L2 normalization. |

---

## Metrics

```python
from pymv.metrics import (
    # External (require ground truth)
    clustering_accuracy,       # ACC — Hungarian alignment
    clustering_f1_score,       # F1  — Hungarian alignment
    nmi_score,                 # Normalized Mutual Information
    ari_score,                 # Adjusted Rand Index
    # Internal (no ground truth)
    silhouette_score,          # Silhouette Coefficient  [-1, 1] higher is better
    calinski_harabasz_score,   # Variance Ratio Criterion  higher is better
    davies_bouldin_score,      # Davies-Bouldin Index  lower is better
)
```

### External metrics

| Function | Range | Notes |
|---|---|---|
| `clustering_accuracy(y_true, y_pred)` | [0, 1] | Label alignment via Hungarian algorithm. Higher is better. |
| `clustering_f1_score(y_true, y_pred, average="macro")` | [0, 1] | F1 after alignment. Supports `macro`, `micro`, `weighted`. |
| `nmi_score(y_true, y_pred)` | [0, 1] | Alignment-free; shared information. Higher is better. |
| `ari_score(y_true, y_pred)` | [-1, 1] | Alignment-free; chance-corrected. Higher is better. |

### Internal metrics

| Function | Range | Notes |
|---|---|---|
| `silhouette_score(X, labels)` | [-1, 1] | Mean silhouette coefficient. Higher is better. |
| `calinski_harabasz_score(X, labels)` | [0, inf) | Between / within dispersion ratio. Higher is better. |
| `davies_bouldin_score(X, labels)` | [0, inf) | Average max cluster similarity. Lower is better. |

---

## Experiment framework

```python
from pymv.experiment import Experiment
```

```python
Experiment(
    datasets,          # str | list[str] | list[callable] | dict | tuple
    models,            # BaseModel | list[BaseModel]
    metrics,           # callable | list[callable] | dict[str, callable]
    n_runs: int = 1,
    verbose: int = 1,  # 0=silent  1=tables  2=progress+tables  3=debug
    log_dir: str = "./logs",
)
```

### Flexible inputs

```python
# Datasets — any of these work
datasets = ["acm", "dblp"]                         # names
datasets = [load_acm, load_dblp]                   # callables
datasets = {"acm": load_acm, "mine": (Xs, As, y)}  # dict mix
datasets = "acm"                                   # single string
datasets = (Xs, As, y)                             # single tuple

# Models
models = model                         # single
models = [model_a, model_b]            # list

# Metrics
metrics = clustering_accuracy          # single
metrics = [clustering_accuracy, nmi_score]
metrics = {"ACC": clustering_accuracy} # dict with custom names
```

### Verbose levels

| Level | Console output |
|---|---|
| `0` | Silent |
| `1` | Config table + result tables only **(default)** |
| `2` | One-liner progress per (dataset x model) + tables |
| `3` | Full debug (same detail as the log file) |

### Logging

- All execution details go to timestamped log files in `log_dir` (default `./logs/`).
- Data-loader messages (downloading, loading, sample counts) are **log-file only** — they never clutter the console.

### Result object

```python
result = exp.run()

print(result)                    # config table + all result tables
print(result.config_table())     # model hyper-parameters only
print(result.results_tables())   # per-dataset results only
```

---

## Random search

```python
from pymv.experiment import RandomSearchExperiment

result = RandomSearchExperiment(
    model_class=LMGEC,
    param_distributions={...},
    dataset="acm",
    metrics=[clustering_accuracy, nmi_score],
    best_metric="clustering_accuracy",
    n_iter=30,
).run()

print(result.best_params)     # best hyper-parameter dict
print(result.best_score)      # best value of the target metric
print(result.best_metrics)    # all metrics for the best trial
```

---

## Visualization

```python
from pymv.visualization.visualizer import plot_clusters

plot_clusters(model.embedding_, model.labels_, method="tsne", title="LMGEC clusters")
plot_clusters(model.embedding_, model.labels_, method="pca",  title="LMGEC clusters (PCA)")
```

Requires `matplotlib` (`pip install matplotlib`).

---

## Dataset variants

```python
# Via the variant parameter directly
from pymv.data import load_pascal
ds = load_pascal(variant="clip")

# Via the registry (load_dataset)
from pymv.data import load_dataset
ds = load_dataset("pascal_clip")
ds = load_dataset("coco_all_vit_sbert")
ds = load_dataset("coco_cross_resnet_minilm")
```

## Command-line interface

`pymv` ships with **`pymv-cli`** — run experiments directly from the terminal without writing Python code.

### Setup

The CLI is installed automatically with `pymv`:

```bash
pip install -e .          # editable install
pymv-cli --help           # verify
```

### Discover available datasets and models

```bash
pymv-cli --list-datasets
pymv-cli --list-models
```

### Run an experiment

```bash
# Simplest — default parameters, 1 run
pymv-cli --dataset acm

# Custom parameters
pymv-cli --dataset dblp --beta 2 --temperature 100 --tf-idf true

# Multiple runs (shows mean ± std)
pymv-cli --dataset acm --beta 2 --temperature 100 --runs 3

# Choose metrics (short names: acc, nmi, ari, f1)
pymv-cli --dataset acm --metrics acc nmi f1

# Set number of clusters explicitly
pymv-cli --dataset acm -k 3 --embedding-dim 32

# Skip internal preprocessing
pymv-cli --dataset acm --no-preprocess
```

### Random hyper-parameter search

```bash
# Search 20 random configs (default)
pymv-cli --dataset acm --random-search

# More trials, fixed seed
pymv-cli --dataset acm --random-search --n-trials 50 --seed 42

# Fix some params, search others
pymv-cli --dataset dblp --random-search --beta 2 --n-trials 30

# Optimise by NMI instead of ACC
pymv-cli --dataset acm --random-search --best-metric nmi
```

### Save results

```bash
# Save Markdown report + log to experiments/ folder
pymv-cli --dataset acm --beta 2 --temperature 100 --save

# With a custom experiment name
pymv-cli --dataset acm --save --name acm_baseline

# Write a log file only (no Markdown)
pymv-cli --dataset acm --log
```

### Parameter reference

| Flag | Short | Description | Default |
|---|---|---|---|
| `--dataset` | `-d` | Dataset name | *(required)* |
| `--model` | `-m` | Model name | `lmgec` |
| `--n-clusters` | `-k` | Number of clusters | auto-detected |
| `--embedding-dim` | | Latent dimension | `16` |
| `--temperature` | | Softmax temperature | `1.0` |
| `--beta` | | Self-loop weight (scalar or list) | `1.0` |
| `--tf-idf` | | TF-IDF normalization | `false` |
| `--center` | | Center features | `false` |
| `--scale` | | Scale features | `false` |
| `--no-preprocess` | | Disable preprocessing | off |
| `--max-iter` | | Max iterations | `30` |
| `--runs` | `-r` | Independent runs | `1` |
| `--metrics` | | Metrics to report | `acc nmi ari` |
| `--verbose` | `-v` | Verbosity (0–3) | `1` |
| `--random-search` | | Enable random search | off |
| `--n-trials` | | Random search trials | `20` |
| `--best-metric` | | Metric to optimise | `acc` |
| `--seed` | | Random seed | none |
| `--save` | | Save Markdown report | off |
| `--name` | | Experiment folder name | auto |
| `--log` | | Write log file | off |
| `--config` | `-c` | Path to YAML/JSON config file | none |

---

## Config-based experiments

Instead of passing every flag on the command line, you can define your
experiment in a **YAML** (or JSON) config file and run it with a single
command:

```bash
pymv-cli --config configs/experiment.yaml
```

### Config file format

```yaml
# configs/experiment.yaml
dataset: acm

model:
  name: lmgec
  params:
    beta: 2
    temperature: 100
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

All sections except `dataset` are optional — sensible defaults are used
for anything you don't specify.

### Minimal config

```yaml
# configs/minimal.yaml
dataset: dblp
```

This is equivalent to `pymv-cli --dataset dblp`.

### Random search via config

```yaml
# configs/random_search.yaml
dataset: acm

model:
  name: lmgec

experiment:
  random_search: true
  n_trials: 20
  best_metric: acc
  seed: 42
  metrics: [acc, nmi, ari]

search_space:
  embedding_dim: [4, 8, 16, 32, 64]
  temperature: [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
  beta: [0.1, 0.5, 1.0, 2.0, 5.0]
  tf_idf: [true, false]

output:
  save: true
  name: acm_search
```

```bash
pymv-cli --config configs/random_search.yaml
```

### CLI overrides

CLI flags **always take priority** over config values, so you can reuse
the same config for multiple datasets:

```bash
# Config says dataset: acm, but we override to dblp
pymv-cli --config configs/experiment.yaml --dataset dblp

# Override runs
pymv-cli --config configs/experiment.yaml --runs 5

# Override beta and add verbose
pymv-cli --config configs/experiment.yaml --beta 5 -v 2
```

### Shorthand names

If your config is inside the `configs/` directory, you can skip the
path and just use the filename (without extension):

```bash
pymv-cli --config default           # → configs/default.yaml
pymv-cli --config experiment        # → configs/experiment.yaml
pymv-cli --config random_search     # → configs/random_search.yaml
```

### Config field reference

| Section | Field | Description | Default |
|---|---|---|---|
| *(top)* | `dataset` | Dataset name | *(required)* |
| `model` | `name` | Model name | `lmgec` |
| `model.params` | `embedding_dim` | Latent dimension | `16` |
| `model.params` | `temperature` | Softmax temperature | `1.0` |
| `model.params` | `beta` | Self-loop weight | `1.0` |
| `model.params` | `tf_idf` | TF-IDF normalization | `false` |
| `model.params` | `center` | Center features | `false` |
| `model.params` | `scale` | Scale features | `false` |
| `model.params` | `preprocess` | Enable preprocessing | `true` |
| `model.params` | `max_iter` | Max iterations | `30` |
| `experiment` | `runs` | Independent runs | `1` |
| `experiment` | `metrics` | Metrics list | `[acc, nmi, ari]` |
| `experiment` | `verbose` | Verbosity (0–3) | `1` |
| `experiment` | `random_search` | Enable random search | `false` |
| `experiment` | `n_trials` | Random search trials | `20` |
| `experiment` | `best_metric` | Metric to optimise | `acc` |
| `experiment` | `seed` | Random seed | none |
| `output` | `save` | Save Markdown report | `false` |
| `output` | `name` | Experiment folder name | auto |
| `output` | `log` | Write log file | `false` |
| `search_space` | `<param>` | List of values to sample | *(built-in ranges)* |

---

## Algorithm — LMGEC intuition

LMGEC (Linear Multi-view Graph Embedding Clustering) jointly learns a shared low-dimensional representation and cluster assignments across multiple views.

**Pipeline.** Given *V* views with features X_v and adjacencies A_v:

1. **Propagate** — build S_v = D_v^{-1}(A_v + beta * I), compute H_v = S_v * X_v.
2. **Project** — learn orthonormal W_v via SVD (Procrustes).
3. **Consensus** — weighted combination: XW = sum_v alpha_v * (H_v * W_v), where alpha_v is proportional to exp(-inertia_v / tau).
4. **Cluster** — assign by nearest centroid in the consensus space.
5. **Alternate** — repeat projection, assignment, centroid updates for `max_iter` iterations.

---

## Tips & troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: tensorflow` | `pip install "tensorflow>=2.10"` |
| Slow on large datasets | Reduce `max_iter`; use `preprocess=False` with pre-computed views |
| GPU acceleration | Install `tensorflow[and-cuda]` |
| Visualization missing | `pip install matplotlib seaborn` |
| Log files filling up | Change `log_dir` or set `verbose=0` |
| Data download fails | Check internet; files are on Zenodo |

---

## Roadmap

- [ ] More multi-view clustering algorithms (spectral, deep, contrastive, ...)
- [ ] Multi-modal support (text + image, graph + tabular, ...)
- [ ] Benchmark tables on all built-in datasets
- [ ] Pre-built configurations for reproducible paper comparisons
- [ ] PyPI release

Contributions are welcome! See [issues](https://github.com/AbdelheqMokhtari/pymv/issues).

---

## License

MIT License — see [LICENSE](LICENSE).
