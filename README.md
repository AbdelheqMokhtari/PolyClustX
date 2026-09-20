<p align="center"><img src="src/pymv/display/assets/logo.png" width="120" alt="pymv logo"></p>

<p align="center">
  <a href="https://github.com/AbdelheqMokhtari/pymv/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/AbdelheqMokhtari/pymv?color=8A1538"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white">
  <a href="https://github.com/AbdelheqMokhtari/pymv/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AbdelheqMokhtari/pymv/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://pymv.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Read%20the%20Docs-475569"></a>
</p>

# pymv

A comparative framework for **multi-view and multimodal clustering and
representation learning** in Python.

Use a common workflow to prepare data, compare algorithms, inspect learned
representations, and report repeated experiments. Work with one or several
feature views, optional graph relations, and optional ground-truth labels.
Use Python, the command line, or the graphical interface.

[Documentation](https://pymv.readthedocs.io/en/latest/) ·
[Getting started](https://pymv.readthedocs.io/en/latest/user_guide/quickstart.html) ·
[Dataset catalog](https://pymv.readthedocs.io/en/latest/catalog/datasets.html)

## Installation

Install from a repository checkout with Python 3.9 or newer:

```bash
git clone https://github.com/AbdelheqMokhtari/pymv.git
cd pymv
pip install -e .
```

To include the graphical interface:

```bash
pip install -e ".[display]"
pymv-display
```

Open the address printed in the terminal, normally `http://localhost:8501`.
For UMAP projections, also install `pip install -e ".[umap]"`.

## A first experiment

```python
from pymv.data import load_data
from pymv.models import LMGEC

data = load_data("acm")  # downloaded once, then cached locally
model = LMGEC(n_clusters=3, embedding_dim=16, random_state=42)
data.fit(model)

predictions = model.labels_
representation = model.embedding_
```

To compare methods or parameter settings over several runs:

```python
from pymv.experiment import Experiment
from pymv.metrics import clustering_accuracy, nmi_score

results = Experiment(
    datasets=["acm"],
    models=[
        LMGEC(n_clusters=3, embedding_dim=8, temperature=1.0),
        LMGEC(n_clusters=3, embedding_dim=16, temperature=2.0),
    ],
    metrics=[clustering_accuracy, nmi_score],
    runs=5,
    save_path="experiments/comparison",
).run()
print(results.json_path)  # one JSON report containing every configuration and run
```

External metrics such as ACC, NMI, and ARI require labels. Internal metrics
such as Silhouette, Calinski–Harabasz, and Davies–Bouldin evaluate clustering
without labels. See the [experiment guide](https://pymv.readthedocs.io/en/latest/user_guide/experiments.html)
for evaluation settings, random/grid search, and choosing a search objective.
Reports include per-run scores, mean ± standard deviation, hyperparameters,
timestamps, and available loss histories and model diagnostics. Use
`results.export("comparison.zip")` for a JSON/CSV bundle. Without `save_path`,
reports go to `./experiments/` in the current working directory.

## Available methods

Every method exposes cluster assignments and a learned representation through
the common model API. New methods can be added through the model registry.

| Method | Input support | Original publication |
|---|---|---|
| LMGEC | Feature views with optional graphs | Fettal, Labiod & Nadif. [Simultaneous Linear Multi-view Attributed Graph Representation Learning and Clustering](https://doi.org/10.1145/3539597.3570367), WSDM 2023. |
| SCM | Feature views; graphs are not used | Luo et al. [Simple Contrastive Multi-View Clustering with Data-Level Fusion](https://www.ijcai.org/proceedings/2024/519), IJCAI 2024. |

Both current implementations require complete or explicitly imputed views.
See [algorithm details and scope](https://pymv.readthedocs.io/en/latest/background/algorithms.html)
and the [model catalog](https://pymv.readthedocs.io/en/latest/catalog/models.html).

## Bring your own data

```python
from pymv.data import MultiViewDataset

# Each array has samples in rows and features in columns.
data = MultiViewDataset([view_1, view_2], labels=None)
data.fit(model)
```

Views share the same sample order. Use `align_views` with sample IDs when they
contain different samples. The dataset also supports modality names, an explicit
observation mask, and zero, one, or several graph relations per view.

Catalog datasets with alternative feature representations use `variant=`:

```python
from pymv.data import list_variants, load_data

print(list_variants("pascal"))
data = load_data("pascal", variant="clip")
```

See [loading and aligning data](https://pymv.readthedocs.io/en/latest/user_guide/data.html),
[missing views](https://pymv.readthedocs.io/en/latest/user_guide/missing_views.html),
and [graph relations](https://pymv.readthedocs.io/en/latest/user_guide/graphs.html).

## Graphical interface

The sidebar lets you move between datasets, models, experiments,
hyperparameter search, results, visualizations, and exports. Prepare multiple
named datasets and model configurations in either order. Load, inspect and
preprocess each dataset on the Dataset page, then click **Apply changes**.
Compare all compatible combinations or choose pairs in a dataset-by-model table. Run
random or grid search with a selectable metric, name and copy the best configuration into
an experiment, and repeat fits 1–20 times to obtain mean ± standard deviation.
Benchmark mode starts enabled for labeled datasets and can be disabled on the
Dataset page. Compare normalized losses across runs and download JSON, CSV or ZIP
results and plots.

Upload individual view files or NPZ/MATLAB containers. Labels may be in a
separate file, included in the same upload batch, or stored in a selected column
of a view. Select their location explicitly so labels are excluded from training.

[Read the interface guide](https://pymv.readthedocs.io/en/latest/user_guide/workbench.html).

## Command line and saved configurations

```bash
pymv-cli --dataset acm --model lmgec --runs 5 --seed 42
pymv-cli --config experiment --dataset acm --runs 5
```

Optional YAML/JSON configs save your dataset, parameters, metrics, seeds, and
search space in a reusable recipe. They are useful for repeating or sharing
experiments without retyping a long command; Python and interface users do not
need them. See the [CLI reference](https://pymv.readthedocs.io/en/latest/api/cli.html).

## Extend the framework

Implement the common model API and register your method to compare it with
existing approaches. The [model contribution guide](https://pymv.readthedocs.io/en/latest/contributing/models.html)
covers implementation, tests, citations, and pull requests.

## License

MIT. See [LICENSE](LICENSE). Cite the original algorithm papers when reporting
results with their implementations.
