<p align="center"><img src="src/pymv/display/assets/logo.png" width="120" alt="pymv logo"></p>

<p align="center">
  <a href="https://github.com/AbdelheqMokhtari/pymv/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/AbdelheqMokhtari/pymv?color=8A1538"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white">
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

Use Python **3.10+** (3.11 recommended). Installation includes the interface,
Python API, CLI, and Optuna. There is no separate interface installation.

From PyPI, once this version is published:

```bash
python -m pip install pymv-clustering
```

Or from the repository:

```bash
git clone https://github.com/AbdelheqMokhtari/pymv.git
cd pymv
python -m pip install .
```

Launch the interface with **`pymv display`** (or `pymv-display`), then open the
address printed in the terminal, normally `http://localhost:8501`.
An isolated environment is recommended; see the
[installation guide](https://pymv.readthedocs.io/en/latest/user_guide/installation.html).
PyTorch is installed automatically; TensorFlow is not required.

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
for evaluation settings, random/grid/Optuna search, and choosing a search objective.
Reports include per-run scores, mean ± standard deviation, hyperparameters,
timestamps, and available loss histories and model diagnostics. Use
`results.export("comparison.zip")` for a JSON/CSV bundle. Without `save_path`,
reports go to `./experiments/` in the current working directory.

## Available methods

Every method exposes cluster assignments and a learned representation through
the common model API. New methods can be added through the model registry.

“Graphs” means the method uses supplied adjacency matrices, not just internally
constructed similarities. Missing-view ✓ means native observation-mask support.

| Category | Method | Graphs | Missing views | Original publication |
|---|---|:---:|:---:|---|
| Graph-aware | LMGEC | ✓ | — | [Fettal et al., WSDM 2023](https://doi.org/10.1145/3539597.3570367) |
| Graph-aware | DGF | ✓ | — | [Zheng et al., Dual Graph Filtering](https://arxiv.org/abs/2511.20030) |
| Graph-aware | SGLA | ✓ | — | [Li et al., ICDE 2025](https://arxiv.org/abs/2508.09452) |
| Feature-based | MultiViewKMeans | — | — | [Bickel & Scheffer, ICDM 2004](https://doi.org/10.1109/ICDM.2004.10095) |
| Feature-based | MultiViewSpectralClustering | — | — | [Kumar & Daumé, ICML 2011](https://icml.cc/2011/papers/272_icmlpaper.pdf) |
| Feature-based | G-CoMVKM | — | — | [Sinaga & Yang, Electronics 2025](https://doi.org/10.3390/electronics14112129) |
| Feature-based | IDEC | — | — | [Guo et al., IJCAI 2017](https://www.ijcai.org/proceedings/2017/243) |
| Feature-based | CONAN | — | — | [Ke et al., BigData 2021](https://doi.org/10.1109/BigData52589.2021.9671851) |
| Feature-based | SCM | — | — | [Luo et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/519) |
| Feature-based | MFLVC | — | — | [Xu et al., CVPR 2022](https://github.com/SubmissionsIn/MFLVC) |
| Feature-based | UMC | — | — | [Zhang et al., ACL 2024](https://aclanthology.org/2024.acl-long.2/) |
| Incomplete-view | COMPLETER | — | ✓ | [Lin et al., CVPR 2021](https://github.com/XLearning-SCU/2021-CVPR-Completer) |
| Incomplete-view | DCP | — | ✓ | [Lin et al., TPAMI](https://doi.org/10.1109/TPAMI.2022.3197238) |

COMPLETER and DCP learn missing-view completion from at least two fully observed
training samples; other methods require complete or explicitly imputed views.
Single-view reductions and extensions beyond a paper's original view count are
documented in the [model tutorial](https://pymv.readthedocs.io/en/latest/user_guide/clustering_models.html).
All models are available in the interface, Python API and model registry.
UMC uses user-supplied features, without downloading pretrained encoders.
DGF builds kNN graphs when none are supplied; SGLA combines feature kNN graphs
with supplied graphs. Implementations are not claims of reproduced paper scores.
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
random, grid or Optuna search with a selectable metric, name and copy the best configuration into
an experiment, and repeat fits 1–20 times to obtain mean ± standard deviation.
Benchmark mode starts enabled for labeled datasets and can be disabled on the
Dataset page. Compare normalized losses across runs and download JSON, CSV or ZIP
results and plots.

Upload individual view files or NPZ/MATLAB containers. Labels may be in a
separate file, included in the same upload batch, or stored in a selected column
of a view. Select their location explicitly so labels are excluded from training.
Synthetic examples can include binary, mutual or RBF-weighted kNN graphs, with
one or several relations per view, built separately or shared across views.

[Read the interface guide](https://pymv.readthedocs.io/en/latest/user_guide/workbench.html).

## Command line

```bash
pymv models                       # list methods
pymv models idec                  # inspect parameters
pymv datasets
pymv run acm --model lmgec --param embedding_dim=8 --runs 5
pymv search acm --model lmgec --strategy grid --vary 'temperature=[0.5,1,2]'
pymv search acm --model lmgec --strategy optuna --vary 'temperature={"low":0.1,"high":5.0}' --trials 10
```

Repeat `--param name=value` for fixed settings and `--vary name=values` for
searched settings. Use `--objective nmi`, or choose internal metrics with
`--no-benchmark --clusters 3 --metrics silhouette --objective silhouette`.
Reports go to `experiments/`; choose another folder with `--output`.
The older `pymv-cli` remains supported.
[CLI reference](https://pymv.readthedocs.io/en/latest/api/cli.html).

## Examples and local documentation

The [notebooks](notebooks/) demonstrate offline multi-dataset comparisons,
repeated-run tables, graph inputs, missing views and hyperparameter search.
In a repository checkout:

```bash
make docs-build    # build HTML, installing documentation tools if necessary
make docs-serve    # serve existing HTML at http://127.0.0.1:8000
make docs-deploy   # trigger Read the Docs (requires project access)
```

Deployment builds a version already pushed to GitHub, not uncommitted files.

## Extend the framework

Implement the common model API and register your method to compare it with
existing approaches. The [model contribution guide](https://pymv.readthedocs.io/en/latest/contributing/models.html)
covers implementation, tests, citations, and pull requests.

## License

MIT. See [LICENSE](LICENSE). Cite the original algorithm papers when reporting
results with their implementations.
