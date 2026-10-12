<p align="center"><img src="src/polyclustx/display/assets/logo.png" width="150" alt="PolyClustX logo"></p>

<p align="center">
  <a href="https://github.com/AbdelheqMokhtari/PolyClustX/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/AbdelheqMokhtari/PolyClustX?color=4E4E92"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-557AA3?logo=python&logoColor=white">
  <a href="https://github.com/AbdelheqMokhtari/PolyClustX/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AbdelheqMokhtari/PolyClustX/actions/workflows/ci.yml/badge.svg?branch=develop"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-634E87"></a>
  <a href="https://polyclustx.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Read%20the%20Docs-4E4E92"></a>
</p>

# PolyClustX

**A Unified Framework for Multi-View and Multimodal Representation Learning and Clustering from Feature Views to Attributed Graphs**

Prepare data, compare algorithms, tune hyperparameters and visualize results
through one Python API, command line or graphical interface. Work with one or
many feature views, optional graphs and optional ground-truth labels.

[Documentation](https://polyclustx.readthedocs.io/en/latest/) ·
[Models](https://polyclustx.readthedocs.io/en/latest/background/algorithms.html) ·
[Examples](https://polyclustx.readthedocs.io/en/latest/user_guide/examples.html)

## Install and launch

Python **3.10+** is required. Installation includes PyTorch, the interface and
Optuna—no separate interface installation is needed.

From PyPI, once the PolyClustX distribution is published:

```bash
python -m pip install polyclustx
polyclustx-display
```

Or install from the repository:

```bash
git clone https://github.com/AbdelheqMokhtari/PolyClustX.git
cd PolyClustX
python -m pip install .
polyclustx-display
```

Open the address printed in the terminal, normally http://localhost:8501.
If the command is not found, use `python -m polyclustx.display`.
The [installation guide](docs/user_guide/installation.rst) covers environments
and migration from the former `pymv` package.

## Use Python

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from polyclustx.data import MultiViewDataset
from polyclustx.models import LMGEC
from polyclustx.metrics import nmi_score

X, y = load_wine(return_X_y=True)  # available offline
X = StandardScaler().fit_transform(X)
data = MultiViewDataset([X[:, :6], X[:, 6:]], labels=y, name="Wine")

model = LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42)
data.fit(model)
print(model.labels_[:10])          # one cluster ID per sample
print(model.embedding_.shape)     # (178, 3)
print("NMI:", nmi_score(y, model.labels_))
```

This splits Wine's feature columns into two aligned views. To use a catalog
dataset instead, import `load_data` from `polyclustx.data` and call
`data = load_data("acm")`.

Compare two methods on the data above, repeating each fit three times:

```python
from polyclustx.experiment import Experiment
from polyclustx.metrics import clustering_accuracy, nmi_score
from polyclustx.models import LMGEC, MultiViewKMeans

results = Experiment(
    datasets=data,
    models=[
        MultiViewKMeans(n_clusters=3, random_state=42),
        LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42),
    ],
    metrics=[clustering_accuracy, nmi_score],
    runs=3,
    save_path="experiments/comparison",
).run()

print(results.results_tables())  # scores across repeated runs
results.export("experiments/comparison.zip")  # JSON and CSV reports
```

Plot the learned representation:

```python
from polyclustx.visualization import plot_clusters

plot_clusters(model.embedding_, model.labels_, method="pca")
```

Use `MultiViewDataset([view_1, view_2], labels=None)` for your own feature
matrices. Add graphs, align samples by ID or simulate missing views as needed.
See the [data guide](docs/user_guide/data.md) and
[experiment guide](docs/user_guide/experiments.md).

## Use the command line

```bash
polyclustx models
polyclustx datasets
lmgec --help
lmgec acm --embedding-dim 8 --runs 3
scm acm --embedding-dim 16 --max-iter 50
idec acm --hidden-dims '128,64' --pretrain-epochs 20
```

Each method has its own command and parameter help. Use `--variant` for a dataset
variant, `--clusters` to override K, and `--output` to select the report directory.
Without an explicit K, labeled datasets use their number of reference classes.

Search candidates or continuous ranges:

```bash
lmgec acm --search grid --vary 'temperature=[0.5,1,2]' --objective nmi
lmgec acm --search optuna --vary 'temperature={"low":0.1,"high":5.0}' --trials 10
multiview-kmeans acm --no-benchmark --clusters 3 --metrics silhouette db
```

Compare several methods and datasets in **one report**:

```bash
polyclustx experiment --datasets acm dblp --models lmgec scm --runs 3
polyclustx experiment --datasets acm --models lmgec scm --param lmgec.embedding_dim=8
polyclustx experiment --config examples/experiment_plan.yaml
```

The [example plan](examples/experiment_plan.yaml) keeps larger studies readable:
name configurations, set per-model parameters, choose datasets and specify repeats.
Dataset-specific default K is handled automatically. Short commands may conflict
with other installed tools; use a dedicated environment or `polyclustx lmgec acm`.

**Interface:** `polyclustx-display` is a standalone command.
See the [CLI guide with complete examples](docs/api/cli.rst).

## Available methods

Graphs ✓ means the method uses supplied adjacency matrices. Missing views ✓
means native support for an observation mask.

| Category | Method / command | Graphs | Missing views |
|---|---|:---:|:---:|
| Graph-aware | LMGEC / `lmgec` | ✓ | — |
| Graph-aware | DGF / `dgf` | ✓ | — |
| Graph-aware | SGLA / `sgla` | ✓ | — |
| Feature-based | MultiViewKMeans / `multiview-kmeans` | — | — |
| Feature-based | MultiViewSpectralClustering / `multiview-spectral` | — | — |
| Feature-based | G-CoMVKM / `g-comvkm` | — | — |
| Feature-based | IDEC / `idec` | — | — |
| Feature-based | CONAN / `conan` | — | — |
| Feature-based | SCM / `scm` | — | — |
| Feature-based | MFLVC / `mflvc` | — | — |
| Feature-based | UMC / `umc` | — | — |
| Incomplete-view | COMPLETER / `completer` | — | ✓ |
| Incomplete-view | DCP / `dcp` | — | ✓ |

Capabilities differ by method. Some single-view and multi-view cases are
framework adaptations; UMC uses supplied features without pretrained encoders.
See [algorithm details, assumptions and references](docs/user_guide/clustering_models.md).
These implementations do not claim to reproduce the papers' benchmark scores.

## Explore and visualize

The interface provides dataset preparation, named models, repeated experiments,
hyperparameter search, visualizations and JSON/CSV/ZIP exports.
Labels enable benchmark metrics; without labels, use internal metrics.
[Interface guide](docs/user_guide/workbench.md).

The [notebooks](notebooks/) are **unexecuted tutorials** for you to run.
Start with [the end-to-end study](notebooks/06_polyclustx_feature_to_graph_study.ipynb)
to compare methods on Iris and Wine, add graphs, visualize embeddings and
losses, search hyperparameters and export results.

## Documentation

Browse the [documentation](https://polyclustx.readthedocs.io/en/latest/),
start with the [worked examples](docs/user_guide/examples.rst), or follow the
[interactive interface guide](docs/user_guide/workbench.md).

## Research affiliations

<p>
  <a href="https://centreborelli.ens-paris-saclay.fr/fr"><img src="src/polyclustx/display/assets/borelli.png" height="90" alt="Centre Borelli"></a>
  &nbsp;&nbsp;
  <a href="https://u-paris.fr/"><img src="src/polyclustx/display/assets/paris_cite.png" height="90" alt="Université Paris Cité"></a>
</p>

Centre Borelli · Université Paris Cité.
Institutional marks remain separate from PolyClustX and do not imply endorsement.

## References

Cite the original papers when using these methods in research. Reference code
links identify the implementations that informed the framework adaptations.

| Method | Paper | Reference implementation |
|---|---|---|
| LMGEC | [Fettal et al., WSDM 2023](https://doi.org/10.1145/3539597.3570367) | [Code](https://github.com/chakib401/LMGEC) |
| DGF | [Zheng et al., Dual Graph Filtering](https://arxiv.org/abs/2511.20030) | [Code](https://github.com/HaoranZ99/DGF) |
| SGLA | [Li et al., ICDE 2025](https://arxiv.org/abs/2508.09452) | [Code](https://github.com/CyanideCentral/SGLA) |
| MultiViewKMeans | [Bickel & Scheffer, ICDM 2004](https://doi.org/10.1109/ICDM.2004.10095) | [Code](https://github.com/mvlearn/mvlearn) |
| MultiViewSpectralClustering | [Kumar & Daumé, ICML 2011](https://icml.cc/2011/papers/272_icmlpaper.pdf) | [Code](https://github.com/mvlearn/mvlearn) |
| G-CoMVKM | [Sinaga & Yang, Electronics 2025](https://doi.org/10.3390/electronics14112129) | [Code](https://github.com/KristinaP09/G-CoMVKM) |
| IDEC | [Guo et al., IJCAI 2017](https://www.ijcai.org/proceedings/2017/243) | [Code](https://github.com/XifengGuo/IDEC) |
| CONAN | [Ke et al., BigData 2021](https://doi.org/10.1109/BigData52589.2021.9671851) | [Code](https://github.com/Guanzhou-Ke/conan) |
| SCM | [Luo et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/519) | [Code](https://github.com/SubmissionsIn/SCM) |
| MFLVC | [Xu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.html) | [Code](https://github.com/SubmissionsIn/MFLVC) |
| UMC | [Zhang et al., ACL 2024](https://aclanthology.org/2024.acl-long.2/) | [Code](https://github.com/thuiar/UMC) |
| COMPLETER | [Lin et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_COMPLETER_Incomplete_Multi-View_Clustering_via_Contrastive_Prediction_CVPR_2021_paper.pdf) | [Code](https://github.com/XLearning-SCU/2021-CVPR-Completer) |
| DCP | [Lin et al., TPAMI](https://doi.org/10.1109/TPAMI.2022.3197238) | [Code](https://github.com/XLearning-SCU/2022-TPAMI-DCP) |


## License

The software license and third-party notices are in [LICENSE](LICENSE). Institutional logos are not
covered by the software's MIT license. Cite the original algorithm papers
when publishing results.
