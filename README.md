<p align="center"><img src="src/pymv/display/assets/logo.png" width="120" alt="pymv symbol"></p>
<h1 align="center">pymv</h1>

<p align="center">Multi-view and multimodal clustering in Python.</p>

`pymv` provides one data representation for single-view, multi-view,
multimodal, and attributed-graph clustering. Features, labels, graphs, and
missing views are optional. The package currently includes LMGEC and SCM, a
dataset catalog, experiment utilities, metrics, visualizations, a CLI, and an
optional graphical interface.

## Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/AbdelheqMokhtari/pymv.git
cd pymv
pip install -e .
```

Optional components:

```bash
pip install -e ".[display]"  # graphical interface
pip install -e ".[umap]"     # UMAP projections
pip install -e ".[docs]"     # documentation
pip install -e ".[dev]"      # tests and development tools
```

## Quick start

```python
from pymv.data import load_data
from pymv.models import LMGEC

data = load_data("acm")
model = LMGEC(n_clusters=3, embedding_dim=16, random_state=42)
data.fit(model)

print(model.labels_)
print(model.embedding_.shape)
```

`MultiViewDataset` keeps every view aligned and records the information a model
may need:

```python
data.Xs             # feature matrices
data.As             # zero, one, or several graph relations per view
data.labels         # optional ground-truth labels
data.sample_ids     # aligned sample identifiers
data.observed_mask  # present/missing sample-view pairs
data.view_names
data.modalities
data.is_complete
data.category
```

The same structure works with one, two, or many views.

## Run the interface

```bash
pip install -e ".[display]"
pymv-display
```

Open the local address printed in the terminal, normally
`http://localhost:8501`. The sidebar contains seven freely accessible steps:

1. **Dataset** — catalog data, simple per-view files, NPZ/MATLAB containers, or
   configurable synthetic data.
2. **Preprocessing** — transformations for all or selected views, graph fusion,
   missing-view simulation, and explicit imputation.
3. **Model** — only compatible models are available; parameters and random
   search candidates are edited directly.
4. **Experiment** — choose evaluation metrics and run each configuration one or
   many times.
5. **Results** — individual fits and mean ± standard deviation summaries.
6. **Visualizations** — loss, predicted/true-label projections, confusion
   matrices, and cluster details.
7. **Export** — reproducible ZIP and CSV outputs.

Several datasets and experiments can coexist in a session. Only the active
dataset is materialized; inactive datasets can be unloaded or removed from the
sidebar. Results remain available if their source dataset is removed.

## Load data

```python
from pymv.data import get_dataset_info, list_datasets, list_variants, load_data

print(list_datasets())
print(get_dataset_info("pascal"))
print(list_variants("pascal"))

data = load_data("pascal", variant="clip")
```

Catalog files download once and are cached under `~/.pymv_datasets`. Set
`PYMV_DATA_DIR` to use another cache directory.

For custom arrays:

```python
from pymv.data import MultiViewDataset

data = MultiViewDataset(
    Xs=[view_1, view_2],
    As=[[relation_1, relation_2], None],
    labels=labels,              # may be None
    observed_mask=mask,         # True means that view is observed
    sample_ids=sample_ids,
    view_names=["text", "image"],
    modalities=["text", "image"],
)
```

Use `align_views` when views do not initially contain the same samples. CSV and
TSV headers are detected automatically by the interface and `read_arrays`.

## Missing views and graph relations

```python
from pymv.data import impute_missing, simulate_missing_views

incomplete = simulate_missing_views(data, missing_rate=0.2, random_state=42)
baseline = impute_missing(incomplete, strategy="mean")
```

The observation mask is explicit; zero-valued features are never interpreted as
missing. LMGEC and SCM currently require complete views, so use explicit
imputation before fitting them. Future missing-aware models can consume the mask
directly.

A view may have several graph relations. Fuse them during fitting:

```python
data.fit(model, fusion="sum", weights={0: [0.3, 0.7]})
data.fit(model, fusion="hadamard")
```

## Experiments and metrics

```python
from pymv.experiment import Experiment
from pymv.metrics import clustering_accuracy, nmi_score

result = Experiment(
    datasets=["acm", "dblp"],
    models=[LMGEC(n_clusters=3, embedding_dim=16)],
    metrics=[clustering_accuracy, nmi_score],
    runs=5,
).run()
```

External metrics such as ACC, NMI, and ARI need ground-truth labels. Internal
metrics such as Silhouette, Calinski–Harabasz, and Davies–Bouldin allow
unsupervised evaluation without labels.

## Add a model

Implement `fit`, subclass `BaseModel`, and register the class:

```python
from pymv.models import BaseModel
from pymv.models.registry import register_model

@register_model("my_method", description="A short method description")
class MyMethod(BaseModel):
    supports_missing_views = False

    def __init__(self, n_clusters, strength=1.0, random_state=None):
        ...

    def fit(self, Xs, As=None, *, observed_mask=None):
        self.labels_ = ...
        self.embedding_ = ...
        return self
```

The registry exposes the model to Python, the interface, CLI, and generated
documentation. See [the model contribution guide](docs/contributing/models.md)
for validation, tests, citations, and pull-request instructions.

## Documentation and development

- [User guide](docs/index.rst)
- [Dataset import guide](docs/user_guide/data.md)
- [Interactive interface guide](docs/user_guide/workbench.md)
- [Algorithm background and papers](docs/background/algorithms.md)
- [Model catalog](docs/catalog/models.rst)
- [Dataset catalog](docs/catalog/datasets.rst)

Run local checks with:

```bash
ruff check src tests
pytest
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

Network dataset tests are excluded by default; run `pytest -m network` when the
downloads are intentionally available.

## License

MIT. See [LICENSE](LICENSE).
