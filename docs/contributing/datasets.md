# Add a dataset

You can use your own data immediately. Catalog registration is only needed
when you want other users to load it by name with `load_data`.

## 1. Create a dataset from arrays

```python
from sklearn.datasets import load_iris
from polyclustx.data import MultiViewDataset
from polyclustx.models import LMGEC

X, y = load_iris(return_X_y=True)
data = MultiViewDataset(
    [X[:, :2], X[:, 2:]], labels=y, name="Iris measurements",
    view_names=["sepal", "petal"],
)
model = data.fit(LMGEC(n_clusters=3, embedding_dim=1, random_state=42))
print(model.labels_.shape)  # (150,)
```

Use `labels=None` without reference labels. Every view must use the same sample
order. For unaligned samples, follow the [ID-alignment example](../user_guide/missing_views.md).

## 2. Save and reload your own file

Continue the example above:

```python
import numpy as np
from polyclustx.data import read_arrays, dataset_from_arrays

np.savez("iris_views.npz", sepal=X[:, :2], petal=X[:, 2:], labels=y)
data = dataset_from_arrays(
    read_arrays("iris_views.npz"),
    view_keys=["sepal", "petal"], label_key="labels", name="Iris measurements",
)
print(data.feature_dims)  # [2, 2]
```

Choose each array's role explicitly. For graph data, add graph arrays to the
file and pass `graph_keys={"sepal": ["sepal_graph"]}`. Multiple relations can
map to the same view. See [loading data](../user_guide/data.md).

## 3. Add a public dataset to the catalog

Place a `DatasetSpec` entry in `_SPECS` in `src/polyclustx/data/registry.py`.
For a MATLAB file containing `image`, `text` and `labels` arrays:

```python
DatasetSpec(
    name="my_dataset",
    record="YOUR_ZENODO_RECORD_ID",
    filename="my_dataset.mat",
    reader=readers.read_feature_views,
    kwargs={"keys": ("image", "text"), "label_key": "labels"},
    n_views=2, modalities=("image", "text"),
    description="Describe the samples, classes and feature representations.",
    reference="https://example.org/original-dataset",
    reference_verified=True,
)
```

Replace the placeholder record, filename and reference with the real source.
The current downloader uses Zenodo records. Reuse a reader from
`src/polyclustx/data/readers.py`; create a new reader only for a different layout.
Readers return `(Xs, As, labels)`, using `None` when graphs or labels are absent.

Users can now call `load_data("my_dataset")`. The dataset also appears in the
interface and documentation catalog. Do not commit dataset files; include their
source, citation and redistribution terms in your contribution.

## 4. Add a variant

Add a second `_SPECS` entry named `my_dataset_compact` with its own file.
After the existing `VARIANTS` grouping code, before `ALIASES` is constructed, add:

```python
VARIANTS["my_dataset"]["compact"] = "my_dataset_compact"
VARIANTS.pop("my_dataset_compact")
```

Users select the representation without needing another loader:

```python
from polyclustx.data import load_data

data = load_data("my_dataset", variant="compact")
```

Test the reader with a tiny local file and check view shapes, sample order and
label length. Then open a pull request targeting `develop`. A complete offline
registry test is in `tests/test_contribution_examples.py`.
