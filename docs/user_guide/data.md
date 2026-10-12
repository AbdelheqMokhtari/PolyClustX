# Loading and organizing data

All entry points return `MultiViewDataset`. Python and the workbench use the
same importer: file format does not determine an array's meaning.

## The data contract

| Field | Meaning |
| --- | --- |
| `Xs[v]` | Numeric features, samples × features for view v |
| `As[v]` | `None`, one adjacency, or a list of graph relations |
| `labels` | Optional evaluation labels, never training input |
| `sample_ids` | Unique IDs defining the global sample order |
| `observed_mask[i, v]` | Whether sample i is observed in view v |
| `view_names`, `modalities` | View identity and modality annotations |
| `metadata` | Variant, source, simulation and imputation provenance |

At least one view must exist and every sample must have an observed view.
Observed features must be finite. Missing feature values within an observed row
are different from a missing view. Graphs are finite, nonnegative square
matrices in the declared sample order. SciPy sparse matrices are supported.

## Built-in datasets and variants

```python
from polyclustx.data import load_data, list_datasets, list_variants

data = load_data("pascal", variant="clip")
graph_data = load_data("acm")
feature_data = load_data("acm", variant="5view")
incomplete = load_data("acm", 0.2, random_state=42)
```

The second positional argument is a missing-view fraction, not a variant.
Use `variant=` explicitly. Downloads are cached after first use.
See the [dataset catalog](../catalog/datasets.rst).

## Arrays already in Python

```python
import numpy as np
from polyclustx.data import MultiViewDataset

rng = np.random.default_rng(4)
data = MultiViewDataset(
    [rng.normal(size=(20, 4)), rng.normal(size=(20, 7))],
    sample_ids=[f"sample-{i}" for i in range(20)],
    view_names=["image_embedding", "text_embedding"],
    modalities=["image", "text"],
)
single_view = MultiViewDataset([data.Xs[0]])
```

One, two, or more views use the same container and fitting interface.

## NumPy, CSV and MATLAB: inspect first, map explicitly

```python
from polyclustx.data import read_arrays, dataset_from_arrays

arrays = read_arrays("measurements.npz")
print({key: value.shape for key, value in arrays.items()})
data = dataset_from_arrays(
    arrays,
    view_keys=["visual", "language"],
    graph_keys={"visual": ["spatial", "similarity"], "language": ["citations"]},
    label_key=None,  # opt in only if you have ground truth
    modalities=["image", "text"],
    name="My study",
)
```

Supported formats are `.npy`, `.npz`, numeric `.csv`/`.tsv`, and MATLAB v7 or
earlier `.mat`. NumPy object pickles are disabled. MATLAB v7.3/HDF5 is not yet
supported: export MATLAB v7 or NumPy instead.

MATLAB cells appear as `X[0]`, `X[1]`, etc., in column-major order. Names such as
`X`, `Y` or `A` do not imply features, labels or graphs. For feature matrices
stored features × samples, use `transpose_views=["visual"]`.

For several aligned CSV views:

```python
from polyclustx.data import load_files

data = load_files(
    ["shape.csv", "texture.csv"],
    view_keys=["shape.csv:shape", "texture.csv:texture"],
    modalities=["image", "image"],
)
```

CSV/TSV headers are detected automatically. Use `skip_header=0` or `1` in
`read_arrays` to override detection when necessary.
Store string IDs in separate NPY/NPZ vectors or pass them to `align_views`;
numeric feature CSVs do not silently strip ID columns.
Without IDs, you assert identical row order in every view and graph.
Equal row counts alone do not prove alignment.

Labels may be part of the same file batch; exclude their key from `view_keys`:

```python
data = load_files(
    ["shape.npy", "texture.npy", "Y.npy"],
    view_keys=["shape.npy:shape", "texture.npy:texture"],
    label_key="Y.npy:Y",
)
```

For labels embedded in a feature matrix, use `label_column=(view_key, index)`.
The zero-based column is removed from that view before fitting. For example,
with two feature columns followed by one label column:

```python
data = load_files(
    ["measurements.csv"],
    view_keys=["measurements.csv:measurements"],
    label_column=("measurements.csv:measurements", 2),
)
```

This mapping is retained when the interface unloads and reloads a dataset.
If labels are duplicated across several files, remove the extra label columns
before importing those views.

## Four settings, one interface

| Graphs? | Same modality? | Category |
| --- | --- | --- |
| No | Yes | `multiview` |
| No | No | `multimodal` |
| Yes | Yes | `attributed_graph_multiview` |
| Yes | No | `attributed_graph_multimodal` |

Different image descriptors are views of one modality; an image and caption
are different modalities. `data.category` derives from graph availability and
modality annotations. `data.is_complete` is independent, not a fifth category.

Continue with [missing views](missing_views.md) or [graph relations](graphs.md).
