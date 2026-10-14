# Loading and organizing datasets

The catalog downloads MATLAB files directly from Hugging Face. You do not need
MATLAB, a Hugging Face account, or an additional download package.

## Four kinds of datasets

| Setting | Feature views | Graphs | Examples |
| --- | --- | --- | --- |
| Multi-view attributed graph clustering (MVAGC) | Same modality; a feature can be reused with different structural relations | Yes | ACM, DBLP, IMDB, Wiki, Amazon Photos |
| Multi-view clustering (MVC) | Different representations of the same modality | No | 100leaves, 3sources, handwritten |
| Multimodal clustering (MMC) | Different modalities, such as images and text | No | Pascal, COCO, ESP-Game |
| Multimodal attributed graph clustering (MMAGC) | Different modalities describing graph nodes | Yes | Toys, Movies, Grocery, Reddit |

Different image descriptors are views of one modality; images and captions are
different modalities. Wikipedia-test is therefore MMC. Completeness is separate:
you can simulate missing observations on any complete dataset with at least
two composed views. See [missing views](missing_views.md).

## Load a catalog dataset

```python
from polyclustx.data import load_data, get_dataset_info, list_datasets, list_variants

data = load_data("acm")
captions = load_data("pascal", variant="clip")
small_grocery = load_data("grocery-S", variant="clip_minilm")

print(data.summary())
print(data.metadata["available_features"])  # ["X1_features"]
print(data.metadata["available_graphs"])    # ["A1_PAP", "A2_PLP"]
print(get_dataset_info("pascal", variant="clip"))  # metadata only; no download
```

Names are case-insensitive; hyphens and underscores are interchangeable.
Variants are **alternate feature representations**, not new dataset families.
Grocery-S and Reddit-S are smaller datasets, not encoder variants of the full
datasets. Use `variant=` explicitly: the second positional argument is the
missing-view fraction.

All 66 artifact links are in the [dataset catalog](../catalog/datasets.rst).
Downloads are cached under `~/.polyclustx_datasets/huggingface/`, or under
`POLYCLUSTX_DATA_DIR/huggingface/`. Old differently structured downloads are
not reused or deleted. If a file's upload has not finished, the loader reports
that it is unavailable; retry after it has been published.

## The standardized MATLAB layout

Each file uses:

- `labels`: one reference label per sample, used only for evaluation.
- `X1`, `X2`, … or names such as `X1_image`: samples × features.
- `A1`, `A2`, … or names such as `A1_PAP`: samples × samples adjacency.

Arrays are ordered naturally: X2 comes before X10. Features and graphs retain
the file's sample order. Sparse matrices stay sparse. The loader does not
invent extra feature transforms, build new graphs, symmetrize edges or add
self-loops. You can do those explicitly during preprocessing.

Default composition is straightforward:

| Stored arrays | Default algorithm views |
| --- | --- |
| Features only | One view per feature matrix |
| One feature and several graphs | Reuse the feature in a separate view for each graph |
| Several features and a shared graph | One view per feature, with the same graph |
| Several features and several graphs | One view per feature, with all relations attached |

These defaults are explicit conventions, not an assumption that a graph's
numeric suffix identifies its feature. Override them with `views=`.

## Compose exactly the views you want

One feature can be used with separate graphs, with several relations in one
view, or without a graph:

```python
from polyclustx.data import load_data

data = load_data(
    "acm",
    views={
        "Co-authorship": ("X1_features", "A1_PAP"),
        "Shared subjects": ("X1_features", "A2_PLP"),
        "Combined structure": ("X1_features", ["A1_PAP", "A2_PLP"]),
        "Keywords only": "X1_features",
    },
)
print(data.view_names)
print(data.feature_dims)  # [1870, 1870, 1870, 1870]
```

A mapping value is either a feature key, or `(feature_key, graph_keys)`.
Graph keys may be one string, a list, or `None`. View names must be nonempty
and unique. A single named entry creates a single-view dataset. Labels cannot
be selected as features or relations. Reused matrices share storage.

For catalog selections, only the chosen matrices and labels are materialized
from the cached file. Available array names and descriptions remain in
`data.metadata`, even when you did not select those arrays.

Multiple relations are combined when fitting, using weighted sum by default:

```python
from polyclustx.models import LMGEC

model = data.fit(
    LMGEC(n_clusters=data.n_clusters, embedding_dim=8, random_state=42),
    fusion="sum",
    weights={2: [0.25, 0.75]},  # third composed view: PAP then PLP
)
```

Use `fusion="hadamard"` without weights for elementwise product. Graph direction,
weights and self-loops are preserved until you explicitly preprocess them.

## Load your own standardized MAT file

```python
from scipy.io import savemat
from sklearn.datasets import load_iris
from polyclustx.data import load_mat

X, labels = load_iris(return_X_y=True)
savemat("iris.mat", {"X1": X[:, :2], "X2": X[:, 2:], "labels": labels})

data = load_mat("iris.mat", name="Iris")
single = load_mat("iris.mat", views={"Sepal only": "X1"}, name="Iris sepal")
print(data.feature_dims)    # [2, 2]
print(single.n_views)      # 1
```

`load_mat` recognizes the standardized names. It also accepts explicit mappings
for other naming conventions. MATLAB v7.3/HDF5 is unsupported; export MATLAB
v7 or an earlier format.

## Arbitrary NumPy, CSV and MATLAB arrays

Use the same importer with an explicit mapping if your arrays have other names:

```python
from polyclustx.data import read_arrays, dataset_from_arrays

arrays = read_arrays("measurements.npz")
data = dataset_from_arrays(
    arrays,
    views={
        "Visual links": ("visual", ["spatial", "similarity"]),
        "Visual features only": "visual",
        "Language": ("language", "citations"),
    },
    label_key="reference_labels",  # or None for an unlabeled study
    modalities=["image", "image", "text"],
    name="My study",
)
```

The existing `view_keys=[...]` and `graph_keys={...}` interface also remains
available. With explicit mappings, labels are opt-in unless you use `load_mat`.

For multiple CSV/NumPy files, keys include the filename:

```python
from polyclustx.data import load_files

data = load_files(
    ["shape.csv", "texture.csv", "labels.npy"],
    view_keys=["shape.csv:shape", "texture.csv:texture"],
    label_key="labels.npy:labels",
    modalities=["image", "image"],
)
```

CSV/TSV headers are detected automatically. `read_arrays` can override this
with `skip_header=0` or `1`. NumPy pickles are disabled; MATLAB cells are
exposed as `key[0]`, `key[1]`, etc.

For transposed features, pass `transpose_views=["feature_key"]`. For an
embedded label column, use `label_column=("feature_key", index)`; the zero-based
column is removed before fitting. Extract this column before reusing its
feature matrix in several composed views.

Without IDs, you assert that all arrays have the same sample row order;
equal row counts alone do not establish alignment. Use `id_keys` for
different sample orders or missing rows, as explained in
[aligned and missing views](missing_views.md).

## The common data container

| Field | Meaning |
| --- | --- |
| `Xs[v]` | Numeric sample × feature matrix |
| `As[v]` | No graph, one adjacency, or a list of relations |
| `labels` | Optional evaluation labels, never training input |
| `sample_ids` | Unique IDs defining the aligned sample order |
| `observed_mask[i, v]` | Whether sample i is observed in view v |
| `view_names`, `modalities` | Names and modality annotations |
| `metadata` | Source, variant, array selection and preparation details |

Every sample must retain at least one observed view. Observed features must
be finite; graphs must be square, finite and nonnegative. One, two or more
views use the same `data.fit(model)` interface.

Continue with [the interactive interface](workbench.md) or
[comparative experiments](experiments.md).
