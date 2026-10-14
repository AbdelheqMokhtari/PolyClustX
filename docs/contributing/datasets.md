# Add a dataset

You can use your own arrays or files immediately. Register a dataset only when
you want others to load it by name in Python, the interface and CLI.

## 1. Write a small standardized file

```python
from scipy.io import savemat
from sklearn.datasets import load_iris
from polyclustx.data import load_mat

X, labels = load_iris(return_X_y=True)
savemat("iris_views.mat", {"X1": X[:, :2], "X2": X[:, 2:], "labels": labels})

data = load_mat("iris_views.mat", name="Iris")
print(data.feature_dims)  # [2, 2]
```

Rows are aligned samples. Features start with `X`, adjacency matrices with `A`,
and reference labels use the exact key `labels`. Omit graphs for feature-only
data. Keep sparse graphs sparse when saving. For custom unlabeled files, omit
labels; catalog benchmark files include them.

You can compose views without registering the file:

```python
single = load_mat("iris_views.mat", views={"Sepal": "X1"})
both = load_mat("iris_views.mat", views={"Sepal": "X1", "Petal": "X2"})
```

For another naming convention or format, use `read_arrays` and
`dataset_from_arrays` with an explicit mapping. See [loading data](../user_guide/data.md).

## 2. Add one catalog entry

Publish the file at a direct HTTP download URL, for example Hugging Face's
`resolve/main/...mat?download=true` URL. Ordinary HTTP downloads use Requests,
which is already installed with PolyClustX; no new dependency is needed.

Add this entry to `_SPECS` in `src/polyclustx/data/registry.py`, replacing the
example download/reference URLs with your real sources:

```python
DatasetSpec(
    name="iris_views",
    url="https://your-host.example/iris_views.mat",
    filename="iris_views.mat",
    n_views=2, n_samples=150, n_clusters=3,
    feature_dims=(2, 2),
    feature_descriptions=("Sepal measurements", "Petal measurements"),
    modalities=("measurements", "measurements"),
    category="multiview",
    description="Iris flower measurements, split into two feature views.",
    reference="https://archive.ics.uci.edu/dataset/53/iris",
    reference_verified=True,
)
```

The standardized reader is the default: you do not need a new reader or loader.
For graph datasets, add `graph_descriptions=(...)` and choose
`attributed_graph_multiview` or `attributed_graph_multimodal` as the category.
For image/text features without a graph, use `category="multimodal"` and
`modalities=("image", "text")`. Set `n_views` to the default composed view count,
not necessarily the number of stored feature matrices.

The dataset now appears in `list_datasets()`, `load_data("iris_views")`,
the interface and the generated documentation catalog. Do not commit data
files; provide citations, representation descriptions and redistribution terms.

## 3. Group an alternate representation

Add another spec named `iris_views_compact`, with its own direct URL and feature
dimensions. Group it in `VARIANTS` after the existing grouping loop and before
`ALIASES` is constructed:

```python
VARIANTS["iris_views"]["compact"] = "iris_views_compact"
VARIANTS.pop("iris_views_compact")
```

Then users can select it with the same loader:

```python
from polyclustx.data import load_data

data = load_data("iris_views", variant="compact")
```

Variants mean alternative representations of the same samples. A different
sample subset should have its own family, like Grocery-S versus Grocery.

## 4. Test and submit

Install test tools with `python -m pip install pytest ruff`; `make docs-build`
installs the documentation tools automatically.

1. Save a tiny local MAT fixture using the same keys as the real file.
2. Stub `download_file` in your test so it works without network access.
3. Check sample counts, view shapes, label length, sparse graphs, modalities,
   default composition and variant selection.
4. Add descriptions and source links to your spec. The documentation catalog
   is generated automatically; update the README dataset table as well.
5. Run `make test`, `make lint` and `make docs-build`, then open a pull request
   targeting `develop`.

A concrete offline registration/variant test is in
`tests/test_contribution_examples.py`. The standardized layouts for all catalog
artifacts are checked in `tests/test_dataset_variants.py`.
