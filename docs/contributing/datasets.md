# Add datasets and variants

Use `MultiViewDataset` for local/private data; registration is not required.
For reusable public data, add a `DatasetSpec` to `polyclustx.data.registry` with its
source record, artifact filename, reader, description and view count.
Readers return `(Xs, As, labels)`; validation and metadata happen centrally.

For a variant, add an artifact mapping under its family's `VARIANTS` entry.
Users then call `load_data("family", variant="representation")`, without a new
public loader for every representation. Update modality annotations when the
artifact contains distinct modalities. Preserve old published names as aliases
when migrating an existing artifact.

Add tests for the reader using small synthetic file fixtures. Optional real
download tests must use the `network` marker so ordinary CI stays offline.
Document source terms, original dataset references and any unavailable/corrupt
artifacts. The catalog, CLI and interface populate from registry metadata;
rebuild docs to check descriptions and variants.

## 1. Start with a local example

Run `python examples/custom_datasets.py` from a repository checkout. It creates
two tiny MATLAB variants in a temporary directory, loads both, prints their
dimensions and cleans up. No downloads or registry changes are needed.

```python
from polyclustx.data import dataset_from_arrays, read_arrays

data = dataset_from_arrays(
    read_arrays("my_dataset.mat"), view_keys=["image", "text"], label_key="labels",
    modalities=["image", "text"],
)
```

Rows must refer to the same samples in the same order. For different sample
sets, supply per-view ID arrays as described in [loading data](../user_guide/data.md).
Use small fixtures to test optional labels, sparse matrices and graph mappings.

## 2. Add a public artifact to the catalog

In `src/polyclustx/data/registry.py`, add entries to `_SPECS`. The following is a
template: replace the record ID, file names and reference with your real data.
Do not commit the dataset files to the source repository.

```python
DatasetSpec(
    name="my_dataset", record="YOUR_ZENODO_RECORD", filename="my_dataset.mat",
    reader=readers.read_feature_views,
    kwargs={"keys": ("image", "text"), "label_key": "labels"},
    n_views=2, modalities=("image", "text"),
    description="Short description of the samples and two modalities.",
    reference="https://example.org/original-dataset", reference_verified=True,
)
```

Readers live in `src/polyclustx/data/readers.py`. Reuse a reader for standard layouts;
add a new one only when the file format needs it. Return `(Xs, As, labels)`;
use `None` for absent graphs or labels. Register actual download artifacts,
not a link to a repository's HTML page. Include redistribution terms and the
original dataset citation in the pull request.

## 3. Group variants under one family

Add a second `_SPECS` entry named `my_dataset_compact` for its actual artifact.
Then, after the existing variant-grouping loops and **before** `ALIASES` is
constructed, add:

```python
VARIANTS["my_dataset"]["compact"] = "my_dataset_compact"
VARIANTS.pop("my_dataset_compact")
```

The existing default mapping is retained. Users can then call:

```python
from polyclustx.data import list_variants, load_data

print(list_variants("my_dataset"))  # compact, default
data = load_data("my_dataset", variant="compact")
```

Legacy artifact names become aliases automatically. No new public loader is
needed for the compact variant. The interface and documentation catalog use
these same family mappings.

## 4. Test and document the contribution

Add `tests/test_my_dataset.py` with a small `scipy.io.savemat` fixture, exercise
the reader and assert view shapes, label length, graph alignment and variant
resolution. Use `monkeypatch` to replace the downloader when testing `load_data`;
ordinary tests must not contact external servers. See
`tests/test_contribution_examples.py` for a complete offline registration test.

Update `description`, `modalities`, `reference`, and optional `reference_details`
in the registry; the dataset table is generated automatically. Add unusual
loading instructions to `docs/user_guide/data.md`, and an example under
`examples/` if necessary. Run tests and the strict documentation build before
opening a pull request against `develop`.
