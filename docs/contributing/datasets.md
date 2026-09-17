# Add datasets and variants

Use `MultiViewDataset` for local/private data; registration is not required.
For reusable public data, add a `DatasetSpec` to `pymv.data.registry` with its
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
