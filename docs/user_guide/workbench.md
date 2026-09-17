# Interactive scientific workbench

The interface is the optional `pymv.display` subpackage of the same distribution.

```bash
pip install "pymv-clustering[display]"
pymv-display
# Equivalent:
python -m pymv.display
# Custom local port:
pymv-display --server.address=127.0.0.1 --server.port=8502
```

For this unpublished branch, install from the checkout with
`pip install -e ".[display]"`. Open the local URL printed in the terminal.

## Three freely accessible tabs

**Datasets** loads a catalog family/variant, uploads your files, or creates a
small synthetic example without network access. Saved datasets are independent
session snapshots reusable across experiments.

**Experiment workspace** selects a snapshot, model and manual K values. The
JSON constructor editor exposes registered model defaults. Random-search
parameters are JSON lists; each sampled configuration is evaluated for every
K and repeat. Dedicated fields control K and seeds.

**History** stacks experiments newest-first with recipes, individual trials,
runtime, metrics, predictions and embeddings. **Edit / duplicate settings**
prefills the editor without changing the original record. Change any settings
and run again. **Delete result** removes only that session result.

## Benchmark toggle and manual K sweeps

Disable **Benchmark evaluation** for real-world work: external scores disappear
and labels are not used for evaluation or K selection. With no labels, the
toggle is disabled. Enable it to opt into external evaluation when labels exist.
Model training remains unsupervised.

Type K values such as `2, 4, 7, 12`. Valid values range from one to the sample
count, subject to model requirements. Internal indices are undefined at
degenerate endpoints. Repeat seeds are base seed + zero-based repeat index.

## Explicit upload mapping

After uploading NPY, NPZ, numeric CSV/TSV or MATLAB files, inspect keys/shapes
and select:

1. Feature arrays and optional transposes.
2. A modality for each view.
3. Zero, one or several graph relation keys belonging to each view.
4. Optional labels.
5. Per-view sample IDs, or an explicit common-order confirmation and optional mask.

With ID alignment, labels also need an ID key and graphs follow their view's
local ID order. With aligned arrays, supply a samples × views mask or explicitly
detect wholly NaN rows. The importer never guesses roles from MATLAB names.

## Missing views and graph fusion

Choose a simulation rate to remove sample-view pairs reproducibly. Existing
models need explicit mean/zero imputation on incomplete views; leaving **None**
selected produces a validation error.

Choose weighted sum or Hadamard fusion. Weight JSON uses zero-based view indices,
for example `{"0": [0.3, 0.7]}`. Model parameters control algorithm-specific
propagation; SCM ignores graphs. Global centering/scaling, when enabled,
precedes model preprocessing and is recorded in the recipe.

## Inspect and export

Explore a successful trial's cluster counts, loss history when available, and
PCA projection. A 2D visualization is not evidence of clustering correctness.
Download an experiment ZIP containing:

- `experiment.json`: recipe, fingerprint, IDs, masks, version, timestamps,
  scores and recorded failures.
- `metrics.csv`: individual trial results.
- `trial_N.npz`: predictions, embedding and optional loss history.

Feature matrices and ground-truth labels are not exported. Keep the original
data and source revision; reload matching data and check the fingerprint to
reproduce a recipe. Review sample IDs for sensitivity before sharing archives.

**History is session-local, not a persistent database.** Download records before
closing or restarting. Runs are synchronous; choose realistic sweep sizes.
For shared hosting, provide authentication, TLS and resource limits outside
the app. Do not expose an unauthenticated Streamlit server publicly.
