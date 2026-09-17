# Interactive interface

The optional `pymv.display` package provides a local interface for building and
comparing multi-view clustering experiments.

```bash
pip install "pymv-clustering[display]"
pymv-display
```

For a repository checkout, use `pip install -e ".[display]"`. Open the local
address printed by Streamlit. To choose a port explicitly:

```bash
pymv-display --server.address=127.0.0.1 --server.port=8502
```

## Sidebar workflow

The seven sidebar steps remain freely accessible. Moving to another step does
not discard the active dataset, preprocessing recipe, model settings, or saved
experiments.

1. **Dataset** loads data and reports its structure.
2. **Preprocessing** prepares all views, selected views, or each view separately.
3. **Model** exposes typed controls for compatible registered methods.
4. **Experiment** configures repeated runs and evaluation.
5. **Results** aggregates repetitions as mean ± standard deviation.
6. **Visualizations** explores loss, embeddings, labels, and confusion.
7. **Export** downloads reproducible results.

The sidebar selects, unloads, or removes session datasets. Only the active
dataset is materialized in RAM. Catalog data can be reloaded from its cache,
synthetic data can be regenerated from its recipe, and uploaded data is staged
in a session-specific temporary directory. Removing an uploaded dataset also
removes its temporary source files. Experiment results remain available.

## Loading and inspecting data

Choose one source at a time:

- **Catalog** selects a dataset family and variant.
- **Files → One file per view** accepts CSV, TSV, or NPY feature matrices and
  an optional separate label file. Rows are samples; headers are detected
  automatically.
- **Files → NPZ / MATLAB container** first lists every array and shape. Select
  feature arrays, optional labels, transposes, modalities, sample-ID arrays,
  and graph relations explicitly.
- **Synthetic** controls samples, classes, view dimensions, per-view
  distributions, overlap, imbalance, modality, noise, and graph relations.

After validation, the Dataset page reports sample/view/class counts, missing
pairs, estimated memory, feature density, mean, variance, per-feature variance,
class imbalance, and graph density/symmetry. This information helps determine
whether a model and preprocessing choice are appropriate.

For differently sampled views, enable sample-ID alignment in the container
mode and choose one ID array for each view. Graph matrices follow the local
order of their associated view. No array name is silently treated as a feature,
label, ID, or graph.

## Preprocessing

Transformations can be disabled, shared by all views, applied to selected
views, or configured per view. Available operations are:

- L2 normalization or TF-IDF + L2;
- column centering and unit-variance scaling;
- graph propagation with a configurable self-loop weight;
- weighted-sum or Hadamard early fusion when a view has several relations.

Missing-view simulation and explicit mean/zero imputation are configured in
this step. Simulation retains at least one observed view per sample. The
original dataset stays unchanged; the complete recipe is copied into each
experiment record.

If a dataset remains incomplete, methods that require complete views are not
offered in the Model step. Selecting imputation makes those methods available.
This prevents an invalid experiment before fitting begins.

## Models and hyperparameters

Model controls change with the selected registry entry. Numeric parameters,
booleans, and layer dimensions are edited directly; there is no JSON editor.
If generic feature preprocessing was selected in the preceding step,
model-internal preprocessing is disabled to prevent applying transformations
twice.

K starts at the number of unique loaded labels when labels exist. This is only
an editable default: the user may choose another K or compare several K values.
Labels are never passed into model fitting.

Choose **Random search** to mark individual parameters and select their allowed
candidate values. The interface currently samples discrete candidates. Its
search-strategy control is designed so additional optimizers can be registered
later without changing fixed model parameters.

## Repeated experiments and evaluation

Experiment names increment automatically (`Experiment 1`, `Experiment 2`, …).
Each sampled configuration and K can run multiple times with independent seeds.
The Results page keeps every fit and groups repetitions into mean ± population
standard deviation.

In real-world mode, choose internal metrics only. Enable benchmark mode when
ground-truth labels exist to add ACC, NMI, and ARI. Benchmark labels are used
for evaluation and visual comparison only; clustering remains unsupervised.

Training runs synchronously. Large datasets, deep methods, many K values, large
searches, and many repetitions multiply runtime, so start with a small trial.
Failed fits are recorded beside successful fits rather than erasing completed
work.

## Visualizations

Choose any successful fit to inspect:

- predicted cluster sizes;
- optimization loss when the model exposes it;
- PCA, t-SNE, or optional UMAP projections;
- side-by-side predicted and ground-truth coloring in benchmark mode;
- a confusion matrix and per-cluster size/purity details.

Projection sampling is capped at 5,000 rows. With more than 20 labels, plots
switch from a large categorical legend to a compact color scale. Confusion
matrices show the 30 largest classes/clusters and combine the remainder into an
`Other` bucket. These limits affect display only, not clustering or metrics.

## Export and deployment safety

The ZIP export contains the recipe, library version, dataset fingerprint,
sample IDs, observation mask, per-fit metrics, predictions, embeddings, loss,
and benchmark labels used by the visualizations. Original feature matrices are
not included. A separate metrics CSV is also available.

Session history is temporary. Download records before restarting the server.
The default launcher binds locally. Public hosting needs authentication, TLS,
upload limits, execution limits, and operating-system resource controls; do not
expose an unauthenticated research server to the internet.
