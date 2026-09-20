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

The sidebar pages remain freely accessible. Moving to another page does
not discard the active dataset, preprocessing recipe, model settings, or saved
experiments.

1. **Dataset** loads data and reports its structure.
2. **Preprocessing** prepares all views, selected views, or each view separately.
3. **Model** prepares a collection of named, compatible configurations.
4. **Hyperparameter search** runs random or grid search and selects an objective.
5. **Experiment** selects dataset/model combinations and repeated runs.
6. **Results** aggregates repetitions as mean ± standard deviation.
7. **Visualizations** compares normalized loss, embeddings, labels, and confusion.
8. **Export** downloads individual results and complete comparison batches.

The sidebar selects, unloads, or removes session datasets. Only the active
dataset is materialized in RAM. Catalog data can be reloaded from its cache,
synthetic data can be regenerated from its recipe, and uploaded data is staged
in a session-specific temporary directory. Removing an uploaded dataset also
removes its temporary source files. Experiment results remain available.
Run arrays are also stored in session-specific temporary files and read only
when requested, rather than retaining every embedding in RAM. Removing a result
releases its temporary arrays. Python datasets added directly with
`Workspace.add_dataset` retain an in-memory source snapshot; use lazy loaders
for large programmatic sources.

## Loading and inspecting data

Choose one source at a time:

- **Catalog** selects a dataset family and variant.
- **Files → One file per view** accepts CSV, TSV, or NPY arrays. Under
  **Ground-truth labels**, choose no labels, a labels file from the same batch
  (such as `Y.npy`), an embedded label column, or a separate labels upload.
  Labels are excluded from the selected features. Rows are samples; CSV/TSV
  headers are detected automatically. Embedded columns use a one-based index
  in the interface and must contain numeric labels in CSV/TSV files.
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

Under **Add graphs to a view**, upload adjacency matrices or GraphML/GEXF files
and select their feature view. Each file adds a relation; nothing is silently
replaced. Adjacencies must be square and follow the dataset's sample order.
Graph node IDs must match sample IDs, including isolated nodes. GraphML/GEXF IDs
are matched against the exact string representation of each sample ID. Multiple
array containers should instead use the explicit container mapping controls.

**Benchmark mode** is enabled by default when the dataset has labels. Uncheck
it here to use real-world evaluation. Unlabeled data always uses real-world mode.
The mode belongs to the dataset, not the model or experiment page.

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

Save each configuration with **Save model settings**, then prepare another
method or parameter setting. Saved configurations can be selected together in
Experiment and removed independently of existing results.

## Hyperparameter search

Open **Hyperparameter search** and choose **Random search** or **Grid search**.
Select the model, K values and parameters to search. Select suggested
candidates or type a custom number and press Enter. Scientific notation such
as `1e-4` is accepted; custom values remain available after saving and returning
to the search page. Invalid values disable running: dimensions and iteration
counts must be positive whole numbers, temperatures and learning rates must be
positive, and rates must stay in [0, 1]. NaN, infinity, and empty candidate
lists are rejected with an explanation. Random search samples discrete
configurations with replacement; grid search evaluates every combination.

Choose **Search metrics** and **Optimize metric**. Internal objectives such as
Silhouette are available in both modes; external objectives such as NMI require
benchmark mode. Davies–Bouldin is minimized; the other built-in metrics are
maximized. Selection uses the mean across repeats, and configurations with
failed runs cannot win.

Results, tuning steps and JSON/CSV downloads remain on this page. **Prepare ZIP**
adds a complete archive with run arrays on demand; only one record's ZIP is
cached at a time. **Copy best
to experiment** adds the winning model parameters and K to the model collection,
selects its dataset, restores its preprocessing and opens Experiment. It does
not start another fit automatically. Search reports also remain available for
visualization and export after their source dataset has been removed.

## Repeated experiments and evaluation

Experiment names increment automatically (`Experiment 1`, `Experiment 2`, …).
Select datasets, saved model configurations and the combinations to run. The
**Independent runs per configuration** slider ranges from 1 to 20. Each
configuration and K runs with independent seeds.
The Results page keeps every fit and groups repetitions into mean ± population
standard deviation.

In real-world mode, choose internal metrics only. Benchmark mode, set on the
Dataset page, adds ACC, NMI, ARI, and F1. Benchmark labels are used
for evaluation and visual comparison only; clustering remains unsupervised.
Benchmark and real-world datasets cannot be combined in one batch. Use separate
experiments or change the datasets' modes first.

**Reproducibility and saving** contains the base seed and output directory.
The seed is retained because it lets you reproduce initialization sequences.
One JSON file stores the entire batch; each model/dataset combination remains
independently inspectable in Results. Summary and detailed tables give each
hyperparameter its own column.

Training runs synchronously. Large datasets, deep methods, many K values, large
searches, and many repetitions multiply runtime, so start with a small trial.
Failed fits are recorded beside successful fits rather than erasing completed
work.
Jobs are limited to 2,000 fits in the interface, and configurations are executed
sequentially. This is not a guarantee that an individual large model fits in
available RAM or GPU memory. Start with a small trial and remove unused results.

## Visualizations

Choose any successful fit to inspect:

- predicted cluster sizes;
- min–max normalized optimization loss when the model exposes it;
- PCA, t-SNE, or optional UMAP projections;
- side-by-side predicted and ground-truth coloring in benchmark mode;
- a confusion matrix and per-cluster size/purity details.

Projection sampling is capped at 5,000 rows. With more than 20 labels, plots
switch from a large categorical legend to a compact color scale. Confusion
matrices show the 30 largest classes/clusters and combine the remainder into an
`Other` bucket. These limits affect display only, not clustering or metrics.

**Compare normalized loss** selects models and runs across experiments and
searches in one plot. Run 1 is selected as each configuration's representative
by default; select another run or several runs to compare variation. Normalized
loss curves compare optimization shape, not the quality of different objectives.

Confusion-matrix columns are aligned to true classes by maximum overlap using
Hungarian matching. Original cluster IDs remain visible. The matrix can show
counts or proportions per true class; overflow buckets preserve the total count.
Matching uses the displayed classes/clusters, while evaluation metrics use all
samples. Projections, confusion matrices and loss comparisons offer plot JSON,
data CSV and a ZIP containing a standalone interactive HTML figure.

## Export and deployment safety

Each run or batch automatically writes a detailed JSON report to the output
directory, defaulting to `./experiments/` on the server. JSON includes per-run
metrics, mean/std summaries, raw and normalized losses, available model
diagnostics, and search tuning steps. Direct JSON/CSV downloads are available
beside the results; Export also provides a single report for a complete batch.
Use **Prepare ZIP** only when run arrays are needed. Large ZIPs still require memory while downloading;
prefer JSON/CSV for large comparisons and export individual run collections.

The per-record ZIP export contains the recipe, library version, dataset fingerprint,
sample IDs, observation mask, per-fit metrics, predictions, embeddings, loss,
and benchmark labels used by the visualizations. Original feature matrices are
not included. A separate metrics CSV is also available.

Batch ZIPs contain the combined JSON and CSV; per-record ZIPs include arrays.
Session history and run arrays are temporary. Download them before restarting
the server. Automatically saved JSON reports persist; removing a session result
does not delete a previously saved report from disk.
The default launcher binds locally. Public hosting needs authentication, TLS,
upload limits, execution limits, and operating-system resource controls; do not
expose an unauthenticated research server to the internet.
