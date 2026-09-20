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
not discard the dataset collection, applied preparation, model settings, or saved
experiments.

1. **Dataset** loads, names, inspects and prepares datasets in one page.
2. **Model** prepares named configurations, even before any dataset is loaded.
3. **Hyperparameter search** runs random or grid search and selects an objective.
4. **Experiment** selects dataset/model combinations and repeated runs.
5. **Results** aggregates repetitions as mean ± standard deviation.
6. **Visualizations** compares normalized loss, embeddings, labels, and confusion.
7. **Export** downloads individual results and complete comparison batches.

The sidebar is navigation only. Use **Browse datasets** on the Dataset page to
inspect a collection item, rename it, unload it or remove it. Select a prepared
dataset directly on the search page; experiments have their own dataset selector.
Only the currently inspected source needs to be materialized in RAM.
Catalog data can be reloaded from its cache,
synthetic data can be regenerated from its recipe, and uploaded data is staged
in a session-specific temporary directory. Removing an uploaded dataset also
removes its temporary source files. Experiment results remain available.
Run arrays are also stored in session-specific temporary files and read only
when requested, rather than retaining every embedding in RAM. Removing a result
releases its temporary arrays. Python datasets added directly with
`Workspace.add_dataset` retain an in-memory source snapshot; use lazy loaders
for large programmatic sources.

## Loading and inspecting data

Name synthetic and uploaded datasets before loading them. Defaults increment
automatically: `synthetic-dataset#1`, `synthetic-dataset#2`, or
`uploaded-dataset#1`, `uploaded-dataset#2`. Catalog datasets retain readable
catalog names. Names are unique within the collection; duplicates receive a
numeric suffix, never an internal ID. Renaming survives unloading and reloading.

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

### Generate synthetic graphs

Under **Dataset → Synthetic → Synthetic graphs**, choose **No graphs**,
**Per-view graphs**, or **Shared graphs**. Per-view graphs use each selected
view's features; shared graphs use one source view and attach identical
relations to the selected views. Select one target view and one relation for
a single graph, or increase **Relations per selected view** for multiple graphs.

Each relation has its own type and neighbor count:

- **Binary kNN** connects samples if either is among the other's nearest neighbors.
- **Mutual kNN** connects samples only when both neighbor directions exist.
- **Weighted kNN (RBF)** uses kNN edges weighted by
  `exp(-distance² / (2 × bandwidth²))`. Adjust the bandwidth to the feature scale.

Graphs are sparse, undirected, aligned with sample order and have no self-loops.
Edges use features, never ground-truth labels. Mutual kNN and very small RBF
bandwidths can produce isolated samples. Change each relation's neighbor count
or type to create different relations; identical settings yield identical graphs.
The random seed reproduces features and graphs, including after unloading and
reloading the dataset. Graph settings do not change the generated features.

Generation rejects requests exceeding two million directed neighbor entries
across attached relations, before allocating the data. Large nearest-neighbor
searches can still be slow; start small. Use **Apply changes** to make the
dataset available to experiments. In preprocessing, select sum or Hadamard
fusion when using multiple relations. See also the
[Python graph-generation example](graphs.md#generate-synthetic-graphs-in-python).

### Inspect loaded data

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

## Preparation on the Dataset page

After loading, inspect the dataset, choose its benchmark setting and configure
preparation below its details. Click **Apply changes** to validate and save it.
New datasets are offered in Experiment and Hyperparameter search only after
this action. Adding graph relations also requires applying the updated
preparation. You can prepare any number of datasets and browse them independently.

After applying, switch **Inspect** between **Prepared data** and **Original
data** to compare profiles. Prepared statistics are cached without retaining a
second full dataset in memory. Editing controls does not change the saved
preparation until you apply again. Each application starts from the original
source, avoiding repeated normalization or propagation.

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

If a dataset remains incomplete, combinations requiring complete views are
excluded from experiments and those methods are unavailable for its search.
Explicit imputation makes the combinations available again.

## Models and hyperparameters

Model controls change with the selected registry entry. Numeric parameters,
booleans, and layer dimensions are edited directly; there is no JSON editor.
Models can be prepared before datasets. Choose the method, edit its name and
parameters, then save it. Default names are `LMGEC#1`, `LMGEC#2`, `SCM#1`, etc.
Dataset compatibility is checked when pairing configurations for experiments,
not when building the model collection. When a dataset has generic feature
preprocessing, model-internal preprocessing is disabled for its fit to avoid
applying transformations twice.

Independent model configurations start with K = 3; choose another K or compare
several values as needed. Search settings start from the selected dataset's
number of classes when labels exist. Both defaults are editable. Labels are
never passed into model fitting, and combinations with K greater than the
dataset's number of samples are excluded.

Save each configuration with **Save model settings**, then prepare another
method or parameter setting. Saved configurations can be selected together in
Experiment and removed independently of existing results.

## Hyperparameter search

Open **Hyperparameter search** and choose **Random search** or **Grid search**.
Select a prepared dataset, model, K values and parameters to search. Select suggested
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
cached at a time. Enter a **Best model name**, defaulting to a name such as
`LMGEC_grid_best_#1` or `LMGEC_random_best_#1`. **Copy best to experiment** adds
the winning model parameters and K to the model collection under that name,
selects its dataset, restores its preprocessing and opens Experiment. It does
not start another fit automatically. Search reports also remain available for
visualization and export after their source dataset has been removed.

## Repeated experiments and evaluation

Experiment names increment automatically (`Experiment 1`, `Experiment 2`, …).
Select prepared datasets and saved model configurations. **All compatible
pairs** runs their valid Cartesian combinations without listing every pair.
Use **Choose in table** to select combinations in a matrix: one row per dataset,
one column per model, and a checkbox per pairing. Blank incompatible cells cannot
enable a fit. The table scrolls when the collection is large, and the fit count
updates with your selection. The
**Independent runs per configuration** slider ranges from 1 to 20. Each
configuration and K runs with independent seeds.
The Results page keeps every fit and groups repetitions into mean ± population
standard deviation. Named configurations appear in result headings and reports.
Each result has a consistent actions row and a separate downloads section.

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
Run choices identify the experiment, dataset, named model, configuration, K and
repetition number, without internal session codes. Repeated jobs with identical
names receive a readable **Result 1**, **Result 2**, etc. in the comparison list.

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
Downloaded files use readable experiment/dataset/model names and timestamps;
plot downloads also identify the configuration and run. Machine-readable JSON
retains internal identifiers for linking records, but these are not run names.
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
