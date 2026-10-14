# Interactive interface

Build datasets, prepare named models, compare experiments and download results
without writing code. Pages are independent: you can return to any page without
discarding your work.

## Install and launch

Use Python 3.10 or newer; Python 3.11 is recommended.

```bash
python -m pip install polyclustx
polyclustx-display
```

The terminal shows the PolyClustX version and an **Open in browser** address,
normally http://127.0.0.1:8501. Open that address to use the interface; press
**Ctrl+C** in the terminal to stop it. Framework development tips are hidden,
while server warnings and errors remain visible.
For environment creation, Windows activation, repository downloads and restarting,
follow the [installation tutorial](installation.rst). To choose another port:

```bash
polyclustx-display --server.port=8502
```

## Your first experiment

Try a small comparison before loading your own data:

1. In **Dataset → Synthetic**, name the dataset `Three groups`. Use 90 samples,
   3 classes and 2 Gaussian views with 8 features each. Leave graphs disabled.
2. Click **Create synthetic dataset**, keep benchmark mode enabled, then click
   **Apply changes**. The dataset is now available for experiments.
3. In **Model**, save `MultiViewKMeans` as `K-means baseline`. Save a second
   configuration using `LMGEC`, embedding dimension 3 and `max_iter=10`.
4. In **Experiment**, select `Three groups` and both models. Keep K=3 and choose
   3 independent runs. Click **Run experiment**: this performs six fits.
5. In **Results**, compare the mean NMI and its standard deviation. In
   **Visualizations**, select Run 1 and compare reference and predicted labels.
6. In **Export**, choose the experiment and download its ZIP report.

To try unlabeled use, return to **Dataset**, disable benchmark mode and apply
the change. Run a new experiment using silhouette instead of NMI.

## 1. Dataset: load, inspect and prepare

### Choose a source

| Source | What to do |
| --- | --- |
| Catalog | Select a dataset family, inspect its features and relations, then load it. Variant controls appear only when alternate representations exist. |
| Files: one file per view | Upload CSV, TSV or NPY files. Select labels separately if present. |
| Files: NPZ / MATLAB container | X* features, A* graphs and labels are recognized. Name each composed view and select its feature and relations; other layouts retain explicit mapping. |
| Synthetic | Choose samples, classes, views, feature dimensions and distributions. Optionally generate graphs. |

Name synthetic or file datasets before loading. Defaults increment as
`synthetic-dataset#1`, `synthetic-dataset#2`, and `uploaded-dataset#1`.
Use **Browse datasets** to switch between loaded items, rename them, unload
their arrays from RAM, or remove them from the collection.

Rows are samples. CSV/TSV headers are detected automatically. A labels file
may be uploaded with feature files, but select it as labels so it is not used
as a feature view. A numeric label column can also be selected within a view;
column numbers in the interface start at 1.

For standardized NPZ/MAT files, the interface recognizes X* features, A* graphs
and the exact `labels` field. For other naming schemes, select each role
explicitly. To align differently sampled views, select one sample-ID array per
view. See [loading data](data.md) and [missing views](missing_views.md).

### Choose feature/graph combinations

After loading a catalog dataset, open **Choose feature/graph combinations**:

1. Set **Number of views** and give each view a unique name.
2. Choose its **Feature array**. You can select the same feature for several views.
3. Choose **Graph relations**: none for feature-only input, one for a separate
   graph view, or several for relation fusion. Graphs can also be reused.
4. Click **Update views**, then configure preparation below and click **Apply changes**.

For ACM, create two views from `X1_features`: one with `A1_PAP` (shared authors)
and one with `A2_PLP` (shared subjects). Alternatively, make one view with both
relations or one feature-only view. Updating composition resets preparation so
old settings do not silently apply to different views. The selection survives
unloading and reloading; the source file and previous results are unchanged.

Standardized file uploads provide the same name/feature/graph controls before
**Add container dataset**. The labels field is selected for evaluation, not as
a feature. No JSON editing is required.

**Add graphs to a view** accepts adjacency files (NPY, NPZ, CSV, TSV) and
GraphML/GEXF graphs. Select the target view. Repeated uploads add relations;
graph nodes and sample IDs must agree.

### Inspect the data

The Dataset page shows sample and class counts, class balance, view dimensions,
feature statistics, missing observations, estimated memory, and graph properties.
Catalog descriptions explain feature encoders and structural relations. Stored
feature counts are separate from composed view counts; sparse/shared matrices
are not counted repeatedly in the memory estimate. Actual loaded dimensions
are shown even when they differ from source-encoder widths.
Inspect either original or applied/prepared data. Benchmark mode starts enabled
when labels exist; uncheck it to use internal evaluation only.

### Generate synthetic graphs

Under **Synthetic graphs**, choose **No graphs**, **Per-view graphs**, or
**Shared graphs**. Per-view graphs use each target view's features. Shared
graphs use one source view and attach the same relations to all selected targets.

One target view and one relation gives a single graph. Increase **Relations per
selected view** to create multiple graphs. Each relation has its own settings:

- **Binary kNN**: an edge exists if either neighbor direction exists.
- **Mutual kNN**: both neighbor directions must exist.
- **Weighted kNN (RBF)**: kNN edges weighted by `exp(-distance² / (2 × bandwidth²))`.

Graphs are sparse, symmetric, have no self-loops, and never use class labels.
Change neighbor counts or relation types to generate different relations.
Mutual kNN and small RBF bandwidths may produce isolated nodes. The same seed
reproduces the dataset, including after unloading and reloading it.
Generation rejects requests above two million directed neighbor entries.
See [Python graph examples](graphs.md#generate-synthetic-graphs-in-python).

### Apply preprocessing

Preprocessing stays on the Dataset page. Apply a shared transformation to all
views, to selected views, or configure views separately. Options include
normalization, TF-IDF, centering, scaling, graph propagation and relation fusion.
Multiple graphs can be fused by weighted sum or Hadamard product.

To test missing-view settings, choose a missing fraction and seed. To run a
complete-input model, choose explicit mean or zero imputation. Models that
cannot accept the prepared data are excluded from compatible experiment pairs.

Click **Apply changes** after loading or changing preparation. Only applied
datasets appear in experiments and searches. This validates and saves the
recipe; the original source stays unchanged. Preparation is applied once for
each fit, and model-internal preprocessing is disabled when necessary to
avoid applying feature transformations twice.

## 2. Model: save named configurations

Models can be prepared before any dataset is loaded. Select a method and edit
its typed parameter controls, then choose a **Configuration name**. Defaults
are names such as `LMGEC#1`. Save as many configurations as needed.
See [available models and original implementations](../background/algorithms.md).

K-means, spectral co-training, IDEC, COMPLETER, DCP and MFLVC use the same
configuration workflow. See the [model tutorial](clustering_models.md) for
single-view reductions, original formulations and training-stage meanings.
Device, affinity and initialization use dropdowns. Spectral gamma and the
informative view have **Automatic** switches; disabling them reveals typed
numeric controls. COMPLETER/DCP require at least two fully observed training
samples and trained predictors for missing-view completion. These requirements
are checked before incompatible experiments run.

Saved K values are used for real-world experiments and manual K comparisons.
For benchmark experiments, dataset class counts are the default on Experiment;
you can edit them there or switch to the saved model K values.

LMGEC may need a smaller embedding on a low-dimensional dataset. Its limit is
`min(samples, smallest view feature count) − 1`. Results show a short adjustment
notice, the effective value used, and the requested value. Your saved configuration
is unchanged. This limit is model-specific and is not applied to SCM.
It is also not applied to neural hidden dimensions or spectral cluster counts.

## 3. Search: tune parameters independently

Select a prepared dataset, a compatible model, and **Random search**,
**Grid search**, or **Optuna (TPE)**. Enable parameters to search and type candidate values into
their controls; press Enter to add each value. Invalid values are rejected.
For Optuna, enable **Use a range** on a numeric parameter to set minimum,
maximum and optional logarithmic scale. Otherwise Optuna chooses among your
explicit candidate values. Grid evaluates every combination; random and Optuna
use the specified trial budget. Each K has its own Optuna study so scores from
different cluster counts do not distort one another's parameter suggestions.

Choose metrics and an **Optimize metric**, then the number of independent runs
(1–20). Internal metrics work without labels; external metrics such as NMI
require benchmark mode. Davies–Bouldin is minimized; the other built-in scores
are maximized. Selection uses mean scores across repeated runs.

Click **Run search** to see the summary and tuning steps. Choose a **Best model
name** (default: `MODEL_strategy_best_#1`) and click **Copy best to experiment**
to save the winning settings. The winning K is preserved. This does not start
a new fit. Download the search report from **Export → Individual results**.

## 4. Experiment: compare combinations

Select prepared datasets and saved models. **All compatible pairs** evaluates
their Cartesian combinations. **Choose in table** offers a dataset-by-model
checkbox matrix, useful for larger collections. Incompatible pairs are excluded.

For labeled benchmark datasets, K defaults to the number of unique true labels.
Edit **K for dataset-name**, or uncheck **Use dataset class counts for benchmark
datasets** to use the model's saved K values, including multiple-K comparisons.
Real-world datasets always use manually configured model K values.
Benchmark and real-world datasets cannot be mixed in one experiment batch.

Name the experiment and select 1–20 independent runs per configuration.
Under **Reproducibility and saving**, choose a base seed and output directory.
Run 1 uses the base seed; subsequent runs increment it. A JSON report is saved
automatically, normally under `./experiments/` in the server's working directory.

Jobs run one fit at a time, with at most 2,000 fits per job. Start with a small
trial: an individual model may still exceed available RAM or GPU memory.
The status panel shows the current fit, training stage and completed iterations.
After a stage makes progress, its elapsed time estimates the remaining stage
time. This is an estimate, not a deadline; it excludes later stages and can
change with early stopping, dataset size and device. No iterations are removed
to make a run appear faster. Initial graph construction, spectral decomposition
and final K-means can take time before the next update.

## 5. Results and visualizations

Results show one column per parameter, mean ± population standard deviation,
and detailed runs. Failed fits remain visible with errors. You can reuse
settings or delete a result. **Open Export** takes you to the download page.

Visualizations include cluster sizes, normalized loss, projections, a confusion
matrix and cluster details. Choose an experiment/configuration/run by name.
**Compare normalized loss** can combine models across experiments; Run 1 is
selected by default. Normalization compares curve shape, not objective quality
between different methods.

Compute a PCA, t-SNE or optional UMAP projection to compare predicted clusters
and true labels. At most 5,000 samples are plotted. With more than 20 classes,
a compact color scale replaces the long legend.

Confusion matrices use full Hungarian matching to maximize the diagonal before
limiting the display. Predicted IDs are displayed as `0` through `K−1`; categorical
axes avoid extra numeric ticks. Raw model labels remain unchanged in reports.
Up to 30 high-frequency matched pairs are shown; other counts are grouped into
separate Other buckets. Very large class matrices are rejected with an export
suggestion. Toggle row normalization to show proportions or counts.

Click **Save plot to Export** below any supported figure to capture its current
settings and plotted values. Saving the same figure again replaces that snapshot.

## 6. Export: download reports, plots and data

| Export type | Contents |
| --- | --- |
| Experiment batches | One JSON/CSV/ZIP report covering the whole comparison. |
| Individual results | Experiment or search JSON, metrics CSV, or ZIP with run arrays. |
| Visualizations | Saved plot JSON, plotted-data CSV, or ZIP with standalone interactive HTML. |
| Prepared datasets | MATLAB `.mat` file with applied preparation, views, graphs, IDs and availability mask. |

For individual result arrays, use **Prepare ZIP** first. For visualizations,
select a saved plot, enter its download name and select a format. Remove snapshots
you no longer need. Snapshot data stays on disk until requested; only the chosen
download is assembled. Up to 100 snapshots of at most 32 MB each are accepted.

For prepared data, select an applied dataset, choose whether labels should be
included, and click **Prepare MATLAB file**, then **Download prepared data**.
This exports Dataset preprocessing, not additional transformations inside a model.
Arrays are named `view_1`, `view_2`, `graph_1_1`, etc. `graph_mapping` contains
one-based view/relation pairs; `observed_mask` marks available observations.
Reload through Files → NPZ / MATLAB container and select those arrays explicitly.
The interface limits source data to 256 MB for this download; Python users can use
`polyclustx.display.exporting.dataset_mat(data)` directly and write the returned bytes.

## Session storage and hosting

Dataset recipes, run arrays and saved plots belong to the current session.
Download what you need before restarting. Removing a result does not delete
previously saved JSON reports. Catalog files remain in the download cache.
Uploaded sources and generated run/plot files use temporary directories.

The interface binds locally by default. Public hosting requires authentication,
TLS, upload limits and memory limits. Do not expose it publicly without those
protections. Browser downloads save to your browser's configured location;
automatic report paths refer to the machine running the server.
