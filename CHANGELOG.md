# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [0.3.3] - 2026-03-11 *(in progress)*
### Fixed
- **LMGEC: GPU out-of-memory (OOM) crash.**
  TensorFlow pre-allocated the entire GPU VRAM at startup, causing `InternalError: Failed copying input tensor` on devices with limited memory (e.g. laptop GPUs). Multiple sequential `.fit()` calls in the same session (notebooks) would also OOM because tensors from previous runs were never released.
  - Added `_configure_gpu()` — enables **GPU memory growth** at import time so TensorFlow allocates VRAM on demand instead of grabbing everything upfront.
  - Added **automatic GPU → CPU fallback** — if the GPU runs out of memory during the training loop, LMGEC catches the error, clears the GPU session, and retries the run on CPU. A warning is logged but the run completes.
  - Added **GPU memory cleanup after `fit()`** — all TensorFlow tensors are explicitly deleted and `tf.keras.backend.clear_session()` is called after extracting NumPy results, freeing VRAM for the next call.

### Changed
- **LMGEC: moved `scipy.sparse` import to module top level** — was previously a lazy import inside `fit()`. Since `scipy` is already a transitive dependency (via scikit-learn), there is no benefit to deferring the import.
- **Experiment output: unified experiment folder.** When `save_markdown=True`, all artifacts are now saved together in a single experiment folder instead of being scattered across separate directories:
  - `results.md` — well-organized Markdown report (config table, per-dataset result tables, best values bolded).
  - `experiment.log` — full debug log written directly inside the experiment folder (previously created in a separate `./logs/` directory).
  - `metadata/config.json` — model hyper-parameters, dataset names, metric names, number of runs.
  - `metadata/datasets.json` — per-dataset shape info (n_samples, n_views, n_clusters, feature dimensions, adjacency shapes).
  - `metadata/environment.json` — Python version, TensorFlow version, NumPy/SciPy versions, GPU device name, OS, platform.
  - `metadata/raw_scores.json` — every individual run score for reproducibility and re-analysis.
- **Logger: `experiment_dir` parameter.** `setup_experiment_logger()` now accepts an `experiment_dir` argument; when provided, the log file is written as `<experiment_dir>/experiment.log` instead of `./logs/`. The logger also stores the log file path as `logger.log_file_path`.
- **Markdown export API renamed** for clarity:
  - `export_experiment_markdown()` → `export_experiment()` (now creates the full folder, not just the `.md` file).
  - `export_random_search_markdown()` → `export_random_search()` (same).
- **Experiment & RandomSearchExperiment: `log` parameter (default `False`).** By default, no log file is created on disk. When `log=True`, a log file is written to `log_dir`. When `save_markdown=True`, logging is **forced on** and the log file is written inside the experiment folder regardless of the `log` setting.
- **Experiment verbose=2: tqdm progress bar replaces one-liner messages.** `verbose>=2` now renders a live `tqdm` progress bar that tracks the total number of runs (datasets × models × runs) and displays the current dataset, model, and run number via `set_postfix_str`. The previous `logger.info` one-liner progress messages are removed.
- **RandomSearchExperiment verbose=2: tqdm progress bar.** `verbose>=2` now renders a `tqdm` progress bar tracking trial progress, displaying the best metric score found so far in the postfix.
- **`plot_clusters()`: new `ax` and `show` parameters.** Accepts an optional `matplotlib.axes.Axes` object for composing multi-panel figures, and a `show` flag (default `True`) to control whether `plt.show()` is called. Returns the reduced 2D array as before.

### Added
- **`Experiment` & `RandomSearchExperiment`: `save_markdown` and `experiment_name` parameters.**
  - `save_markdown=True` — saves the full experiment folder (results + log + metadata).
  - `experiment_name="my_exp"` — custom folder name under `experiments/`. When `None`, auto-increments as `experiment#1`, `experiment#2`, etc.
- **Metadata collection in `Experiment.run()`** — automatically gathers dataset shapes, model configs, environment info, and per-run raw scores during execution and passes them to the export functions.
- **Metadata collection in `RandomSearchExperiment.run()`** — same as above, plus all trial parameters and scores in `raw_scores.json`.
- **`pymv.visualization.plot_loss()`** — new function that plots a training loss curve. Supports customizable figure size, colors, markers, line width, log scale, grid, and the same `ax`/`show` composability as `plot_clusters()`. Returns the `matplotlib.axes.Axes` object.
- **`tqdm` dependency** added to `pyproject.toml` (`tqdm>=4.60.0`).

## [0.3.2] - 2026-03-10
### Changed
- **LMGEC: per-view preprocessing parameters.**
  - `beta` now accepts `float` (same for all views) **or** `list[float]` (one per view).
  - `tf_idf` now accepts `bool` (same for all views) **or** `list[bool]` (one per view).
  - Renamed `center_data` → `center` and added `scale` for consistency.
- **LMGEC: `preprocess` master switch** (`bool`, default `True`).
  When `False`, all internal preprocessing (normalization, graph propagation, centering, scaling) is skipped — the user passes already-prepared views directly into the optimisation loop.
- **LMGEC: `As` is now optional** (`None` by default). When `preprocess=True` and `As=None`, features are normalized but not propagated (no graph).
- **LMGEC: TensorFlow training loop rewritten** — factory `_make_train_loop(n_views)` unrolls the view loop at trace time (`range()` instead of `tf.range()`), fixing the `ValueError` crash caused by symbolic indexing into Python lists.
- **Experiment: clean console output.** Console now shows only two tables:
  1. **Model Configuration** — one row per model, columns = hyper-parameters.
  2. **Results — \<dataset\>** — one table per dataset, rows = models, columns = metrics + Time (best values marked with `*`).
  All noisy progress messages (loading, per-run scores, timing) go exclusively to the **log file**.
- **Experiment: flexible input API.** `datasets`, `models`, and `metrics` parameters now accept multiple formats:
  - `datasets`: `str`, `list[str]`, `list[callable]`, `dict[str, callable | tuple]`, single `callable`, single `(Xs, As, y)` tuple.
  - `models`: single `BaseModel` instance or a list.
  - `metrics`: single `callable`, `list[callable]`, or `dict[str, callable]`.
- **Experiment: verbose levels 0–3** (default now **1** instead of 2):
  - `0` — silent (nothing printed).
  - `1` — config table + result tables only.
  - `2` — one-liner progress per (dataset × model) + tables.
  - `3` — full debug (every run, all params, same as log file).
- **RandomSearchExperiment: verbose default changed to 1** (table only).
- **Logger: default log directory** changed from `~/pymv_logs` to `./logs` (relative to current working directory, so logs live next to your project).
- **Data loaders: all `print()` statements replaced with `logging`** (11 files: `downloader.py`, `acm.py`, `dblp.py`, `imdb.py`, `wiki.py`, `bdgp.py`, `pascal.py`, `prokaryotic.py`, `coco_all.py`, `coco_cross.py`, `amazon_photos.py`). When called from an `Experiment`, dataset loading messages go only to the log file. When called standalone, they go through Python's default logging.

### Added
- `LMGEC.fit_predict(Xs, As)` — convenience method returning labels directly.
- `LMGEC.name` property (get/set) — used by `Experiment` tables to distinguish model configurations.
- `preprocess_features()` now accepts a single matrix (not just a list) and returns a single matrix in that case.
- `ExperimentResult.config_table()` — renders a model-configuration table (one row per model, columns = hyper-parameters).
- `ExperimentResult.results_tables()` — renders per-dataset result tables.
- `ExperimentResult.set_models()` — stores model list for the config table.
- `RandomSearchResult.best_score` property — best value of the target metric across all trials.
- Comprehensive NumPy-style docstrings with `.. math::`, `Examples`, and `See Also` sections for all functions in `pymv.preprocessing.features`:
  - Public API functions marked with **Public API** tag.
  - Internal helpers marked with `.. note:: **Internal helper**` tag pointing to the parent function.
- Module-level docstring for `pymv.preprocessing.features` listing the public/internal API split.
- Comprehensive NumPy-style docstrings for all functions in `pymv.metrics.clustering`:
  - External metrics (`clustering_accuracy`, `clustering_f1_score`, `nmi_score`, `ari_score`) with mathematical formulas and Hungarian alignment explanation.
  - Internal metrics (`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`) with formulas and higher/lower-is-better guidance.
  - Internal helpers (`ordered_confusion_matrix`, `cmat_to_pseudo_labels`) marked with **Internal helper** tag.
- Module-level docstring for `pymv.metrics.clustering` listing the external/internal/helper API split.

### Fixed
- **Data loader console noise during experiments** — dataset loaders printed verbose messages (`Loading ACM dataset...`, `Dataset already downloaded.`, sample/view counts) directly to stdout via `print()`. These now use the `logging` module and are silenced on the console when run inside an `Experiment` (captured in the log file instead).

## [0.3.1] - 2026-03-09
### Added
- **8 new dataset loaders** in `pymv.data`, each with on-demand Zenodo download:
  - `load_bdgp()` — BDGP multi-view dataset.
  - `load_coco_all()` — COCO (all views) multi-view dataset.
  - `load_coco_cross()` — COCO (cross-modal views) multi-view dataset.
  - `load_dblp()` — DBLP attributed graph (3 meta-path adjacencies: APTPA, APCPA, APA).
  - `load_imdb()` — IMDB attributed graph (2 meta-path adjacencies: MAM, MDM).
  - `load_pascal()` — Pascal multi-view dataset.
  - `load_prokaryotic()` — Prokaryotic multi-view dataset.
  - `load_wiki()` — Wikipedia attributed graph (features + adjacency, with log-transformed second view).
- All new loaders registered in `pymv.data.registry` and exposed via `pymv.data.__init__`.
- All datasets accessible through the unified `load_dataset("name")` API.
- **Experiment framework** (`pymv.experiment`) — Cornac-inspired evaluation harness:
  - `Experiment(datasets, models, metrics)` — runs all (dataset × model) combinations, reports per-dataset result tables with timing; supports multiple independent runs with mean ± std.
  - `RandomSearchExperiment(model_class, param_distributions, ...)` — random hyper-parameter search for a single model class, tracks best configuration by a chosen metric.
  - `ExperimentResult`, `RandomSearchResult`, `Result` — structured result containers with pretty-printed ASCII tables.
  - `setup_experiment_logger()` — timestamped dual-handler logging (file at DEBUG level + console at INFO level).
  - All execution details (parameters, per-trial scores, timings, errors) saved to log files in a configurable directory.

## [0.3.0] - 2026-03-09
### Changed — Full library restructure
- Renamed the package from `graph-lmgec` (`graphs.lmgec`) to **`pymv`** (`src/pymv/`).
- Reorganized the flat module layout into a domain-driven subpackage structure:
  - `pymv.core` — abstract `BaseModel` class that all models must implement (`fit` contract).
  - `pymv.models` — model implementations; LMGEC moved to `pymv.models.lmgec.LMGEC`.
  - `pymv.data` — dataset loading subsystem (`base`, `registry`, `downloader`, per-dataset loaders).
  - `pymv.preprocessing` — feature and graph utilities (`features.py`, `graph_construction.py`).
  - `pymv.metrics` — clustering evaluation (`clustering.py`).
  - `pymv.visualization` — plotting utilities (`visualizer.py`).
  - `pymv.experiment` — (placeholder) experiment runner subpackage.
  - `pymv.utils` — (placeholder) general-purpose helpers.

### Added
- **`pymv.core.BaseModel`** — abstract base class enforcing a `fit(X)` interface for all models.
- **`pymv.data` subsystem**:
  - `base.get_dataset_path(name)` — resolves `~/.pymv_datasets/<name>` and creates the directory.
  - `downloader.download_file(url, dest)` — downloads remote `.mat` files with progress bar (requests + tqdm), skips if cached.
  - `registry.load_dataset(name)` / `registry.list_datasets()` — central dispatcher for dataset loaders.
  - `acm.load_acm()` — downloads ACM from Zenodo, returns `(Xs, As, labels)`.
- **`pymv.preprocessing`** (rewritten from scratch with full docstrings):
  - `preprocess_graph(adj, beta)` — computes row-normalized propagation matrix S = D⁻¹(A + βI).
  - `normalize_features(features, tf_idf)` — TF-IDF + ℓ2 or plain ℓ2 normalization.
  - `preprocess_features(features, center, scale)` — optional StandardScaler across views (sparse-safe).
  - `get_propagated_features(features, adj, beta, tf_idf)` — full pipeline: normalize → propagate → dense H = S X.
- **`pymv.metrics.clustering`** (expanded):
  - Added `nmi_score(y_true, y_pred)` wrapper around normalized mutual information.
  - Added `ari_score(y_true, y_pred)` wrapper around adjusted Rand index.
  - Added internal (unsupervised) metrics: `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`.
- **`pymv.visualization.plot_clusters`** (rewritten):
  - Supports t-SNE, PCA, and UMAP (optional dependency) for 2D reduction.
  - Returns the reduced array for further use.
  - Colored per-cluster scatter with legend.
- **LMGEC model** refactored (`pymv.models.lmgec.LMGEC`):
  - Now inherits from `BaseModel`; `fit()` returns `self` (scikit-learn style).
  - Constructor uses explicit keyword arguments: `n_clusters`, `embedding_dim`, `temperature`, `max_iter`, `tol`, `beta`, `center_data`, `tf_idf`, `scale`.
  - `fit(Xs, adjs=None)` handles graph propagation and preprocessing internally.
  - Learned attributes stored on the instance: `labels_`, `embedding_`, `centroids_`, `alphas_`, `loss_history_`.
  - Explicit float32 conversion for TensorFlow tensors; safe fallback when all view weights are zero.
- TensorFlow added to `dependencies` in `pyproject.toml` (`tensorflow>=2.10.0`).

### Removed
- Old `graphs.lmgec` package layout and `graphs` namespace package.
- Standalone `graph_construction.GraphBuilder` class (file kept as placeholder for future use).
- Standalone `Visualizer` class with `plot_confusion_matrix`, `plot_embedding`, `plot_graph_structure` replaced by the simpler `plot_clusters` function.
- `lmgec_test()` smoke helper.
- `evaluate_clustering()` convenience wrapper (use individual metric functions instead).
- Bundled datasets removed from the installable package; now downloaded on demand to `~/.pymv_datasets/`.



## [0.2.2] - 2026-02-16
### Added
- Stable multi-view consensus weighting.

### Fixed
- LMGEC `fit()` now explicitly converts arrays to `tf.float32` tensors before entering `@tf.function`, preventing dtype mismatch errors.
- Division-by-zero guard when all view alphas sum to zero.
- `embedding_dim >= n_features` edge case handled with a warning and identity fallback instead of a crash.

## v0.5.0 (2026-04-10)

### Feat

- **data**: add embedding variants (clip, resnet_minilm, vit_minilm, vit_sbert) for pascal, coco_all, and coco_cross
- **data**: add embedding variants (clip, resnet_minilm, vit_minilm, vit_sbert) for pascal, coco_all, and coco_cross
- **data**: add flickr30k, iaprtc12, nus_wide, and esp_game dataset loaders

### Fix

- **preprocessing**: correct order of normalization and scaling in LMGEC pipeline
- **experiment**: fix tqdm progress bar to update in place instead of printing every line
- **ci**: update py_compile path from src/graphs to src/pymv
- **lint**: fix all flake8 violations (W292, W291, W293, W391, F401, F841, E221, E302, E303)
- **markdown**: adding horizontal scrollbar to fix wrapping problem in the notebook output

## v0.4.0 (2026-04-01)

### Feat

- add model registry, pytest suite, and CI workflow
- **tracking**: add expirement tracking system for saving configs and results
- **models**: add models registry system for dynamic model management
- add CLI, config-based experiments, and core framework improvements #2
- **core**: add version endpoint to expose library version
- **cli**: add config-based experiment system using yaml/json
- **cli**: add command-line interface for running experiments
- **experiements**: improve configuration table formatting and conditional display

### Refactor

- **git**: modify .gitignore for experiments and temorary files
- **cli**: adding the verbose details with cli
- **dataloader**: standardize dataset outputs with unified structure
- **preprocessing**: modify some typos in the code in preprocessing folder
- **metrics**: modify some typos in the code in metrics folder
- **experiment**: modify some typos in the code in expirement folder

## v0.3.3 (2026-03-16)

## v0.3.2 (2026-03-10)

## v0.3.1 (2026-03-09)

## v0.3.0 (2026-03-09)

## v0.2.2 (2026-02-16)

## v0.2.1 (2026-01-06)

## v0.2.0 (2026-01-02)

## v0.1.0 (2025-12-01)
