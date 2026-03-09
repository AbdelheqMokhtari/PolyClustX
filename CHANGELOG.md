# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

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
- Bundled `.mat` dataset files moved from `src/graphs/lmgec/datasets/` to `tests/datasets/`.

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
