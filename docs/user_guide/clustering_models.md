# Choosing and running clustering models

All models take a list of aligned feature matrices. Rows are samples; columns
may differ between views. Use `data.fit(model)` with a `MultiViewDataset` to
forward graph mappings and the observation mask automatically. Ground-truth
labels are used only by the evaluation tools, never by these estimators.

| Estimator / registry key | One view | Two views | More views | Missing views |
|---|---|---|---|---|
| `MultiViewKMeans` / `multiview_kmeans` | Ordinary K-means | Co-EM | Cyclic Co-EM extension | Explicit imputation |
| `MultiViewSpectralClustering` / `multiview_spectral` | Normalized spectral clustering | Spectral co-training | Other-view projection aggregation | Explicit imputation |
| `IDEC` / `idec` | IDEC | Early-fusion IDEC | Early-fusion IDEC | Explicit imputation |
| `COMPLETER` / `completer` | Autoencoder + K-means | COMPLETER | All-pairs extension | Native latent completion; ≥2 complete training samples |
| `DCP` / `dcp` | Autoencoder + K-means | DCP clustering | DCP complete-graph route | Native latent completion; ≥2 complete training samples |
| `MFLVC` / `mflvc` | Reconstruction + pseudo-label training | MFLVC | MFLVC | Explicit imputation |
| `CONAN` / `conan` | Encoder/fusion alignment | Contrastive fusion | Contrastive fusion | Explicit imputation |
| `DGF` / `dgf` | Feature-to-graph contrast | Dual filtering | Mean/all-pairs extension | Explicit imputation |
| `GCoMVKM` / `g_comvkm` | Feature-weighted objective | Collaborative updates | Collaborative updates | Explicit imputation |
| `SGLA` / `sgla` | Feature kNN Laplacian | Laplacian aggregation | Laplacian aggregation | Explicit imputation |
| `UMC` / `umc` | Feature-dropout curriculum | Modality masking | Core-view masking | Explicit imputation |

The one-view reductions and extensions above are framework conventions, not
claims that the original papers evaluated those settings. DGF and SGLA consume
supplied adjacencies; the other estimators in this table do not. Spectral
co-training constructs its own affinities. LMGEC and SCM are also available;
see [all models, papers and original implementations](../background/algorithms.md).

## Start with K-means

```python
import numpy as np
from pymv.data import MultiViewDataset
from pymv.models import MultiViewKMeans

rng = np.random.default_rng(42)
data = MultiViewDataset([rng.normal(size=(100, 8)), rng.normal(size=(100, 12))])
model = data.fit(MultiViewKMeans(n_clusters=3, n_init=10, random_state=42))
print(model.labels_.shape)              # (100,)
print([c.shape for c in model.centroids_])  # [(3, 8), (3, 12)]
new_labels = model.predict([rng.normal(size=(5, 8)), rng.normal(size=(5, 12))])
```

With one view, sklearn KMeans is used directly. With two views, each update
uses the other view's partition to recompute its centers and then reassigns
samples by Euclidean distance. With more views the partition passes cyclically
from view to view. `max_iter` counts sweeps across all views; `patience` limits
stalled sweeps. Restarts are selected by their final summed, unsquared Euclidean
objective, available as `coem_objective_` and `restart_objectives_`.

Final centers are means of samples assigned to the same cluster in every
view. If a cluster has no consensus samples, its previous centers are retained
with a warning; requested K is not silently changed. Prediction sums the
view-wise distance matrices. `inertia_` describes the final prediction centers;
single-view K-means instead uses sklearn's usual squared-distance inertia.
`embedding_` is the concatenated input features, not a learned projection.

Compared with the mvlearn behavioral reference, initialization uses correct
k-means++ sampling, empty consensus clusters retain centers, and stopping uses
finite objectives. Historical implementation bugs are not reproduced.

## Spectral co-training

```python
from pymv.models import MultiViewSpectralClustering

model = data.fit(MultiViewSpectralClustering(
    n_clusters=3, affinity="rbf", gamma=None, max_iter=5, random_state=42,
))
embedding = model.embedding_
```

Choose `rbf`, `nearest_neighbors`, or `poly`. Automatic gamma is estimated
separately per view as `1 / (2 * median_distance**2)`; a zero median uses 1
to avoid division by zero. `n_neighbors` includes the sample itself and cannot
exceed the sample count. Neighbor graphs are symmetrized. Polynomial kernels
accept `degree` and `coef0`.

The spectral representation is obtained from the normalized affinity, using
the largest-magnitude spectral components to match mvlearn's singular-subspace
convention. For signed refined similarities, absolute row degrees are used;
isolated rows have inverse degree zero. Every co-training round synchronously
uses the previous embeddings of all other views and the original affinities.
It does **not** cluster each view independently and vote on labels.

After row normalization, views are concatenated, or `info_view=0` selects one
view. sklearn KMeans supplies the final labels. `view_embeddings_` and
`gammas_` expose intermediate information. `subspace_change_history_` records
projection changes; this is not an optimization loss, so `loss_history_` is
`None`. This estimator is transductive: fit new data jointly rather than calling
an unsupported out-of-sample `predict`.

Dense affinity matrices require quadratic memory and eigendecomposition takes
cubic time. `max_memory_mb` rejects oversized estimates before allocation.
Use a smaller sample set for large datasets; increasing the budget does not
make a large spectral job fast.

## IDEC

```python
from pymv.models import IDEC

model = data.fit(IDEC(
    n_clusters=3, embedding_dim=10, hidden_dims=(128, 64),
    pretrain_epochs=20, max_iter=500, update_interval=20, random_state=42,
))
```

IDEC pretrains a symmetric autoencoder, initializes latent centroids using
K-means, and jointly optimizes reconstruction plus `gamma * KL(P || Q)`.
`Q` is the Student-t assignment distribution (`alpha` controls its shape);
`P` sharpens assignments and compensates for cluster frequencies. Targets
are refreshed over the full training set every `update_interval` minibatch
updates. Stopping uses the fraction of changed assignments and `tol`, not labels.

`pretrain_epochs` counts full epochs. **`max_iter` counts minibatch updates**,
matching the reference refinement loop. `predict` uses the trained Student-t
centroids; `transform` returns latent features. Several views are concatenated
before the encoder. Pretraining is performed locally with Adam, rather than
requiring the reference's legacy Keras/Caffe weight files; published pretrained
results are therefore not automatically reproduced. `pretrain_loss_history_`
is separate from the refinement `loss_history_`.

## COMPLETER and DCP: native missing views

```python
from pymv.data import simulate_missing_views
from pymv.models import COMPLETER, DCP

incomplete = simulate_missing_views(data, missing_rate=0.2, random_state=42)
model = incomplete.fit(COMPLETER(
    n_clusters=3, max_iter=50, prediction_start=10, random_state=42,
))
# The same data contract works with DCP(n_clusters=3, ...).
labels = model.predict(incomplete.Xs, observed_mask=incomplete.observed_mask)
```

Missingness is explicit: an all-zero observed vector is **not** treated as
missing. Training follows the reference's complete-sample approach: at least
two rows must be observed in every view. If the overlap is insufficient,
provide additional paired training samples or choose an explicit imputation
baseline. This requirement also appears in the model catalog and interface.

View-specific autoencoders produce probability-valued latent codes.
Consistency learning and reconstruction precede cross-view prediction from
`prediction_start` onward. Missing latent codes are predicted from the
available views; with several observed sources, their predictions are averaged.
Concatenated completed codes are clustered with K-means. Features themselves
are not filled or returned as an imputed dataset. Fitting never uses missing
placeholder values as observed data.

`prediction_start` must be less than `max_iter` and `prediction_weight` must
be positive when completing missing inputs. COMPLETER weights log-marginals
by `alpha + 1`; DCP uses `alpha` directly. With two views these clustering
objectives coincide when the effective weights match. DCP's category-supervised
classification loss is deliberately not used in unsupervised clustering.

COMPLETER's >2-view extension averages all pairwise consistency/prediction
terms. DCP implements the supplied complete-graph clustering route: for >2
views, pair (0, 1) has weight 1 and the others .1, consistency and prediction
terms are divided by the view count, and reconstruction by `views - 2`.
The two-view case uses the two-view reference loss, avoiding division by zero.
The supplied core-view and supervised routes are not implemented here.

`hidden_dims=(256, 256)` is a compact default, not a paper benchmark setting.
For the supplied reference architecture, choose `(1024, 1024, 1024)` and the
appropriate dataset-specific widths, latent dimension, epochs and learning rate.
`prediction_dims` controls each prediction bottleneck; `batchnorm` is optional.

## MFLVC

```python
from pymv.models import MFLVC

model = data.fit(MFLVC(
    n_clusters=3, hidden_dims=(128, 64), embedding_dim=64, projection_dim=32,
    pretrain_epochs=20, max_iter=30, finetune_epochs=10, random_state=42,
))
```

MFLVC first pretrains per-view autoencoders. Its contrastive phase combines
reconstruction, high-level feature contrast, and cluster-probability contrast
with a marginal entropy term. K-means on per-view min-max-scaled high-level
features then produces pseudo-labels. Hungarian matching aligns these to the
current cluster-head IDs before fine-tuning. **No true labels are used.**

Final labels are the argmax of mean view-wise cluster probabilities.
`embedding_` concatenates normalized high-level features; `probabilities_`
and `view_labels_` expose the cluster heads. `loss_stages_` identifies each
epoch in `loss_history_`; do not interpret phase boundaries as a single smooth
objective. `max_iter` counts only contrastive epochs. Fine-tuning uses proper
cross-entropy on logits rather than applying softmax twice as in the supplied
script, and minibatches replace the reference's full-dataset fine-tuning batch.
Default epochs are shorter than the paper's schedules. Match experimental
settings before comparing reported results.

## Shared behavior and the interface

Neural feature models with a `preprocess` option default to min-max scaling using observed
training values. Prediction reuses fitted scaling. Set `preprocess=False` for
already prepared inputs; the display does this when external preprocessing
has changed features. Neural kernels use float32; Co-EM and spectral kernels
use float64. DGF operates on supplied feature scales; normalize or scale its
inputs explicitly on Dataset when needed. sklearn provides K-means initialization/final clustering where
specified; no TensorFlow or mvlearn installation is required.

Choose `device="cpu"`, `"cuda"`, `"cuda:0"`, or `"auto"`. Automatic CUDA fits
restart on CPU after a CUDA allocation failure. Single-view K-means always
runs on CPU. Public results are NumPy arrays; fitted neural `model_` modules
are moved to CPU in evaluation mode. Memory checks are estimates, not hard
process limits. Inputs and caller RNG state are preserved. Set `random_state`
for repeatability within the same software and hardware environment.

In **Model**, select a method, edit the named controls and save a configuration.
Spectral gamma and informative-view controls have **Automatic** switches;
affinity and device use dropdowns. **Search** supports numeric candidates and
affinity choices with random/grid/Optuna search, including numeric ranges for
Optuna. **Experiment** handles repeated runs
and compatible dataset/model pairs. **Export** includes phase/component losses,
Co-EM restart objectives, consensus counts, and spectral subspace changes when
available. MultiViewSpectralClustering has no scalar training loss and records
subspace change instead; SGLA does record its spectral aggregation objective.

See `examples/new_models.py` for a small, offline experiment across these
methods and the [API reference](../api/models.rst) for constructor defaults.

## CONAN: nonlinear contrastive fusion

**Use when:** complete feature views should share a learned nonlinear fusion.
Each view has an MLP encoder. A fusion network combines their outputs and
aligns the fused representation with each view using an instance contrastive
loss. A divergence-based clustering head learns assignments jointly.

```python
from pymv.models import CONAN
model = CONAN(n_clusters=3, embedding_dim=32, hidden_dims=(128, 64),
              projection_dim=32, max_iter=50, random_state=42)
data.fit(model)
labels = model.predict(new_views)  # same view order and feature dimensions
```

This is the **MLP + SimCLR + DDC** route, not the optional image backbones,
DEC or SimSiam ablations. DDC combines kernel separation, assignment
orthogonality and a simplex-corner term. Positive contrastive pairs are matched
by sample; the reference's feature-axis concatenation issue is not reproduced.
With one view the encoder is aligned to its fusion representation. Graphs are
ignored. `max_iter` counts full training epochs. Exposes `probabilities_`,
`embedding_`, `labels_` and two-component loss histories.
[Paper](https://doi.org/10.1109/BigData52589.2021.9671851) ·
[Original code](https://github.com/Guanzhou-Ke/conan).

## DGF: node and feature graph filtering

**Use when:** complete feature views and graph structure should jointly inform
clustering. Linear per-view projections feed two filters: a node-neighborhood
filter and a feature-correlation filter. Cross-modal, random-walk and
pseudo-community contrast train the projections without true labels.

```python
from pymv.models import DGF
model = DGF(n_clusters=3, embedding_dim=32, num_layers=5,
            max_iter=50, batch_size=128, random_state=42)
data.fit(model)  # supplied graphs are used; otherwise feature kNN graphs are built
```

Graphs supplied per view are symmetrized and averaged into a common graph;
multiple relations are fused by the dataset first. Without supplied graphs,
cosine kNN graphs use `n_neighbors`. Two-view filtering follows the reference;
more views use mean feature-graph fusion and all-pairs cross-modal contrast.
One view retains feature-to-graph contrast. `attribute_filter=True` filters
random-walk edges using attribute similarity. Isolated walks stay at their
current node. Original sample IDs are used directly, fixing the reference's
subset-remapping problem. Final-epoch weights are returned, never a checkpoint
chosen using true-label NMI.

`max_iter` counts epochs. `num_layers`, `walks_per_node`, `walk_length`, and
`context_size` control compute cost. This is full-graph training; `batch_size`
chunks cross-modal comparisons, not graph propagation. The memory guard rejects
large estimated allocations. Loss terms are divided by detached magnitudes;
therefore `loss_history_` can be nearly constant even as training changes.
Inspect `loss_components_` for raw cross-modal, graph and community trajectories.
The model is **transductive**: refit to include new nodes.
[Paper](https://arxiv.org/abs/2511.20030) · [Original code](https://github.com/HaoranZ99/DGF).

## G-CoMVKM: feature selection and collaborative clustering

**Use when:** an interpretable feature-weighted method is preferable to a neural
network. Views exchange membership information, update centers, learn entropy
regularized feature weights and reduce low-weight features. View weights then
balance the resulting distortions; their membership vote yields the partition.

```python
from pymv.models import GCoMVKM
model = GCoMVKM(n_clusters=3, gamma=5., theta=4., random_state=42)
data.fit(model)
print(model.view_weights_, model.selected_features_)
```

The default parameters follow the authors' demonstration. `gamma` must exceed
one; the view-weight exponent is singular at one and the supplied Python
default below one reverses its preference for distortion. `theta` is positive.
Shared sample indices initialize aligned centers. Feature updates preserve a
per-feature regularizer, use stable softmax and retain at least one feature per
view. Constant features and empty clusters have finite fallbacks. The input
arrays are never reduced in place. Set `feature_selection=False` to retain all
features. Prediction uses the saved feature indices and training sample count,
so its result does not depend on prediction batch size. `max_iter` counts update
sweeps; `tol` controls objective convergence. Single-view input keeps the same
feature-weighted objective rather than claiming ordinary K-means equivalence.
[Paper](https://doi.org/10.3390/electronics14112129) ·
[Original code](https://github.com/KristinaP09/G-CoMVKM).

## SGLA: spectrum-guided graph aggregation

**Use when:** you want inspectable weights over supplied graphs and feature
similarity graphs. Each feature view creates a cosine kNN graph. SGLA learns
nonnegative weights summing to one over their normalized Laplacians, minimizing
the relative eigengap minus algebraic connectivity plus squared-weight
regularization. PyTorch computes eigensystems; SciPy's COBYLA searches weights.

```python
from pymv.models import SGLA
model = SGLA(n_clusters=3, n_neighbors=10, gamma=.5, max_iter=50, random_state=42)
data.fit(model)
print(list(zip(model.graph_sources_, model.view_weights_)))
```

The smallest eigenvectors are row-normalized, then clustered by K-means.
This readout replaces the standalone implementation's discretization routine.
Negative cosine weights are clipped to preserve valid graph degrees. Requires
`n_clusters < n_samples` for the relative eigengap. It accepts one or more
feature views, with or without supplied graphs, but is transductive.

This implementation uses a **bounded dense eigensolver**, with quadratic memory
and cubic eigendecomposition cost; it does not implement SGLA+'s approximation,
the reference's sparse large-scale solver, or its NetMF embedding task.
`max_memory_mb` guards allocation. `max_iter` is the requested objective
evaluation budget (COBYLA requires at least number-of-graphs + 1 evaluations).
[Paper](https://arxiv.org/abs/2508.09452) · [Original code](https://github.com/CyanideCentral/SGLA).

## UMC: density-guided contrastive curriculum

**Use when:** your complete feature views benefit from curriculum learning with
pseudo-labels. The framework accepts **your extracted feature matrices**; it
does not download BERT, WavLM, Swin or the authors' datasets.

```python
from pymv.models import UMC
model = UMC(n_clusters=3, embedding_dim=32, hidden_dims=(64,),
            core_view=0, initial_fraction=.25, pretrain_epochs=10,
            max_iter=20, random_state=42)
data.fit(model)
```

MLP projections and nonlinear fusion replace modality-specific encoders.
Pretraining contrasts the full representation with versions that mask each
non-core view. The first view is retained by default; choose `core_view` to
change it. With one view, feature-dropout augmentations replace token deletion.
These are explicit feature-input adaptations, not end-to-end utterance encoding.

After pretraining, K-means produces pseudo-labels and inherits centroids between
rounds. Per-cluster neighbor density selects a growing fraction of samples;
supervised contrast uses their pseudo-labels, while the remainder uses instance
contrast. Candidate neighborhoods follow the reference's maximum nearest-neighbor
distance selection criterion. Singleton clusters are retained and ties are
handled deterministically. The fraction increases from `initial_fraction` toward
one over `max_iter` rounds. `pretrain_epochs` counts initial full epochs.
`selected_counts_` records curriculum sizes. `transform` and `predict` accept
new samples. Training augmentations do **not** imply missing-view support.
[Paper](https://aclanthology.org/2024.acl-long.2/) · [Original code](https://github.com/thuiar/UMC).

All five methods above require complete or explicitly imputed inputs. The
offline comparison notebooks use deliberately small networks and budgets to
teach the API, not to reproduce published accuracy.
