# Available models, background and references

The framework is method-independent. Its data representation and evaluation
tools should not be confused with the assumptions of an individual algorithm.

## Choose a model

```{polyclustx-catalog} models
```

Check **Missing views** before fitting incomplete inputs; a model without
native support needs explicit imputation. All listed models accept single or
multiple feature views. The table is generated from the model registry.
See the [model API](../api/models.rst) for constructor parameters.

## LMGEC

LMGEC couples linear representation learning and clustering after per-view
neighborhood smoothing. It learns view weights and a consensus representation.
The original work is:

Chakib Fettal, Lazhar Labiod, and Mohamed Nadif. **Simultaneous Linear Multi-view
Attributed Graph Representation Learning and Clustering**. WSDM 2023, 303–311.
[DOI: 10.1145/3539597.3570367](https://doi.org/10.1145/3539597.3570367).
[Author-hosted paper](https://chakib401.github.io/assets/pdf/LMGEC.pdf).
[Original implementation](https://github.com/chakib401/LMGEC).

In polyclustx, feature normalization and optional graph propagation precede the
linear optimization. With graphs, propagation uses
$S=D^{-1}(A+\beta I)$; without a graph, features are normalized without
propagation. Parameters include embedding dimension, temperature, per-view
self-loop weights, TF-IDF, centering and scaling.

The optimization runs in PyTorch using float64. It preserves the previous
TensorFlow implementation's preprocessing, scikit-learn initialization, view
weights, alternating SVD/assignment/centroid updates, and pre-update loss
definition. An empty cluster receives a zero centroid. Public results remain
NumPy arrays; no automatic differentiation is needed by this model.

When the embedding dimension exceeds the number of clusters (or the update
matrix is otherwise rank-deficient), the SVD projection is not unique.
TensorFlow and PyTorch can choose different valid null-space bases. Both solve
the same orthogonal projection objective, but embeddings and subsequent
optimization trajectories can differ; cross-backend bitwise equivalence is
not a reproducibility guarantee. Regression tests check orthogonality and the
optimal projection objective for this case, rather than requiring a specific
null-space basis.

Early relation fusion is framework preprocessing, not a claim that the original
paper learns relation weights. Incomplete views require explicit imputation;
the implemented loss does not use an observation mask.

## SCM

SCM concatenates normalized views and uses a shared network with contrastive
learning and noise/view-drop augmentations. The original work is:

Caixuan Luo, Jie Xu, Yazhou Ren, Junbo Ma, and Xiaofeng Zhu. **Simple Contrastive
Multi-View Clustering with Data-Level Fusion**. IJCAI 2024, 4697–4705.
[Original proceedings and paper](https://www.ijcai.org/proceedings/2024/519).
[Authors' reference implementation](https://github.com/SubmissionsIn/SCM).

The polyclustx implementation uses PyTorch, a shared encoder/projector, and
K-means on its learned representation. Adjacencies are accepted for interface
compatibility but ignored. The paper discusses several variants; the available
class is the implemented feature-clustering route, not every paper variant.
Its `miss_rate` parameter controls augmentation of complete training inputs,
not the framework's missing-view simulation rate. Native missing-input fitting
is not currently implemented.

The default encoder has hidden widths 500, 500, and 2000, followed by
`embedding_dim`; a linear projector produces L2-normalized `projection_dim`
features. Training uses the reference's symmetric feature contrastive loss,
uniform nonempty/non-full view masks, independent per-view Gaussian-noise
selection, and Adam with coupled L2 `weight_decay`. Setting
`reconstruction_weight` above zero adds clean and augmented reconstruction
losses. The fitted `model_` is now a PyTorch module, not a Keras model; use its
`state_dict()` for network weights. It is returned on CPU in evaluation mode.

For reproducible comparisons, note the framework's deliberate conventions:

- `max_iter` counts **full epochs**, whereas the reference training script
  processes one minibatch per iteration.
- Partial minibatches with at least two samples are trained; a trailing
  singleton is skipped because it has no contrastive negatives.
- `preprocess=True` performs per-feature min-max scaling. The reference uses
  dataset-specific preprocessing, so match it explicitly when reproducing papers.
- The base contrastive/K-means route is implemented, not the paper's ETC route.

Switching from TensorFlow changes SCM's initialization and numerical training
trajectory. Nonzero weight decay and augmentation sampling also now follow the
PyTorch reference; old TensorFlow SCM scores are not expected to match exactly.

## Device selection and reproducibility

Both models accept `device="auto"` (default), `"cpu"`, `"cuda"`, or
`"cuda:0"` for a specific accelerator. Automatic selection uses CUDA when
available and restarts on CPU after CUDA out-of-memory. Explicit CUDA requests
raise an error if unavailable or out of memory, rather than silently changing
the requested device. The fitted `device_` and exported diagnostics record
where training ran. Only CPU and CUDA are supported.

Set `random_state` and fix the software environment for repeated experiments.
Exact reproducibility across devices and PyTorch versions is not guaranteed.
SCM initialization preserves the caller's PyTorch CPU random-generator state;
augmentations use a model-local NumPy generator.

Migration checks compare LMGEC's loop against independent NumPy equations
and SCM's contrastive loss and gradients against the explicit objective. During
migration, six seeded CPU LMGEC fits (one/two/three views, preprocessing on/off)
also matched the saved TensorFlow implementation's labels exactly, with
embeddings, centroids and losses agreeing within `rtol=1e-9, atol=1e-10`.
Additional three-view checks cover weighted graphs, centering/scaling and
per-view TF-IDF with the same tolerance.
SCM's default network parameters and outputs matched the supplied reference
exactly; contrastive loss and gradients agreed within `1e-12`. These checks do
not establish identical results for every dataset or hardware configuration.

## Co-EM K-means and spectral co-training

`MultiViewKMeans` follows the two-view partition exchange described by Steffen
Bickel and Tobias Scheffer, **Multi-View Clustering**, ICDM 2004.
[Paper DOI](https://doi.org/10.1109/ICDM.2004.10095).
`MultiViewSpectralClustering` follows Abhishek Kumar and Hal Daumé III,
**A Co-training Approach for Multi-view Spectral Clustering**, ICML 2011.
[Paper](https://icml.cc/2011/papers/272_icmlpaper.pdf).
The [mvlearn implementations](https://mvlearn.github.io/references/cluster.html)
are behavioral references, not runtime dependencies. Single-view reductions,
the cyclic >2-view K-means extension, numerical safeguards and convergence
conventions are explained in the [model tutorial](../user_guide/clustering_models.md).

## IDEC

Xifeng Guo, Long Gao, Xinwang Liu and Jianping Yin. **Improved Deep Embedded
Clustering with Local Structure Preservation**, IJCAI 2017.
[Paper](https://www.ijcai.org/proceedings/2017/243) ·
[Reference](https://github.com/XifengGuo/IDEC).
The PyTorch implementation combines reconstruction with Student-t clustering
and a sharpened target distribution. Multiple input views use early fusion;
this is an application of IDEC, not a claim of a multi-view IDEC objective.

## COMPLETER and DCP

Yijie Lin et al. **COMPLETER: Incomplete Multi-view Clustering via Contrastive
Prediction**, CVPR 2021.
[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Lin_COMPLETER_Incomplete_Multi-View_Clustering_via_Contrastive_Prediction_CVPR_2021_paper.html) ·
[Reference](https://github.com/XLearning-SCU/2021-CVPR-Completer).

Yijie Lin et al. **Dual Contrastive Prediction for Incomplete Multi-view
Representation Learning**, IEEE TPAMI.
[DOI](https://doi.org/10.1109/TPAMI.2022.3197238) ·
[Reference](https://github.com/XLearning-SCU/2022-TPAMI-DCP).
Both learn cross-view latent prediction on fully observed training samples,
then complete missing latent views before clustering. DCP here implements
the unsupervised two-view / complete-graph routes, not label-supervised
classification. Details and overlap requirements appear in the
[model tutorial](../user_guide/clustering_models.md).

## MFLVC

Jie Xu, Huayi Tang, Yazhou Ren, Liang Peng, Xiaofeng Zhu and Lifang He.
**Multi-Level Feature Learning for Contrastive Multi-View Clustering**, CVPR 2022.
[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.pdf) ·
[Reference](https://github.com/SubmissionsIn/MFLVC).
View-specific autoencoders feed shared feature and cluster heads. Training
uses reconstruction, feature/cluster contrast and aligned pseudo-label
fine-tuning. The final prediction averages view cluster probabilities.
Implementation conventions and the one-view reduction are documented in the
[model tutorial](../user_guide/clustering_models.md).

## Responsible comparisons

Fix dataset variants, sample alignment, preprocessing, graph fusion, seeds and
evaluation space when reporting comparisons. State whether you supplied K from
ground truth. Imputed runs should report the imputation baseline. Framework
tests check implementation behavior, not reproduction of every published table.

Use the capability table above when choosing a method and the
[API](../api/models.rst) for implemented parameters.
