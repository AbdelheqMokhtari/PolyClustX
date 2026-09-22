# Available models, background and references

The framework is method-independent. Its data representation and evaluation
tools should not be confused with the assumptions of an individual algorithm.

## Choose a model

```{pymv-catalog} models
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

In pymv, feature normalization and optional graph propagation precede the
linear optimization. With graphs, propagation uses
$S=D^{-1}(A+\beta I)$; without a graph, features are normalized without
propagation. Parameters include embedding dimension, temperature, per-view
self-loop weights, TF-IDF, centering and scaling.

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

The pymv implementation uses TensorFlow, a shared encoder/projector, and
K-means on its learned representation. Adjacencies are accepted for interface
compatibility but ignored. The paper discusses several variants; the available
class is the implemented feature-clustering route, not every paper variant.
Its `miss_rate` parameter controls augmentation of complete training inputs,
not the framework's missing-view simulation rate. Native missing-input fitting
is not currently implemented.

## Responsible comparisons

Fix dataset variants, sample alignment, preprocessing, graph fusion, seeds and
evaluation space when reporting comparisons. State whether you supplied K from
ground truth. Imputed runs should report the imputation baseline. Framework
tests check implementation behavior, not reproduction of every published table.

Use the capability table above when choosing a method and the
[API](../api/models.rst) for implemented parameters.
