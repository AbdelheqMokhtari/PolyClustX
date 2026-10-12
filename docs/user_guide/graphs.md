# Multiple graph relations per view

The outer graph list follows feature-view order. Each entry is `None`, one
adjacency, or several relations assigned to that view.

An entry can also be a NetworkX-compatible graph object. Node IDs must equal
the dataset's `sample_ids` (by default, integers `0` through `n_samples - 1`),
including isolated samples. Conversion orders the adjacency by sample IDs, not
the graph's insertion order. Directed edges retain their direction, parallel
edge weights are summed, and absent weights default to one. All weights must
be finite and nonnegative.

```python
import networkx as nx  # included with the display extra; otherwise install networkx
import numpy as np
from polyclustx.data import MultiViewDataset

graph = nx.Graph()
graph.add_nodes_from(["sample-b", "sample-a", "sample-c"])
graph.add_edge("sample-a", "sample-b", weight=0.8)
data = MultiViewDataset(
    [np.ones((3, 2))], As=[graph],
    sample_ids=["sample-a", "sample-b", "sample-c"],
)
```

The interface additionally accepts GraphML/GEXF files and dense or sparse
adjacency files under **Dataset → Add graphs to a view**.

```python
import numpy as np
from polyclustx.data import MultiViewDataset
from polyclustx.models import LMGEC

rng = np.random.default_rng(9)
image, text = rng.normal(size=(30, 6)), rng.normal(size=(30, 8))
spatial = np.eye(30)
similarity = np.ones((30, 30)) - np.eye(30)
data = MultiViewDataset([image, text], As=[[spatial, similarity], None],
                       modalities=["image", "text"])
model = data.fit(LMGEC(n_clusters=3, embedding_dim=4, random_state=42),
                 fusion="sum", weights={0: [0.3, 0.7]})
```

Weighted sum computes $A_{\mathrm{fused}} = \sum_r w_r A_r$.
Weights must be finite and nonnegative, one per relation, with at least one
positive value. Defaults are all one: a **sum**, not an automatic average.

`fusion="hadamard"` computes the elementwise product
$A_{\mathrm{fused}} = A_1 \odot A_2 \odot \cdots$.
An edge must appear in every relation to survive. Hadamard fusion rejects weights.
Relations must have identical shapes and sample order.

```python
from polyclustx.preprocessing import fuse_graphs, get_propagated_features

a = fuse_graphs([spatial, similarity], weights=[0.3, 0.7])
h = get_propagated_features(image, adj=a, beta=1.0)
```

Fusion precedes self-loops and row normalization:
$S = D^{-1}(A_{\mathrm{fused}} + \beta I)$, then $H = SX$.
Sparse graph inputs stay sparse through fusion; feature propagation may yield
dense features.

Direct LMGEC calls sum nested relation lists by default. Use `data.fit` for
custom weighting. Shared graphs must be repeated explicitly, such as `As=[A, A]`;
the last graph is no longer silently assigned to unmentioned views.
SCM and the feature-only estimators ignore supplied graphs. Spectral co-training
constructs its own feature affinities. LMGEC uses supplied graphs only with
preprocessing enabled. Check the [model catalog](../background/algorithms.md).

## Generate synthetic graphs in Python

The synthetic generator is usable without installing Streamlit. One recipe
creates aligned features and sparse graph relations suitable for the same
preprocessing, fusion and model APIs as loaded datasets:

```python
from polyclustx.display.synthetic import SyntheticGraphSpec, SyntheticSpec, generate_synthetic
from polyclustx.models import LMGEC

data = generate_synthetic(SyntheticSpec(
    n_samples=120,
    n_classes=3,
    dimensions=[10, 8],
    distributions=["Gaussian", "Nonlinear"],
    graph_specs=[
        SyntheticGraphSpec(kind="knn", neighbors=8),
        SyntheticGraphSpec(kind="rbf_knn", neighbors=15, bandwidth=2.0),
    ],
    random_state=7,
))
model = data.fit(LMGEC(n_clusters=3, embedding_dim=4, random_state=7),
                 fusion="sum", weights={0: [0.7, 0.3], 1: [0.7, 0.3]})
```

Each view receives two relations in this example. `graph_specs=[]` produces
feature-only data; one specification produces one relation per target view.
Use `kind="mutual_knn"` for edges present in both neighbor directions. Binary
and weighted kNN instead keep either direction. All generated graphs are CSR,
symmetric and have zero diagonal. RBF weights use
$\exp(-d^2/(2\sigma^2))$, where `bandwidth` is $\sigma$ in feature-space units.

`graph_views=[0]` attaches graphs only to the first view. Add
`shared_graph_view=0` to build the relations from the first view and copy them
to every target view; without it each target uses its own features. Indices are
zero-based. Single-view datasets use one dimension and distribution entry.

Neighbor counts must be between 1 and `n_samples - 1`; bandwidths must be
finite and positive. The generator limits total directed neighbor entries to
two million, counting each target view. Neighbor searches use bounded distance
scratch space, but large feature arrays and subsequent models still consume RAM.
Graphs never use labels, and changing graph settings leaves features unchanged.
The full recipe is saved in `data.metadata["synthetic"]`.

Existing recipes using `graph_relations` and `graph_neighbors` remain supported:
they create binary kNN relations, increasing K by two per relation (clipped to
`n_samples - 1`). Explicit `graph_specs` overrides these legacy fields.
