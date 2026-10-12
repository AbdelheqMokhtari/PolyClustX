# Add your own model

Models follow one small contract: implement `fit(Xs, As=None)`, return `self`,
and expose `labels_` and `embedding_` in the same sample order. Do not read
evaluation labels, infer K from them, or mutate the caller's input arrays.

## A complete feature-only example

```python
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from polyclustx.core import BaseModel
from polyclustx.models import register_model

@register_model(
    "concatenated_kmeans",
    description="K-means baseline on concatenated feature views.",
    uses_graphs=False,
)
class ConcatenatedKMeans(BaseModel):
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, Xs, As=None):
        self.embedding_ = np.concatenate(
            [x.toarray() if sparse.issparse(x) else x for x in Xs], axis=1
        )
        self.labels_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        ).fit_predict(self.embedding_)
        return self
```

Import the module once before calling `get_model("concatenated_kmeans")`.
Registration is process-local. For an in-package contribution, export the class
from `polyclustx.models`; this makes it discoverable by the CLI, workbench and catalog.
Registry defaults are inferred from explicit constructor arguments, excluding
`n_clusters`; use `defaults={...}` for curated UI defaults. Parameters should be
JSON-serializable for reproducible recipes.

For detailed experiment reports, optionally expose `loss_history_` (one value
per training iteration), `view_weights_` or `alphas_`, `inertia_`,
`per_view_results_`, and `per_view_loss_history_`. Missing attributes are exported
as `null`. Label per-view diagnostics with their stage and meaning; initialization
scores must not be presented as final clustering performance. Reports normalize
one-dimensional loss histories automatically and never fabricate unavailable
per-view results.

Add `paper="https://doi.org/..."`, `implementation="https://github.com/..."`,
a concise description and `uses_graphs=True`
when applicable. A registry entry does not automatically validate a method's
scientific claims: document differences from the reference implementation.

## Where to put your contribution

1. Add `src/polyclustx/models/your_method/` with the implementation and `__init__.py`.
2. Import/export its registered class in `src/polyclustx/models/__init__.py`.
3. Add deterministic, small offline tests in `tests/test_your_method.py`.
4. Add a runnable example under `examples/` and link it from
   `docs/user_guide/examples.rst`.
5. Explain the objective, assumptions and citation in
   `docs/background/algorithms.md`. Its capability table is generated automatically.
6. Add the class's autodoc entry to `docs/api/models.rst`.
7. For specialized parameter ranges, add controls in
   `src/polyclustx/display/model_controls.py` and interface tests. Simple constructor
   parameters otherwise use the inferred controls.

If a method adjusts parameters for a dataset, it may implement the optional
`resolve_fit_parameters(Xs, params)` static method, returning a new parameter
dictionary. The interface records requested and effective values and informs
the user. Do not apply another algorithm's dimension limits to your model.
Test the adjustment and its report; LMGEC provides an example.

## Missing-aware methods

Declare `supports_missing_views = True` only when the implementation genuinely
uses an observation mask and has tests for it. Accept
`fit(Xs, As=None, *, observed_mask=None)`. The dataset calls this keyword only
for incomplete data. Use the mask in objectives/normalization: zero placeholders
must never be treated as observed data. Graph observations and feature
observations are separate; feature missingness does not remove graph edges.

## Validation checklist

- One, two and multiple views; heterogeneous feature dimensions.
- Correct labels/embedding shapes and sample order.
- Seeded smoke tests on synthetic data; no downloads required.
- Clear errors for unsupported graphs, missing views or parameter choices.
- Sparse input support or an explicit limitation.
- Original citation, implementation provenance, licensing and numerical deviations.

The [pull-request guide](pull_requests.md) explains how to propose inclusion.
