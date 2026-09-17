# Aligned and missing views

The framework can represent missing views, but a model must explicitly
implement an objective that uses the observation mask. **Neither current model
natively consumes incomplete views.** SCM's training-time view-drop augmentation
does not establish support for genuinely unavailable input views.

## Align samples using IDs

```python
import numpy as np
from pymv.data import align_views

data = align_views(
    [np.array([[1., 2.], [3., 4.]]), np.array([[50.], [30.]])],
    view_ids=[["a", "b"], ["c", "b"]],
    view_names=["image", "text"],
    modalities=["image", "text"],
)
print(data.sample_ids)      # a, b, c: first-occurrence union order
print(data.observed_mask)   # [[True, False], [True, True], [False, True]]
```

Missing rows use zero placeholders **plus a separate mask**. An observed zero
vector stays observed. IDs must be unique, nonmissing and consistently typed.
For alignment, `graphs` uses each view's **local** ID order; graphs are expanded
into the union order. Optional labels require `label_ids` covering that union.
File imports use `id_keys=[...]` and, when needed, `label_id_key=...`.

Already aligned data can provide a boolean `(samples, views)` mask (or 0/1 in
file imports). Opting into `detect_nan_rows=True` marks only wholly NaN rows
missing. Partial NaNs in observed rows are rejected. ID alignment and explicit
row-mask detection are separate import modes.

## Simulate missing observations

```python
from pymv.data import load_data, simulate_missing_views

data = load_data("acm", 0.2, random_state=42)
# Equivalent:
complete = load_data("acm")
data = simulate_missing_views(complete, missing_rate=0.2, random_state=42)
```

Seeded constrained MCAR-style sampling removes
`floor(rate × currently_observed_pairs)` sample-view pairs, without examining
features or labels, and retains at least one view per sample. This constraint
means the process is **not independent Bernoulli MCAR**.

“20%” means observed sample-view pairs, not whole samples or exactly 20% per
view. The source stays unchanged. Requested/achieved rates and seed are stored
in `data.metadata["missing_simulation"]`. Graphs and labels are unchanged:
this simulates missing features, not deleted edges.

Impossible rates, values outside [0, 1), and positive removal for single-view
data are rejected. A view may become entirely unavailable, in which case
mean imputation cannot estimate its feature means.

## Choose a deliberate baseline

```python
from pymv.data import impute_missing
from pymv.models import LMGEC

# data.fit(LMGEC(n_clusters=3)) rejects incomplete inputs.
prepared = impute_missing(data, strategy="mean")  # or explicit "zero"
model = prepared.fit(LMGEC(n_clusters=3, embedding_dim=8, random_state=42))
```

Means use observed rows only, without labels; sparse inputs may be densified.
The output is marked complete and retains the original mask in imputation
metadata. Report this as **LMGEC/SCM after imputation**, not a new missing-aware
version of either algorithm.

Keep the dataset object when fitting. Legacy tuple unpacking loses the mask;
raw zero-filled matrices do not reveal missingness. New missing-aware models
declare `supports_missing_views = True` and accept
`fit(Xs, As=None, *, observed_mask=...)`. See [adding models](../contributing/models.md).
