# Aligned and missing views

Use missing views when some samples have no observation in one or more views.
Not every model accepts incomplete inputs. Check the **Missing views** column
in the [model guide](../background/algorithms.md), or choose explicit imputation
before fitting a model that requires complete data.

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

This creates an incomplete version of a complete dataset so you can test
missing-view settings. `0.2` hides about 20% of available sample/view observations
at random, while keeping at least one view for every sample. The original
dataset, labels and graphs stay unchanged. Use the same `random_state` to
reproduce the missing observations.

Use at least two views and a rate below 1. Impossible removal requests are
rejected. The availability mask records exactly what is missing.

## Choose a deliberate baseline

For native latent completion, choose `COMPLETER` or `DCP`. They train on
fully observed rows, so at least two such samples must remain after missing-view
simulation. Set `prediction_start < max_iter` and `prediction_weight > 0`.
The [model tutorial](clustering_models.md) includes a complete example and
explains how the learned predictors use observed views at inference time.

## Explicit imputation baseline

```python
from pymv.data import impute_missing
from pymv.models import LMGEC

prepared = impute_missing(data, strategy="mean")  # or explicit "zero"
model = prepared.fit(LMGEC(n_clusters=3, embedding_dim=8, random_state=42))
```

Means use observed rows only, without labels; sparse inputs may be densified.
The output is marked complete and retains the original mask in imputation
metadata. If a view has no observed samples, choose zero imputation instead.
Report the imputation strategy alongside the model in comparisons.

Keep the dataset object when fitting. Legacy tuple unpacking loses the mask;
raw zero-filled matrices do not reveal missingness. New missing-aware models
declare `supports_missing_views = True` and accept
`fit(Xs, As=None, *, observed_mask=...)`. See [adding models](../contributing/models.md).
