# Add a model

A model inherits `BaseModel`, implements `fit(Xs, As=None)`, and stores
`labels_` and `embedding_`. The base class provides `fit_predict`.

## 1. Implement a small baseline

This example concatenates feature views and runs K-means. It accepts one or
more complete views and does not use graphs.

```python
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from polyclustx.core import BaseModel
from polyclustx.models import register_model

@register_model("concatenated_kmeans", uses_graphs=False,
                description="K-means on concatenated feature views.")
class ConcatenatedKMeans(BaseModel):
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, Xs, As=None):
        self.embedding_ = np.concatenate(
            [x.toarray() if sparse.issparse(x) else x for x in Xs], axis=1
        )
        self.labels_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        ).fit_predict(self.embedding_)
        return self
```

Keep rows in their original sample order. Do not modify the input arrays or use
evaluation labels during training. If your method produces diagnostics, expose
them as `loss_history_`, `view_weights_` or `inertia_`; reports include them automatically.

## 2. Fit it and use it in an experiment

Run the class definition above, then:

```python
from sklearn.datasets import load_iris
from polyclustx.data import MultiViewDataset
from polyclustx.experiment import Experiment
from polyclustx.metrics import nmi_score

X, y = load_iris(return_X_y=True)
data = MultiViewDataset([X[:, :2], X[:, 2:]], labels=y, name="Iris")
model = data.fit(ConcatenatedKMeans(n_clusters=3))
print(model.labels_.shape)  # (150,)

results = Experiment(
    data, [ConcatenatedKMeans(n_clusters=3)],
    metrics={"NMI": nmi_score}, runs=3, save_path="experiments/my-model",
).run()
print(results.results_tables())
```

This works in your own script without changing the library. Registration lasts
for the current Python process: import your model module before using its name.

## 3. Include it in PolyClustX

1. Put the class in `src/polyclustx/models/concatenated_kmeans/concatenated_kmeans.py`.
2. In that folder's `__init__.py`, add:

   ```python
   from .concatenated_kmeans import ConcatenatedKMeans
   ```

3. Import the class in `src/polyclustx/models/__init__.py` and add it to `__all__`.
   Its registry entry makes it available to the interface and the namespaced
   command `polyclustx concatenated-kmeans`.
4. Add the original paper URL with `paper=` in the decorator, describe the
   method in `docs/background/algorithms.md`, and add its class to `docs/api/models.rst`.
5. Add a small test, then open a pull request targeting `develop`.

For example, after the class is part of the package:

```python
from sklearn.datasets import load_iris
from polyclustx.models import ConcatenatedKMeans

def test_concatenated_kmeans():
    X, _ = load_iris(return_X_y=True)
    model = ConcatenatedKMeans(n_clusters=3)
    assert model.fit([X[:, :2], X[:, 2:]]) is model
    assert model.labels_.shape == (150,)
    assert model.embedding_.shape == (150, 4)
```

Only declare `supports_missing_views = True` if your implementation actually
uses `fit(..., observed_mask=None)` to handle unavailable views. Add a missing-view
test before enabling that capability. For a graph method, set `uses_graphs=True`
and use the supplied `As` matrices.
