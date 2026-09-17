"""Run small, deterministic examples without downloading datasets."""

import numpy as np

from pymv.data import MultiViewDataset, impute_missing, simulate_missing_views
from pymv.experiment import Experiment
from pymv.metrics import silhouette_score
from pymv.models import LMGEC, SCM


def main():
    rng = np.random.default_rng(42)
    centers = np.repeat(np.arange(3), 10)[:, None]
    views = [rng.normal(size=(30, width)) + centers * 3 for width in [4, 6, 5]]
    data = MultiViewDataset(views, name="Synthetic measurements",
                            view_names=["shape", "texture", "color"], modalities=["image"] * 3)

    # Same interface for one, two or many views; no labels required.
    for count in [1, 2, 3]:
        subset = MultiViewDataset(data.Xs[:count])
        model = subset.fit(LMGEC(n_clusters=3, embedding_dim=3, max_iter=2, random_state=42))
        assert model.labels_.shape == (30,)

    # Explicit missing-view baseline, not a missing-aware variant of LMGEC.
    incomplete = simulate_missing_views(data, .2, random_state=42)
    prepared = impute_missing(incomplete)
    prepared.fit(LMGEC(n_clusters=3, embedding_dim=3, max_iter=2, random_state=42))

    # Multiple relations assigned to the first feature view.
    relations = MultiViewDataset(data.Xs, [[np.eye(30), np.ones((30, 30))], None, None])
    relations.fit(LMGEC(n_clusters=3, embedding_dim=3, max_iter=2, random_state=42), weights={0: [.2, .8]})

    # SCM does not use graphs. Small sizes here are for a smoke test, not research defaults.
    scm = data.fit(SCM(n_clusters=3, embedding_dim=4, projection_dim=3, hidden_dims=(8,),
                      batch_size=15, max_iter=1, random_state=42))
    assert scm.embedding_.shape[0] == data.n_samples

    Experiment(data, [LMGEC(n_clusters=k, embedding_dim=3, max_iter=2, random_state=42) for k in [2, 3]],
               {"Silhouette": silhouette_score}, benchmark=False).run()


if __name__ == "__main__":
    main()
