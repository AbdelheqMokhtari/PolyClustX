"""Small offline API example; settings illustrate workflows, not paper benchmarks.

Run from the repository: python examples/new_models.py
"""
from __future__ import annotations

import numpy as np

from pymv.data import MultiViewDataset, simulate_missing_views
from pymv.experiment import Experiment
from pymv.metrics import nmi_score
from pymv.models import COMPLETER, DCP, IDEC, MFLVC, MultiViewKMeans, MultiViewSpectralClustering


def run(output_dir="experiments/new-models"):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 16)
    views = [labels[:, None] * 3 + rng.normal(size=(48, dim)) for dim in (8, 10, 12)]
    data = MultiViewDataset(views, labels=labels, name="three-view-example")
    small = dict(n_clusters=3, embedding_dim=4, hidden_dims=(16,), max_iter=3,
                 batch_size=16, n_init=2, device="cpu")
    models = [
        MultiViewKMeans(3, n_init=2),
        MultiViewSpectralClustering(3, max_iter=2, n_init=2),
        IDEC(**small, pretrain_epochs=2, update_interval=1),
        COMPLETER(**small, prediction_dims=(8, 4), prediction_start=1),
        DCP(**small, prediction_dims=(8, 4), prediction_start=1),
        MFLVC(**small, projection_dim=3, pretrain_epochs=2, finetune_epochs=2),
    ]
    complete = Experiment(data, models, [nmi_score], runs=2, save_path=output_dir, verbose=0).run()
    incomplete = simulate_missing_views(data, .2, random_state=42)
    missing = Experiment(incomplete, models[3:5], [nmi_score], runs=2,
                         save_path=output_dir, verbose=0).run()
    return complete, missing


if __name__ == "__main__":
    for result in run():
        print(result.json_path)
