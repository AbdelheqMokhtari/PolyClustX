import numpy as np
import pytest

from polyclustx.core import BaseModel
from polyclustx.data import MultiViewDataset, simulate_missing_views
from polyclustx.experiment import Experiment, RandomSearchExperiment
from polyclustx.experiment.result import RandomSearchResult
from polyclustx.metrics import davies_bouldin_score, nmi_score, silhouette_score


class DummyModel(BaseModel):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, Xs, As=None):
        self.labels_ = np.arange(len(Xs[0])) % self.n_clusters
        self.embedding_ = Xs[0]
        return self


def data():
    return MultiViewDataset([np.array([[0], [10], [1], [11], [2], [12]], dtype=float)])


def test_unlabeled_experiment():
    result = Experiment(data(), DummyModel(), {"silhouette": silhouette_score}, verbose=0, benchmark=False).run()
    assert result.all_results["dataset"][0].metric_avg["silhouette"] > .7


def test_external_metrics_rejected_without_labels_or_benchmark():
    with pytest.raises(ValueError, match="External metrics require"):
        Experiment(data(), DummyModel(), [nmi_score], verbose=0).run()
    labeled = data()
    labeled.labels = np.arange(6) % 2
    with pytest.raises(ValueError, match="benchmark=True"):
        Experiment(labeled, DummyModel(), [nmi_score], benchmark=False, verbose=0).run()


def test_unlabeled_random_search_and_metric_direction():
    result = RandomSearchExperiment(DummyModel, {"n_clusters": [2, 3]}, data(),
                                    {"db": davies_bouldin_score}, "db", n_iter=3, seed=0,
                                    benchmark=False, verbose=0).run()
    assert result.best_score == min(t["metrics"]["db"] for t in result.trials)
    result = RandomSearchResult("data", "model", "db", higher_is_better=False)
    for value in [np.nan, 3.0, 1.0, 2.0]:
        result.add_trial({}, {"db": value}, 0)
    assert result.best_score == 1.0


def test_experiment_preserves_missing_mask():
    incomplete = simulate_missing_views(MultiViewDataset([np.ones((6, 2))] * 2), .2)
    with pytest.raises(ValueError, match="does not support missing views"):
        Experiment(incomplete, DummyModel(), [silhouette_score], verbose=0).run()


def test_saved_recipe_preserves_variant_and_evaluation(tmp_path):
    ds = data()
    ds.name = "pascal"
    ds.metadata["variant"] = "clip"
    experiment = Experiment(ds, DummyModel(), [davies_bouldin_score], benchmark=False,
                            save_markdown=True, save_path=str(tmp_path), verbose=0)
    experiment.run()
    config = experiment._build_reproducible_config()
    assert config["variant"] == "clip" and config["dataset"] == "pascal" and config["no_benchmark"]
