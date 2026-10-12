"""Small offline workflows used verbatim in the user documentation.

Run: python examples/practical_workflows.py
Short training schedules illustrate the API, not research benchmark settings.
"""

from pathlib import Path

from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

from polyclustx.data import MultiViewDataset, impute_missing, simulate_missing_views
from polyclustx.experiment import (
    Experiment,
    GridSearchExperiment,
    OptunaSearchExperiment,
    RandomSearchExperiment,
)
from polyclustx.metrics import nmi_score, silhouette_score
from polyclustx.models import COMPLETER, LMGEC, MultiViewKMeans, MultiViewSpectralClustering


def feature_views(loader=load_iris):
    """Split measured columns into two aligned views; labels are for evaluation."""
    source = loader()
    X = StandardScaler().fit_transform(source.data)
    midpoint = X.shape[1] // 2
    return MultiViewDataset(
        [X[:, :midpoint], X[:, midpoint:]], labels=source.target,
        name="Iris" if loader is load_iris else "Wine",
    )


def cluster_features():
    data = feature_views()
    model = data.fit(MultiViewKMeans(n_clusters=3, n_init=3, random_state=42))
    print("Predicted labels:", model.labels_[:10])
    print("Embedding shape:", model.embedding_.shape)
    print("NMI:", nmi_score(data.labels, model.labels_))

    # One view uses ordinary K-means through the same estimator API.
    single = MultiViewDataset([data.Xs[0]])
    single_model = single.fit(MultiViewKMeans(n_clusters=3, random_state=42))
    print("Single-view labels:", single_model.labels_.shape)
    return data, model


def compare_methods(output_dir="experiments/tutorial"):
    results = Experiment(
        datasets={"Iris": feature_views(), "Wine": feature_views(load_wine)},
        models=[
            MultiViewKMeans(n_clusters=3, n_init=3, random_state=42),
            MultiViewSpectralClustering(n_clusters=3, max_iter=2, n_init=3, random_state=42),
            LMGEC(n_clusters=3, embedding_dim=1, max_iter=5, random_state=42, device="cpu"),
        ],
        metrics={"NMI": nmi_score}, runs=3,
        save_path=output_dir, experiment_name="Iris and Wine comparison",
    ).run()
    print(results.results_tables())
    results.export(Path(output_dir) / "comparison.zip")
    return results


def choose_k_without_labels(output_dir="experiments/tutorial"):
    data = MultiViewDataset(feature_views().Xs, name="Unlabeled measurements")
    return Experiment(
        data,
        [MultiViewKMeans(n_clusters=k, n_init=3, random_state=42) for k in [2, 3, 4]],
        metrics={"Silhouette": silhouette_score}, benchmark=False, runs=2,
        save_path=output_dir,
    ).run()


def cluster_graphs():
    from polyclustx.display.synthetic import SyntheticGraphSpec, SyntheticSpec, generate_synthetic

    data = generate_synthetic(SyntheticSpec(
        n_samples=90, n_classes=3, dimensions=[8, 6, 5],
        distributions=["Gaussian", "Nonlinear", "Gaussian"], random_state=42,
        graph_specs=[
            SyntheticGraphSpec(kind="knn", neighbors=5),
            SyntheticGraphSpec(kind="rbf_knn", neighbors=10, bandwidth=2.0),
        ],
    ))
    model = data.fit(
        LMGEC(n_clusters=3, embedding_dim=3, max_iter=5, random_state=42, device="cpu"),
        fusion="sum", weights={0: [0.7, 0.3], 1: [0.7, 0.3], 2: [0.7, 0.3]},
    )
    print("Three views, two relations per view:", model.labels_.shape)
    return data, model


def cluster_missing_views():
    data = feature_views()
    incomplete = simulate_missing_views(data, missing_rate=0.2, random_state=42)
    print("Observed samples per view:", incomplete.observed_mask.sum(axis=0))

    # Native incomplete-view learning: pass the mask through data.fit().
    model = incomplete.fit(COMPLETER(
        n_clusters=3, embedding_dim=4, hidden_dims=(16,), prediction_dims=(8, 4),
        prediction_start=1, max_iter=3, batch_size=32, n_init=2,
        random_state=42, device="cpu",
    ))

    # Alternative baseline for a complete-input method: impute explicitly.
    prepared = impute_missing(incomplete, strategy="mean")
    baseline = prepared.fit(MultiViewKMeans(n_clusters=3, n_init=3, random_state=42))
    print("COMPLETER NMI:", nmi_score(data.labels, model.labels_))
    print("Mean-imputation baseline NMI:", nmi_score(data.labels, baseline.labels_))
    return incomplete, model, baseline


def tune_parameters(output_dir="experiments/tutorial"):
    data = feature_views(load_wine)
    candidates = {
        "n_clusters": [3], "embedding_dim": [2, 4], "temperature": [0.5, 1.0],
        "max_iter": [5], "device": ["cpu"],
    }
    options = dict(dataset=data, metrics={"NMI": nmi_score}, best_metric="NMI",
                   runs=2, seed=42, save_path=output_dir)
    grid = GridSearchExperiment(LMGEC, candidates, **options).run()
    random = RandomSearchExperiment(LMGEC, candidates, n_iter=3, **options).run()
    continuous = {**candidates, "temperature": {"low": 0.1, "high": 3.0, "log": True}}
    optuna = OptunaSearchExperiment(LMGEC, continuous, n_iter=3, **options).run()
    print("Grid winner:", grid.best_params, grid.best_score)
    print("Random winner:", random.best_params, random.best_score)
    print("Optuna winner:", optuna.best_params, optuna.best_score)

    # Reuse the winning parameters for an ordinary fit.
    best_model = data.fit(LMGEC(**grid.best_params, random_state=42))
    return grid, random, optuna, best_model


def save_figures(data, model, results, output_dir="experiments/tutorial"):
    import matplotlib.pyplot as plt

    from polyclustx.visualization import plot_clusters, plot_loss_comparison

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Both panels use the same PCA coordinates: only the labels change.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_clusters(model.embedding_, data.labels, method="pca", ax=axes[0],
                  title="Reference classes", show=False)
    plot_clusters(model.embedding_, model.labels_, method="pca", ax=axes[1],
                  title="Predicted clusters", show=False)
    fig.tight_layout()
    fig.savefig(directory / "clusters.png", dpi=160)
    plt.close(fig)

    # Run 1 represents each configuration. Curves compare shape, not loss scales.
    ax = plot_loss_comparison(results, run=1, show=False)
    ax.figure.savefig(directory / "losses.svg", bbox_inches="tight")
    plt.close(ax.figure)


def main(output_dir="experiments/tutorial"):
    cluster_features()
    results = compare_methods(output_dir)
    choose_k_without_labels(output_dir)
    graph_data, graph_model = cluster_graphs()
    cluster_missing_views()
    tune_parameters(output_dir)
    save_figures(graph_data, graph_model, results, output_dir)
    print("Reports and figures:", Path(output_dir).resolve())


if __name__ == "__main__":
    main()
