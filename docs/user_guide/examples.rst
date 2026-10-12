Examples
========

Run these blocks in order in Python or a notebook. They use Wine measurements
and generated data, so no downloads are needed. Each block performs its task
directly: there are no helper functions to define.

.. contents:: On this page
   :local:
   :depth: 1

1. Prepare two feature views
----------------------------

Split Wine's 13 measured features into two aligned views. Rows must represent
the same samples in the same order. Reference labels are optional and are used
only for evaluation.

.. code-block:: python

   from sklearn.datasets import load_wine
   from sklearn.preprocessing import StandardScaler
   from polyclustx.data import MultiViewDataset

   X, y = load_wine(return_X_y=True)
   X = StandardScaler().fit_transform(X)
   data = MultiViewDataset([X[:, :6], X[:, 6:]], labels=y, name="Wine")
   print(data.n_samples, data.feature_dims)  # 178, [6, 7]

To use a catalog dataset instead: ``data = load_data("acm")`` after importing
``load_data`` from ``polyclustx.data``. See :doc:`data` for CSV and MATLAB inputs.

2. Fit LMGEC and evaluate
-------------------------

.. code-block:: python

   from polyclustx.models import LMGEC
   from polyclustx.metrics import clustering_accuracy, nmi_score

   model = LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42)
   data.fit(model)
   print("Cluster IDs:", model.labels_[:10])
   print("Embedding:", model.embedding_.shape)  # (178, 3)
   print("ACC:", clustering_accuracy(data.labels, model.labels_))
   print("NMI:", nmi_score(data.labels, model.labels_))

For ordinary complete views, the equivalent direct call is
``labels = model.fit_predict(data.Xs, data.As)``. Prefer ``data.fit(model)``
when using missing-view masks or multiple graph relations.

3. Use one view or several views
--------------------------------

.. code-block:: python

   from polyclustx.models import MultiViewKMeans

   single = MultiViewDataset([X])
   single.fit(MultiViewKMeans(n_clusters=3, random_state=42))

   three_views = MultiViewDataset([X[:, :4], X[:, 4:8], X[:, 8:]])
   three_model = three_views.fit(
       LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42)
   )
   print(three_model.labels_.shape)  # (178,)

MultiViewKMeans uses ordinary K-means for a single view. Check the
:doc:`model catalog </background/algorithms>` for each method's input support.

4. Preprocess views and add graphs
----------------------------------

Construct two kNN relations per view, then fuse them with a weighted sum.
Graph construction uses only the feature values.

.. code-block:: python

   from sklearn.neighbors import kneighbors_graph

   relations = []
   for view in data.Xs:
       near = kneighbors_graph(view, n_neighbors=5, include_self=False)
       wide = kneighbors_graph(view, n_neighbors=10, include_self=False)
       relations.append([near.maximum(near.T), wide.maximum(wide.T)])

   graph_data = MultiViewDataset(data.Xs, As=relations, labels=y, name="Wine graphs")
   graph_model = graph_data.fit(
       LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42),
       fusion="sum", weights={0: [0.7, 0.3], 1: [0.7, 0.3]},
   )

For explicit propagation, preprocess each view and disable the model's internal
preprocessing to avoid doing it twice:

.. code-block:: python

   from polyclustx.preprocessing import fuse_graphs, get_propagated_features

   prepared_views = []
   for view, graphs in zip(data.Xs, relations):
       adjacency = fuse_graphs(graphs, weights=[0.7, 0.3])
       prepared_views.append(get_propagated_features(view, adj=adjacency, beta=1.0))

   prepared_model = LMGEC(n_clusters=3, embedding_dim=3, preprocess=False, random_state=42)
   prepared_model.fit(prepared_views)

For count features, LMGEC also accepts per-view options such as
``tf_idf=[True, False]`` and ``beta=[1.0, 0.5]``. TF-IDF requires nonnegative
features: do not apply it to these standardized Wine measurements.

5. Compare methods and datasets
-------------------------------

.. code-block:: python

   from sklearn.datasets import make_blobs
   from polyclustx.experiment import Experiment

   Z, z_labels = make_blobs(n_samples=90, n_features=12, centers=3, random_state=42)
   synthetic = MultiViewDataset([Z[:, :6], Z[:, 6:]], labels=z_labels, name="Synthetic")
   results = Experiment(
       datasets={"Wine": data, "Synthetic": synthetic},
       models=[
           LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42),
           MultiViewKMeans(n_clusters=3, n_init=3, random_state=42),
       ],
       metrics={"ACC": clustering_accuracy, "NMI": nmi_score}, runs=3,
       save_path="experiments/comparison",
   ).run()
   print(results.results_tables())
   results.export("experiments/comparison.zip")

This performs 12 fits and reports mean ± standard deviation for each pair.
Both datasets have three reference classes. In Python, set K explicitly on
each estimator; use separate jobs when datasets need different K values.

6. Choose K without labels
--------------------------

.. code-block:: python

   from polyclustx.metrics import silhouette_score

   unlabeled = MultiViewDataset(data.Xs, name="Measurements")
   k_results = Experiment(
       unlabeled,
       [LMGEC(n_clusters=k, embedding_dim=3, max_iter=20, random_state=42)
        for k in [2, 3, 4]],
       metrics={"Silhouette": silhouette_score}, benchmark=False, runs=2,
       save_path="experiments/choose-k",
   ).run()
   print(k_results.results_tables())

Higher silhouette indicates better-separated clusters in the learned embedding.

7. Simulate missing views
-------------------------

.. code-block:: python

   from polyclustx.data import simulate_missing_views, impute_missing
   from polyclustx.models import COMPLETER

   missing = simulate_missing_views(data, 0.2, random_state=42)
   print(missing.observed_mask.sum(axis=0))
   missing_model = missing.fit(COMPLETER(
       n_clusters=3, embedding_dim=4, hidden_dims=(16,), prediction_dims=(8, 4),
       max_iter=3, prediction_start=1, batch_size=32, random_state=42,
   ))

   # An explicit imputation baseline lets a complete-input method use the data.
   complete = impute_missing(missing, strategy="mean")
   baseline = complete.fit(LMGEC(n_clusters=3, embedding_dim=3, random_state=42))
   print("COMPLETER NMI:", nmi_score(y, missing_model.labels_))
   print("Mean-imputed LMGEC NMI:", nmi_score(y, baseline.labels_))

The original data stays unchanged. The fraction refers to observed sample-view
pairs; each sample keeps at least one view. The short neural training schedule
above demonstrates the API, not tuned benchmark settings.

8. Search hyperparameters
-------------------------

.. code-block:: python

   from polyclustx.experiment import GridSearchExperiment, RandomSearchExperiment, OptunaSearchExperiment

   space = {"n_clusters": [3], "embedding_dim": [2, 4],
            "temperature": [0.5, 1.0], "max_iter": [20]}
   grid = GridSearchExperiment(
       LMGEC, space, data, metrics={"NMI": nmi_score}, best_metric="NMI",
       runs=2, seed=42, save_path="experiments/grid",
   ).run()
   print(grid.best_params, grid.best_score)

   random = RandomSearchExperiment(
       LMGEC, space, data, metrics={"NMI": nmi_score}, best_metric="NMI",
       n_iter=3, runs=2, seed=42, save_path="experiments/random",
   ).run()

   continuous = {**space, "temperature": {"low": 0.1, "high": 3.0, "log": True}}
   optuna = OptunaSearchExperiment(
       LMGEC, continuous, data, metrics={"NMI": nmi_score}, best_metric="NMI",
       n_iter=3, runs=2, seed=42, save_path="experiments/optuna",
   ).run()
   best_model = data.fit(LMGEC(**grid.best_params, random_state=42))

Grid evaluates every combination, random samples candidates, and Optuna can
suggest continuous values. All three select by mean NMI across repeated fits.
For unlabeled data, use an internal metric and ``benchmark=False``.

9. Visualize and save results
-----------------------------

.. code-block:: python

   from polyclustx.visualization import plot_clusters, plot_loss, plot_loss_comparison
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(10, 4))
   plot_clusters(model.embedding_, model.labels_, method="pca", ax=axes[0],
                 title="Predicted clusters", show=False)
   plot_clusters(model.embedding_, y, method="pca", ax=axes[1],
                 title="Reference classes", show=False)
   fig.savefig("experiments/predicted-clusters.png", bbox_inches="tight")
   ax = plot_loss(model.loss_history_, normalize=True, show=False)
   ax.figure.savefig("experiments/loss.svg", bbox_inches="tight")
   ax = plot_loss_comparison([results, grid], run=1, show=False)
   ax.figure.savefig("experiments/compared-losses.svg", bbox_inches="tight")

   results.export("experiments/comparison.csv")
   results.export("experiments/comparison.json")

Run 1 represents each configuration in the loss comparison. Normalized curves
compare optimization trajectories, not the absolute value of different objectives.

Notebook tutorials
------------------

* :download:`Feature-to-graph study <../../notebooks/06_polyclustx_feature_to_graph_study.ipynb>`
* :download:`All-method comparison <../../notebooks/04_all_methods_comparison.ipynb>`
* :download:`Search strategies <../../notebooks/05_search_strategies.ipynb>`
