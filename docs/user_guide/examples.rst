Examples
========

Start with the offline tutorial below, then use the focused benchmark examples.
All examples using catalog datasets download their data on first use.

Notebook tutorials
------------------

Download and open these notebooks in Jupyter (``python -m pip install notebook``
if needed). All notebooks are supplied **without executed outputs**. Run them
yourself with a kernel that has PolyClustX installed. The tutorials below do not
download datasets; catalog examples in other notebooks may download on first use.

* :download:`From features to attributed graphs <../../notebooks/06_polyclustx_feature_to_graph_study.ipynb>`:
  start here for Iris/Wine comparisons, repeated experiments, projection and loss
  visualizations, multiple graph relations, grid/random/Optuna search and exports.

* :download:`All-method comparison <../../notebooks/04_all_methods_comparison.ipynb>`:
  thirteen models, two datasets, repeated-run tables, graph relations, missing
  inputs, single-view use, loss plots and ZIP reports.
* :download:`Search strategies <../../notebooks/05_search_strategies.ipynb>`:
  grid, random and Optuna search, continuous/integer ranges, internal objectives
  and copying the best parameters into a repeated experiment.

New clustering methods
----------------------

The :doc:`clustering_models` tutorial explains the method-specific assumptions.
This small example compares six estimators and then runs COMPLETER/DCP on
explicitly missing views. Small networks and short schedules illustrate the
workflow; they are not publication benchmark settings.

.. literalinclude:: ../../examples/new_models.py
   :language: python

Offline: single, multiple, missing and graph views
--------------------------------------------------

This complete script creates synthetic views, runs LMGEC and SCM, explicitly
imputes missing views, fuses relations and compares manually chosen K values.
Small training settings keep the tutorial quick; they are not recommended
research hyperparameters. From a development checkout, run
``python examples/offline_workflows.py``.

.. literalinclude:: ../../examples/offline_workflows.py
   :language: python
   :lines: 3-

For file imports, see :doc:`data`; for interactive experiments see
:doc:`workbench`.

.. contents:: On this page
   :local:
   :depth: 1

Example 1 — Load, fit, evaluate
--------------------------------

The simplest workflow: load a dataset, fit a model, evaluate.

.. code-block:: python

   from polyclustx.data import load_acm
   from polyclustx.models import LMGEC
   from polyclustx.metrics import clustering_accuracy, nmi_score, ari_score

   Xs, As, y = load_acm()

   model = LMGEC(n_clusters=3, embedding_dim=16)
   labels = model.fit_predict(Xs, As)

   print("ACC:", clustering_accuracy(y, labels))
   print("NMI:", nmi_score(y, labels))
   print("ARI:", ari_score(y, labels))


Example 2 — Per-view preprocessing
-----------------------------------

LMGEC can apply different preprocessing to each view.
Pass lists to ``beta`` and ``tf_idf``:

.. code-block:: python

   model = LMGEC(
       n_clusters=3,
       embedding_dim=16,
       tf_idf=[True, False],      # TF-IDF on view 0, L2 on view 1
       beta=[1.0, 0.5],           # different self-loop weights
       center=False,             # TF-IDF requires nonnegative features
   )
   labels = model.fit_predict(Xs, As)


Example 3 — Custom preprocessed views
--------------------------------------

Set ``preprocess=False`` to skip all internal preprocessing:

.. code-block:: python

   from polyclustx.preprocessing import get_propagated_features, preprocess_features

   # Xs and As come from Example 1. Center before row normalization;
   # do not center features that are intended for TF-IDF.
   H0 = get_propagated_features(Xs[0], As[0], beta=1.0, tf_idf=True)
   H1 = get_propagated_features(Xs[1], As[1], beta=0.5, center=True)
   Hs = [H0, H1]

   model = LMGEC(n_clusters=3, embedding_dim=16, preprocess=False)
   labels = model.fit_predict(Hs)      # no adjacency needed


Example 4 — Feature-only clustering
------------------------------------

If your data has no natural graph structure, just pass raw feature matrices:

.. code-block:: python

   model = LMGEC(n_clusters=5, embedding_dim=32, preprocess=False)
   labels = model.fit_predict([X_view1, X_view2, X_view3])


Example 5 — Full comparative experiment
----------------------------------------

Compare multiple model configurations across multiple datasets:

.. code-block:: python

   from polyclustx.experiment import Experiment
   from polyclustx.models import LMGEC
   from polyclustx.metrics import clustering_accuracy, nmi_score, ari_score

   m1 = LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0)
   m1.name = "LMGEC-T1"

   m2 = LMGEC(n_clusters=3, embedding_dim=16, temperature=10.0)
   m2.name = "LMGEC-T10"

   m3 = LMGEC(n_clusters=3, embedding_dim=32, temperature=5.0, tf_idf=True)
   m3.name = "LMGEC-d32-tfidf"

   result = Experiment(
       datasets=["acm", "dblp", "imdb"],
       models=[m1, m2, m3],
       metrics=[clustering_accuracy, nmi_score, ari_score],
       runs=10,
       verbose=1,
   ).run()

   print(result.config_table())
   print(result.results_tables())


Example 6 — Hyper-parameter search
-----------------------------------

Find the best configuration for a single dataset with random search:

.. code-block:: python

   from polyclustx.experiment import RandomSearchExperiment
   from polyclustx.models import LMGEC
   from polyclustx.metrics import clustering_accuracy, nmi_score

   rs = RandomSearchExperiment(
       model_class=LMGEC,
       param_distributions={
           "n_clusters": [3],
           "embedding_dim": [4, 8, 16, 32, 64],
           "temperature": [0.1, 0.5, 1.0, 5.0, 10.0],
           "beta": [0.1, 0.5, 1.0, 2.0],
           "max_iter": [30],
       },
       dataset="acm",
       metrics=[clustering_accuracy, nmi_score],
       best_metric="clustering_accuracy",
       n_iter=30,
   )
   result = rs.run()

   print("Best params:", result.best_params)
   print("Best ACC:   ", result.best_score)


Example 7 — Custom data in Experiment
--------------------------------------

Pass your own pre-loaded data instead of built-in loaders:

.. code-block:: python

   from polyclustx.experiment import Experiment
   from polyclustx.models import LMGEC
   from polyclustx.metrics import clustering_accuracy

   my_data = (Xs, As, y)

   result = Experiment(
       datasets={"my_dataset": my_data},
       models=[LMGEC(n_clusters=5, embedding_dim=16)],
       metrics=[clustering_accuracy],
   ).run()


Local datasets and variants (offline)
-------------------------------------

Run ``python examples/custom_datasets.py`` in a repository checkout. The example
creates two small MATLAB representations of the same samples and loads them
without downloads. See :doc:`/contributing/datasets` to extend the example into
a registered dataset family with variants.

Example 8 — Visualize the embedding
------------------------------------

Plot the learned consensus embedding with t-SNE or PCA:

.. code-block:: python

   from polyclustx.data import load_acm
   from polyclustx.models import LMGEC
   from polyclustx.visualization.visualizer import plot_clusters

   Xs, As, y = load_acm()
   model = LMGEC(n_clusters=3, embedding_dim=16)
   model.fit(Xs, As)

   plot_clusters(model.embedding_, model.labels_, method="tsne",
                 title="LMGEC on ACM")

   plot_clusters(model.embedding_, y, method="pca",
                 title="Ground truth on ACM")
