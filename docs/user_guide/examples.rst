Examples
========

This page walks through 8 progressively detailed examples.

.. contents:: On this page
   :local:
   :depth: 1

Example 1 — Load, fit, evaluate
--------------------------------

The simplest workflow: load a dataset, fit a model, evaluate.

.. code-block:: python

   from pymv.data import load_acm
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score, ari_score

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
       center=True,
   )
   labels = model.fit_predict(Xs, As)


Example 3 — Custom preprocessed views
--------------------------------------

Set ``preprocess=False`` to skip all internal preprocessing:

.. code-block:: python

   from pymv.preprocessing import get_propagated_features, preprocess_features

   H0 = get_propagated_features(X0, A0, beta=1.0, tf_idf=True)
   H1 = get_propagated_features(X1, A1, beta=0.5, tf_idf=False)

   Hs = preprocess_features([H0, H1], center=True, scale=False)

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

   from pymv.experiment import Experiment
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score, ari_score

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
       n_runs=10,
       verbose=1,
   ).run()

   print(result.config_table())
   print(result.results_tables())


Example 6 — Hyper-parameter search
-----------------------------------

Find the best configuration for a single dataset with random search:

.. code-block:: python

   from pymv.experiment import RandomSearchExperiment
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score

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

   from pymv.experiment import Experiment
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy

   my_data = (Xs, As, y)

   result = Experiment(
       datasets={"my_dataset": my_data},
       models=[LMGEC(n_clusters=5, embedding_dim=16)],
       metrics=[clustering_accuracy],
   ).run()


Example 8 — Visualize the embedding
------------------------------------

Plot the learned consensus embedding with t-SNE or PCA:

.. code-block:: python

   from pymv.data import load_acm
   from pymv.models import LMGEC
   from pymv.visualization.visualizer import plot_clusters

   Xs, As, y = load_acm()
   model = LMGEC(n_clusters=3, embedding_dim=16)
   model.fit(Xs, As)

   plot_clusters(model.embedding_, model.labels_, method="tsne",
                 title="LMGEC on ACM")

   plot_clusters(model.embedding_, y, method="pca",
                 title="Ground truth on ACM")
