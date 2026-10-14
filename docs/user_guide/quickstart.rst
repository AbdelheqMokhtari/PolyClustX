Quick start
===========

Choose the interface, Python API or CLI
---------------------------------------

The complete installation includes all three interfaces:

.. code-block:: bash

   polyclustx-display
   lmgec acm --embedding-dim 8 --runs 3

The interface opens in your browser. The CLI prints experiment tables and saves
reports. For programmatic control, use the same data and model classes below.

A small comparative experiment
------------------------------

This example is offline: Wine's measured columns are divided into two aligned
views. Labels are only used to evaluate the clustering.

.. code-block:: python

   from sklearn.datasets import load_wine
   from sklearn.preprocessing import StandardScaler
   from polyclustx.data import MultiViewDataset
   from polyclustx.models import LMGEC, MultiViewKMeans
   from polyclustx.metrics import clustering_accuracy, nmi_score
   from polyclustx.experiment import Experiment

   X, labels = load_wine(return_X_y=True)  # small, offline example
   X = StandardScaler().fit_transform(X)
   data = MultiViewDataset([X[:, :6], X[:, 6:]], labels=labels, name="Wine")

   results = Experiment(
       datasets=data,
       models=[
           LMGEC(n_clusters=3, embedding_dim=3, max_iter=20, random_state=42),
           MultiViewKMeans(n_clusters=3, random_state=42),
       ],
       metrics={"ACC": clustering_accuracy, "NMI": nmi_score},
       runs=3,
       save_path="experiments/comparison",
   ).run()

By default, the runner prints a configuration table followed by one results
table per dataset. Here is the result portion from this example:

.. code-block:: text

   Results — Wine
   ┌────────┬──────────────────┬────────────────────┐
   │ Metric │            LMGEC │    MultiViewKMeans │
   ├────────┼──────────────────┼────────────────────┤
   │ ACC    │ 0.9551 (±0.0000) │ 0.9663 (±0.0000) * │
   │ NMI    │ 0.8352 (±0.0000) │ 0.8770 (±0.0000) * │
   │ Alphas │   [0.481, 0.519] │                    │
   │ Time   │            0.03s │              0.02s │
   └────────┴──────────────────┴────────────────────┘

Each score is a mean and standard deviation over three fits, including zero
standard deviation. A star marks the best mean in each column. Timing and
numerical results can vary between environments. A narrow terminal transposes
the table; notebooks keep the model rows and add a horizontal scrollbar.

.. code-block:: python

   results.export("experiments/comparison.zip")  # JSON and CSV reports

Fit a model directly to inspect its learned representation:

.. code-block:: python

   from polyclustx.visualization import plot_clusters

   model = data.fit(LMGEC(n_clusters=3, embedding_dim=3, random_state=42))
   print(model.labels_.shape)      # (178,)
   print(model.embedding_.shape)   # (178, 3)
   plot_clusters(model.embedding_, model.labels_, method="pca")

Use a catalog dataset
---------------------

.. code-block:: python

   from polyclustx.data import load_data

   data = load_data("acm", views={
       "Co-authorship": ("X1_features", "A1_PAP"),
       "Shared subjects": ("X1_features", "A2_PLP"),
   })
   model = data.fit(LMGEC(n_clusters=data.n_clusters, embedding_dim=8, random_state=42))

See :doc:`data` for direct Hugging Face downloads, encoder variants, local MAT
files, feature-only views and multiple graph relations in one view.

Clustering without labels
-------------------------

Use internal metrics and choose K manually:

.. code-block:: python

   from polyclustx.metrics import silhouette_score

   unlabeled = MultiViewDataset(data.Xs, name="My measurements")
   Experiment(
       datasets=unlabeled,
       models=[LMGEC(n_clusters=k, embedding_dim=4, random_state=42)
               for k in [2, 3, 5]],
       metrics={"Silhouette": silhouette_score},
       benchmark=False,
   ).run()

One, two or more views use the same data.fit(model) interface. Labels never
reach the model. Continue with :doc:`examples`, :doc:`experiments` and
:doc:`workbench`.
