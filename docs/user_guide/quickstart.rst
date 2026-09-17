Quick start
===========

Clustering without labels
-------------------------

.. code-block:: python

   import numpy as np
   from pymv.data import MultiViewDataset
   from pymv.models import LMGEC
   from pymv.metrics import silhouette_score
   from pymv.experiment import Experiment

   rng = np.random.default_rng(42)
   data = MultiViewDataset(
       [rng.normal(size=(60, 8)), rng.normal(size=(60, 5))],
       view_names=["shape", "texture"],
       modalities=["image", "image"],
       name="My measurements",
   )
   model = data.fit(LMGEC(n_clusters=3, embedding_dim=4, random_state=42))
   print(model.labels_.shape)      # (60,)
   print(model.embedding_.shape)   # consensus representation

   Experiment(
       datasets=data,
       models=[LMGEC(n_clusters=k, embedding_dim=4, random_state=42)
               for k in [2, 3, 5]],
       metrics={"Silhouette": silhouette_score},
       benchmark=False,
   ).run()

You choose K. Labels are neither required nor supplied to the model. Internal
metrics evaluate the learned embedding, not agreement with a hidden ground truth.

A labeled benchmark
-------------------

.. code-block:: python

   from pymv.data import load_data
   from pymv.metrics import nmi_score

   data = load_data("acm")
   model = data.fit(LMGEC(n_clusters=3, embedding_dim=8, random_state=42))
   print("NMI:", nmi_score(data.labels, model.labels_))

All loaders return a :class:`~pymv.data.MultiViewDataset`, with optional labels,
a sample order, an observation mask, and one graph entry per view. Legacy
``Xs, As, y = data`` unpacking still works, but loses the IDs and missing mask.
Prefer ``data.fit(model)`` when handling incomplete or multi-relational data.

Continue with :doc:`data`, :doc:`examples`, and :doc:`workbench`.
