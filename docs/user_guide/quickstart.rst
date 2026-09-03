Quick Start
===========

This page shows the two most common workflows — use them to get up and
running in under a minute.

Load → Fit → Evaluate
---------------------

.. code-block:: python

   from pymv.data import load_acm
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score

   # 1. Load data (auto-downloads on first call)
   Xs, As, y = load_acm()

   # 2. Fit and predict
   model = LMGEC(n_clusters=3, embedding_dim=16)
   labels = model.fit_predict(Xs, As)

   # 3. Evaluate
   print("ACC:", clustering_accuracy(y, labels))
   print("NMI:", nmi_score(y, labels))

Every dataset loader returns the same triple:

- ``Xs`` — ``list`` of feature matrices (one per view)
- ``As`` — ``list`` of adjacency matrices (one per view)
- ``y``  — ground-truth label vector

Comparative experiment
----------------------

Compare multiple configurations across multiple datasets in one call:

.. code-block:: python

   from pymv.experiment import Experiment
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score

   Experiment(
       datasets=["acm", "dblp", "imdb"],
       models=[
           LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0),
           LMGEC(n_clusters=3, embedding_dim=16, temperature=10.0),
       ],
       metrics=[clustering_accuracy, nmi_score],
   ).run()

The console prints **two clean tables**: model configuration and per-dataset
results (with best values marked ``*``).  All detailed logs go to ``./logs/``.

What's next?
------------

- :doc:`examples` — 8 in-depth examples covering every feature.
- :doc:`/api/models` — full LMGEC API reference.
- :doc:`/api/experiment` — Experiment & RandomSearchExperiment docs.
