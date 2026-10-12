Command line
============

Use ``pymv`` for short, model-independent commands. The same installation
provides the interface and Python API; configuration files are not required.

Discover and run
----------------

.. code-block:: bash

   pymv --help
   pymv display
   pymv datasets
   pymv models
   pymv models idec
   pymv run acm --model lmgec --param embedding_dim=8 --runs 5

``pymv models NAME`` lists that method's parameters and defaults.
Repeat ``--param`` for additional settings:

.. code-block:: bash

   pymv run acm --model idec --param hidden_dims='[128,64]' --param max_iter=100

Values may be numbers, text, ``true``/``false`` or JSON lists. K defaults to
the number of ground-truth classes when labels exist; override with
``--clusters``. Select a dataset variant with ``--variant``.

Search
------

.. code-block:: bash

   pymv search acm --model lmgec --strategy grid --vary 'embedding_dim=[4,8]' --vary 'temperature=[0.5,1,2]' --runs 3
   pymv search acm --model lmgec --strategy random --vary 'temperature=[0.5,1,2]' --trials 10
   pymv search acm --model lmgec --strategy optuna --vary 'temperature={"low":0.1,"high":5,"log":true}' --trials 10 --objective nmi

``--param`` fixes a value; ``--vary`` searches it. Grid exhausts all choices;
random samples choices; Optuna adapts subsequent trials to completed scores
and accepts choices or numeric ranges. Add ``"type":"int"`` to a range for
integer parameters. Quote the whole assignment as shown. Optuna does not
prune partially trained runs; every trial evaluates its requested repeats.

``--metrics acc nmi ari`` selects external metrics. For unlabeled data use:

.. code-block:: bash

   pymv run acm --model multiview_kmeans --no-benchmark --clusters 3 --metrics silhouette db

A search uses ``--objective`` (default: first metric); Davies–Bouldin is
minimized, while ACC, NMI and Silhouette are maximized. ``--seed`` makes
initializations repeatable. ``--output reports/my-study`` chooses the JSON
report directory (default ``experiments``). See :doc:`/user_guide/experiments`.

Existing scripts
----------------

The older ``pymv-cli`` entry point and YAML/JSON ``--config`` recipes remain
available for compatibility. New users can use the simpler commands above.

.. automodule:: pymv.command
   :members: main

.. automodule:: pymv.config
   :members:

.. autoclass:: pymv.tracking.ExperimentTracker
   :members:
