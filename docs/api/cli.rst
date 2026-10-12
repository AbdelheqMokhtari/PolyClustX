Command line
============

Install PolyClustX once. Use a method's name to fit a model,
``polyclustx experiment`` to compare methods, and ``polyclustx-display``
to open the interface. All commands use the same estimators and report format.

Discover methods and datasets
-----------------------------

.. code-block:: bash

   polyclustx --version
   polyclustx models
   polyclustx datasets
   lmgec --help
   idec --help

Each method's help lists only its own parameters and the shared run/search
options. Installed method commands are ``lmgec``, ``scm``, ``idec``,
``completer``, ``dcp``, ``mflvc``, ``conan``, ``dgf``, ``g-comvkm``,
``sgla``, ``umc``, ``multiview-kmeans`` and ``multiview-spectral``.

Short command names may collide with unrelated tools in the same environment.
The namespaced equivalent is always available: ``polyclustx lmgec ...``.
If console commands are not on PATH, use ``python -m polyclustx lmgec ...``.

Run a method
------------

.. code-block:: bash

   lmgec acm
   lmgec acm --embedding-dim 8 --max-iter 30 --runs 5
   scm acm --embedding-dim 16 --max-iter 50
   idec acm --hidden-dims '128,64' --pretrain-epochs 20
   multiview-spectral acm --affinity rbf --n-init 5
   lmgec pascal --variant clip --clusters 5

The dataset is positional; ``lmgec --dataset acm`` is equivalent.
Parameter flags use hyphens: Python's ``embedding_dim`` becomes
``--embedding-dim``. Booleans use ``true`` or ``false``; layer widths
accept ``'128,64'`` or ``'[128,64]'``. For per-view values, pass a quoted
JSON list, for example ``--beta '[1.0,0.5]'``. The generic
``--param embedding_dim=8`` form also works; do not specify the same setting
using both forms. ``--no-preprocess`` disables internal preprocessing.

K defaults to the number of reference classes if labels exist. Override it
with ``--clusters`` (also ``--n-clusters`` or ``-k``).
The model itself never uses reference labels for training.
Without benchmark labels, choose K explicitly:

.. code-block:: bash

   multiview-kmeans acm --no-benchmark --clusters 3 --metrics silhouette db

``--runs 5`` repeats the configuration with different initialization seeds.
``--seed 42`` sets the reproducible base seed. ``--output reports/my-study``
selects the JSON report directory; the default is ``./experiments``.
A failed fixed-parameter fit is recorded in the report and returns a nonzero
process status; inspect its ``error`` field.

Missing-view simulation and imputation are explicit:

.. code-block:: bash

   lmgec acm --missing-rate 0.2 --impute mean --seed 42 --data-seed 7

Only use incomplete data directly with a model that supports it. See
:doc:`/user_guide/missing_views`.

Search hyperparameters
----------------------

.. code-block:: bash

   lmgec acm --search grid --vary 'embedding_dim=[4,8]' --vary 'temperature=[0.5,1,2]' --runs 3
   lmgec acm --search random --vary 'temperature=[0.5,1,2]' --trials 3 --objective nmi
   lmgec acm --search optuna --vary 'temperature={"low":0.1,"high":5,"log":true}' --trials 10 --objective nmi

Ordinary parameter flags fix values. ``--vary`` selects parameters to search.
Grid evaluates every combination; random samples candidates; Optuna adapts
later trials using completed scores. Optuna also accepts integer ranges:
``--vary 'embedding_dim={"type":"int","low":4,"high":16}'``.

Metrics accept short names ``acc nmi ari f1 silhouette db ch``.
``--objective`` selects the metric to optimize; it must be included in
``--metrics`` and defaults to the first selected metric.
Davies–Bouldin is minimized; ACC, NMI and Silhouette are maximized.
Each trial uses the mean across ``--runs``. No trial pruning is performed.

For scripts preferring one executable, the generic forms remain supported:

.. code-block:: bash

   polyclustx run acm --model lmgec --param embedding_dim=8
   polyclustx search acm --model lmgec --strategy grid --vary 'temperature=[0.5,1,2]'

Compare methods and datasets
----------------------------

.. code-block:: bash

   polyclustx experiment --datasets acm dblp --models lmgec scm --runs 3
   polyclustx experiment --datasets acm --models lmgec scm --param lmgec.embedding_dim=8 --param scm.max_iter=50

This evaluates every dataset/model combination and saves **one combined JSON**
with configuration details, per-run metrics, means, standard deviations and
available loss histories. One dataset is loaded at a time. Default K is inferred
separately for each labeled dataset. All benchmark datasets must have labels;
use ``--no-benchmark --clusters K`` for an entirely unsupervised comparison.

Use ``--param model.parameter=value`` to adjust a selected method.
For variants, write ``--datasets pascal:clip acm``. The registry identifier
``g_comvkm`` and command spelling ``g-comvkm`` both work in plans.
Completed dataset results are saved as the plan advances; failed fits remain
visible and the command returns a nonzero status.

Larger studies: one readable plan
---------------------------------

In a repository checkout:

.. code-block:: bash

   polyclustx experiment --config examples/experiment_plan.yaml
   polyclustx experiment --config examples/experiment_plan.yaml --runs 5 --output reports/study

.. literalinclude:: ../../examples/experiment_plan.yaml
   :language: yaml

Use a model mapping with ``name``, optional ``label``, and ``params``.
Different configurations of the same method need distinct labels. For a
dataset-specific variant or K, replace a dataset name with a mapping:

.. code-block:: yaml

   datasets:
     - name: pascal
       variant: clip
       clusters: 5
     - name: acm
       clusters: 3

Top-level fields are ``name``, ``datasets``, ``models``, ``runs``,
``seed``, ``clusters``, ``metrics``, ``benchmark`` and ``output``.
Unspecified benchmark K comes from that dataset's labels. Relative output
paths are relative to your working directory, not the plan file.
JSON mappings are also accepted. Plans contain data only—no executable Python.

CLI settings override the corresponding top-level field; ``--models``
replaces the model list. Qualified ``--param`` overrides apply to all
configurations of that method. Unknown fields and parameter names are rejected.

Launch the interface
--------------------

.. code-block:: bash

   polyclustx-display
   polyclustx-display --server.port=8502
   python -m polyclustx.display

The launcher is a standalone command, not a required subcommand.
The previous ``polyclustx display`` spelling is retained only as a compatibility
alias. The former ``polyclustx-cli`` executable is no longer installed.
Old single-model YAML recipes can still be replayed using
``python -m polyclustx.cli --config path/to/config.yaml``; those are distinct
from the new multi-model plan schema above.

API helpers
-----------

.. automodule:: polyclustx.command
   :members: main, model_main

.. automodule:: polyclustx.command_experiments
   :members: read_plan, prepare_plan, run_plan
