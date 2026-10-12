Worked examples
===============

These examples use Iris, Wine and generated data: no dataset downloads are
needed. Training schedules are deliberately short to illustrate the workflow,
not to reproduce paper results.

Download :download:`the complete script <../../examples/practical_workflows.py>`
and run it with ``python practical_workflows.py``. Reports and plots are written
to ``experiments/tutorial/``. To try individual examples, run the setup below
first, then define and call the function you want.

.. contents:: Choose a task
   :local:
   :depth: 1

Setup: two aligned feature views
--------------------------------

Rows correspond to the same samples in both views. We split Iris and Wine
columns for demonstration; these are two views of tabular measurements, not
different modalities. Scaling uses the whole dataset for this clustering study.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :start-at: from pathlib import Path
   :end-before: def cluster_features

1. Cluster one or two views
---------------------------

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: cluster_features

.. code-block:: python

   data, model = cluster_features()

``labels_`` has one cluster ID per row. Labels stored on the dataset are used
only for the NMI evaluation, never to train the model.

2. Compare three methods on two datasets
----------------------------------------

This runs 18 fits: two datasets × three methods × three repetitions.
Iris and Wine both have three reference classes, so the same K is appropriate.
For datasets with different class counts, prepare separate jobs with each K,
or use the :doc:`multi-dataset CLI </api/cli>` to infer K per dataset.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: compare_methods

.. code-block:: python

   results = compare_methods()

The table contains mean and standard deviation across runs. ``comparison.zip``
contains the JSON report and a CSV table of individual runs.

3. Choose K without true labels
-------------------------------

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: choose_k_without_labels

.. code-block:: python

   internal_results = choose_k_without_labels()
   print(internal_results.results_tables())

Higher silhouette indicates better-separated groups in the representation
being evaluated; it does not establish that the clusters have semantic meaning.

4. Use three views with multiple graphs
---------------------------------------

Generate a binary kNN and a weighted kNN relation for each view. LMGEC receives
their weighted sum. Graph construction uses features, not reference labels.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: cluster_graphs

.. code-block:: python

   graph_data, graph_model = cluster_graphs()

For your own data, pass ``MultiViewDataset([X1, X2], As=[[A1, A2], None])``:
two relations belong to view 1, and view 2 has no graph. See :doc:`graphs`.

5. Compare missing-view learning with imputation
------------------------------------------------

Hide 20% of observed sample-view pairs while retaining at least one view per
sample. The original dataset stays unchanged.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: cluster_missing_views

.. code-block:: python

   incomplete, completed_model, baseline = cluster_missing_views()

COMPLETER receives the observation mask. The baseline receives mean-imputed
features instead. Check :doc:`model capabilities </background/algorithms>`
before fitting a different estimator to incomplete data.

6. Grid, random and continuous search
-------------------------------------

All three searches maximize mean NMI over two runs per candidate. Replace NMI
with an internal metric and set ``benchmark=False`` for unlabeled data.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: tune_parameters

.. code-block:: python

   grid, random, optuna, best_model = tune_parameters()

7. Save projections and compare losses
--------------------------------------

Use the three-dimensional graph embedding from example 4 and the experiment
report from example 2. This creates ``clusters.png`` and ``losses.svg``.

.. literalinclude:: ../../examples/practical_workflows.py
   :language: python
   :pyobject: save_figures

.. code-block:: python

   save_figures(graph_data, graph_model, results)

For another run, change ``run=1`` to ``run=2``. Pass a list of result objects to
compare losses across experiments. Normalization compares curve shapes, not
the absolute quality of different algorithms' objectives.

Continue in a notebook
----------------------

* :download:`Feature-to-graph study <../../notebooks/06_polyclustx_feature_to_graph_study.ipynb>`
* :download:`All-method comparison <../../notebooks/04_all_methods_comparison.ipynb>`
* :download:`Search strategies <../../notebooks/05_search_strategies.ipynb>`

For CSV, NumPy and MATLAB inputs, see :doc:`data`. For the same tasks without
code, follow :doc:`workbench`.
