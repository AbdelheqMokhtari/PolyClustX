PolyClustX
==========

**A Unified Framework for Multi-View and Multimodal Representation Learning
and Clustering from Feature Views to Attributed Graphs**

Prepare data, compare methods, tune hyperparameters and visualize results through
one Python API, command line or interactive interface.
Use a single view, several feature views, multimodal data, or attributed graphs
without changing the structure of your workflow.

.. code-block:: bash

   pip install polyclustx
   polyclustx-display

.. code-block:: python

   from polyclustx.data import load_data
   from polyclustx.models import LMGEC

   data = load_data("acm")
   model = data.fit(LMGEC(n_clusters=3, embedding_dim=8, random_state=42))
   predicted_clusters = model.labels_

Start with :doc:`user_guide/quickstart`, explore
:doc:`user_guide/workbench`, or find a method and dataset in the
:doc:`background/algorithms` and :doc:`catalog/datasets` catalogs.

.. toctree::
   :maxdepth: 2
   :caption: User guide & tutorials

   user_guide/installation
   user_guide/quickstart
   user_guide/data
   user_guide/missing_views
   user_guide/graphs
   user_guide/clustering_models
   user_guide/experiments
   user_guide/workbench

.. toctree::
   :maxdepth: 2
   :caption: Algorithms & resources

   background/algorithms
   catalog/datasets

.. toctree::
   :maxdepth: 2
   :caption: Examples of usage

   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: Extending PolyClustX

   contributing/models
   contributing/datasets

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/data
   api/core
   api/models
   api/preprocessing
   api/metrics
   api/experiment
   api/display
   api/visualization
   api/cli

Scientific scope
----------------

Support for a data representation does not imply that every model uses all of
its information. See :doc:`background/algorithms` for the capability table,
original papers, source implementations and algorithm explanations.

Authors and affiliation
-----------------------

Abdelheq Mokhtari, Lazhar Labiod and Mohamed Nadif.
All authors are affiliated with Centre Borelli, Université Paris Cité.
