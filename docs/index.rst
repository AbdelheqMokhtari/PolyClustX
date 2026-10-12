pymv: comparative clustering and representation learning
========================================================================

pymv is a comparative framework for multi-view and multimodal clustering and
representation learning. It provides one data contract, multiple methods, and
a reproducible experimental workspace.
Use a single view, several feature views, multimodal data, or attributed graphs
without changing the structure of your workflow.

pymv separates **data representation**, **model fitting**, and **evaluation**.
Ground-truth labels are optional. Graphs are optional, and a feature view can
have several graph relations. Incomplete views can be represented, aligned by
sample ID, and explicitly imputed; native missing-view learning depends on the
selected model's capabilities.

.. code-block:: bash

   pip install pymv-clustering
   pymv display

.. code-block:: python

   from pymv.data import load_data
   from pymv.models import LMGEC

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
   :caption: Extending pymv

   contributing/models
   contributing/datasets
   contributing/pull_requests
   contributing/documentation

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

.. toctree::
   :maxdepth: 1
   :caption: Project

   user_guide/changelog

Scientific scope
----------------

Support for a data representation does not imply that every model uses all of
its information. See :doc:`background/algorithms` for the capability table,
original papers, source implementations and algorithm explanations.

The documentation organization takes inspiration from the
`UMAP documentation <https://umap-learn.readthedocs.io/en/latest/>`_.
The provisional symbol and restrained burgundy/slate palette are shared with the
interface. The mark is independent of, and does not imply endorsement by,
Université Paris Cité.
