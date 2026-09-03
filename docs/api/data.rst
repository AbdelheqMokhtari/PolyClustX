Data Loading
============

The ``pymv.data`` subpackage provides convenient loader functions for
multi-view and attributed-graph datasets. Every loader downloads the
data on first use and caches it locally under ``~/.pymv_datasets/``.

Registry
--------

.. autofunction:: pymv.data.load_dataset

.. autofunction:: pymv.data.list_datasets


.. autofunction:: pymv.data.get_dataset_info

The catalog currently holds **55 datasets** across the ``graph``,
``features`` and ``multimodal`` categories.  ``list_datasets()`` is the
authoritative list — the loaders named below are the most commonly used
ones, not the full set:

.. code-block:: python

   from pymv.data import list_datasets
   list_datasets()                     # every name
   list_datasets(category="graph")     # filter by category

Dataset Container
-----------------

.. autoclass:: pymv.data.MultiViewDataset
   :members:
   :undoc-members:
   :special-members: __init__, __iter__, __len__, __getitem__

Dataset Loaders
---------------

Every entry in the catalog also gets a dedicated ``load_<name>``
function, generated from its ``DatasetSpec``.

.. autofunction:: pymv.data.load_acm

.. autofunction:: pymv.data.load_amazon_photos

.. autofunction:: pymv.data.load_bdgp

.. autofunction:: pymv.data.load_coco_all

.. autofunction:: pymv.data.load_coco_cross

.. autofunction:: pymv.data.load_dblp

.. autofunction:: pymv.data.load_imdb

.. autofunction:: pymv.data.load_pascal

.. autofunction:: pymv.data.load_prokaryotic

.. autofunction:: pymv.data.load_wiki
