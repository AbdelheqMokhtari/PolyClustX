Models
======

LMGEC
-----

.. automodule:: pymv.models.lmgec.lmgec
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: pymv.models.LMGEC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

LMGECplus
---------

The modular research framework.  ``LMGECplus`` is a facade over a
:class:`~pymv.models.lmgecplus.ComponentBundle`; every stage below can be
swapped independently, which is what makes ablation studies a one-liner.

.. autoclass:: pymv.models.LMGECplus
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: pymv.models.lmgecplus.LMGECResult
   :members:

Component registry
^^^^^^^^^^^^^^^^^^

.. autofunction:: pymv.models.lmgecplus.register_component

.. autofunction:: pymv.models.lmgecplus.get_component

.. autofunction:: pymv.models.lmgecplus.list_components

.. autofunction:: pymv.models.lmgecplus.resolve_component

Ablation studies
^^^^^^^^^^^^^^^^

.. autofunction:: pymv.models.lmgecplus.build_ablation_suite

.. autodata:: pymv.models.lmgecplus.DEFAULT_ABLATIONS

Component interfaces
^^^^^^^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.interfaces
   :members:
   :undoc-members:
   :show-inheritance:

Reconstruction losses
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.losses
   :members:
   :undoc-members:
   :show-inheritance:

Normalizations
^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.normalization
   :members:
   :undoc-members:
   :show-inheritance:

Graph propagation
^^^^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.propagation
   :members:
   :undoc-members:
   :show-inheritance:

Attention
^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.attention
   :members:
   :undoc-members:
   :show-inheritance:

View weighting
^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.weighting
   :members:
   :undoc-members:
   :show-inheritance:

Alignment
^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.alignment
   :members:
   :undoc-members:
   :show-inheritance:

Clustering heads
^^^^^^^^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.clustering
   :members:
   :undoc-members:
   :show-inheritance:

Optimizer
^^^^^^^^^

.. automodule:: pymv.models.lmgecplus.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Model registry
--------------

.. autofunction:: pymv.models.register_model

.. autofunction:: pymv.models.get_model

.. autofunction:: pymv.models.get_model_info

.. autofunction:: pymv.models.list_models

.. autofunction:: pymv.models.list_models_info
