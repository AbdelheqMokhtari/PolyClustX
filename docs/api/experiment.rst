Experiment Framework
====================

Experiment runners and result containers for systematic model comparison
and hyper-parameter search.

Experiment
----------

.. autoclass:: pymv.experiment.Experiment
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

RandomSearchExperiment
----------------------

.. autoclass:: pymv.experiment.RandomSearchExperiment
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Grid search
-----------

.. autoclass:: pymv.experiment.GridSearchExperiment
   :members:
   :inherited-members:
   :special-members: __init__

Result containers and reports
-----------------------------

.. autoclass:: pymv.experiment.OptunaSearchExperiment
   :members:
   :inherited-members:
   :special-members: __init__

Runner results expose ``to_dict()``, ``json_path``, ``save_json(output_dir)``
and ``export(path)`` for JSON, CSV or ZIP output. Automatic JSON export defaults
to ``./experiments``; pass ``save_path=`` or ``save_json=False`` to a runner.

.. autoclass:: pymv.experiment.ExperimentResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pymv.experiment.RandomSearchResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pymv.experiment.Result
   :members:
   :undoc-members:
   :show-inheritance:
