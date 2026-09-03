CLI, Config & Tracking
======================

The command-line front end (``pymv-cli``), the YAML/JSON config loader
that backs ``--config``, and the experiment tracker that writes run
artifacts to disk.

Command line
------------

.. automodule:: pymv.cli
   :members: main, build_parser
   :undoc-members:

Config files
------------

.. automodule:: pymv.config
   :members:
   :undoc-members:

Built-in configs are shipped inside the package, so the shorthand form
resolves from any working directory:

.. code-block:: bash

   pymv-cli --config default
   pymv-cli --config experiment
   pymv-cli --config random_search

Experiment tracking
-------------------

.. autoclass:: pymv.tracking.ExperimentTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
