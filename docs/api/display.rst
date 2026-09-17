Interactive workbench API
===========================

.. autofunction:: pymv.display.launch

The workspace service is independent of Streamlit and can also be used from
Python. Its records are session-local; exports do not persist original data.

.. autoclass:: pymv.display.workspace.ExperimentSpec
   :members:

.. autoclass:: pymv.display.workspace.Workspace
   :members:

.. autofunction:: pymv.display.workspace.export_record

.. autofunction:: pymv.display.workspace.fingerprint
