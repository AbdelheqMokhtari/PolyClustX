Interactive interface API
=========================

.. autofunction:: pymv.display.launch

The workspace service is independent of Streamlit and can also be used from
Python. Its records are session-local; exports do not persist original data.

.. autoclass:: pymv.display.workspace.ExperimentSpec
   :members:

.. autoclass:: pymv.display.workspace.Workspace
   :members:

.. autoclass:: pymv.display.preprocessing.PreprocessingSpec
   :members:

.. autoclass:: pymv.display.preprocessing.ViewPreprocessing
   :members:

.. autofunction:: pymv.display.profiles.dataset_profile

.. autoclass:: pymv.display.synthetic.SyntheticSpec
   :members:

.. autoclass:: pymv.display.synthetic.SyntheticGraphSpec
   :members:

.. autofunction:: pymv.display.synthetic.generate_synthetic

.. autofunction:: pymv.display.workspace.export_record

.. autofunction:: pymv.display.workspace.fingerprint
