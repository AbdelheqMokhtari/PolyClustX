Interactive interface API
=========================

.. autofunction:: polyclustx.display.launch

The workspace service is independent of Streamlit and can also be used from
Python. Its records are session-local; exports do not persist original data.

.. autoclass:: polyclustx.display.workspace.ExperimentSpec
   :members:

.. autoclass:: polyclustx.display.workspace.Workspace
   :members:

.. autoclass:: polyclustx.display.preprocessing.PreprocessingSpec
   :members:

.. autoclass:: polyclustx.display.preprocessing.ViewPreprocessing
   :members:

.. autofunction:: polyclustx.display.profiles.dataset_profile

.. autoclass:: polyclustx.display.synthetic.SyntheticSpec
   :members:

.. autoclass:: polyclustx.display.synthetic.SyntheticGraphSpec
   :members:

.. autofunction:: polyclustx.display.synthetic.generate_synthetic

.. autofunction:: polyclustx.display.workspace.export_record

.. autofunction:: polyclustx.display.workspace.fingerprint

.. autofunction:: polyclustx.display.exporting.dataset_mat
