Installation
============

Python 3.9 or newer is supported by the framework; dependency versions available
for your Python version determine the supported TensorFlow/Streamlit releases.
Python 3.11 is a convenient choice for a new environment.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install pymv-clustering

On Windows, activate with ``.venv\Scripts\activate``.
The distribution is named **pymv-clustering**, while Python imports use **pymv**.

Interactive interface
---------------------

.. code-block:: bash

   pip install "pymv-clustering[display]"
   pymv-display
   # Equivalent:
   python -m pymv.display

The interface is part of the same distribution. No separate pymv-display
package or repository is required. See :doc:`workbench`.

Development checkout
--------------------

Until this branch is published, install the local checkout to use these changes:

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/pymv.git
   cd pymv
   # Check out the feature branch after its author has pushed it.
   pip install -e ".[dev,docs,display]"
   pytest
   sphinx-build -W --keep-going -b html docs docs/_build/html

Dataset downloads are cached in ``~/.pymv_datasets``. Set ``PYMV_DATA_DIR``
to choose a different location. Listing the catalog does not download data.

UMAP visualizations are optional: ``pip install "pymv-clustering[umap]"``.
