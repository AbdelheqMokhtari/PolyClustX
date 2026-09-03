Installation
============

Requirements
------------

- **Python ≥ 3.8**
- Core: ``numpy``, ``scipy``, ``scikit-learn``, ``networkx``, ``pandas``
- Model backend: ``tensorflow ≥ 2.10``
- Optional: ``matplotlib``, ``seaborn`` (for visualization)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/pymv.git
   cd pymv
   pip install -e .

This installs ``pymv`` in *editable* mode — any code changes take effect
immediately without reinstalling.

TensorFlow
----------

The LMGEC training loop uses TensorFlow.  If it's not already installed:

.. code-block:: bash

   # CPU-only
   pip install "tensorflow>=2.10"

   # GPU (NVIDIA)
   pip install "tensorflow[and-cuda]>=2.10"

Optional dependencies
---------------------

.. code-block:: bash

   # Visualization
   pip install matplotlib seaborn

   # Documentation (for contributors)
   pip install sphinx furo myst-parser sphinx-copybutton sphinx-autodoc-typehints

Verify
------

.. code-block:: python

   import pymv
   print(pymv.__version__)   # e.g. "1.0.0"

   from pymv.data import list_datasets
   print(list_datasets())    # ['acm', 'amazon_photos', ...]
