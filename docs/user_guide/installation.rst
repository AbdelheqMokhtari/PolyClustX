Installation
============

Requirements
------------

- **Python ≥ 3.9**
- Core: ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``,
  ``requests``, ``pyyaml``, ``tqdm``
- Model backend: ``tensorflow ≥ 2.10``

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

Declared as extras in ``pyproject.toml``:

.. code-block:: bash

   pip install -e ".[dev]"     # pytest, coverage, ruff, build
   pip install -e ".[docs]"    # sphinx, furo, myst-parser, copybutton
   pip install -e ".[umap]"    # UMAP projection for plot_clusters

Verify
------

.. code-block:: python

   import pymv
   print(pymv.__version__)   # e.g. "1.1.0"

   from pymv.data import list_datasets
   print(list_datasets())    # ['acm', 'amazon_photos', ...]
