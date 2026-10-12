Installation
============

Use Python **3.10+** (3.11 recommended). Both installation methods include
the Python API, command line, interface and Optuna. No separate interface
installation is needed.

From PyPI
---------

When the release is published on PyPI:

.. code-block:: bash

   python -m pip install pymv-clustering

From GitHub or a downloaded ZIP
-------------------------------

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/pymv.git
   cd pymv
   python -m pip install .

For a ZIP download, extract it and open a terminal in the folder containing
``pyproject.toml``, then run ``python -m pip install .``.

Launch
------

.. code-block:: bash

   pymv display
   # The existing command also works:
   pymv-display

Open http://localhost:8501, or the address printed in the terminal.
Keep the terminal open; Ctrl+C stops the server. Launch again with the same
command. Download session results before stopping. See :doc:`workbench`.

Optional: an isolated environment
---------------------------------

Before installing, create an environment to keep dependencies separate:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

On Windows PowerShell, activate with ``.venv\Scripts\Activate.ps1`` instead.
Activate the same environment before launching again; no reinstall is needed.
If the launch command is not found, try ``python -m pymv.display``.

Compute devices
---------------

PyTorch is installed automatically. Models accept ``device="auto"``, ``"cpu"``
or ``"cuda"``. GPU use needs a compatible NVIDIA driver and PyTorch build;
see the `official PyTorch installer <https://pytorch.org/get-started/locally/>`_
for platform-specific options. TensorFlow is not required.

The interface launcher defaults CPU worker pools to two threads to reduce
overhead on small minibatches. Set ``PYMV_CPU_THREADS`` before launching to
choose another default. Explicit ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS`` settings take precedence. Python API use leaves your
process-wide thread settings unchanged.

Optional UMAP plots use ``python -m pip install umap-learn``.
Dataset downloads are cached in ``~/.pymv_datasets``; set ``PYMV_DATA_DIR``
to choose another directory.
