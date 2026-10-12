Installation
============

Use Python **3.10+** (3.11 recommended). Both installation methods include
the Python API, command line, interface and Optuna. No separate interface
installation is needed.

From PyPI
---------

When the release is published on PyPI:

.. code-block:: bash

   python -m pip install polyclustx

From GitHub or a downloaded ZIP
-------------------------------

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/PolyClustX.git
   cd PolyClustX
   python -m pip install .

For a ZIP download, extract it and open a terminal in the folder containing
``pyproject.toml``, then run ``python -m pip install .``.

Launch
------

.. code-block:: bash

   polyclustx-display

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
If the launch command is not found, try ``python -m polyclustx.display``.

Compute devices
---------------

PyTorch is installed automatically. Models accept ``device="auto"``, ``"cpu"``
or ``"cuda"``. GPU use needs a compatible NVIDIA driver and PyTorch build;
see the `official PyTorch installer <https://pytorch.org/get-started/locally/>`_
for platform-specific options. TensorFlow is not required.

The interface launcher defaults CPU worker pools to two threads to reduce
overhead on small minibatches. Set ``POLYCLUSTX_CPU_THREADS`` before launching to
choose another default. Explicit ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS`` settings take precedence. Python API use leaves your
process-wide thread settings unchanged.

Optional UMAP plots use ``python -m pip install umap-learn``.
Dataset downloads are cached in ``~/.polyclustx_datasets``; set ``POLYCLUSTX_DATA_DIR``
to choose another directory.

Migrating from pymv
-------------------

PolyClustX is the new project name. Install ``polyclustx`` (or the renamed
repository), replace ``from pymv...`` with ``from polyclustx...`` in your code,
and launch with ``polyclustx-display``. The old ``pymv`` namespace is not shipped
as an alias; numerical model behavior is unchanged by the rename.

Rename environment variables to ``POLYCLUSTX_DATA_DIR`` and
``POLYCLUSTX_CPU_THREADS``. To reuse an existing dataset cache without moving or
downloading data, point ``POLYCLUSTX_DATA_DIR`` at your old ``.pymv_datasets``
directory. New JSON reports use ``polyclustx_version`` instead of
``pymv_version``; consumers reading that metadata key should accept either
name for older reports. Existing report files are not rewritten.

Stop any running interface after exporting its results, then start a fresh
session with the new command. Do not reuse pickled estimators or workspace
objects across the package rename; retain the source data and exported reports.
