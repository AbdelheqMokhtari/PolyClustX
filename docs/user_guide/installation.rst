Installation
============

Use Python 3.11 for the steps below. The core library supports Python 3.9+;
the interactive interface requires Python 3.10+ for its Streamlit version.
Check your installation with ``python --version`` (``py --version`` on Windows).

Install from PyPI
-----------------

1. Open a terminal in the folder where you want to keep your experiments.
2. Create an isolated environment:

.. code-block:: bash

   python -m venv .venv

3. Activate it on macOS or Linux:

.. code-block:: bash

   source .venv/bin/activate

   # Windows PowerShell instead:
   .venv\Scripts\Activate.ps1
   # Windows Command Prompt instead:
   .venv\Scripts\activate.bat

4. Install the library and interface:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install "pymv-clustering[ui]"

For Python-only use, install ``pymv-clustering`` without ``[ui]``.
The package name on PyPI is **pymv-clustering**; imports use **pymv**.
``[ui]`` selects the interface dependencies. The older ``[display]`` option
remains an equivalent alias; no separate package is needed.

Interactive interface
---------------------

.. code-block:: bash

   pymv-display
   # Equivalent:
   python -m pymv.display

Open the address printed in the terminal, normally http://localhost:8501.
Keep the terminal open while using the interface. Press Ctrl+C to stop it.
To restart later, open the same folder, activate ``.venv`` again, and run
``pymv-display``. You do not need to reinstall each time. If the command is
not found, use ``python -m pymv.display`` in the activated environment.
Download your results before restarting: interface session history is temporary.
Continue with :doc:`workbench` for a first experiment.

Install a downloaded repository
-------------------------------

Choose **Code → Download ZIP** on GitHub and extract it, or clone with Git:

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/pymv.git
   cd pymv

For a ZIP download, open a terminal in the extracted folder containing
``pyproject.toml`` instead. Then:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   # On Windows use the activation command shown above.
   python -m pip install --upgrade pip
   python -m pip install -e ".[ui]"
   pymv-display

``-e`` uses the code in this folder, so edits take effect without reinstalling.
Keep the folder in place. On your next session, return here, activate the same
environment and launch again. Contributors can additionally install
``python -m pip install -e ".[dev,docs,ui]"``; see
:doc:`/contributing/pull_requests` for tests and pull requests.

Optional projections and cached data
-----------------------------------------

Dataset downloads are cached in ``~/.pymv_datasets``. Set ``PYMV_DATA_DIR``
to choose a different location. Listing the catalog does not download data.

UMAP projections are optional: ``python -m pip install "pymv-clustering[ui,umap]"``
(or ``python -m pip install -e ".[ui,umap]"`` for a checkout).
