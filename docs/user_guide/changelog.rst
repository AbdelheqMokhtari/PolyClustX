Changelog
=========

Unreleased — unified scientific workbench
-----------------------------------------

* Integrated optional ``pymv.display`` interface in the ``pymv-clustering``
  distribution; launch with ``pymv-display`` or ``python -m pymv.display``.
* Independent session experiment history, editable recipes, manual K sweeps,
  benchmark toggle, internal evaluation, and downloadable results/plots.
* Explicit file mapping, optional labels, sample IDs, observation masks,
  constrained missing-view simulation and explicit mean/zero imputation.
* Multiple graph relations per view with weighted-sum or Hadamard early fusion.
* Canonical dataset families with ``variant=`` and deprecated artifact aliases.
* Sphinx Read the Docs theme, shared provisional logo, generated catalogs,
  scientific background and contribution tutorials.

Migration notes
~~~~~~~~~~~~~~~

The old ``pymv_display`` import path is replaced by ``pymv.display``. Install
``pymv-clustering[display]`` instead of a separate interface package.

Graph lists must match the number of feature views. Repeat a shared graph
explicitly (``As=[A, A]``); implicit reuse of the last adjacency was removed.
``list_datasets()`` lists families; use ``include_aliases=True`` for old names.
The ``multimodal`` category now means feature-only multimodal data; graph-bearing
multimodal datasets are ``attributed_graph_multimodal``.

Current algorithms still need complete or explicitly imputed views. Retain the
dataset object and call ``data.fit(model)`` to keep missing-mask safeguards.
Session history is not durable storage; export before restarting the interface.

.. include:: ../../CHANGELOG.md
   :parser: myst_parser.sphinx_
