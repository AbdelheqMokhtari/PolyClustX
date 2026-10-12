# Propose a method or contribute code

To request a method without implementing it, open a
[feature request](https://github.com/AbdelheqMokhtari/pymv/issues/new) with the
paper citation, reference-code link, code license, required data settings,
dependencies and expected behavior. Explain whether it uses graphs and whether
it genuinely handles incomplete views.

For an implementation:

1. Fork the repository and create a focused feature branch.
2. Create and activate an environment using the [installation guide](../user_guide/installation.rst),
   then install `python -m pip install -e .` and the check tools with
   `python -m pip install pytest ruff build` and
   `python -m pip install -r docs/requirements.txt`.
3. Implement the model contract and registry metadata; keep optional heavy
   dependencies isolated where possible.
4. Preserve applicable third-party notices and document implementation origins.
   A paper link alone is not a source-code license.
5. Add synthetic tests, a usage example and algorithm background/citation.
6. Run `ruff check src/ tests/`, `pytest`, and
   `sphinx-build -W --keep-going -b html docs docs/_build/html`.
7. Push your branch to your fork and open a pull request targeting `develop`;
   describe behavior changes and reproducibility
   evidence. Do not claim a reproduced paper table without the corresponding run.

Keep commits focused. Include expected limitations, differences from the paper,
performance/memory considerations, and any new dataset requirements. Maintainers
review before inclusion; registration in a local session is not publication.

## Bug fixes, datasets and documentation

For a bug, add a regression test that fails before your fix. Include a minimal
reproducer, affected versions and the expected behavior. For a dataset, follow
the [offline fixture and variant example](datasets.md); for documentation,
use the [file-by-file guide](documentation.md). Pure documentation corrections
do not need a new Python test, but must pass the documentation build.

All PRs receive lint, core and interface tests, distribution checks, and a
strict Sphinx build. Download the **documentation-html** artifact in GitHub
Actions to inspect the preview. Test XML and built distributions are also
available. New test files are discovered automatically; network downloads are
not part of routine CI. No publishing credentials are needed for fork PRs.

Use focused conventional commits, describe compatibility changes, and respond
to review in the same branch. Maintainers handle merging, version updates and
PyPI publication; a contributor should not run `make release`.
