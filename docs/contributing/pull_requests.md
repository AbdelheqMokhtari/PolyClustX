# Propose a method or contribute code

To request a method without implementing it, open a
[feature request](https://github.com/AbdelheqMokhtari/pymv/issues/new) with the
paper citation, reference-code link, code license, required data settings,
dependencies and expected behavior. Explain whether it uses graphs and whether
it genuinely handles incomplete views.

For an implementation:

1. Fork the repository and create a focused feature branch.
2. Install `pip install -e ".[dev,docs,display]"`.
3. Implement the model contract and registry metadata; keep optional heavy
   dependencies isolated where possible.
4. Preserve applicable third-party notices and document implementation origins.
   A paper link alone is not a source-code license.
5. Add synthetic tests, a usage example and algorithm background/citation.
6. Run `ruff check src/ tests/`, `pytest`, and
   `sphinx-build -W --keep-going -b html docs docs/_build/html`.
7. Push your branch to your fork and open a pull request. Target the maintainers'
   requested integration branch; describe behavior changes and reproducibility
   evidence. Do not claim a reproduced paper table without the corresponding run.

Keep commits focused. Include expected limitations, differences from the paper,
performance/memory considerations, and any new dataset requirements. Maintainers
review before inclusion; registration in a local session is not publication.
