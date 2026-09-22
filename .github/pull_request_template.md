## What changed?

Describe the user-facing behavior and link the issue, if any.
Target `develop`; stable releases are handled by maintainers.

## Contribution type

- [ ] Model / method
- [ ] Dataset / variant / reader
- [ ] Bug fix
- [ ] Documentation / interface / tooling

## Evidence

- [ ] Added or updated small offline tests; new network tests use `@pytest.mark.network`.
- [ ] Ran `ruff check src tests` and `pytest -m 'not network'`.
- [ ] Updated the relevant guide/API/registry metadata and checked the Sphinx build.
- [ ] Added a runnable example when introducing a workflow.
- [ ] No datasets, private information, experiment outputs or screenshots committed.

For methods: original paper/code links, license/notices, supported views/graphs/missingness,
parameter limits, seed behavior and deviations from the paper:

For datasets: original reference and terms, artifact URL, layout, sample alignment,
modalities, graph relations and variant mappings:

## Limitations and compatibility

Mention memory/performance effects, new dependencies and behavior changes.
Describe actual reproduction evidence; do not claim unverified published scores.
