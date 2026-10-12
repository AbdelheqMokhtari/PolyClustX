# Contributing Guide

Contributions to models, datasets, bug fixes, documentation and the interface
are welcome. For a substantial method or dataset, open an issue with the paper,
source code/data link and license first. Use the issue templates on GitHub.

## Set up a contribution

1. Fork `AbdelheqMokhtari/pymv` on GitHub and clone your fork.
2. Add the upstream repository and start from `develop`:

   ```bash
   git remote add upstream https://github.com/AbdelheqMokhtari/pymv.git
   git fetch upstream
   git switch -c fix/short-description upstream/develop
   python -m venv .venv
   source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install -e .
   python -m pip install pytest ruff build
   python -m pip install -r docs/requirements.txt
   ```

3. Make focused changes with tests and documentation, then run:

   ```bash
   ruff check src tests
   pytest -m 'not network'
   sphinx-build -W --keep-going -b html docs docs/_build/html
   python -m build
   ```

4. Commit your changes, push the branch to your fork, and open a pull request
   against **develop**. Fill in the PR checklist. Do not bump versions, publish
   packages or merge into `main` as part of an ordinary contribution.

Detailed recipes: [models](docs/contributing/models.md),
[datasets and variants](docs/contributing/datasets.md),
[documentation and screenshots](docs/contributing/documentation.md).
Run `python examples/custom_datasets.py` for a complete offline data example.

## What CI checks

Pull requests run offline core tests, the full suite with interface dependencies,
lint, a strict documentation build, and wheel/sdist validation. Interface checks
cover the supported minimum Streamlit version and the current allowed release.
Review test XML, built distributions and the HTML documentation artifact in
GitHub Actions. A green build is evidence of tested behavior, not paper-result
reproduction or approval to publish.

New tests are discovered automatically. Use tiny local fixtures for new readers;
mark download tests with `@pytest.mark.network`. Live dataset downloads remain
an opt-in maintainer workflow. Fork pull requests need no repository secrets,
and CI does not publish to PyPI, deploy, push or merge contributions.

## Branching Strategy

A simplified GitFlow adapted to a research framework:

```
main ──────────────────────────────────────── (stable, tagged releases)
  └── develop ────────────────────────────── (integration branch)
        ├── feature/<description>
        ├── fix/<description>
        ├── docs/<description>
        └── tests/<description>
```


### Branch rules

| Branch | Purpose | Merge into | PR required? |
|--------|---------|------------|:------------:|
| `main` | Stable snapshots, tagged releases only | — | Yes (from `develop` only) |
| `develop` | Integration — all feature/fix/docs branches merge here | `main` | Yes |
| `feature/<desc>` | New feature or experiment | `develop` | Yes |
| `fix/<desc>` | Bug fix | `develop` | Yes |
| `docs/<desc>` | Documentation only | `develop` | Yes |

> Feature branches may be deleted after merge unless they represent a significant experiment or research milestone.

---

## Commit Messages — Conventional Commits via Commitizen

We use [Conventional Commits](https://www.conventionalcommits.org/) enforced by [Commitizen](https://commitizen-tools.github.io/commitizen/).

### Interactive commit (recommended)

```bash
cz commit
```

### Manual format

```
<type>(<scope>): <short description>

[optional body]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or experiment step |
| `fix` | Bug fix |
| `build` | Dependencies, tooling, packaging |
| `ci` | CI/CD configuration |
| `docs` | Documentation only |
| `refactor` | Code change with no behavior change |
| `test` | Adding or fixing tests |
| `chore` | Misc housekeeping |
| `experiment` | Experimental run / results |

### Examples

```
feat(clustering): add multiview spectral clustering
fix(metrics): correct normalized mutual information
perf(graph): optimize adjacency matrix construction
refactor(loader): simplify dataset parsing
docs(api): document MVKMeans class
test(clustering): add spectral clustering tests
experiment(baseline): evaluate mv-kmeans on BBCSport
```

---

## Versioning & Releases

To cut a new release:

```bash
# 1. On develop, bump version — updates pyproject.toml, CHANGELOG.md, creates commit + tag
cz bump

# 2. Merge develop into main
git checkout main
git merge --no-ff develop -m "release: merge develop into main for vX.Y.Z"

# 3. Push everything
git push origin main && git push origin develop && git push origin --tags
```

Or with the Makefile shortcut (does all three steps):

```bash
make release
```
