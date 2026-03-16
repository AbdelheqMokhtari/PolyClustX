# Contributing Guide

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
