.PHONY: help guide cli run search list-datasets list-models \
        install test lint fix build docs docs-build docs-serve docs-deploy clean version bump release

PYTHON ?= python
PORT ?= 8000
RTD_PROJECT ?= polyclustx
RTD_VERSION ?= latest

# ── Help ──────────────────────────────────────────────────────────

help:            ## Show this list of targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────

install:         ## Install the complete package, including the interface
	$(PYTHON) -m pip install .

test:            ## Run the test suite
	pytest -q $(ARGS)

lint:            ## Lint with ruff
	ruff check src/ tests/

fix:             ## Auto-fix lint issues
	ruff check --fix src/ tests/

format:          ## Format code (Black-compatible)
	ruff format src/ tests/

build:           ## Build wheel and sdist
	python -m build

docs: docs-build  ## Alias for docs-build

docs-build:      ## Install documentation tools and build HTML
	$(PYTHON) -m pip install -r docs/requirements.txt
	$(PYTHON) -m sphinx -W --keep-going -b html docs docs/_build/html

docs-serve:      ## Serve existing HTML without rebuilding at http://127.0.0.1:8000
	@test -f docs/_build/html/index.html || { echo "Run make docs-build first."; exit 1; }
	$(PYTHON) -m http.server $(PORT) --directory docs/_build/html --bind 127.0.0.1

docs-deploy:     ## Trigger Read the Docs; requires READTHEDOCS_TOKEN in the environment
	$(PYTHON) scripts/deploy_docs.py --project $(RTD_PROJECT) --version $(RTD_VERSION)

clean:           ## Remove build, cache and docs artifacts
	rm -rf build dist *.egg-info src/*.egg-info docs/_build .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ── CLI helpers ───────────────────────────────────────────────────

guide:           ## Show short command-line help
	@$(PYTHON) -m polyclustx.command

cli:             ## Forward arguments, e.g. make cli ARGS="models idec"
	@$(PYTHON) -m polyclustx.command $(ARGS)

run:             ## Run a model: make run DATASET=acm MODEL=lmgec ARGS="--runs 3"
	@$(PYTHON) -m polyclustx.command run $(DATASET) --model $(or $(MODEL),multiview_kmeans) $(ARGS)

search:          ## Tune a model: pass --strategy and --vary through ARGS
	@$(PYTHON) -m polyclustx.command search $(DATASET) --model $(or $(MODEL),multiview_kmeans) $(ARGS)

list-datasets:   ## List dataset families
	@$(PYTHON) -m polyclustx.command datasets

list-models:     ## List registered models
	@$(PYTHON) -m polyclustx.command models

# ── Release ───────────────────────────────────────────────────────

version:         ## Show the current library version
	@polyclustx --version

bump:            ## Bump version using commitizen (updates version files)
	@echo "Bumping version with commitizen..."
	cz bump
	@echo ""
	@echo "Version bumped. Files updated:"
	@echo "  - pyproject.toml  (project.version + tool.commitizen.version)"
	@echo "  - src/polyclustx/__init__.py  (__version__)"
	@echo ""
	@echo "Run  make release  to merge into main and push tags."

release:         ## Full release: bump -> merge develop -> main, push tags
	@echo "Starting release process..."

	@if [ "$$(git branch --show-current)" != "develop" ]; then \
		echo "Error: You must run this from the develop branch"; \
		exit 1; \
	fi

	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Working tree is dirty. Commit or stash changes first."; \
		exit 1; \
	fi

	@echo "Bumping version (commitizen -> version files)"
	cz bump

	@echo "Switching to main"
	git checkout main

	@echo "Pulling latest main"
	git pull origin main

	@echo "Merging develop"
	git merge --no-ff develop -m "release: merge develop into main"

	@echo "Pushing changes"
	git push origin main
	git push origin develop
	git push origin --tags

	@echo ""
	@echo "Release completed successfully"
	@polyclustx --version
