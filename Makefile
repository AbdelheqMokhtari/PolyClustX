.PHONY: help guide cli run search list-datasets list-models \
        install test lint fix build docs clean version bump release

# ── Help ──────────────────────────────────────────────────────────

help:            ## Show this list of targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────

install:         ## Editable install with dev + docs extras
	pip install -e ".[dev,docs]"

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

docs:            ## Build the HTML documentation
	python -m sphinx -b html docs docs/_build/html

clean:           ## Remove build, cache and docs artifacts
	rm -rf build dist *.egg-info src/*.egg-info docs/_build .pytest_cache .ruff_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ── CLI helpers ───────────────────────────────────────────────────

guide:           ## Show the pymv-cli quick-start guide
	@pymv-cli

cli:             ## Run pymv-cli with forwarded arguments   ->  make cli ARGS="--dataset acm"
	@pymv-cli $(ARGS) || { \
		echo ""; \
		echo "------------------------------------------------------------"; \
		echo "Error: Something went wrong with pymv-cli"; \
		echo ""; \
		echo "Try one of:"; \
		echo "  make guide               # quick-start guide"; \
		echo "  make list-datasets       # available datasets"; \
		echo "  make list-models         # available models"; \
		echo "  pymv-cli --help          # full usage"; \
		echo "------------------------------------------------------------"; \
		exit 1; \
	}

run:             ## Run an experiment   ->  make run DATASET=acm BETA=2 RUNS=3
	@pymv-cli --dataset $(DATASET) \
		$(if $(BETA),--beta $(BETA)) \
		$(if $(TEMPERATURE),--temperature $(TEMPERATURE)) \
		$(if $(TFIDF),--tf-idf $(TFIDF)) \
		$(if $(RUNS),--runs $(RUNS)) \
		$(if $(METRICS),--metrics $(METRICS)) \
		$(if $(SAVE),--save) \
	|| { \
		echo ""; \
		echo "Error: Experiment failed."; \
		echo "  Use  make list-datasets  to check the dataset name."; \
		echo "  Use  pymv-cli --help     for the full list of options."; \
		exit 1; \
	}

search:          ## Run a random search ->  make search DATASET=acm TRIALS=20
	@pymv-cli --dataset $(DATASET) --random-search \
		$(if $(TRIALS),--n-trials $(TRIALS)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(METRICS),--metrics $(METRICS)) \
		$(if $(SAVE),--save) \
	|| { \
		echo ""; \
		echo "Error: Random search failed."; \
		echo "  Use  make list-datasets  to check the dataset name."; \
		echo "  Use  pymv-cli --help     for the full list of options."; \
		exit 1; \
	}

list-datasets:   ## List all available datasets
	@pymv-cli --list-datasets

list-models:     ## List all available models
	@pymv-cli --list-models

# ── Release ───────────────────────────────────────────────────────

version:         ## Show the current library version
	@pymv-cli --version

bump:            ## Bump version using commitizen (updates all version files + CHANGELOG.md)
	@echo "Bumping version with commitizen..."
	cz bump
	@echo ""
	@echo "Version bumped. Files updated:"
	@echo "  - pyproject.toml  (project.version + tool.commitizen.version)"
	@echo "  - src/pymv/__init__.py  (__version__)"
	@echo "  - CHANGELOG.md"
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

	@echo "Bumping version (commitizen -> all version files + CHANGELOG.md)"
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
	@pymv-cli --version