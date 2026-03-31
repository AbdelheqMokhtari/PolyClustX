.PHONY: release help cli run search list-datasets list-models

# ── CLI helpers ───────────────────────────────────────────────────

help:            ## Show the pymv-cli quick-start guide
	@pymv-cli

cli:             ## Run pymv-cli with forwarded arguments   →  make cli ARGS="--dataset acm"
	@pymv-cli $(ARGS) || { \
		echo ""; \
		echo "╔══════════════════════════════════════════════════════════╗"; \
		echo "║  \033[1;31mSomething went wrong with pymv-cli\033[0m                      ║"; \
		echo "║                                                          ║"; \
		echo "║  Try one of:                                             ║"; \
		echo "║    make help                # quick-start guide          ║"; \
		echo "║    make list-datasets       # available datasets         ║"; \
		echo "║    make list-models         # available models           ║"; \
		echo "║    pymv-cli --help          # full usage                 ║"; \
		echo "╚══════════════════════════════════════════════════════════╝"; \
		exit 1; \
	}

run:             ## Run an experiment   →  make run DATASET=acm BETA=2 RUNS=3
	@pymv-cli --dataset $(DATASET) \
		$(if $(BETA),--beta $(BETA)) \
		$(if $(TEMPERATURE),--temperature $(TEMPERATURE)) \
		$(if $(TFIDF),--tf-idf $(TFIDF)) \
		$(if $(RUNS),--runs $(RUNS)) \
		$(if $(METRICS),--metrics $(METRICS)) \
		$(if $(SAVE),--save) \
	|| { \
		echo ""; \
		echo "\033[1;31mError:\033[0m Experiment failed."; \
		echo "  Use  make list-datasets  to check the dataset name."; \
		echo "  Use  pymv-cli --help     for the full list of options."; \
		exit 1; \
	}

search:          ## Run a random search →  make search DATASET=acm TRIALS=20
	@pymv-cli --dataset $(DATASET) --random-search \
		$(if $(TRIALS),--n-trials $(TRIALS)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(METRICS),--metrics $(METRICS)) \
		$(if $(SAVE),--save) \
	|| { \
		echo ""; \
		echo "\033[1;31mError:\033[0m Random search failed."; \
		echo "  Use  make list-datasets  to check the dataset name."; \
		echo "  Use  pymv-cli --help     for the full list of options."; \
		exit 1; \
	}

list-datasets:   ## List all available datasets
	@pymv-cli --list-datasets

list-models:     ## List all available models
	@pymv-cli --list-models

# ── Release ───────────────────────────────────────────────────────

release:
	@echo "🚀 Starting release process..."

	@if [ "$$(git branch --show-current)" != "develop" ]; then \
		echo "❌ You must run this from the develop branch"; \
		exit 1; \
	fi

	@echo "📦 Bumping version"
	cz bump

	@echo "🔀 Switching to main"
	git checkout main

	@echo "⬇️ Pulling latest main"
	git pull origin main

	@echo "🔗 Merging develop"
	git merge --no-ff develop -m "release: merge develop into main"

	@echo "⬆️ Pushing changes"
	git push origin main
	git push origin develop
	git push origin --tags

	@echo "✅ Release completed successfully"