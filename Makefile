.PHONY: release

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