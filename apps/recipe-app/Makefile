# =============================================================================
# Makefile for Recipe App
#
# Common development tasks. Run `make help` to see available targets.
# Introduced: session 12
# =============================================================================

.PHONY: dev test seed migrate clean setup docker-up docker-down lint help

# ---- Development ----

dev:  ## Start dev server with auto-reload (Node 18+ --watch)
	node --watch server.js

test:  ## Run test suite
	npx jest --verbose

lint:  ## Lint source files
	npx eslint src/ --ext .js

# ---- Database ----

migrate:  ## Run database migrations
	node migrate.js

seed:  ## Populate database with sample recipes
	node seeds/seed.js

clean:  ## Delete the local SQLite database
	rm -f recipes.db

setup: migrate seed  ## Fresh database with sample data
	@echo "Database ready with sample data"

# ---- Docker ----

docker-up:  ## Start containerized app
	docker-compose up -d

docker-down:  ## Stop containerized app
	docker-compose down

# ---- Help ----

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
