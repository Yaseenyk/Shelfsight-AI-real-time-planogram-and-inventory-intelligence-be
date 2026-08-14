# ShelfSight AI — handover commands.
#
#   make setup      one-time local install (venv + dependencies + database)
#   make build      build both Docker images
#   make run        start the stack   (dashboard http://localhost:3000)
#   make logs       follow the logs
#   make stop       stop the stack, keeping data
#   make evaluate   run the full benchmark suite and publish paper figures
#
# Every target is safe to re-run. `make reset` is the only destructive one and
# asks before deleting anything.

SHELL := /bin/sh
COMPOSE := docker compose
PYTHON ?= python
VENV := .venv
VENV_PY := $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
	VENV_PY := $(VENV)/Scripts/python.exe
endif

.DEFAULT_GOAL := help
.PHONY: help setup install db build run run-llm stop restart logs ps health \
        evaluate export curate train-freshness test lint clean reset

help:  ## Show this help
	@echo "ShelfSight AI"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "First time?  make setup   (local)   or   make build && make run   (Docker)"

# ------------------------------------------------------------------ local --
setup: install db  ## One-time local setup: venv, dependencies, seeded database
	@echo ""
	@echo "Ready. Start the API with:  make serve"

install:  ## Create the virtualenv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	$(VENV_PY) -m pip install -r requirements-ml.txt

db:  ## Create and seed the database (safe on an existing one)
	$(VENV_PY) -m app.db.init_db --seed

serve:  ## Run the API locally with reload
	$(VENV_PY) -m uvicorn app.main:app --reload --port 8000

# ----------------------------------------------------------------- docker --
build:  ## Build the backend and frontend images
	$(COMPOSE) build

run:  ## Start the stack in the background
	$(COMPOSE) up -d
	@echo ""
	@echo "  Dashboard : http://localhost:3000"
	@echo "  API docs  : http://localhost:8000/docs"
	@echo ""
	@echo "The backend needs ~1-2 minutes on first boot (loading models)."
	@echo "Watch it with:  make logs"

run-llm:  ## Start the stack including a local Ollama container
	$(COMPOSE) --profile llm up -d
	@echo "Pull a model once Ollama is up:  make pull-model"

pull-model:  ## Download the LLM used for insight briefings
	$(COMPOSE) exec ollama ollama pull $${OLLAMA_MODEL:-llama3.2}

stop:  ## Stop the stack (data is kept)
	$(COMPOSE) down

restart: stop run  ## Restart the stack

logs:  ## Follow container logs
	$(COMPOSE) logs -f --tail=100

ps:  ## Show container status
	$(COMPOSE) ps

health:  ## Check that the API is answering
	@curl -fsS http://localhost:8000/health || echo "backend not responding yet"

# ------------------------------------------------------------- ml / paper --
evaluate:  ## Run all benchmarks and publish figures to docs/publication_metrics/
	$(VENV_PY) models/export_pipeline.py metrics --suites all

export:  ## Export models to ONNX and measure the CPU speed-up
	$(VENV_PY) models/export_pipeline.py export --format onnx --benchmark

curate:  ## List the pinned public datasets available to download
	$(VENV_PY) tools/dataset_curator.py list

train-freshness:  ## Fine-tune the freshness classifier (DATA=<dataset dir>)
	@test -n "$(DATA)" || (echo "usage: make train-freshness DATA=data/freshness"; exit 1)
	$(VENV_PY) models/train_freshness.py --data-dir $(DATA) --epochs $${EPOCHS:-25}

# ------------------------------------------------------------------ dev ----
test:  ## Run the test suite
	$(VENV_PY) -m pytest -q

lint:  ## Lint the codebase
	$(VENV_PY) -m ruff check .

clean:  ## Remove caches and build artefacts (keeps data and weights)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ 2>/dev/null || true

reset:  ## DESTRUCTIVE: delete containers, volumes, database and weights
	@echo "This deletes the database, uploads and all trained weights."
	@printf "Type 'yes' to continue: " && read answer && [ "$$answer" = "yes" ]
	$(COMPOSE) down -v
	rm -f shelfsight.db shelfsight.db-wal shelfsight.db-shm
	@echo "Reset complete. Run 'make run' for a clean system."
