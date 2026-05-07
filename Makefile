.ONESHELL:
SHELL := /bin/bash

.PHONY: help install install-pip install-poetry env ingest ingest-all run run-chatbot benchmark benchmark-single clean-db clean-chroma

CONFIG ?= configs/baseline.yaml
DATA_DIR ?= data/docs
EVAL_DATASET ?= data/eval/dataset.json
RESULTS_DIR ?= results
PORT ?= 8501
ADDRESS ?= 0.0.0.0
EVALUATOR ?= openai
EVALUATOR_MODEL ?= gpt-4o-mini

help:
	@echo "Targets:"
	@echo "  install              Install deps (pip)"
	@echo "  ingest               Ingest using CONFIG=$(CONFIG) from DATA_DIR=$(DATA_DIR)"
	@echo "  ingest-all           Ingest for all configs in configs/"
	@echo "  benchmark            Run RAGAS benchmark on all configs"
	@echo "  benchmark-single     Run RAGAS benchmark on CONFIG=$(CONFIG)"
	@echo "  run                  Run Streamlit (PORT=$(PORT) ADDRESS=$(ADDRESS))"
	@echo "  clean-db             Remove local ChromaDB folder (./chroma_db)"
	@echo ""
	@echo "Variables:"
	@echo "  EVALUATOR=$(EVALUATOR)        Evaluator LLM provider (openai/ollama)"
	@echo "  EVALUATOR_MODEL=$(EVALUATOR_MODEL)  Evaluator model name"
	@echo "  DATA_DIR=$(DATA_DIR)        Source PDFs directory"
	@echo "  EVAL_DATASET=$(EVAL_DATASET)   Evaluation dataset path"

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ingest:
	PYTHONPATH=. .venv/bin/python3 scripts/ingest.py --config $(CONFIG) --data-dir $(DATA_DIR)

ingest-all:
	@set -e; \
	for f in configs/*.yaml; do \
		echo "==> ingesting $$f"; \
		$(MAKE) ingest CONFIG=$$f DATA_DIR=$(DATA_DIR); \
	done

run-chatbot:
	.venv/bin/streamlit run src/app/chatbot.py --server.port $(PORT) --server.address $(ADDRESS)

run: run-chatbot

ingest-and-run: ingest run-chatbot

benchmark:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark.py \
		--eval-dataset $(EVAL_DATASET) \
		--config-dir configs/ \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL)

benchmark-single:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark.py \
		--eval-dataset $(EVAL_DATASET) \
		--config $(CONFIG) \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL)

clean-db:
	rm -rf ./chroma_db

clean-chroma: clean-db
