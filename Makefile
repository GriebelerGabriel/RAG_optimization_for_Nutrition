.ONESHELL:
SHELL := /bin/bash

.PHONY: help install install-pip install-poetry env ingest ingest-all run run-chatbot benchmark benchmark-single benchmark-all benchmark-eval benchmark-parallel benchmark-deterministic benchmark-report benchmark-unified-single benchmark-with-no-rag benchmark-no-rag-only clean-db clean-chroma

CONFIG ?= configs/baseline.yaml
DATA_DIR ?= data/docs
EVAL_DATASET ?= data/eval/dataset.json
RESULTS_DIR ?= results
PORT ?= 8501
ADDRESS ?= 0.0.0.0
EVALUATOR ?= openrouter
EVALUATOR_MODEL ?= google/gemini-2.5-flash-preview

help:
	@echo "Targets:"
	@echo "  install              Install deps (pip)"
	@echo "  ingest               Ingest using CONFIG=$(CONFIG) from DATA_DIR=$(DATA_DIR)"
	@echo "  ingest-all           Ingest for all configs in configs/"
	@echo "  benchmark            Run RAGAS benchmark on all configs"
	@echo "  benchmark-single     Run RAGAS benchmark on CONFIG=$(CONFIG)"
	@echo "  benchmark-all        Unified: pipeline + RAGAS + deterministic + report"
	@echo "  benchmark-eval       Evaluate cached pipeline: deterministic + RAGAS + report"
	@echo "  benchmark-parallel   Same as benchmark-eval but 3 parallel batches (faster)"
	@echo "  benchmark-deterministic  Deterministic NLP metrics only (no LLM cost)"
	@echo "  benchmark-report     Re-generate report from existing CSVs"
	@echo "  benchmark-with-no-rag  Run full benchmark including no-RAG baselines"
	@echo "  benchmark-no-rag-only  Run only no-RAG configs (LLM without retrieval)"
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
	for f in $$(find configs/ -name '*.yaml' | sort); do \
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

benchmark-all:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--eval-dataset $(EVAL_DATASET) \
		--config-dir configs/ \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL)

benchmark-eval:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL) \
		--skip-pipeline

benchmark-parallel:
	./scripts/ragas_parallel.sh

benchmark-deterministic:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--eval-dataset $(EVAL_DATASET) \
		--config-dir configs/ \
		--results-dir $(RESULTS_DIR) \
		--skip-ragas --skip-pipeline

benchmark-report:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--results-dir $(RESULTS_DIR) \
		--skip-pipeline --skip-ragas --skip-deterministic

benchmark-with-no-rag: benchmark-all

benchmark-no-rag-only:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--eval-dataset $(EVAL_DATASET) \
		--config configs/fair_comparison/no_rag_llama8b.yaml \
		--config configs/fair_comparison/no_rag_gpt35_turbo.yaml \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL)

benchmark-unified-single:
	PYTHONPATH=. .venv/bin/python3 scripts/benchmark_unified.py \
		--eval-dataset $(EVAL_DATASET) \
		--config $(CONFIG) \
		--results-dir $(RESULTS_DIR) \
		--evaluator $(EVALUATOR) \
		--evaluator-model $(EVALUATOR_MODEL)
