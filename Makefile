.ONESHELL:
SHELL := /bin/bash

.PHONY: help install install-pip install-poetry env ingest ingest-all run run-chatbot run-chatbot-py benchmark benchmark-generation clean-db clean-chroma

CONFIG ?= configs/baseline.yaml
DATA_DIR ?= data/raw
EVAL_DATASET ?= data/eval/golden_dataset.json
RESULTS_DIR ?= results
PORT ?= 8501
ADDRESS ?= 0.0.0.0
FORCE ?= 0

help:
	@echo "Targets:"
	@echo "  install                    Install deps (prefers poetry; falls back to pip)"
	@echo "  ingest                     Ingest using CONFIG=$(CONFIG) from DATA_DIR=$(DATA_DIR)"
	@echo "  ingest-all                 Ingest for all configs in configs/"
	@echo "  run                        Run Streamlit (PORT=$(PORT) ADDRESS=$(ADDRESS))"
	@echo "  clean-db                   Remove local ChromaDB folder (./chroma_db)"
	@echo ""
	@echo "Notes:"
	@echo "  - Run make targets from the repository root (same folder as this Makefile)."
	@echo "  - Override Streamlit port/address: make run-chatbot PORT=8501 ADDRESS=0.0.0.0"
	@echo "  - Aliases kept: run-chatbot, clean-chroma"

install: install-poetry
	@true

install-pip:
	pip install -r requirements.txt

install-poetry:
	poetry install

ingest:
	python -c "from src.core.pipeline import RAGPipeline; from src.ingestion import PDFParser, TextPreprocessor; p=PDFParser(); docs=p.parse_directory('$(DATA_DIR)'); pre=TextPreprocessor(); [setattr(d,'content', pre.preprocess(d.content)) for d in docs]; pipe=RAGPipeline.from_config('$(CONFIG)'); pipe.ingest_documents(docs); print('ingested_docs=', len(docs))"

ingest-all:
	@set -e; \
	for f in configs/*.yaml; do \
		echo "==> ingesting $$f"; \
		$(MAKE) ingest CONFIG=$$f DATA_DIR=$(DATA_DIR); \
	done

run-chatbot:
	streamlit run src/app/chatbot.py --server.port $(PORT) --server.address $(ADDRESS)

run: run-chatbot

run-chatbot-py:
	PYTHONPATH=. streamlit run src/app/chatbot.py --server.port $(PORT) --server.address $(ADDRESS)

ingest-and-run: ingest run-chatbot

benchmark:
	python -c "from src.evaluation import Benchmark; b=Benchmark('$(EVAL_DATASET)'); df=b.run_all_configs('configs/'); agg=b.aggregate_results(df); import pathlib; pathlib.Path('$(RESULTS_DIR)').mkdir(parents=True, exist_ok=True); b.save_results(agg,'$(RESULTS_DIR)/benchmark_results.csv')"

benchmark-generation:
	python -c "from src.evaluation import Benchmark; b=Benchmark('$(EVAL_DATASET)'); df=b.run_all_configs('configs/', include_generation=True); agg=b.aggregate_results(df); import pathlib; pathlib.Path('$(RESULTS_DIR)').mkdir(parents=True, exist_ok=True); b.save_results(agg,'$(RESULTS_DIR)/benchmark_results_with_generation.csv')"


clean-db:
	rm -rf ./chroma_db

clean-chroma: clean-db
