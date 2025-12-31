.PHONY: help install-pip install-poetry env ingest run-chatbot run-chatbot-py benchmark benchmark-generation clean-chroma

CONFIG ?= configs/baseline.yaml
DATA_DIR ?= data/raw
EVAL_DATASET ?= data/eval/golden_dataset.json
RESULTS_DIR ?= results

help:
	@echo "Targets:"
	@echo "  install-pip                Install dependencies with pip"
	@echo "  install-poetry             Install dependencies with poetry"
	@echo "  ingest CONFIG=...          Ingest PDFs from $(DATA_DIR) into ChromaDB using CONFIG"
	@echo "  run-chatbot                Run Streamlit chatbot"
	@echo "  ingest-and-run             Ingest then run Streamlit chatbot"
	@echo "  benchmark                  Run retrieval-only benchmark for all configs"
	@echo "  benchmark-generation       Run benchmark incl. generation for all configs"
	@echo "  clean-chroma               Remove local ChromaDB folder (./chroma_db)"

install-pip:
	pip install -r requirements.txt

install-poetry:
	poetry install

ingest:
	python -c "from src.core.pipeline import RAGPipeline; from src.ingestion import PDFParser, TextPreprocessor; p=PDFParser(); docs=p.parse_directory('$(DATA_DIR)'); pre=TextPreprocessor(); [setattr(d,'content', pre.preprocess(d.content)) for d in docs]; pipe=RAGPipeline.from_config('$(CONFIG)'); pipe.ingest_documents(docs); print('ingested_docs=', len(docs))"

run-chatbot:
	streamlit run src/app/chatbot.py

run-chatbot-py:
	PYTHONPATH=. streamlit run src/app/chatbot.py

ingest-and-run: ingest run-chatbot

benchmark:
	python -c "from src.evaluation import Benchmark; b=Benchmark('$(EVAL_DATASET)'); df=b.run_all_configs('configs/'); agg=b.aggregate_results(df); import pathlib; pathlib.Path('$(RESULTS_DIR)').mkdir(parents=True, exist_ok=True); b.save_results(agg,'$(RESULTS_DIR)/benchmark_results.csv')"

benchmark-generation:
	python -c "from src.evaluation import Benchmark; b=Benchmark('$(EVAL_DATASET)'); df=b.run_all_configs('configs/', include_generation=True); agg=b.aggregate_results(df); import pathlib; pathlib.Path('$(RESULTS_DIR)').mkdir(parents=True, exist_ok=True); b.save_results(agg,'$(RESULTS_DIR)/benchmark_results_with_generation.csv')"

clean-chroma:
	rm -rf ./chroma_db
