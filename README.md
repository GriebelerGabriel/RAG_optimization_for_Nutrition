# RAG Nutrition System

A modular, experiment-driven RAG (Retrieval-Augmented Generation) system for nutritional education. Swap any component (chunker, embedding, retriever, reranker, LLM) via YAML configuration and benchmark to find the optimal setup.

## Data Flow Overview

```
PDFs (data/docs/)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ INGESTION                                                │
│                                                          │
│  PDFParser → TextPreprocessor → Chunker → Embedder      │
│                                      │          │        │
│                                      ▼          ▼        │
│                              ChromaDB (chroma_db/)       │
│                              + BM25 index (if hybrid)    │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ QUERY (chatbot / benchmark)                              │
│                                                          │
│  User Question                                           │
│       ▼                                                  │
│  Retriever (naive / hyde / hybrid)                       │
│       ▼                                                  │
│  Reranker (optional)                                     │
│       ▼                                                  │
│  LLM → Response                                          │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ EVALUATION                                               │
│                                                          │
│  RAGAS metrics  OR  Custom metrics                       │
│  → results/*.csv + results/artifacts/                    │
└─────────────────────────────────────────────────────────┘
```

Each config gets its own isolated ChromaDB collection (keyed by chunker+embedder hash) so experiments never contaminate each other.

## Quick Start

```bash
# 1. Install
make install

# 2. Set up API keys
cp .env.example .env.local
# Edit .env.local with your OPENAI_API_KEY (required for OpenAI/LiteLLM configs)

# 3. Ingest documents + launch chatbot (one command)
make ingest-and-run CONFIG=configs/baseline.yaml
```

Or step by step:

```bash
make install
make ingest CONFIG=configs/baseline.yaml
make run PORT=8501
```

Run all commands from the repository root (the folder with the `Makefile`).

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make install` | Create `.venv` and install dependencies |
| `make ingest CONFIG=... DATA_DIR=...` | Ingest PDFs for one config |
| `make ingest-all` | Ingest for every `configs/*.yaml` |
| `make run` / `make run-chatbot` | Launch Streamlit chatbot |
| `make ingest-and-run` | Ingest + run in one shot |
| `make benchmark` | Run RAGAS benchmark on all configs |
| `make benchmark-single CONFIG=...` | Run RAGAS benchmark on one config |
| `make clean-db` | Delete the ChromaDB vector store |

**Variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONFIG` | `configs/baseline.yaml` | Config YAML for ingest/benchmark |
| `DATA_DIR` | `data/docs` | Directory containing source PDFs |
| `EVAL_DATASET` | `data/eval/dataset.json` | Evaluation dataset path |
| `EVALUATOR` | `openai` | Evaluator LLM provider (`openai`/`ollama`) |
| `EVALUATOR_MODEL` | `gpt-4o-mini` | Evaluator model name |
| `RESULTS_DIR` | `results` | Output directory for benchmark results |
| `PORT` | `8501` | Streamlit port |
| `ADDRESS` | `0.0.0.0` | Streamlit bind address |

## Ingestion Flow

### Data directory

Place PDF documents in `data/docs/`. The included documents cover Brazilian dietary guidelines, WHO sodium guidelines, and cardiovascular health research.

### Via CLI (recommended)

```bash
PYTHONPATH=. .venv/bin/python3 scripts/ingest.py \
    --config configs/baseline.yaml \
    --data-dir data/docs
```

### Via Python API

```python
from src.core.pipeline import RAGPipeline
from src.ingestion import PDFParser, TextPreprocessor

parser = PDFParser()
docs = parser.parse_directory("data/docs")

preprocessor = TextPreprocessor()
for doc in docs:
    doc.content = preprocessor.preprocess(doc.content)

pipeline = RAGPipeline.from_config("configs/baseline.yaml")
pipeline.ingest_documents(docs)
```

### What happens during ingestion

1. **PDF Parsing** (`src/ingestion/parser.py`) — extracts text and metadata (filename, page count, author) from each PDF using PyMuPDF
2. **Preprocessing** (`src/ingestion/preprocessor.py`) — removes page numbers, headers/footers, normalizes whitespace, filters short lines
3. **Chunking** — splits text into chunks based on `chunker.type` (fixed / recursive / sentence)
4. **Embedding** — generates vector embeddings based on `embedding.type`
5. **Indexing** — stores chunks + embeddings in ChromaDB (`chroma_db/`). If `retriever.type` is `hybrid`, also builds a BM25 sparse index saved as `chroma_db/<collection>__bm25.jsonl`

Each unique combination of chunker type + chunk size + overlap + embedder produces a separate ChromaDB collection, so different configs don't interfere.

## Chatbot Flow

```bash
make run CONFIG=configs/baseline.yaml
```

The Streamlit app (`src/app/chatbot.py`):

1. Shows a config dropdown in the sidebar (all `configs/*.yaml` files)
2. Displays the active config details (chunker, embedding, retriever, reranker, LLM)
3. Accepts questions via chat input
4. Retrieves relevant chunks from ChromaDB
5. Optionally reranks results
6. Generates a response using the configured LLM
7. Shows response with expandable source citations (filename + relevance score)
8. Maintains chat history in session state

## Benchmarking

There are two benchmarking systems: **RAGAS** (external, LLM-judged) and **Custom** (built-in, heuristic + optional LLM-judge).

### Evaluation Dataset

`data/eval/dataset.json` — a RAGAS-style flat JSON array:

```json
[
  {
    "question": "Quais são os grupos alimentares?",
    "answer": "Os grupos alimentares incluem...",
    "contexts": ["Os grupos alimentares incluem cereais..."],
    "ground_truth": "Os grupos alimentares incluem...",
    "metadata": {"type": "single-hop", "difficulty": "easy"}
  }
]
```

Supported fields: `question`, `answer`, `contexts` (reference passages), `ground_truth` (ideal answer), `metadata`.

### RAGAS Benchmark (recommended)

Uses the [RAGAS](https://docs.ragas.io/) framework with LLM-as-judge metrics.

```bash
# All configs
make benchmark

# Single config
make benchmark-single CONFIG=configs/baseline.yaml

# With different evaluator
make benchmark EVALUATOR=openai EVALUATOR_MODEL=gpt-4o
```

Or directly:

```bash
PYTHONPATH=. .venv/bin/python3 scripts/benchmark.py \
    --eval-dataset data/eval/dataset.json \
    --config-dir configs/ \
    --results-dir results \
    --evaluator openai \
    --evaluator-model gpt-4o-mini
```

**RAGAS metrics computed:**

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the retrieved context? |
| AnswerRelevancy | Does the answer address the question? |
| ContextPrecision | Are the retrieved chunks relevant? |
| ContextRecall | Are all necessary facts retrieved? |
| AnswerCorrectness | How close is the answer to the reference? |

**Outputs:**
- `results/ragas_benchmark_results.csv` — aggregated metrics per config
- `results/ragas_detail_<config_name>.csv` — per-question breakdown

### Custom Benchmark

Built-in framework with its own metrics (no RAGAS dependency needed).

```python
from src.evaluation import Benchmark

benchmark = Benchmark("data/eval/dataset.json")

# Retrieval-only (fast)
results = benchmark.run_config("configs/baseline.yaml")
df = benchmark.run_all_configs("configs/")
aggregated = benchmark.aggregate_results(df)
benchmark.save_results(aggregated, "results/custom_results.csv")

# Include generation metrics (slower)
df = benchmark.run_all_configs("configs/", include_generation=True)

# With LLM-as-judge
df = benchmark.run_all_configs("configs/", include_generation=True, use_llm_eval=True)
```

**Custom retrieval metrics:** precision@1, precision@3, precision@5, recall@5, MRR

**Custom generation metrics:** faithfulness, answer correctness, abstain correctness, citation quality

**Artifacts:** When `save_artifacts=True` (default), per-question JSON files are saved to `results/artifacts/<experiment_name>/<timestamp>/` containing retrieved chunks, prompt snapshots, response, and all metrics — useful for debugging hallucinations.

## Configuration

Configs are YAML files in `configs/`. Each config defines all five pipeline components.

```yaml
experiment_name: "baseline"

chunker:
  type: "recursive"          # fixed, recursive, sentence
  chunk_size: 512
  overlap: 50

embedding:
  type: "huggingface"        # e5, mpnet, bge, huggingface, openai
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

retriever:
  type: "naive"              # naive, hyde, hybrid
  top_k: 5
  collection_name: "nutrition_docs"

reranker:
  type: "none"               # none, cross_encoder, bge
  model_name: ""
  top_k: 5

llm:
  type: "litellm"            # openai, ollama, litellm
  model_name: "openai/gpt-4o-mini"
  temperature: 0.3
  max_tokens: 256
  context_template: "[Fonte {i}]: {content}"  # optional
```

All configs share the same default system prompt (defined in `src/core/interfaces.py`). It instructs the LLM to answer based on context, always cite sources as `[Fonte X]`, and only say "Não encontrei" when context is empty.

### Included Experiment Configs

| Config | Chunker | Embedding | Retriever | Reranker | LLM |
|--------|---------|-----------|-----------|----------|-----|
| `baseline` | recursive | MiniLM-L12-v2 | naive | none | gpt-4o-mini |
| `experiment_gpt4o` | recursive | MiniLM-L12-v2 | naive | none | gpt-4o (litellm) |
| `experiment_o3_mini` | recursive | MiniLM-L12-v2 | naive | none | o3-mini (litellm) |
| `experiment_hyde` | recursive | E5-base | hyde | none | gpt-4o-mini |
| `experiment_hybrid` | recursive | E5-base | hybrid | cross-encoder | gpt-4o-mini |
| `experiment_ollama` | recursive | E5-base | naive | none | llama3.1:8b (local) |

## Available Components

### Chunkers

| Type | Description |
|------|-------------|
| `fixed` | Fixed character window with overlap |
| `recursive` | LangChain recursive character splitter |
| `sentence` | Groups N sentences per chunk with sentence-level overlap |

### Embedders

| Type | Default Model | Notes |
|------|---------------|-------|
| `e5` | intfloat/multilingual-e5-base | Adds "query:"/"passage:" prefix automatically |
| `mpnet` | paraphrase-multilingual-mpnet-base-v2 | |
| `bge` | BAAI/bge-m3 | |
| `huggingface` | (configurable) | Generic HuggingFace sentence-transformer |
| `openai` | text-embedding-3-small | Requires `OPENAI_API_KEY` |

### Retrievers

| Type | Description | Storage |
|------|-------------|---------|
| `naive` | Direct query embedding → cosine similarity in ChromaDB | ChromaDB only |
| `hyde` | Generates hypothetical answer via LLM, embeds that instead of the query | ChromaDB (uses naive internally) |
| `hybrid` | BM25 sparse + dense retrieval, combined with Reciprocal Rank Fusion | ChromaDB + BM25 JSONL file |

### Rerankers

| Type | Default Model | Notes |
|------|---------------|-------|
| `none` | — | No reranking |
| `cross_encoder` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder re-scoring |
| `bge` | BAAI/bge-reranker-base | BGE reranker |

### LLMs

All LLM types are powered by [LiteLLM](https://github.com/BerriAI/litellm) under the hood, providing a unified interface with built-in retries and timeout handling.

| Type | Model prefix | Examples | Notes |
|------|-------------|----------|-------|
| `litellm` | any | `openai/gpt-4o`, `ollama/llama3.1:8b`, `anthropic/claude-sonnet-4-20250514` | Full provider list: [docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers) |
| `openai` | `openai/` | `gpt-4o-mini`, `gpt-4o` | Auto-prefixed; requires `OPENAI_API_KEY` |
| `ollama` | `ollama/` | `llama3.1:8b`, `mistral:7b` | Auto-prefixed; requires Ollama running locally (`ollama serve`) |

## Project Structure

```
RAG_Nutrition/
├── configs/                  # Experiment YAML configurations
├── configs_openai/           # OpenAI-only config symlinks
├── data/
│   ├── docs/                 # Source PDF documents
│   └── eval/
│       └── dataset.json      # Evaluation questions (126+)
├── scripts/
│   ├── ingest.py             # CLI: document ingestion
│   └── benchmark.py          # CLI: RAGAS benchmark runner
├── src/
│   ├── core/                 # Interfaces, config, pipeline
│   │   ├── config.py         # Pydantic config models + factory
│   │   ├── document.py       # Document dataclass
│   │   ├── interfaces.py     # Abstract base classes
│   │   └── pipeline.py       # RAGPipeline orchestrator
│   ├── chunkers/             # Chunking strategies
│   ├── embedders/            # Embedding models (HF + OpenAI)
│   ├── retrievers/           # Retrieval (naive, hyde, hybrid)
│   ├── rerankers/            # Reranking models
│   ├── llms/                 # LLM providers (LiteLLM — 100+ providers)
│   ├── ingestion/            # PDF parsing + text preprocessing
│   ├── evaluation/
│   │   ├── benchmark.py      # Custom benchmark runner
│   │   ├── metrics.py        # Retrieval metrics (P@k, R@k, MRR)
│   │   ├── generation_metrics.py  # Generation metrics
│   │   ├── visualization.py  # Plotting utilities
│   │   └── text_utils.py     # Shared text overlap utilities
│   └── app/
│       └── chatbot.py        # Streamlit web interface
├── chroma_db/                # ChromaDB vector store (gitignored)
├── results/                  # Benchmark outputs (gitignored)
├── Makefile                  # CLI orchestration
└── requirements.txt
```

## Environment Setup

```bash
cp .env.example .env.local
```

Required environment variables (depending on which configs you use):

| Variable | When needed |
|----------|-------------|
| `OPENAI_API_KEY` | Any config using `openai` or `litellm` with OpenAI models |
| `ANTHROPIC_API_KEY` | LiteLLM configs using `anthropic/*` models |
| `GOOGLE_API_KEY` | LiteLLM configs using `gemini/*` models |

For Ollama configs, ensure the Ollama server is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1:8b`).

## License

MIT
