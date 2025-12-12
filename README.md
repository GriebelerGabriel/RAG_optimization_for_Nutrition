# RAG Nutrition System

A modular, experiment-driven RAG (Retrieval-Augmented Generation) system for nutritional education. This system allows easy swapping and benchmarking of different components (chunkers, embeddings, retrievers, rerankers, LLMs) to find the optimal configuration.

## Features

- **Pluggable Architecture**: Swap any component via YAML configuration
- **Multiple Chunking Strategies**: Fixed, Recursive, Sentence-based
- **Multiple Embedding Models**: E5, MPNet, BGE (HuggingFace), OpenAI
- **Multiple Retrieval Strategies**: Naive, HYDE, Hybrid (BM25 + Dense)
- **Multiple Rerankers**: Cross-encoder, BGE reranker
- **Multiple LLMs**: OpenAI (GPT-4o-mini, GPT-4o), Ollama (Llama, Mistral)
- **Automated Benchmarking**: Compare configurations with retrieval metrics
- **Streamlit Chatbot**: Interactive interface for the best configuration

## Quick Start

### 1. Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Add Documents

Place your PDF documents in `data/raw/`:
- Guia Alimentar para a População Brasileira
- Other nutritional guidelines

### 4. Ingest Documents

```python
from src.core.pipeline import RAGPipeline
from src.ingestion import PDFParser, TextPreprocessor

# Parse PDFs
parser = PDFParser()
docs = parser.parse_directory("data/raw")

# Preprocess
preprocessor = TextPreprocessor()
for doc in docs:
    doc.content = preprocessor.preprocess(doc.content)

# Create pipeline and ingest
pipeline = RAGPipeline.from_config("configs/baseline.yaml")
pipeline.ingest_documents(docs)
```

### 5. Run Chatbot

```bash
streamlit run src/app/chatbot.py
```

## Configuration

Configurations are YAML files in `configs/`. Example:

```yaml
experiment_name: "baseline"

chunker:
  type: "recursive"      # fixed, recursive, sentence
  chunk_size: 512
  overlap: 50

embedding:
  type: "e5"             # e5, mpnet, bge, openai
  model_name: "intfloat/multilingual-e5-large"

retriever:
  type: "naive"          # naive, hyde, hybrid
  top_k: 5

reranker:
  type: "none"           # none, cross_encoder, bge

llm:
  type: "openai"         # openai, ollama
  model_name: "gpt-4o-mini"
  temperature: 0.1
```

## Running Benchmarks

```python
from src.evaluation import Benchmark

benchmark = Benchmark("data/eval/golden_dataset.json")

# Run single config
results = benchmark.run_config("configs/baseline.yaml")

# Run all configs
df = benchmark.run_all_configs("configs/")
aggregated = benchmark.aggregate_results(df)
benchmark.save_results(aggregated, "results/benchmark_results.csv")
```

## Project Structure

```
RAG_Nutrition/
├── src/
│   ├── core/           # Interfaces, config, pipeline
│   ├── chunkers/       # Chunking strategies
│   ├── embedders/      # Embedding models
│   ├── retrievers/     # Retrieval strategies
│   ├── rerankers/      # Reranking models
│   ├── llms/           # Language models
│   ├── ingestion/      # Document parsing
│   ├── evaluation/     # Metrics & benchmarking
│   └── app/            # Streamlit chatbot
├── configs/            # Experiment configurations
├── data/
│   ├── raw/            # Original PDFs
│   ├── processed/      # Processed chunks
│   └── eval/           # Evaluation datasets
├── results/            # Benchmark outputs
└── notebooks/          # Analysis notebooks
```

## Available Components

### Chunkers
| Type | Description |
|------|-------------|
| `fixed` | Fixed character size with overlap |
| `recursive` | LangChain recursive splitter |
| `sentence` | Sentence-based grouping |

### Embedders
| Type | Model |
|------|-------|
| `e5` | intfloat/multilingual-e5-large |
| `mpnet` | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| `bge` | BAAI/bge-m3 |
| `openai` | text-embedding-3-small |

### Retrievers
| Type | Description |
|------|-------------|
| `naive` | Direct embedding similarity |
| `hyde` | Hypothetical Document Embeddings |
| `hybrid` | BM25 + Dense with RRF |

### Rerankers
| Type | Model |
|------|-------|
| `none` | No reranking |
| `cross_encoder` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `bge` | BAAI/bge-reranker-base |

### LLMs
| Type | Models |
|------|--------|
| `openai` | gpt-4o-mini, gpt-4o |
| `ollama` | llama3.1:8b, mistral:7b |

## License

MIT
