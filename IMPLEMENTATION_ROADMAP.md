# RAG Nutrition System - Exploration & Benchmarking Roadmap

**Goal:** Build a modular, experiment-driven RAG system where components (chunkers, embeddings, retrievers, LLMs) can be easily swapped and benchmarked. The best-performing configuration will power the final chatbot interface.

---

## Architecture Philosophy: Pluggable Components

The system uses **abstract base classes** and **configuration-driven instantiation** to enable:
- Hot-swapping any component without code changes
- Running automated benchmarks across all combinations
- Tracking experiment results with metrics

```python
# Example: Swapping components via config
config = {
    "chunker": "semantic",        # or "fixed", "sentence"
    "embedding": "e5-large",      # or "mpnet", "openai"
    "retriever": "hyde",          # or "naive", "hybrid"
    "reranker": "bge",            # or "none", "cross-encoder"
    "llm": "gpt-4o-mini",         # or "llama3", "claude"
}
pipeline = RAGPipeline.from_config(config)
```

---

## Phase 1: Project Setup & Modular Infrastructure

### ☐ 1.1 Initialize Project Structure
**Decision:** Python 3.11+ with Poetry for dependency management
- **Why:** Best ML/NLP ecosystem. Poetry ensures reproducible builds.

### ☐ 1.2 Create Abstract Base Classes for Swappable Components
**Decision:** Define interfaces for each pipeline stage

```python
# src/core/interfaces.py
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[Document]: ...

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list[Document]) -> list[Document]: ...
```

### ☐ 1.3 Configuration System
**Decision:** YAML config files + Pydantic for validation

```yaml
# configs/experiment_01.yaml
experiment_name: "baseline_naive_rag"
chunker:
  type: "fixed"
  chunk_size: 512
  overlap: 50
embedding:
  type: "e5-large"
retriever:
  type: "naive"
  top_k: 5
reranker:
  type: "none"
llm:
  type: "gpt-4o-mini"
  temperature: 0.1
```

---

## Phase 2: Knowledge Base Preparation

### ☐ 2.1 Corpus Collection
**Documents to collect:**
- Guia Alimentar para a População Brasileira (PDF)
- Dietary Guidance to Improve Cardiovascular Health (AHA)
- Other validated nutritional guidelines

### ☐ 2.2 Document Parsing & Extraction
**Decision:** `PyMuPDF` (fitz) + `unstructured` library
- **Why:** PyMuPDF handles complex PDF layouts well. `unstructured` provides intelligent document parsing that preserves structure (headers, tables, lists).
- **Alternative considered:** pdfplumber - good but less robust for complex layouts.

### ☐ 2.3 Text Preprocessing
**Decision:** `unstructured` built-in cleaning + simple regex
- **Tasks:**
  - Remove headers/footers/page numbers
  - Normalize whitespace
  - Handle special characters (nutritional symbols, units)
- **Why:** `unstructured` already handles most cleaning. No need for heavy NLP libraries like spaCy for simple preprocessing.

### ☐ 2.4 Chunking Strategy (SWAPPABLE)
**Implementations to benchmark:**

| Chunker | Description | Parameters to Test |
|---------|-------------|--------------------|
| `FixedChunker` | Fixed character/token size | size: 256, 512, 1024; overlap: 10-20% |
| `RecursiveChunker` | LangChain recursive splitter | respects paragraphs > sentences |
| `SemanticChunker` | Embedding-based boundaries | similarity threshold |
| `SentenceChunker` | Split by sentences | sentences_per_chunk: 3-10 |

**Why multiple:** Chunking significantly impacts retrieval quality. Must benchmark to find optimal for nutrition domain.

### ☐ 2.5 Embedding Models (SWAPPABLE)
**Implementations to benchmark:**

| Model | Dimensions | Notes |
|-------|------------|-------|
| `intfloat/multilingual-e5-large` | 1024 | HuggingFace - Best multilingual retrieval |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | HuggingFace - Good balance speed/quality |
| `BAAI/bge-m3` | 1024 | HuggingFace - State-of-art multilingual |
| `OpenAI text-embedding-3-small` | 1536 | API-based, high quality |

**Why multiple:** Embedding quality directly affects retrieval. Portuguese content needs multilingual models.

### ☐ 2.6 Vector Database Setup
**Decision:** `ChromaDB` (embedded mode)
- **Why:** Zero-config, perfect for experimentation, supports multiple collections (one per embedding model).
- No need for production DB - this is a research project.

---

## Phase 3: Retrieval Pipeline

### ☐ 3.1 Basic Retrieval Implementation
**Decision:** Cosine similarity search
- **Parameters:** `top_k=5` (retrieve 5 most relevant chunks)
- **Why:** Standard, interpretable, works well with normalized embeddings.

### ☐ 3.2 Retrieval Strategies (SWAPPABLE)
**Implementations to benchmark:**

| Retriever | Description |
|-----------|-------------|
| `NaiveRetriever` | Direct query embedding → cosine search (baseline) |
| `HYDERetriever` | Generate hypothetical answer, embed that |
| `RRRRetriever` | Rewrite-Retrieve-Read: LLM rewrites query first |
| `HybridRetriever` | BM25 + Dense with reciprocal rank fusion |
| `MultiQueryRetriever` | Generate multiple query variations, merge results |

**Why multiple:** Query optimization is a key research focus. Must compare techniques.

### ☐ 3.3 Re-ranking Models (SWAPPABLE)
**Implementations to benchmark:**

| Reranker | Description |
|----------|-------------|
| `NoReranker` | Baseline - no reranking |
| `CrossEncoderReranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `BGEReranker` | `BAAI/bge-reranker-base` |
| `CohereReranker` | API-based (if budget allows) |

**Why:** Reranking can significantly improve precision@k.

---

## Phase 4: Generation & Prompt Engineering

### ☐ 4.1 LLM Selection (SWAPPABLE)
**Implementations to benchmark:**

| LLM | Access | Notes |
|-----|--------|-------|
| `GPT-4o-mini` | OpenAI API | Best cost/quality ratio |
| `GPT-4o` | OpenAI API | Higher quality, higher cost |
| `Llama 3.1 8B` | Ollama / HuggingFace | Free, private, good quality |
| `Llama 3.1 70B` | Ollama / HuggingFace | Better reasoning |
| `Mistral 7B` | HuggingFace | Fast, good for Portuguese |
| `Claude 3.5 Sonnet` | Anthropic API | Excellent, compare if budget allows |

**Why multiple:** LLM choice affects answer quality and hallucination rate.

### ☐ 4.2 Prompt Template Design
**Decision:** Structured prompt with role, context, constraints, and format instructions

```text
Sistema: Você é um nutricionista especializado em educação alimentar.
Responda APENAS com base no contexto fornecido.
Se a informação não estiver no contexto, diga "Não encontrei essa informação nas fontes disponíveis."

Contexto:
{retrieved_chunks}

Pergunta do usuário: {query}

Instruções:
- Seja factual e cite as fontes quando possível
- Use linguagem acessível ao público geral
- Não faça recomendações médicas individualizadas
```

### ☐ 4.3 Response Formatting
**Decision:** Markdown output with source citations
- **Why:** Improves readability and traceability.

---

## Phase 5: Evaluation & Benchmarking Framework

### ☐ 5.1 Create Evaluation Dataset
**Tasks:**
- Create 50-100 question-answer pairs with ideal context chunks
- Include edge cases (out-of-scope, ambiguous queries)
- Store in `data/eval/golden_dataset.json`

```json
{
  "questions": [
    {
      "id": "q001",
      "question": "Quais são os grupos alimentares do Guia Alimentar?",
      "ideal_chunks": ["chunk_id_1", "chunk_id_2"],
      "ideal_answer": "Os grupos alimentares são..."
    }
  ]
}
```

### ☐ 5.2 Retrieval Metrics
**Metrics to compute:**
- **Precision@k** (k=1,3,5)
- **Recall@k**
- **MRR (Mean Reciprocal Rank)**
- **NDCG (Normalized Discounted Cumulative Gain)**

### ☐ 5.3 Generation Quality Metrics
**Using `ragas` library:**
- **Faithfulness** - answer grounded in context?
- **Answer Relevancy** - addresses the question?
- **Context Precision** - retrieved context relevant?
- **Context Recall** - all needed info retrieved?

### ☐ 5.4 Automated Benchmark Runner
**Decision:** Create a benchmark script that:
1. Loads all experiment configs from `configs/`
2. Runs each configuration against the evaluation dataset
3. Computes all metrics
4. Saves results to `results/benchmark_results.csv`

```python
# src/evaluation/benchmark.py
def run_benchmark(config_path: str) -> dict:
    config = load_config(config_path)
    pipeline = RAGPipeline.from_config(config)
    
    results = []
    for question in eval_dataset:
        retrieved = pipeline.retrieve(question.text)
        answer = pipeline.generate(question.text, retrieved)
        
        metrics = compute_metrics(retrieved, answer, question)
        results.append(metrics)
    
    return aggregate_metrics(results)
```

### ☐ 5.5 Results Dashboard
**Decision:** Simple Streamlit dashboard to visualize benchmark results
- Compare configurations side-by-side
- Show metric distributions
- Identify best-performing config

### ☐ 5.6 Human Evaluation Protocol
**Tasks:**
- Design evaluation form for nutritionist (Google Forms or similar)
- Criteria: factual accuracy, clarity, usefulness (1-5 scale)
- Sample: 30-50 responses from top 3 configurations

---

## Phase 6: Final Chatbot Interface

### ☐ 6.1 Select Best Configuration
**Tasks:**
- Analyze benchmark results
- Select configuration with best retrieval + generation metrics
- Document the winning configuration and why

### ☐ 6.2 Build Streamlit Chatbot
**Decision:** `Streamlit` with the best-performing pipeline
- **Features:**
  - Chat interface with conversation history
  - Show retrieved sources (expandable)
  - Display confidence/relevance scores
  - Option to thumbs up/down responses (for future improvement)

```python
# src/app/chatbot.py
import streamlit as st
from src.core.pipeline import RAGPipeline

# Load best config
pipeline = RAGPipeline.from_config("configs/best_config.yaml")

st.title("🥗 Assistente de Nutrição")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Faça sua pergunta sobre nutrição..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Buscando informações..."):
        response, sources = pipeline.query(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.expander("📚 Fontes consultadas"):
        for source in sources:
            st.markdown(f"- {source.metadata['title']} (relevância: {source.score:.2f})")
```

### ☐ 6.3 Conversation Memory
**Decision:** Simple in-memory buffer (last 5 turns)
- **Why:** Sufficient for demo/evaluation purposes.

---

## Phase 7: Experiment Matrix & Analysis

### ☐ 7.1 Define Experiment Matrix
**Key variables to test:**

| Variable | Options |
|----------|--------|
| Chunk size | 256, 512, 1024 |
| Chunk overlap | 0%, 10%, 20% |
| Embedding model | e5-large, mpnet, bge-m3 |
| Retriever | naive, hyde, hybrid |
| Reranker | none, cross-encoder, bge |
| LLM | gpt-4o-mini, llama3-8b |
| top_k | 3, 5, 10 |

**Note:** Full matrix = 3×3×3×3×3×2×3 = 1458 combinations. Use ablation studies instead:
1. Fix all but one variable
2. Find best value for that variable
3. Move to next variable

### ☐ 7.2 Run Ablation Studies
**Order of experiments:**
1. **Chunking:** Fix embedding=e5, retriever=naive, reranker=none, llm=gpt-4o-mini → test chunk sizes
2. **Embedding:** Use best chunk size → test embedding models
3. **Retriever:** Use best chunk+embedding → test retrieval strategies
4. **Reranker:** Use best chunk+embedding+retriever → test rerankers
5. **LLM:** Use best retrieval pipeline → test LLMs

### ☐ 7.3 Document Findings
**Create:** `EXPERIMENT_RESULTS.md` with:
- Best configuration per component
- Metric improvements at each step
- Insights about nutrition domain specifics
- Recommendations for future work

---

## Technology Stack Summary

| Component | Decision | Rationale |
|-----------|----------|----------|
| **Language** | Python 3.11+ | Best ML/NLP ecosystem |
| **Framework** | Custom modular + LangChain components | Full control over swapping |
| **PDF Parsing** | PyMuPDF + unstructured | Handles complex layouts |
| **Chunking** | Multiple (swappable) | Benchmark to find best |
| **Embeddings** | Multiple (swappable) | Benchmark to find best |
| **Vector DB** | ChromaDB (embedded) | Simple, sufficient for research |
| **LLM** | Multiple (swappable) | Benchmark to find best |
| **Retrieval** | Multiple (swappable) | Core research focus |
| **Frontend** | Streamlit | Rapid prototyping, good enough |
| **Evaluation** | ragas + custom metrics | Automated benchmarking |
| **Config** | YAML + Pydantic | Easy experiment management |

---

## Project File Structure

```
RAG_Nutrition/
├── src/
│   ├── core/
│   │   ├── interfaces.py       # Abstract base classes
│   │   ├── pipeline.py         # RAGPipeline orchestrator
│   │   └── config.py           # Pydantic config models
│   ├── chunkers/               # Swappable chunking implementations
│   │   ├── __init__.py
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── sentence.py
│   ├── embedders/              # Swappable embedding implementations
│   │   ├── __init__.py
│   │   ├── e5.py
│   │   ├── mpnet.py
│   │   └── openai.py
│   ├── retrievers/             # Swappable retrieval implementations
│   │   ├── __init__.py
│   │   ├── naive.py
│   │   ├── hyde.py
│   │   ├── hybrid.py
│   │   └── multi_query.py
│   ├── rerankers/              # Swappable reranker implementations
│   │   ├── __init__.py
│   │   ├── none.py
│   │   ├── cross_encoder.py
│   │   └── bge.py
│   ├── llms/                   # Swappable LLM implementations
│   │   ├── __init__.py
│   │   ├── openai.py
│   │   └── ollama.py
│   ├── ingestion/              # Document processing
│   │   ├── parser.py
│   │   └── preprocessor.py
│   ├── evaluation/             # Metrics & benchmarking
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── dashboard.py
│   └── app/                    # Streamlit chatbot
│       └── chatbot.py
├── configs/                    # Experiment configurations
│   ├── baseline.yaml
│   ├── experiment_hyde.yaml
│   ├── experiment_hybrid.yaml
│   └── best_config.yaml        # Final best configuration
├── data/
│   ├── raw/                    # Original PDFs
│   ├── processed/              # Cleaned text chunks
│   └── eval/                   # Evaluation datasets
│       └── golden_dataset.json
├── results/                    # Benchmark outputs
│   ├── benchmark_results.csv
│   └── plots/
├── notebooks/                  # Exploration & analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_analysis.ipynb
│   └── 03_results_analysis.ipynb
├── chroma_db/                  # Vector database storage
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_ROADMAP.md
└── EXPERIMENT_RESULTS.md       # Final findings
```