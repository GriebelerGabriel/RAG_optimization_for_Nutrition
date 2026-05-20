"""RAGAS evaluator construction helpers.

Matches the proven fast approach from the reference notebook:
- Uses LangChain ChatOpenAI directly (no wrapper)
- Passes llm/embeddings to evaluate() directly
- Default RunConfig (no overrides)
"""

import os

ALL_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]


def build_ragas_evaluator(evaluator_type: str, evaluator_model: str):
    """Build LangChain LLM and embeddings for RAGAS evaluation."""
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

    if evaluator_type == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        lc_llm = ChatOpenAI(
            model=evaluator_model,
            temperature=0,
            max_tokens=32768,
            request_timeout=120,
        )
        lc_emb = OpenAIEmbeddings(model=embedding_model)
        return lc_llm, lc_emb

    elif evaluator_type == "openrouter":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        lc_llm = ChatOpenAI(
            model=evaluator_model,
            temperature=0,
            max_tokens=32768,
            request_timeout=120,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        lc_emb = OpenAIEmbeddings(model=embedding_model)
        return lc_llm, lc_emb

    elif evaluator_type == "ollama":
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        lc_llm = Ollama(model=evaluator_model)
        lc_emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        return lc_llm, lc_emb

    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}. Options: openai, openrouter, ollama")


def build_ragas_metrics(llm=None, embeddings=None, metric_names: list[str] | None = None):
    """Build RAGAS metrics. Uses defaults (no custom llm/embeddings per metric)."""
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    all_metrics = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "answer_correctness": AnswerCorrectness(),
    }

    names = metric_names or list(all_metrics.keys())
    unknown = set(names) - set(all_metrics)
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {unknown}. Available: {list(all_metrics)}")

    return [all_metrics[n] for n in names]
