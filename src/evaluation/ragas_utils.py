"""RAGAS evaluator construction helpers.

Shared between the standalone RAGAS benchmark script and the unified
benchmark script.
"""

ALL_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]


def build_ragas_evaluator(evaluator_type: str, evaluator_model: str):
    """Build the LLM and embeddings that RAGAS will use for judging."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if evaluator_type == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        lc_llm = ChatOpenAI(model=evaluator_model, temperature=0)
        lc_emb = OpenAIEmbeddings(model="text-embedding-3-small")
        return LangchainLLMWrapper(lc_llm), LangchainEmbeddingsWrapper(lc_emb)

    elif evaluator_type == "ollama":
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import HuggingFaceEmbeddings

        lc_llm = Ollama(model=evaluator_model)
        lc_emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        return LangchainLLMWrapper(lc_llm), LangchainEmbeddingsWrapper(lc_emb)

    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def build_ragas_metrics(llm, embeddings, metric_names: list[str] | None = None):
    """Build RAGAS metrics, optionally filtered by name.

    Args:
        llm: RAGAS-wrapped LLM.
        embeddings: RAGAS-wrapped embeddings.
        metric_names: Subset of metric names to use. None = all 5.
            Options: faithfulness, answer_relevancy, context_precision,
                     context_recall, answer_correctness.
    """
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    all_metrics = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
    }

    names = metric_names or list(all_metrics.keys())
    unknown = set(names) - set(all_metrics)
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {unknown}. Available: {list(all_metrics)}")

    return [all_metrics[n]() for n in names]


def build_ragas_run_config(max_workers: int = 8, timeout: int = 60):
    """Build a tuned RunConfig for RAGAS evaluation.

    Defaults are optimized for OpenAI gpt-4o-mini to avoid rate limits:
    - Fewer workers (8 vs default 16) to reduce timeout errors
    - Lower timeout (60s vs 180s) to fail fast
    - Fewer retries (3 vs 10) to avoid long waits
    """
    from ragas import RunConfig

    return RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_wait=30,
        max_retries=3,
        log_tenacity=False,
    )
