"""Run RAGAS benchmarks for RAG pipeline configurations.

Usage:
    python3 scripts/benchmark.py --eval-dataset data/eval/dataset.json
    python3 scripts/benchmark.py --evaluator openai --evaluator-model gpt-4o-mini
    python3 scripts/benchmark.py --config configs/baseline.yaml  # single config
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_ragas_evaluator(
    evaluator_type: str,
    evaluator_model: str,
):
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


def build_ragas_metrics(llm, embeddings):
    """Build the standard RAGAS metric suite."""
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]


def run_config_benchmark(config_path: Path, questions: list[dict]) -> dict | None:
    """Run the RAG pipeline for one config, collecting (retrieved, generated) per question.

    Returns a dict with config_name, samples, and per-question latencies, or None on failure.
    """
    from src.core.pipeline import RAGPipeline

    logger.info("Running pipeline for: %s", config_path.name)
    try:
        pipeline = RAGPipeline.from_config(str(config_path))
    except Exception as e:
        logger.error("Failed to create pipeline for %s: %s", config_path.name, e)
        return None

    config_name = pipeline.config.experiment_name if pipeline.config else config_path.stem
    samples = []
    latencies = []

    for i, q in enumerate(questions):
        start = time.time()

        # Retrieve (may work even without LLM)
        try:
            retrieved_docs = pipeline.retrieve(q["question"])
        except Exception as e:
            logger.warning("  Q%d retrieve failed: %s", i, e)
            retrieved_docs = []

        # Generate (needs LLM)
        try:
            response = pipeline.generate(q["question"], retrieved_docs)
        except Exception as e:
            logger.warning("  Q%d generate failed: %s", i, e)
            response = ""

        latencies.append((time.time() - start) * 1000)

        samples.append({
            "user_input": q["question"],
            "retrieved_contexts": [doc.content for doc in retrieved_docs],
            "reference_contexts": q.get("contexts", []),
            "response": response,
            "reference": q.get("ground_truth") or q.get("answer", ""),
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    logger.info("  %s: %d questions, avg %.0fms", config_name, len(samples), avg_latency)

    return {
        "config_name": config_name,
        "config_path": str(config_path),
        "samples": samples,
        "avg_latency_ms": avg_latency,
    }


def evaluate_with_ragas(config_result: dict, metrics: list, llm, embeddings) -> dict:
    """Run RAGAS evaluate() on a single config's results."""
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas import evaluate

    samples = [
        SingleTurnSample(**s) for s in config_result["samples"]
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info("  Evaluating with RAGAS: %s ...", config_result["config_name"])
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        raise_exceptions=False,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmarks with RAGAS")
    parser.add_argument(
        "--eval-dataset",
        default="data/eval/dataset.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/",
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Single config YAML (skips --config-dir)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--evaluator",
        default="openai",
        choices=["openai", "ollama"],
        help="LLM provider for RAGAS evaluation",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-4o-mini",
        help="Model name for the evaluator LLM",
    )
    args = parser.parse_args()

    # --- Load dataset ---
    dataset_path = Path(args.eval_dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)
    logger.info("Loaded %d questions from %s", len(questions), dataset_path)

    # --- Discover configs ---
    if args.config:
        config_paths = [Path(args.config)]
    else:
        config_paths = sorted(Path(args.config_dir).glob("*.yaml"))
    logger.info("Configs to evaluate: %s", [p.name for p in config_paths])

    # --- Run each config through the RAG pipeline ---
    all_results = []
    for config_path in config_paths:
        result = run_config_benchmark(config_path, questions)
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No configs could run. Check API keys / Ollama availability.")
        sys.exit(1)

    # --- Build RAGAS evaluator ---
    try:
        ragas_llm, ragas_emb = build_ragas_evaluator(args.evaluator, args.evaluator_model)
        metrics = build_ragas_metrics(ragas_llm, ragas_emb)
    except Exception as e:
        logger.error("Failed to build RAGAS evaluator: %s", e)
        logger.error("Make sure you have the right API keys set (.env) or Ollama running.")
        sys.exit(1)

    # --- Run RAGAS evaluation per config ---
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_result in all_results:
        try:
            ragas_result = evaluate_with_ragas(config_result, metrics, ragas_llm, ragas_emb)
            row = {
                "config": config_result["config_name"],
                "config_path": config_result["config_path"],
                "avg_latency_ms": config_result["avg_latency_ms"],
            }
            # _repr_dict has metric_name -> mean_score
            row.update(ragas_result._repr_dict)

            # Also save per-question detail CSV
            detail_df = ragas_result.to_pandas()
            detail_path = results_dir / f"ragas_detail_{config_result['config_name']}.csv"
            detail_df.to_csv(detail_path, index=False)

            rows.append(row)
            logger.info("  %s done", config_result["config_name"])
        except Exception as e:
            logger.error("RAGAS eval failed for %s: %s", config_result["config_name"], e)

    # --- Save results ---
    import pandas as pd

    if rows:
        df = pd.DataFrame(rows)
        output_path = results_dir / "ragas_benchmark_results.csv"
        df.to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)

        # Print summary
        print("\n" + "=" * 70)
        print("RAGAS BENCHMARK RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
    else:
        logger.warning("No results to save.")


if __name__ == "__main__":
    main()
