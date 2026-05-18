"""Run deterministic NLP metric benchmarks for RAG pipeline configurations.

Computes BLEU, ROUGE-L, METEOR, BERTScore, and Sentence-BERT semantic
similarity on the same (response, reference) pairs used by the RAGAS script.

Usage:
    python3 scripts/benchmark_deterministic.py
    python3 scripts/benchmark_deterministic.py --config configs/baseline.yaml
    python3 scripts/benchmark_deterministic.py --artifacts-dir results/artifacts
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_config_benchmark(config_path: Path, questions: list[dict]) -> dict | None:
    """Run the RAG pipeline for one config, collecting (response, reference) per question."""
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

        try:
            retrieved_docs = pipeline.retrieve(q["question"])
        except Exception as e:
            logger.warning("  Q%d retrieve failed: %s", i, e)
            retrieved_docs = []

        try:
            response = pipeline.generate(q["question"], retrieved_docs)
        except Exception as e:
            logger.warning("  Q%d generate failed: %s", i, e)
            response = ""

        latencies.append((time.time() - start) * 1000)
        reference = q.get("ground_truth") or q.get("answer", "")

        samples.append({
            "question": q["question"],
            "response": response,
            "reference": reference,
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    logger.info("  %s: %d questions, avg %.0fms", config_name, len(samples), avg_latency)

    return {
        "config_name": config_name,
        "config_path": str(config_path),
        "samples": samples,
        "avg_latency_ms": avg_latency,
    }


def evaluate_deterministic(
    config_result: dict,
    bertscore_model: str = "xlm-roberta-base",
    st_model: str = "intfloat/multilingual-e5-base",
) -> pd.DataFrame:
    """Compute deterministic metrics for all samples in a config result."""
    from src.evaluation.deterministic_metrics import compute_deterministic_metrics

    rows = []
    for sample in config_result["samples"]:
        metrics = compute_deterministic_metrics(
            response=sample["response"],
            reference=sample["reference"],
            bertscore_model=bertscore_model,
            st_model_name=st_model,
        )
        rows.append({
            "question": sample["question"],
            "response": sample["response"],
            "reference": sample["reference"],
            **metrics.to_dict(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic NLP metric benchmarks")
    parser.add_argument("--eval-dataset", default="data/eval/dataset.json")
    parser.add_argument("--config-dir", default="configs/")
    parser.add_argument("--config", default=None, help="Single config YAML")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Reuse saved outputs from artifacts instead of running pipeline")
    parser.add_argument("--bertscore-model", default="xlm-roberta-base")
    parser.add_argument("--st-model", default="intfloat/multilingual-e5-base")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load from artifacts or run pipeline ---
    all_config_results = []

    if args.artifacts_dir:
        artifacts_path = Path(args.artifacts_dir)
        # May be a parent dir with multiple config subdirs, or a single run dir
        json_files = list(artifacts_path.rglob("q_*.json"))
        if json_files:
            # Group by parent directory (config_name/timestamp/)
            grouped: dict[str, list[Path]] = {}
            for jf in json_files:
                run_dir = jf.parent
                key = run_dir.parent.name  # config name
                grouped.setdefault(key, []).append(jf)

            for config_name, files in grouped.items():
                samples = []
                for jf in sorted(files):
                    with open(jf, encoding="utf-8") as f:
                        artifact = json.load(f)
                    if artifact.get("response") and artifact.get("ideal_answer"):
                        samples.append({
                            "question": artifact["question"],
                            "response": artifact["response"],
                            "reference": artifact["ideal_answer"],
                        })
                if samples:
                    all_config_results.append({
                        "config_name": config_name,
                        "config_path": "",
                        "samples": samples,
                        "avg_latency_ms": 0,
                    })
                    logger.info("Loaded %d samples from artifacts for %s", len(samples), config_name)
        else:
            logger.warning("No artifact files found in %s", artifacts_path)
    else:
        # Load dataset and run pipeline
        dataset_path = Path(args.eval_dataset)
        if not dataset_path.exists():
            logger.error("Dataset not found: %s", dataset_path)
            sys.exit(1)

        with open(dataset_path, encoding="utf-8") as f:
            questions = json.load(f)
        logger.info("Loaded %d questions from %s", len(questions), dataset_path)

        if args.config:
            config_paths = [Path(args.config)]
        else:
            config_paths = sorted(Path(args.config_dir).glob("*.yaml"))
        logger.info("Configs to evaluate: %s", [p.name for p in config_paths])

        for config_path in config_paths:
            result = run_config_benchmark(config_path, questions)
            if result is not None:
                all_config_results.append(result)

    if not all_config_results:
        logger.error("No results to evaluate.")
        sys.exit(1)

    # --- Compute deterministic metrics ---
    summary_rows = []
    for config_result in all_config_results:
        logger.info("Computing deterministic metrics for: %s", config_result["config_name"])
        detail_df = evaluate_deterministic(
            config_result,
            bertscore_model=args.bertscore_model,
            st_model=args.st_model,
        )

        # Save per-question detail CSV
        detail_path = results_dir / f"deterministic_detail_{config_result['config_name']}.csv"
        detail_df.to_csv(detail_path, index=False)

        # Aggregate means
        metric_cols = ["bleu", "rouge_l", "meteor", "bertscore_f1", "semantic_similarity"]
        means = detail_df[metric_cols].mean()
        row = {
            "config": config_result["config_name"],
            "config_path": config_result["config_path"],
            "avg_latency_ms": config_result["avg_latency_ms"],
            **means.to_dict(),
        }
        summary_rows.append(row)
        logger.info("  %s done", config_result["config_name"])

    # --- Save summary ---
    df = pd.DataFrame(summary_rows)
    output_path = results_dir / "deterministic_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    print("\n" + "=" * 70)
    print("DETERMINISTIC BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
