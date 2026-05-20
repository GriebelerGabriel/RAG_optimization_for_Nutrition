"""Unified RAG benchmark: pipeline + RAGAS + deterministic metrics + report.

Runs the RAG pipeline ONCE per config, then evaluates outputs with both
RAGAS (LLM-as-judge) and deterministic NLP metrics, and generates a full
suite of visual artifacts (PNG plots + self-contained HTML report).

Usage:
    # Full run on all configs
    python3 scripts/benchmark_unified.py

    # Single config
    python3 scripts/benchmark_unified.py --config configs/fair_comparison/fair_gpt4o_mini.yaml

    # Deterministic metrics only (no LLM eval cost)
    python3 scripts/benchmark_unified.py --skip-ragas

    # Re-generate report from existing CSVs (no pipeline, no eval)
    python3 scripts/benchmark_unified.py --skip-pipeline --skip-ragas --skip-deterministic
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineSample:
    """Output from running one question through the RAG pipeline."""
    question: str
    response: str
    reference: str
    retrieved_contexts: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)


@dataclass
class ConfigRunResult:
    """All samples for one config, plus metadata."""
    config_name: str
    config_path: str
    samples: list[PipelineSample]
    latencies_ms: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Phase 1: Pipeline
# ---------------------------------------------------------------------------

def run_pipeline_for_configs(
    config_paths: list[Path],
    questions: list[dict],
) -> list[ConfigRunResult]:
    """Run RAG pipeline for each config, collecting outputs for both evaluators."""
    from src.core.pipeline import RAGPipeline

    results = []
    for config_path in config_paths:
        logger.info("Running pipeline for: %s", config_path.name)
        try:
            pipeline = RAGPipeline.from_config(str(config_path))
        except Exception as e:
            logger.error("Failed to create pipeline for %s: %s", config_path.name, e)
            continue

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
            samples.append(PipelineSample(
                question=q["question"],
                response=response,
                reference=q.get("ground_truth") or q.get("answer", ""),
                retrieved_contexts=[doc.content for doc in retrieved_docs],
                reference_contexts=q.get("contexts", []),
            ))

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        logger.info("  %s: %d questions, avg %.0fms", config_name, len(samples), avg_latency)
        results.append(ConfigRunResult(
            config_name=config_name,
            config_path=str(config_path),
            samples=samples,
            latencies_ms=latencies,
            avg_latency_ms=avg_latency,
        ))

    return results


def save_pipeline_cache(results: list[ConfigRunResult], cache_path: Path) -> None:
    """Save pipeline outputs to JSON for reuse."""
    data = []
    for r in results:
        data.append({
            "config_name": r.config_name,
            "config_path": r.config_path,
            "avg_latency_ms": r.avg_latency_ms,
            "samples": [
                {
                    "question": s.question,
                    "response": s.response,
                    "reference": s.reference,
                    "retrieved_contexts": s.retrieved_contexts,
                    "reference_contexts": s.reference_contexts,
                }
                for s in r.samples
            ],
        })
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pipeline_cache(cache_path: Path) -> list[ConfigRunResult]:
    """Load pipeline outputs from a JSON cache file."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    results = []
    for entry in data:
        samples = [PipelineSample(**s) for s in entry["samples"]]
        results.append(ConfigRunResult(
            config_name=entry["config_name"],
            config_path=entry["config_path"],
            samples=samples,
            avg_latency_ms=entry["avg_latency_ms"],
        ))
    return results


# ---------------------------------------------------------------------------
# Phase 2: RAGAS evaluation
# ---------------------------------------------------------------------------

def evaluate_ragas(
    results: list[ConfigRunResult],
    evaluator_type: str,
    evaluator_model: str,
    results_dir: Path,
    metric_names: list[str] | None = None,
) -> pd.DataFrame | None:
    """Run RAGAS evaluation matching the proven fast notebook approach."""
    import os
    from datasets import Dataset
    from ragas import evaluate
    from src.evaluation.ragas_utils import build_ragas_evaluator, build_ragas_metrics

    os.environ["RAGAS_WITH_EVAL_MODE"] = "true"

    try:
        ragas_llm, ragas_emb = build_ragas_evaluator(evaluator_type, evaluator_model)
        metrics = build_ragas_metrics(metric_names=metric_names)
    except Exception as e:
        logger.error("Failed to build RAGAS evaluator: %s", e)
        return None

    logger.info("  Metrics: %s", [m.name for m in metrics])

    total = len(results)
    rows = []
    for i, config_result in enumerate(results, 1):
        try:
            dataset = Dataset.from_dict({
                "question": [s.question for s in config_result.samples],
                "answer": [s.response for s in config_result.samples],
                "contexts": [s.retrieved_contexts for s in config_result.samples],
                "ground_truth": [s.reference for s in config_result.samples],
            })

            logger.info("  [%d/%d] RAGAS evaluating: %s ...", i, total, config_result.config_name)
            ragas_result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
                show_progress=True,
                raise_exceptions=False,
            )

            row = {
                "config": config_result.config_name,
                "config_path": config_result.config_path,
                "avg_latency_ms": config_result.avg_latency_ms,
            }
            row.update(ragas_result._repr_dict)
            rows.append(row)

            detail_df = ragas_result.to_pandas()
            detail_path = results_dir / f"ragas_detail_{config_result.config_name}.csv"
            detail_df.to_csv(detail_path, index=False)
            logger.info("  [%d/%d] Done: %s", i, total, config_result.config_name)
        except Exception as e:
            logger.error("  RAGAS eval failed for %s: %s", config_result.config_name, e)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ragas_benchmark_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Phase 3: Deterministic evaluation
# ---------------------------------------------------------------------------

def evaluate_deterministic(
    results: list[ConfigRunResult],
    results_dir: Path,
    bertscore_model: str = "xlm-roberta-base",
    st_model: str = "intfloat/multilingual-e5-base",
) -> pd.DataFrame | None:
    """Run deterministic NLP metrics on all config results."""
    from src.evaluation.deterministic_metrics import compute_deterministic_metrics

    rows = []
    for config_result in results:
        logger.info("  Deterministic metrics: %s", config_result.config_name)
        detail_rows = []
        for sample in config_result.samples:
            metrics = compute_deterministic_metrics(
                response=sample.response,
                reference=sample.reference,
                bertscore_model=bertscore_model,
                st_model_name=st_model,
            )
            detail_rows.append({
                "question": sample.question,
                "response": sample.response,
                "reference": sample.reference,
                **metrics.to_dict(),
            })

        detail_df = pd.DataFrame(detail_rows)
        detail_path = results_dir / f"deterministic_detail_{config_result.config_name}.csv"
        detail_df.to_csv(detail_path, index=False)

        metric_cols = ["bleu", "rouge_l", "meteor", "bertscore_f1", "semantic_similarity"]
        means = detail_df[metric_cols].mean()
        rows.append({
            "config": config_result.config_name,
            "config_path": config_result.config_path,
            "avg_latency_ms": config_result.avg_latency_ms,
            **means.to_dict(),
        })
        logger.info("  %s done", config_result.config_name)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "deterministic_benchmark_results.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Phase 4: Comparison + visualization + HTML report
# ---------------------------------------------------------------------------

def generate_report(
    ragas_df: pd.DataFrame | None,
    det_df: pd.DataFrame | None,
    results_dir: Path,
    plots_dir: Path,
) -> None:
    """Generate correlation analysis, all plots, and HTML report."""
    from src.evaluation.visualization import (
        generate_comparison_report,
        plot_correlation_heatmap,
        plot_ranking_comparison,
        generate_html_report,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Merge what we have for the full comparison plot suite ---
    if ragas_df is not None and det_df is not None:
        # Correlation analysis
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.compare_frameworks import compute_correlations, find_best_configs

        corr_df = compute_correlations(ragas_df, det_df)
        corr_df.to_csv(results_dir / "correlation_analysis.csv", index=False)

        # Best config analysis
        best = find_best_configs(ragas_df, det_df)

        # Correlation + ranking plots
        plot_correlation_heatmap(corr_df, save_path=plots_dir / "correlation_heatmap.png")
        plot_ranking_comparison(ragas_df, det_df, save_path=plots_dir / "ranking_comparison.png")

        # Full comparison suite (bar, radar, heatmap, latency, scatter)
        merged = ragas_df.merge(det_df, on="config", suffixes=("_ragas", "_det"))
        comparison_plots = generate_comparison_report(merged, output_dir=str(plots_dir))

        # HTML report
        all_plots = {
            "correlation_heatmap.png": plots_dir / "correlation_heatmap.png",
            "ranking_comparison.png": plots_dir / "ranking_comparison.png",
            **comparison_plots,
        }
        generate_html_report(
            ragas_df=ragas_df,
            det_df=det_df,
            corr_df=corr_df,
            best=best,
            plots=all_plots,
            output_path=results_dir / "benchmark_report.html",
        )

        print("\n" + "=" * 70)
        print("FRAMEWORK AGREEMENT ANALYSIS")
        print("=" * 70)
        print(f"Best config by RAGAS mean:         {best['best_ragas']}")
        print(f"Best config by deterministic mean: {best['best_deterministic']}")
        print(f"Agreement: {'YES' if best['agreement'] else 'NO'}")
        print("=" * 70)

    elif ragas_df is not None:
        generate_comparison_report(ragas_df, output_dir=str(plots_dir), prefix="ragas")
    elif det_df is not None:
        generate_comparison_report(det_df, output_dir=str(plots_dir), prefix="deterministic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified RAG benchmark: pipeline + RAGAS + deterministic + report",
    )
    parser.add_argument("--eval-dataset", default="data/eval/dataset.json")
    parser.add_argument("--config-dir", default="configs/")
    parser.add_argument("--config", default=None, help="Single config YAML")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--evaluator", default="openai", choices=["openai", "openrouter", "ollama"])
    parser.add_argument("--evaluator-model", default="gpt-4.1-mini")
    parser.add_argument("--bertscore-model", default="xlm-roberta-base")
    parser.add_argument("--st-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--ragas-metrics", nargs="+", default=None,
                        help="RAGAS metrics to run (default: all). "
                        "Options: faithfulness answer_relevancy context_precision "
                        "context_recall answer_correctness")

    # Phase control
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline, load from cached results")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS evaluation")
    parser.add_argument("--skip-deterministic", action="store_true",
                        help="Skip deterministic evaluation")
    parser.add_argument("--skip-report", action="store_true",
                        help="Skip comparison and visualization")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"

    # --- Phase 1: Pipeline ---
    pipeline_results = None

    if not args.skip_pipeline:
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
            config_paths = sorted(Path(args.config_dir).rglob("*.yaml"))
        logger.info("Configs to evaluate: %s", [p.name for p in config_paths])

        if not config_paths:
            logger.error("No config files found")
            sys.exit(1)

        pipeline_results = run_pipeline_for_configs(config_paths, questions)

        if not pipeline_results:
            logger.error("No configs could run. Check API keys / Ollama availability.")
            sys.exit(1)

        save_pipeline_cache(pipeline_results, results_dir / "pipeline_cache.json")
        logger.info("Pipeline cache saved to %s", results_dir / "pipeline_cache.json")
    else:
        cache_path = results_dir / "pipeline_cache.json"
        if cache_path.exists():
            pipeline_results = load_pipeline_cache(cache_path)
            logger.info("Loaded %d configs from pipeline cache", len(pipeline_results))
        else:
            logger.warning("No pipeline cache found at %s", cache_path)

    # --- Phase 2: RAGAS ---
    ragas_df = None
    if not args.skip_ragas:
        if pipeline_results:
            logger.info("=== RAGAS Evaluation ===")
            ragas_df = evaluate_ragas(
                pipeline_results, args.evaluator, args.evaluator_model, results_dir,
                metric_names=args.ragas_metrics,
            )
            if ragas_df is not None:
                logger.info("RAGAS results saved to %s", results_dir / "ragas_benchmark_results.csv")
        else:
            # Try loading from existing CSV
            ragas_csv = results_dir / "ragas_benchmark_results.csv"
            if ragas_csv.exists():
                ragas_df = pd.read_csv(ragas_csv)
                logger.info("Loaded existing RAGAS results from %s", ragas_csv)

    # --- Phase 3: Deterministic ---
    det_df = None
    if not args.skip_deterministic:
        if pipeline_results:
            logger.info("=== Deterministic Evaluation ===")
            det_df = evaluate_deterministic(
                pipeline_results, results_dir, args.bertscore_model, args.st_model,
            )
            if det_df is not None:
                logger.info("Deterministic results saved to %s",
                            results_dir / "deterministic_benchmark_results.csv")
        else:
            det_csv = results_dir / "deterministic_benchmark_results.csv"
            if det_csv.exists():
                det_df = pd.read_csv(det_csv)
                logger.info("Loaded existing deterministic results from %s", det_csv)

    # --- Phase 4: Report ---
    if not args.skip_report:
        logger.info("=== Generating Report ===")
        generate_report(ragas_df, det_df, results_dir, plots_dir)
        logger.info("Report artifacts saved to %s", results_dir)

    # --- Print summary tables ---
    print("\n" + "=" * 70)
    print("UNIFIED BENCHMARK RESULTS")
    print("=" * 70)
    if ragas_df is not None:
        print("\n--- RAGAS (LLM-as-Judge) ---")
        print(ragas_df.to_string(index=False))
    if det_df is not None:
        print("\n--- Deterministic NLP Metrics ---")
        print(det_df.to_string(index=False))
    if ragas_df is None and det_df is None:
        print("No results generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
