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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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

NO_RAG_SYSTEM_PROMPT = (
    "Você é um assistente especialista em nutrição. "
    "Responda com base no seu próprio conhecimento. "
    "Seja sempre sucinto e claro."
)


def _process_rag_question(
    idx: int,
    q: dict,
    pipeline: "RAGPipeline",
    query_embedding: list[float] | None = None,
) -> tuple[int, PipelineSample | None, float]:
    """Process a single question through the RAG pipeline. Thread-safe."""
    start = time.time()
    try:
        retrieved_docs = pipeline.retrieve(q["question"], query_embedding=query_embedding)
    except Exception as e:
        logger.warning("  Q%d retrieve failed: %s", idx, e)
        return idx, None, 0.0

    try:
        response = pipeline.generate(q["question"], retrieved_docs)
    except Exception as e:
        logger.warning("  Q%d generate failed: %s", idx, e)
        return idx, None, 0.0

    latency_ms = (time.time() - start) * 1000
    sample = PipelineSample(
        question=q["question"],
        response=response,
        reference=q.get("ground_truth") or q.get("answer", ""),
        retrieved_contexts=[doc.content for doc in retrieved_docs],
        reference_contexts=q.get("contexts", []),
    )
    return idx, sample, latency_ms


def _process_no_rag_question(
    idx: int,
    q: dict,
    llm: "BaseLLM",
) -> tuple[int, PipelineSample | None, float]:
    """Process a single question through LLM only (no retrieval). Thread-safe."""
    start = time.time()
    try:
        prompt = f"{NO_RAG_SYSTEM_PROMPT}\n\nPergunta do usuário: {q['question']}\n\nResposta:"
        response = llm.generate(prompt)
    except Exception as e:
        logger.warning("  Q%d generate failed: %s", idx, e)
        return idx, None, 0.0

    latency_ms = (time.time() - start) * 1000
    sample = PipelineSample(
        question=q["question"],
        response=response,
        reference=q.get("ground_truth") or q.get("answer", ""),
        retrieved_contexts=[],
        reference_contexts=q.get("contexts", []),
    )
    return idx, sample, latency_ms


def _run_no_rag_parallel(
    config_path: Path,
    questions: list[dict],
    max_workers: int = 8,
) -> ConfigRunResult | None:
    """Run questions through LLM only (no retrieval), parallelized."""
    from src.core.config import load_config
    from src.llms import create_llm

    config = load_config(str(config_path))
    llm = create_llm(config.llm)
    config_name = config.experiment_name

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, q in enumerate(questions):
            future = executor.submit(_process_no_rag_question, i, q, llm)
            futures[future] = i

        indexed_results: dict[int, tuple[PipelineSample, float]] = {}
        failed_count = 0
        for future in as_completed(futures):
            idx, sample, latency_ms = future.result()
            if sample is None:
                failed_count += 1
                continue
            indexed_results[idx] = (sample, latency_ms)

    samples = []
    latencies = []
    for i in range(len(questions)):
        if i in indexed_results:
            sample, latency_ms = indexed_results[i]
            samples.append(sample)
            latencies.append(latency_ms)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    logger.info("  %s (no-RAG): %d/%d questions succeeded, avg %.0fms", config_name, len(samples), len(questions), avg_latency)
    if failed_count:
        logger.warning("  %s: %d questions failed and were excluded", config_name, failed_count)
    return ConfigRunResult(
        config_name=config_name,
        config_path=str(config_path),
        samples=samples,
        latencies_ms=latencies,
        avg_latency_ms=avg_latency,
    )


def _ensure_ingested(pipeline: "RAGPipeline", data_dir: Path) -> None:
    """Ingest raw documents into the pipeline if the collection is empty."""
    from src.ingestion import PDFParser, TextPreprocessor

    if not hasattr(pipeline.retriever, "_collection"):
        return
    if pipeline.retriever._collection.count() > 0:
        return

    logger.info("  Collection empty, ingesting from %s ...", data_dir)
    pdf_parser = PDFParser()
    docs = pdf_parser.parse_directory(str(data_dir))
    if not docs:
        logger.error("  No PDF documents found in %s", data_dir)
        return

    preprocessor = TextPreprocessor()
    for doc in docs:
        doc.content = preprocessor.preprocess(doc.content)

    pipeline.ingest_documents(docs)
    logger.info("  Ingested %d documents (%d chunks)",
                len(docs), pipeline.retriever._collection.count())


def run_pipeline_for_configs(
    config_paths: list[Path],
    questions: list[dict],
    max_workers: int = 8,
    cache_path: Path | None = None,
    data_dir: Path | None = None,
) -> list[ConfigRunResult]:
    """Run RAG pipeline (or no-RAG baseline) for each config, with parallel questions."""
    from src.core.config import load_config
    from src.core.pipeline import RAGPipeline

    # Load existing cache to skip already-completed configs.
    results: list[ConfigRunResult] = []
    completed_configs: set[str] = set()
    if cache_path and cache_path.exists():
        results = load_generation_cache(cache_path)
        completed_configs = {r.config_name for r in results}
        if completed_configs:
            logger.info("Loaded %d configs from cache, skipping: %s", len(results), sorted(completed_configs))

    for config_path in config_paths:
        # Peek at config name to check cache.
        config = load_config(str(config_path))
        config_name = config.experiment_name
        if config_name in completed_configs:
            logger.info("Skipping cached config: %s", config_name)
            continue

        logger.info("Running pipeline for: %s (workers=%d)", config_path.name, max_workers)

        if getattr(config, "mode", "rag") == "no_rag":
            result = _run_no_rag_parallel(config_path, questions, max_workers)
            if result is not None:
                results.append(result)
            save_generation_cache(results, cache_path)
            continue

        try:
            pipeline = RAGPipeline.from_config(str(config_path))
        except Exception as e:
            logger.error("Failed to create pipeline for %s: %s", config_path.name, e)
            continue

        # Ensure the vector store is populated for this config's chunking strategy.
        if data_dir:
            _ensure_ingested(pipeline, data_dir)

        # Pre-embed all queries in one batch to avoid lock contention during threading.
        query_embeddings: list[list[float]] | None = None
        if hasattr(pipeline.embedder, "embed_queries"):
            logger.info("  Pre-embedding %d queries in batch...", len(questions))
            try:
                query_embeddings = pipeline.embedder.embed_queries(
                    [q["question"] for q in questions],
                )
                logger.info("  Pre-embedding done (%d queries)", len(query_embeddings))
            except Exception as e:
                logger.warning("  Batch embedding failed, falling back to per-query: %s", e)

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, q in enumerate(questions):
                emb = query_embeddings[i] if query_embeddings else None
                future = executor.submit(_process_rag_question, i, q, pipeline, emb)
                futures[future] = i

            indexed_results: dict[int, tuple[PipelineSample, float]] = {}
            failed_count = 0
            for future in as_completed(futures):
                idx, sample, latency_ms = future.result()
                if sample is None:
                    failed_count += 1
                    continue
                indexed_results[idx] = (sample, latency_ms)
                logger.info("  Q%d done (%.0fms)", idx, latency_ms)

        samples = []
        latencies = []
        for i in range(len(questions)):
            if i in indexed_results:
                sample, latency_ms = indexed_results[i]
                samples.append(sample)
                latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        logger.info("  %s: %d/%d questions succeeded, avg %.0fms", config_name, len(samples), len(questions), avg_latency)
        if failed_count:
            logger.warning("  %s: %d questions failed and were excluded", config_name, failed_count)
        results.append(ConfigRunResult(
            config_name=config_name,
            config_path=str(config_path),
            samples=samples,
            latencies_ms=latencies,
            avg_latency_ms=avg_latency,
        ))

        # Save after each config so progress is preserved on crashes.
        if cache_path:
            save_generation_cache(results, cache_path)
            logger.info("  Cache updated (%d configs saved)", len(results))

    return results


def save_generation_cache(results: list[ConfigRunResult], cache_path: Path) -> None:
    """Save pipeline outputs to JSON for reuse."""
    data = []
    for r in results:
        data.append({
            "config_name": r.config_name,
            "config_path": r.config_path,
            "avg_latency_ms": r.avg_latency_ms,
            "latencies_ms": r.latencies_ms,
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


def load_generation_cache(cache_path: Path) -> list[ConfigRunResult]:
    """Load pipeline outputs from a JSON cache file."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    results = []
    for entry in data:
        samples = [PipelineSample(**s) for s in entry["samples"]]
        results.append(ConfigRunResult(
            config_name=entry["config_name"],
            config_path=entry["config_path"],
            samples=samples,
            latencies_ms=entry.get("latencies_ms", []),
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
    max_workers: int = 8,
) -> pd.DataFrame | None:
    """Single-pass LLM-as-judge evaluation (one call per question, all metrics)."""
    from src.evaluation.ragas_utils import build_ragas_evaluator
    from src.evaluation.single_judge import evaluate_batch, METRIC_NAMES

    try:
        ragas_llm, _ = build_ragas_evaluator(evaluator_type, evaluator_model)
    except Exception as e:
        logger.error("Failed to build evaluator LLM: %s", e)
        return None

    metrics_to_use = metric_names or METRIC_NAMES
    logger.info("  Single-pass LLM-as-judge — metrics: %s", metrics_to_use)

    total = len(results)
    rows = []
    for i, config_result in enumerate(results, 1):
        try:
            logger.info("  [%d/%d] Evaluating: %s (%d questions)...",
                        i, total, config_result.config_name, len(config_result.samples))

            judge_results = evaluate_batch(
                ragas_llm, config_result.samples, max_workers=max_workers,
            )

            detail_rows = [r.to_dict() for r in judge_results]
            detail_df = pd.DataFrame(detail_rows)

            # Keep only requested metrics in detail
            keep_cols = ["question", "response", "reference"] + [m for m in metrics_to_use if m in detail_df.columns]
            detail_df = detail_df[[c for c in keep_cols if c in detail_df.columns]]
            detail_path = results_dir / f"ragas_detail_{config_result.config_name}.csv"
            detail_df.to_csv(detail_path, index=False)

            row = {
                "config": config_result.config_name,
                "config_path": config_result.config_path,
                "avg_latency_ms": config_result.avg_latency_ms,
            }
            for m in metrics_to_use:
                if m in detail_df.columns:
                    row[m] = detail_df[m].mean()
            rows.append(row)
            logger.info("  [%d/%d] Done: %s", i, total, config_result.config_name)
        except Exception as e:
            logger.error("  Eval failed for %s: %s", config_result.config_name, e)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    csv_path = results_dir / "ragas_benchmark_results.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        new_configs = set(df["config"])
        existing = existing[~existing["config"].isin(new_configs)]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
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
    from src.evaluation.deterministic_metrics import compute_deterministic_metrics_batch

    rows = []
    for config_result in results:
        logger.info("  Deterministic metrics: %s", config_result.config_name)
        responses = [s.response for s in config_result.samples]
        references = [s.reference for s in config_result.samples]
        all_metrics = compute_deterministic_metrics_batch(
            responses=responses,
            references=references,
            bertscore_model=bertscore_model,
            st_model_name=st_model,
        )
        detail_rows = []
        for sample, metrics in zip(config_result.samples, all_metrics):
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
    csv_path = results_dir / "deterministic_benchmark_results.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        new_configs = set(df["config"])
        existing = existing[~existing["config"].isin(new_configs)]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Phase 4: Comparison + visualization + HTML report
# ---------------------------------------------------------------------------

def _load_qa_metadata(dataset_path: Path) -> dict[str, dict]:
    """Build a lookup: question text -> {type, difficulty}."""
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)
    lookup = {}
    for q in questions:
        meta = q.get("metadata", {})
        lookup[q["question"]] = {
            "type": meta.get("type", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
        }
    return lookup


def _load_detail_csvs(results_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    """Load all detail CSVs matching {prefix}_detail_*.csv, keyed by config name."""
    detail_dfs = {}
    pattern = f"{prefix}_detail_*.csv"
    for csv_path in sorted(results_dir.glob(pattern)):
        # Extract config name: ragas_detail_llama_hyde.csv -> llama_hyde
        config_name = csv_path.stem.replace(f"{prefix}_detail_", "")
        df = pd.read_csv(csv_path)
        # Normalize question column
        if "user_input" in df.columns:
            df = df.rename(columns={"user_input": "question"})
        detail_dfs[config_name] = df
    return detail_dfs


def build_statistical_analysis(
    results_dir: Path,
    plots_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Path]]:
    """Compute statistical summaries and generate boxplots."""
    from src.evaluation.visualization import (
        compute_statistical_summary,
        plot_metric_boxplots,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    ragas_details = _load_detail_csvs(results_dir, "ragas")
    det_details = _load_detail_csvs(results_dir, "deterministic")

    ragas_stats = None
    det_stats = None

    if ragas_details:
        first_df = next(iter(ragas_details.values()))
        ragas_metrics = [c for c in first_df.columns
                         if c not in ("question", "response", "reference", "user_input")
                         and pd.api.types.is_numeric_dtype(first_df[c])]
        ragas_stats = compute_statistical_summary(ragas_details, ragas_metrics)
        ragas_stats.to_csv(results_dir / "ragas_statistical_summary.csv", index=False)

        try:
            path = plots_dir / "ragas_boxplots.png"
            plot_metric_boxplots(ragas_details, ragas_metrics,
                                 title="RAGAS Metric Distributions", save_path=path)
            plot_paths["ragas_boxplots"] = path
        except Exception as e:
            logger.warning("Could not create RAGAS boxplots: %s", e)

    if det_details:
        first_df = next(iter(det_details.values()))
        det_metrics = [c for c in first_df.columns
                       if c not in ("question", "response", "reference")
                       and pd.api.types.is_numeric_dtype(first_df[c])]
        det_stats = compute_statistical_summary(det_details, det_metrics)
        det_stats.to_csv(results_dir / "deterministic_statistical_summary.csv", index=False)

        try:
            path = plots_dir / "deterministic_boxplots.png"
            plot_metric_boxplots(det_details, det_metrics,
                                 title="Deterministic Metric Distributions", save_path=path)
            plot_paths["deterministic_boxplots"] = path
        except Exception as e:
            logger.warning("Could not create deterministic boxplots: %s", e)

    return ragas_stats, det_stats, plot_paths


def build_category_analysis(
    results_dir: Path,
    dataset_path: Path,
    plots_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Path]]:
    """Group results by question type/difficulty and generate breakdown plots."""
    from src.evaluation.visualization import plot_category_breakdown

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    qa_meta = _load_qa_metadata(dataset_path)
    if not qa_meta:
        return None, None, {}

    # Load all detail CSVs (both RAGAS and deterministic)
    all_details = {}
    for prefix in ("ragas", "deterministic"):
        for config_name, df in _load_detail_csvs(results_dir, prefix).items():
            if config_name not in all_details:
                all_details[config_name] = df
            else:
                # Merge on question if both exist
                cols_to_add = [c for c in df.columns if c not in all_details[config_name].columns]
                if cols_to_add and "question" in df.columns:
                    all_details[config_name] = all_details[config_name].merge(
                        df[["question"] + cols_to_add], on="question", how="outer"
                    )

    if not all_details:
        return None, None, {}

    # Join QA metadata
    rows = []
    for config_name, df in all_details.items():
        for _, row in df.iterrows():
            q = row.get("question", "")
            meta = qa_meta.get(q, {})
            entry = row.to_dict()
            entry["config"] = config_name
            entry["type"] = meta.get("type", "unknown")
            entry["difficulty"] = meta.get("difficulty", "unknown")
            rows.append(entry)

    combined = pd.DataFrame(rows)
    metrics = [c for c in combined.columns
               if c not in ("question", "response", "reference", "config", "type", "difficulty")
               and pd.api.types.is_numeric_dtype(combined[c])]

    # Group by type
    type_df = None
    difficulty_df = None

    if "type" in combined.columns:
        type_df = combined.groupby(["config", "type"])[metrics].mean().reset_index()
        type_df.to_csv(results_dir / "category_by_type.csv", index=False)

        try:
            path = plots_dir / "category_by_type.png"
            plot_category_breakdown(combined, "type", metrics, title="Performance by Question Type",
                                    save_path=path)
            plot_paths["by_type"] = path
        except Exception as e:
            logger.warning("Could not create type breakdown plot: %s", e)

    if "difficulty" in combined.columns:
        difficulty_df = combined.groupby(["config", "difficulty"])[metrics].mean().reset_index()
        difficulty_df.to_csv(results_dir / "category_by_difficulty.csv", index=False)

        try:
            path = plots_dir / "category_by_difficulty.png"
            plot_category_breakdown(combined, "difficulty", metrics,
                                    title="Performance by Difficulty", save_path=path)
            plot_paths["by_difficulty"] = path
        except Exception as e:
            logger.warning("Could not create difficulty breakdown plot: %s", e)

    return type_df, difficulty_df, plot_paths


def classify_failure(row: pd.Series, threshold: float = 0.3) -> list[str]:
    """Classify failure type(s) for a single question-result row.

    Returns list of failure type strings. Empty if no failure detected.
    """
    failures = []

    faith = row.get("faithfulness", np.nan)
    a_corr = row.get("answer_correctness", np.nan)
    ctx_recall = row.get("context_recall", np.nan)
    ctx_prec = row.get("context_precision", np.nan)
    a_rel = row.get("answer_relevancy", np.nan)
    sem_sim = row.get("semantic_similarity", np.nan)
    bert_f1 = row.get("bertscore_f1", np.nan)

    if pd.notna(faith) and pd.notna(a_corr):
        if faith >= 0.7 and a_corr < threshold:
            failures.append("hallucination")

    if pd.notna(ctx_recall) and ctx_recall < threshold:
        failures.append("incomplete_retrieval")

    if pd.notna(ctx_prec) and ctx_prec < threshold:
        failures.append("retrieval_mismatch")

    if pd.notna(a_rel) and a_rel < threshold:
        failures.append("poor_generation")

    if pd.notna(sem_sim) and pd.notna(bert_f1):
        if sem_sim < threshold and bert_f1 < threshold:
            failures.append("low_similarity")

    # Catastrophic: all key RAGAS metrics below threshold
    key_metrics = [v for v in [faith, a_corr, ctx_recall, a_rel] if pd.notna(v)]
    if key_metrics and all(v < threshold for v in key_metrics):
        if "catastrophic" not in failures:
            failures.append("catastrophic")

    return failures


def build_failure_analysis(
    results_dir: Path,
    dataset_path: Path,
    plots_dir: Path,
    threshold: float = 0.3,
) -> tuple[pd.DataFrame | None, dict[str, Path]]:
    """Identify and categorize failure cases across all configs."""
    from src.evaluation.visualization import plot_failure_summary

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    qa_meta = _load_qa_metadata(dataset_path)

    ragas_details = _load_detail_csvs(results_dir, "ragas")
    det_details = _load_detail_csvs(results_dir, "deterministic")

    if not ragas_details and not det_details:
        return None, {}

    all_configs = set(ragas_details.keys()) | set(det_details.keys())
    failure_rows = []

    for config_name in sorted(all_configs):
        ragas_df = ragas_details.get(config_name)
        det_df = det_details.get(config_name)

        if ragas_df is not None and det_df is not None:
            merged = ragas_df.merge(det_df, on="question", how="outer", suffixes=("", "_det"))
        elif ragas_df is not None:
            merged = ragas_df
        else:
            merged = det_df

        for _, row in merged.iterrows():
            q = row.get("question", "")
            failures = classify_failure(row, threshold)
            if not failures:
                continue

            meta = qa_meta.get(q, {})
            failure_rows.append({
                "question": q[:120] + ("..." if len(q) > 120 else ""),
                "config": config_name,
                "type": meta.get("type", "unknown"),
                "difficulty": meta.get("difficulty", "unknown"),
                "failure_types": failures,
                "faithfulness": row.get("faithfulness", np.nan),
                "answer_relevancy": row.get("answer_relevancy", np.nan),
                "context_precision": row.get("context_precision", np.nan),
                "context_recall": row.get("context_recall", np.nan),
                "answer_correctness": row.get("answer_correctness", np.nan),
                "semantic_similarity": row.get("semantic_similarity", np.nan),
                "bertscore_f1": row.get("bertscore_f1", np.nan),
            })

    if not failure_rows:
        return None, {}

    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(results_dir / "failure_analysis.csv", index=False)

    try:
        path = plots_dir / "failure_summary.png"
        plot_failure_summary(failure_df, save_path=path)
        plot_paths["failure_summary"] = path
    except Exception as e:
        logger.warning("Could not create failure summary plot: %s", e)

    return failure_df, plot_paths


def build_no_rag_comparison(
    results_dir: Path,
    plots_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, Path]]:
    """Compare no-RAG baselines against their RAG counterparts."""
    from src.evaluation.visualization import plot_no_rag_comparison

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    ragas_summary = results_dir / "ragas_benchmark_results.csv"
    det_summary = results_dir / "deterministic_benchmark_results.csv"

    ragas_df = pd.read_csv(ragas_summary) if ragas_summary.exists() else None
    det_df = pd.read_csv(det_summary) if det_summary.exists() else None

    if ragas_df is None and det_df is None:
        return None, {}

    # Dynamically identify no-RAG configs and their RAG counterparts by LLM family
    all_configs = sorted(set(
        (ragas_df["config"].tolist() if ragas_df is not None else [])
        + (det_df["config"].tolist() if det_df is not None else [])
    ))
    no_rag_configs = [c for c in all_configs if "no_rag" in c]
    rag_configs = [c for c in all_configs if "no_rag" not in c]

    llm_map = {c: c.replace("no_rag_", "") for c in no_rag_configs}

    rag_counterparts = {}
    for no_rag_name, llm_label in llm_map.items():
        # Match by shared substring (e.g., "llama" or "gpt4o")
        keywords = llm_label.replace("_", "").lower().split("gpt4o")
        if "gpt4o" in llm_label.lower():
            keyword = "gpt4o"
        elif "llama" in llm_label.lower():
            keyword = "llama"
        else:
            keyword = llm_label.lower()
        rag_counterparts[no_rag_name] = [
            c for c in rag_configs if keyword in c.lower()
        ]

    rows = []
    for no_rag_config, llm_label in llm_map.items():
        # Collect scores from both frameworks
        for df, framework in [(ragas_df, "ragas"), (det_df, "det")]:
            if df is None:
                continue
            if no_rag_config not in df["config"].values:
                continue

            no_rag_row = df[df["config"] == no_rag_config].iloc[0]
            metric_cols = [c for c in df.columns
                           if c not in ("config", "config_path", "avg_latency_ms")]

            for counterpart in rag_counterparts.get(no_rag_config, []):
                if counterpart not in df["config"].values:
                    continue
                rag_row = df[df["config"] == counterpart].iloc[0]

                for metric in metric_cols:
                    rag_val = rag_row.get(metric, np.nan)
                    no_rag_val = no_rag_row.get(metric, np.nan)
                    delta = rag_val - no_rag_val if pd.notna(rag_val) and pd.notna(no_rag_val) else np.nan
                    rows.append({
                        "llm": llm_label,
                        "rag_config": counterpart,
                        "metric": metric,
                        "rag_score": rag_val,
                        "no_rag_score": no_rag_val,
                        "delta": delta,
                    })

    if not rows:
        return None, {}

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(results_dir / "no_rag_comparison.csv", index=False)

    try:
        path = plots_dir / "no_rag_comparison.png"
        plot_no_rag_comparison(comparison_df, save_path=path)
        plot_paths["no_rag_comparison"] = path
    except Exception as e:
        logger.warning("Could not create no-RAG comparison plot: %s", e)

    return comparison_df, plot_paths


def generate_report(
    ragas_df: pd.DataFrame | None,
    det_df: pd.DataFrame | None,
    results_dir: Path,
    plots_dir: Path,
    dataset_path: Path | None = None,
) -> None:
    """Generate correlation analysis, all plots, and HTML report."""
    from src.evaluation.visualization import (
        generate_comparison_report,
        plot_correlation_heatmap,
        plot_ranking_comparison,
        generate_html_report,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- New analytical features ---
    ragas_stats_df, det_stats_df, stats_plots = build_statistical_analysis(results_dir, plots_dir)
    category_type_df, category_diff_df, category_plots = None, None, {}
    failure_df, failure_plots = None, {}
    no_rag_comp_df, no_rag_plots = None, {}

    if dataset_path and dataset_path.exists():
        category_type_df, category_diff_df, category_plots = build_category_analysis(
            results_dir, dataset_path, plots_dir,
        )
        failure_df, failure_plots = build_failure_analysis(
            results_dir, dataset_path, plots_dir,
        )

    no_rag_comp_df, no_rag_plots = build_no_rag_comparison(results_dir, plots_dir)

    # --- Merge what we have for the full comparison plot suite ---
    if ragas_df is not None and det_df is not None:
        # Correlation analysis
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.compare_frameworks import compute_correlations, find_best_configs

        corr_df = compute_correlations(ragas_df, det_df)
        corr_df.to_csv(results_dir / "correlation_analysis.csv", index=False)

        # Best config analysis
        best = find_best_configs(ragas_df, det_df)

        # Correlation + ranking plots (need >=3 configs for meaningful correlations)
        if not corr_df.empty:
            plot_correlation_heatmap(corr_df, save_path=plots_dir / "correlation_heatmap.png")
        else:
            logger.info("  Skipping correlation heatmap (need >=3 configs)")

        plot_ranking_comparison(ragas_df, det_df, save_path=plots_dir / "ranking_comparison.png")

        # Full comparison suite (bar, radar, heatmap, latency, scatter)
        merged = ragas_df.merge(det_df, on="config", suffixes=("_ragas", "_det"))
        comparison_plots = generate_comparison_report(merged, output_dir=str(plots_dir))

        # HTML report
        all_plots = {
            **comparison_plots,
        }
        if not corr_df.empty:
            all_plots["correlation_heatmap.png"] = plots_dir / "correlation_heatmap.png"
        all_plots["ranking_comparison.png"] = plots_dir / "ranking_comparison.png"
        generate_html_report(
            ragas_df=ragas_df,
            det_df=det_df,
            corr_df=corr_df,
            best=best,
            plots=all_plots,
            output_path=results_dir / "benchmark_report.html",
            category_by_type_df=category_type_df,
            category_by_difficulty_df=category_diff_df,
            category_plots=category_plots,
            ragas_stats_df=ragas_stats_df,
            det_stats_df=det_stats_df,
            stats_plots=stats_plots,
            failure_df=failure_df,
            failure_plots=failure_plots,
            no_rag_comparison_df=no_rag_comp_df,
            no_rag_plots=no_rag_plots,
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
    parser.add_argument("--data-dir", default="data/docs",
                        help="Directory with raw PDFs for auto-ingestion")
    parser.add_argument("--config-dir", default="configs/")
    parser.add_argument("--config", nargs="+", default=None, help="One or more config YAML files")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--evaluator", default="openai", choices=["openai", "openrouter", "ollama"])
    parser.add_argument("--evaluator-model", default="gpt-4.1-mini")
    parser.add_argument("--bertscore-model", default="xlm-roberta-base")
    parser.add_argument("--st-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--ragas-metrics", nargs="+", default=None,
                        help="RAGAS metrics to run (default: all). "
                        "Options: faithfulness answer_relevancy context_precision "
                        "context_recall answer_correctness")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Max concurrent questions per config (default: 8). "
                             "Set to 1 for sequential behavior.")

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
            config_paths = [Path(c) for c in args.config]
        else:
            config_paths = sorted(Path(args.config_dir).rglob("*.yaml"))
        logger.info("Configs to evaluate: %s", [p.name for p in config_paths])

        if not config_paths:
            logger.error("No config files found")
            sys.exit(1)

        pipeline_results = run_pipeline_for_configs(
            config_paths, questions, max_workers=args.max_workers,
            cache_path=results_dir / "generation_cache.json",
            data_dir=Path(args.data_dir),
        )

        if not pipeline_results:
            logger.error("No configs could run. Check API keys / Ollama availability.")
            sys.exit(1)
    else:
        cache_path = results_dir / "generation_cache.json"
        if cache_path.exists():
            pipeline_results = load_generation_cache(cache_path)
            logger.info("Loaded %d configs from generation cache", len(pipeline_results))
        else:
            logger.warning("No pipeline cache found at %s", cache_path)

    # --- Phase 2: RAGAS ---
    ragas_df = None
    if not args.skip_ragas:
        if pipeline_results:
            logger.info("=== RAGAS Evaluation ===")
            ragas_df = evaluate_ragas(
                pipeline_results, args.evaluator, args.evaluator_model, results_dir,
                metric_names=args.ragas_metrics, max_workers=args.max_workers,
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
        # Fallback: load existing CSVs when phases were skipped
        if ragas_df is None:
            ragas_csv = results_dir / "ragas_benchmark_results.csv"
            if ragas_csv.exists():
                ragas_df = pd.read_csv(ragas_csv)
                logger.info("Loaded existing RAGAS results for report from %s", ragas_csv)
        if det_df is None:
            det_csv = results_dir / "deterministic_benchmark_results.csv"
            if det_csv.exists():
                det_df = pd.read_csv(det_csv)
                logger.info("Loaded existing deterministic results for report from %s", det_csv)

        logger.info("=== Generating Report ===")
        generate_report(ragas_df, det_df, results_dir, plots_dir,
                        dataset_path=Path(args.eval_dataset))
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
