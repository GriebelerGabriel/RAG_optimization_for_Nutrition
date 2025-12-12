"""Benchmarking framework for RAG experiments."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Config, load_config
from src.core.pipeline import RAGPipeline
from .metrics import RetrievalMetrics, compute_retrieval_metrics


@dataclass
class EvalQuestion:
    """A question in the evaluation dataset."""
    id: str
    question: str
    ideal_chunk_ids: list[str]
    ideal_answer: str | None = None


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config_name: str
    question_id: str
    question: str
    retrieved_ids: list[str]
    metrics: RetrievalMetrics
    response: str | None = None
    latency_ms: float = 0.0


class Benchmark:
    """Benchmark runner for RAG experiments."""
    
    def __init__(self, eval_dataset_path: str | Path):
        """Initialize benchmark with evaluation dataset.
        
        Args:
            eval_dataset_path: Path to golden_dataset.json
        """
        self.eval_dataset_path = Path(eval_dataset_path)
        self.questions: list[EvalQuestion] = []
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load evaluation dataset from JSON."""
        if not self.eval_dataset_path.exists():
            print(f"Warning: Evaluation dataset not found at {self.eval_dataset_path}")
            return
        
        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for q in data.get("questions", []):
            self.questions.append(EvalQuestion(
                id=q["id"],
                question=q["question"],
                ideal_chunk_ids=q.get("ideal_chunks", []),
                ideal_answer=q.get("ideal_answer"),
            ))
    
    def run_config(
        self, 
        config: Config | str | Path,
        include_generation: bool = False,
    ) -> list[BenchmarkResult]:
        """Run benchmark for a single configuration.
        
        Args:
            config: Config object or path to config file.
            include_generation: Whether to also benchmark generation.
            
        Returns:
            List of benchmark results.
        """
        if isinstance(config, (str, Path)):
            config = load_config(config)
        
        pipeline = RAGPipeline.from_config(config)
        results = []
        
        for question in self.questions:
            start_time = time.time()
            
            # Retrieve
            retrieved_docs = pipeline.retrieve(question.question)
            retrieved_ids = [doc.id for doc in retrieved_docs]
            
            # Compute metrics
            relevant_ids = set(question.ideal_chunk_ids)
            metrics = compute_retrieval_metrics(retrieved_docs, relevant_ids)
            
            # Optionally generate response
            response = None
            if include_generation:
                response = pipeline.generate(question.question, retrieved_docs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            results.append(BenchmarkResult(
                config_name=config.experiment_name,
                question_id=question.id,
                question=question.question,
                retrieved_ids=retrieved_ids,
                metrics=metrics,
                response=response,
                latency_ms=latency_ms,
            ))
        
        return results
    
    def run_all_configs(
        self, 
        config_dir: str | Path,
        include_generation: bool = False,
    ) -> pd.DataFrame:
        """Run benchmark for all configs in a directory.
        
        Args:
            config_dir: Directory containing YAML config files.
            include_generation: Whether to benchmark generation.
            
        Returns:
            DataFrame with all results.
        """
        config_dir = Path(config_dir)
        all_results = []
        
        for config_path in config_dir.glob("*.yaml"):
            print(f"Running benchmark for: {config_path.name}")
            try:
                results = self.run_config(config_path, include_generation)
                all_results.extend(results)
            except Exception as e:
                print(f"Error running {config_path.name}: {e}")
        
        return self._results_to_dataframe(all_results)
    
    def _results_to_dataframe(self, results: list[BenchmarkResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in results:
            row = {
                "config": r.config_name,
                "question_id": r.question_id,
                "question": r.question,
                "latency_ms": r.latency_ms,
                **r.metrics.to_dict(),
            }
            if r.response:
                row["response"] = r.response
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def aggregate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by config, computing mean metrics.
        
        Args:
            df: DataFrame from run_all_configs.
            
        Returns:
            Aggregated DataFrame with mean metrics per config.
        """
        metric_cols = ["precision@1", "precision@3", "precision@5", "recall@5", "mrr", "latency_ms"]
        return df.groupby("config")[metric_cols].mean().reset_index()
    
    def save_results(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save results to CSV."""
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
