"""Benchmarking framework for RAG experiments."""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Config, load_config
from src.core.pipeline import RAGPipeline
from .metrics import RetrievalMetrics, compute_retrieval_metrics
from .generation_metrics import GenerationMetrics, compute_generation_metrics


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
    generation_metrics: GenerationMetrics | None = None
    # Artifact data for debugging hallucinations
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    prompt_used: str | None = None
    ideal_answer: str | None = None


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
        use_llm_eval: bool = False,
        save_artifacts: bool = True,
        artifacts_dir: str | Path | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmark for a single configuration.
        
        Args:
            config: Config object or path to config file.
            include_generation: Whether to also benchmark generation.
            use_llm_eval: Whether to use LLM-as-judge for generation metrics.
            save_artifacts: Whether to save per-question artifacts for debugging.
            artifacts_dir: Directory to save artifacts (default: results/artifacts).
            
        Returns:
            List of benchmark results.
        """
        if isinstance(config, (str, Path)):
            config = load_config(config)
        
        pipeline = RAGPipeline.from_config(config)
        results = []
        
        # Setup artifacts directory
        if save_artifacts:
            if artifacts_dir is None:
                artifacts_dir = Path("results/artifacts")
            artifacts_dir = Path(artifacts_dir) / config.experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        for question in self.questions:
            start_time = time.time()
            
            # Retrieve
            retrieved_docs = pipeline.retrieve(question.question)
            retrieved_ids = [doc.id for doc in retrieved_docs]
            
            # Compute retrieval metrics
            relevant_ids = set(question.ideal_chunk_ids)
            metrics = compute_retrieval_metrics(retrieved_docs, relevant_ids)
            
            # Store chunk data for artifacts
            retrieved_chunks = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ]
            
            # Optionally generate response and compute generation metrics
            response = None
            generation_metrics = None
            prompt_used = None
            
            if include_generation:
                response = pipeline.generate(question.question, retrieved_docs)
                
                # Build the prompt that was used (for artifact logging)
                prompt_used = self._build_prompt_snapshot(question.question, retrieved_docs, pipeline)
                
                # Compute generation-grounded metrics
                has_relevant_context = len(relevant_ids & set(retrieved_ids)) > 0
                generation_metrics = compute_generation_metrics(
                    response=response,
                    context=retrieved_docs,
                    ideal_answer=question.ideal_answer,
                    has_relevant_context=has_relevant_context,
                    llm=pipeline.llm if use_llm_eval else None,
                    use_llm_eval=use_llm_eval,
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = BenchmarkResult(
                config_name=config.experiment_name,
                question_id=question.id,
                question=question.question,
                retrieved_ids=retrieved_ids,
                metrics=metrics,
                response=response,
                latency_ms=latency_ms,
                generation_metrics=generation_metrics,
                retrieved_chunks=retrieved_chunks,
                prompt_used=prompt_used,
                ideal_answer=question.ideal_answer,
            )
            results.append(result)
            
            # Save per-question artifact
            if save_artifacts:
                self._save_artifact(result, artifacts_dir)
        
        return results
    
    def _build_prompt_snapshot(
        self, 
        query: str, 
        context: list, 
        pipeline: RAGPipeline,
    ) -> str:
        """Build a snapshot of the prompt used for generation."""
        from src.core.document import Document
        context_text = "\n\n".join([
            f"[Fonte {i+1}]: {doc.content}" 
            for i, doc in enumerate(context)
        ])
        system_prompt = pipeline.llm._default_system_prompt()
        if hasattr(pipeline.config, 'llm') and pipeline.config.llm.system_prompt:
            system_prompt = pipeline.config.llm.system_prompt
        
        return f"""{system_prompt}

Contexto:
{context_text}

Pergunta do usuário: {query}

Resposta:"""
    
    def _save_artifact(self, result: BenchmarkResult, artifacts_dir: Path) -> None:
        """Save per-question artifact for debugging."""
        artifact = {
            "question_id": result.question_id,
            "question": result.question,
            "ideal_answer": result.ideal_answer,
            "response": result.response,
            "retrieved_chunks": result.retrieved_chunks,
            "retrieved_ids": result.retrieved_ids,
            "prompt_used": result.prompt_used,
            "retrieval_metrics": result.metrics.to_dict(),
            "generation_metrics": result.generation_metrics.to_dict() if result.generation_metrics else None,
            "latency_ms": result.latency_ms,
        }
        
        artifact_path = artifacts_dir / f"{result.question_id}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
    
    def run_all_configs(
        self, 
        config_dir: str | Path,
        include_generation: bool = False,
        use_llm_eval: bool = False,
        save_artifacts: bool = True,
    ) -> pd.DataFrame:
        """Run benchmark for all configs in a directory.
        
        Args:
            config_dir: Directory containing YAML config files.
            include_generation: Whether to benchmark generation.
            use_llm_eval: Whether to use LLM-as-judge for generation metrics.
            save_artifacts: Whether to save per-question artifacts.
            
        Returns:
            DataFrame with all results.
        """
        config_dir = Path(config_dir)
        all_results = []
        
        for config_path in config_dir.glob("*.yaml"):
            print(f"Running benchmark for: {config_path.name}")
            try:
                results = self.run_config(
                    config_path, 
                    include_generation=include_generation,
                    use_llm_eval=use_llm_eval,
                    save_artifacts=save_artifacts,
                )
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
            if r.generation_metrics:
                row.update(r.generation_metrics.to_dict())
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def aggregate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by config, computing mean metrics.
        
        Args:
            df: DataFrame from run_all_configs.
            
        Returns:
            Aggregated DataFrame with mean metrics per config.
        """
        retrieval_cols = ["precision@1", "precision@3", "precision@5", "recall@5", "mrr", "latency_ms"]
        generation_cols = ["faithfulness", "answer_correctness", "abstain_correctness", "citation_quality"]
        
        # Only include columns that exist in the dataframe
        metric_cols = [c for c in retrieval_cols + generation_cols if c in df.columns]
        
        # For answer_correctness, filter out -1 values (N/A) before averaging
        if "answer_correctness" in df.columns:
            df_filtered = df.copy()
            df_filtered.loc[df_filtered["answer_correctness"] < 0, "answer_correctness"] = float("nan")
            return df_filtered.groupby("config")[metric_cols].mean().reset_index()
        
        return df.groupby("config")[metric_cols].mean().reset_index()
    
    def save_results(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save results to CSV."""
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
