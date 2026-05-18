"""Evaluation and benchmarking tools."""
from .metrics import RetrievalMetrics, compute_retrieval_metrics
from .generation_metrics import GenerationMetrics, compute_generation_metrics
from .deterministic_metrics import DeterministicMetrics, compute_deterministic_metrics
from .benchmark import Benchmark
from .visualization import (
    plot_metric_comparison,
    plot_radar_chart,
    plot_retrieval_vs_generation,
    plot_latency_vs_quality,
    plot_metric_heatmap,
    plot_per_question_breakdown,
    generate_comparison_report,
    print_winner_summary,
    plot_correlation_heatmap,
    plot_ranking_comparison,
    generate_html_report,
)

__all__ = [
    "RetrievalMetrics",
    "compute_retrieval_metrics",
    "GenerationMetrics",
    "compute_generation_metrics",
    "DeterministicMetrics",
    "compute_deterministic_metrics",
    "Benchmark",
    "plot_metric_comparison",
    "plot_radar_chart",
    "plot_retrieval_vs_generation",
    "plot_latency_vs_quality",
    "plot_metric_heatmap",
    "plot_per_question_breakdown",
    "generate_comparison_report",
    "print_winner_summary",
    "plot_correlation_heatmap",
    "plot_ranking_comparison",
    "generate_html_report",
]
