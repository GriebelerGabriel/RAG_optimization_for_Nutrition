"""Visualization tools for comparing RAG experiment results."""
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def set_style() -> None:
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "RAG Configuration Comparison",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create bar chart comparing metrics across configurations.
    
    Args:
        df: Aggregated results DataFrame with 'config' column and metric columns.
        metrics: List of metrics to plot. If None, uses all numeric columns.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if metrics is None:
        # Default metrics to compare
        default_metrics = [
            "precision@5", "recall@5", "mrr", 
            "faithfulness", "answer_correctness", "citation_quality"
        ]
        metrics = [m for m in default_metrics if m in df.columns]
    
    # Filter to only existing metrics
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        raise ValueError("No valid metrics found in DataFrame")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric, alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, df[metric]):
            if not pd.isna(val):
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)

    return fig


def plot_radar_chart(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Configuration Radar Comparison",
    figsize: tuple[int, int] = (10, 10),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create radar/spider chart for multi-dimensional comparison.
    
    Args:
        df: Aggregated results DataFrame with 'config' column and metric columns.
        metrics: List of metrics to plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if metrics is None:
        default_metrics = [
            "precision@5", "recall@5", "mrr",
            "faithfulness", "abstain_correctness", "citation_quality"
        ]
        metrics = [m for m in default_metrics if m in df.columns]
    
    metrics = [m for m in metrics if m in df.columns]
    
    if len(metrics) < 3:
        raise ValueError("Need at least 3 metrics for radar chart")
    
    # Number of metrics
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, row in df.iterrows():
        values = [row[m] if not pd.isna(row[m]) else 0 for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['config'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)

    return fig


def plot_retrieval_vs_generation(
    df: pd.DataFrame,
    retrieval_metric: str = "mrr",
    generation_metric: str = "faithfulness",
    title: str = "Retrieval vs Generation Quality",
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create scatter plot comparing retrieval and generation metrics.
    
    This helps identify if poor generation is caused by retrieval issues.
    
    Args:
        df: Results DataFrame.
        retrieval_metric: X-axis metric (retrieval quality).
        generation_metric: Y-axis metric (generation quality).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if retrieval_metric not in df.columns or generation_metric not in df.columns:
        raise ValueError(f"Required metrics not found: {retrieval_metric}, {generation_metric}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each config with different color
    configs = df['config'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    
    for config, color in zip(configs, colors):
        config_data = df[df['config'] == config]
        ax.scatter(
            config_data[retrieval_metric], 
            config_data[generation_metric],
            c=[color], 
            label=config, 
            s=100, 
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5
        )
    
    # Add quadrant lines at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(0.25, 0.75, 'Good Gen\nBad Retrieval', ha='center', va='center', fontsize=9, alpha=0.6)
    ax.text(0.75, 0.75, 'Good Both', ha='center', va='center', fontsize=9, alpha=0.6)
    ax.text(0.25, 0.25, 'Bad Both', ha='center', va='center', fontsize=9, alpha=0.6)
    ax.text(0.75, 0.25, 'Good Retrieval\nBad Gen', ha='center', va='center', fontsize=9, alpha=0.6)
    
    ax.set_xlabel(f'Retrieval Quality ({retrieval_metric})')
    ax.set_ylabel(f'Generation Quality ({generation_metric})')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)
    else:
        plt.close(fig)

    return fig


def plot_latency_vs_quality(
    df: pd.DataFrame,
    quality_metric: str = "mrr",
    title: str = "Latency vs Quality Trade-off",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create scatter plot showing latency vs quality trade-off.
    
    Args:
        df: Aggregated results DataFrame.
        quality_metric: Quality metric to use for Y-axis.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if "latency_ms" not in df.columns or quality_metric not in df.columns:
        raise ValueError(f"Required columns not found")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        df['latency_ms'], 
        df[quality_metric],
        c=range(len(df)), 
        cmap='viridis',
        s=150, 
        alpha=0.8,
        edgecolors='white',
        linewidth=2
    )
    
    # Add config labels
    for idx, row in df.iterrows():
        ax.annotate(
            row['config'],
            (row['latency_ms'], row[quality_metric]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel(f'Quality ({quality_metric})')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)

    return fig


def plot_metric_heatmap(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Configuration Metrics Heatmap",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create heatmap of all metrics across configurations.
    
    Args:
        df: Aggregated results DataFrame.
        metrics: List of metrics to include.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if metrics is None:
        # Get all numeric columns except latency
        metrics = [c for c in df.columns if c not in ['config', 'latency_ms'] and df[c].dtype in ['float64', 'int64']]
    
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        raise ValueError("No valid metrics found")
    
    # Create heatmap data
    heatmap_data = df.set_index('config')[metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Score'}
    )
    
    ax.set_title(title)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Configuration')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)

    return fig


def plot_per_question_breakdown(
    df: pd.DataFrame,
    metric: str = "mrr",
    title: str | None = None,
    figsize: tuple[int, int] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create grouped bar chart showing per-question performance.
    
    Useful for identifying which questions are problematic across configs.
    
    Args:
        df: Non-aggregated results DataFrame with question_id column.
        metric: Metric to visualize.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    set_style()
    
    if metric not in df.columns or 'question_id' not in df.columns:
        raise ValueError(f"Required columns not found: {metric}, question_id")
    
    if title is None:
        title = f"Per-Question {metric} by Configuration"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pivot_df = df.pivot(index='question_id', columns='config', values=metric)
    pivot_df.plot(kind='bar', ax=ax, alpha=0.85)
    
    ax.set_xlabel('Question ID')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title='Configuration', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved plot to %s", save_path)

    return fig


def generate_comparison_report(
    df: pd.DataFrame,
    output_dir: str | Path = "results/plots",
    prefix: str = "comparison",
) -> dict[str, Path]:
    """Generate a full set of comparison plots.
    
    Args:
        df: Results DataFrame (can be aggregated or not).
        output_dir: Directory to save plots.
        prefix: Prefix for filenames.
        
    Returns:
        Dictionary mapping plot names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # Check if aggregated or per-question data
    is_aggregated = 'question_id' not in df.columns or df['question_id'].nunique() == 1
    
    # Aggregate if needed
    if not is_aggregated:
        agg_df = df.groupby('config').mean(numeric_only=True).reset_index()
    else:
        agg_df = df
    
    # 1. Bar chart comparison
    try:
        path = output_dir / f"{prefix}_bar_comparison.png"
        plot_metric_comparison(agg_df, save_path=path)
        saved_plots['bar_comparison'] = path
    except Exception as e:
        logger.warning("Could not create bar comparison: %s", e)

    # 2. Radar chart
    try:
        path = output_dir / f"{prefix}_radar.png"
        plot_radar_chart(agg_df, save_path=path)
        saved_plots['radar'] = path
    except Exception as e:
        logger.warning("Could not create radar chart: %s", e)

    # 3. Heatmap
    try:
        path = output_dir / f"{prefix}_heatmap.png"
        plot_metric_heatmap(agg_df, save_path=path)
        saved_plots['heatmap'] = path
    except Exception as e:
        logger.warning("Could not create heatmap: %s", e)

    # 4. Latency vs Quality
    try:
        path = output_dir / f"{prefix}_latency_quality.png"
        plot_latency_vs_quality(agg_df, save_path=path)
        saved_plots['latency_quality'] = path
    except Exception as e:
        logger.warning("Could not create latency vs quality plot: %s", e)

    # 5. Retrieval vs Generation (if generation metrics present)
    if 'faithfulness' in agg_df.columns:
        try:
            path = output_dir / f"{prefix}_retrieval_vs_generation.png"
            plot_retrieval_vs_generation(agg_df, save_path=path)
            saved_plots['retrieval_vs_generation'] = path
        except Exception as e:
            logger.warning("Could not create retrieval vs generation plot: %s", e)

    # 6. Per-question breakdown (if not aggregated)
    if not is_aggregated:
        try:
            path = output_dir / f"{prefix}_per_question.png"
            plot_per_question_breakdown(df, save_path=path)
            saved_plots['per_question'] = path
        except Exception as e:
            logger.warning("Could not create per-question plot: %s", e)

    logger.info("Generated %d plots in %s", len(saved_plots), output_dir)
    return saved_plots


def print_winner_summary(df: pd.DataFrame) -> str:
    """Print a summary of which config wins on each metric.
    
    Args:
        df: Aggregated results DataFrame.
        
    Returns:
        Summary string.
    """
    metrics = [c for c in df.columns if c not in ['config'] and df[c].dtype in ['float64', 'int64']]
    
    lines = ["=" * 50, "CONFIGURATION WINNER SUMMARY", "=" * 50, ""]
    
    for metric in metrics:
        if metric == 'latency_ms':
            # Lower is better for latency
            best_idx = df[metric].idxmin()
            direction = "lowest"
        else:
            # Higher is better for quality metrics
            best_idx = df[metric].idxmax()
            direction = "highest"
        
        best_config = df.loc[best_idx, 'config']
        best_value = df.loc[best_idx, metric]
        
        lines.append(f"  {metric:.<25} {best_config} ({direction}: {best_value:.3f})")
    
    lines.extend(["", "=" * 50])
    
    summary = "\n".join(lines)
    logger.info(summary)
    return summary
