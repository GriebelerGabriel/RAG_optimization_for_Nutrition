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


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Heatmap of Spearman correlations between RAGAS and deterministic metrics.

    Args:
        corr_df: DataFrame from compare_frameworks with columns
                 ragas_metric, deterministic_metric, spearman_r.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    pivot = corr_df.pivot(
        index="ragas_metric",
        columns="deterministic_metric",
        values="spearman_r",
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Spearman Correlation: RAGAS vs Deterministic Metrics")
    ax.set_ylabel("RAGAS Metrics")
    ax.set_xlabel("Deterministic Metrics")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved correlation heatmap to %s", save_path)

    return fig


def plot_ranking_comparison(
    ragas_df: pd.DataFrame,
    det_df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Side-by-side bar chart showing config rankings by each framework.

    Args:
        ragas_df: Aggregated RAGAS results with 'config' column.
        det_df: Aggregated deterministic results with 'config' column.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    ragas_cols = [c for c in ragas_df.columns if c not in ("config", "config_path", "avg_latency_ms")]
    det_cols = [c for c in det_df.columns if c not in ("config", "config_path", "avg_latency_ms")]

    ragas_df = ragas_df.copy()
    det_df = det_df.copy()
    ragas_df["framework_mean"] = ragas_df[ragas_cols].mean(axis=1)
    det_df["framework_mean"] = det_df[det_cols].mean(axis=1)

    merged = ragas_df[["config", "framework_mean"]].merge(
        det_df[["config", "framework_mean"]],
        on="config",
        suffixes=("_ragas", "_det"),
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(merged))
    width = 0.35

    bars1 = ax.bar(x - width / 2, merged["framework_mean_ragas"], width, label="RAGAS", alpha=0.85)
    bars2 = ax.bar(x + width / 2, merged["framework_mean_det"], width, label="Deterministic", alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Mean Score")
    ax.set_title("RAGAS vs Deterministic: Mean Metric Score by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["config"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ranking comparison to %s", save_path)

    return fig


def plot_metric_boxplots(
    detail_dfs: dict[str, pd.DataFrame],
    metrics: list[str],
    title: str = "Metric Distribution by Configuration",
    figsize: tuple[int, int] = (16, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create boxplots showing metric distributions across configs.

    Args:
        detail_dfs: Dict mapping config_name -> per-question detail DataFrame.
        metrics: Metric column names to plot.
        title: Figure title.
        figsize: Figure size per subplot row.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    all_rows = []
    for config_name, df in detail_dfs.items():
        for _, row in df.iterrows():
            entry = {"config": config_name}
            for m in metrics:
                entry[m] = row.get(m, np.nan)
            all_rows.append(entry)

    combined = pd.DataFrame(all_rows)

    n_metrics = len(metrics)
    n_configs = len(combined["config"].unique())
    plot_width = max(14, n_configs * 1.2)
    plot_height = 4 * n_metrics

    fig, axes = plt.subplots(n_metrics, 1, figsize=(plot_width, plot_height), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        sns.boxplot(data=combined, x="config", y=metric, ax=ax, hue="config", legend=False)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved boxplots to %s", save_path)

    return fig


def compute_statistical_summary(
    detail_dfs: dict[str, pd.DataFrame],
    metrics: list[str],
) -> pd.DataFrame:
    """Compute mean, std, min, max, median for each (config, metric) pair.

    Args:
        detail_dfs: Dict mapping config_name -> per-question detail DataFrame.
        metrics: Metric column names.

    Returns:
        DataFrame with columns: config, metric, mean, std, min, max, median, count.
    """
    rows = []
    for config_name, df in detail_dfs.items():
        for metric in metrics:
            vals = df[metric].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "config": config_name,
                "metric": metric,
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
                "median": vals.median(),
                "count": len(vals),
            })
    return pd.DataFrame(rows)


def plot_category_breakdown(
    df: pd.DataFrame,
    category_col: str,
    metrics: list[str],
    title: str | None = None,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: one subplot per metric, bars = configs, x-axis = categories.

    Args:
        df: DataFrame with columns: 'config', category_col, and metric columns.
            Must contain per-question rows with category metadata already joined.
        category_col: Column to group by ("type" or "difficulty").
        metrics: Metric columns to plot.
        title: Plot title.
        figsize: Figure size per subplot row.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    grouped = df.groupby(["config", category_col])[metrics].mean().reset_index()

    categories = sorted(grouped[category_col].unique())
    configs = sorted(grouped["config"].unique())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(16, 4 * n_metrics), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        x = np.arange(len(categories))
        width = 0.8 / max(len(configs), 1)

        for j, config in enumerate(configs):
            subset = grouped[grouped["config"] == config]
            vals = [subset[subset[category_col] == cat][metric].values[0]
                    if len(subset[subset[category_col] == cat]) > 0 else 0
                    for cat in categories]
            offset = (j - len(configs) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=config, alpha=0.85)

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel(category_col.title())
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)

    if configs:
        axes[0][0].legend(fontsize=7, loc="upper right")

    if title is None:
        title = f"Performance by {category_col.title()}"
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved category breakdown to %s", save_path)

    return fig


def plot_failure_summary(
    failure_df: pd.DataFrame,
    title: str = "Failure Analysis Summary",
    figsize: tuple[int, int] = (14, 7),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart showing failure type counts per config.

    Args:
        failure_df: DataFrame with 'config' and 'failure_types' (list column).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    # Explode list column so each failure type gets its own row
    exploded = failure_df.explode("failure_types")
    counts = exploded.groupby(["config", "failure_types"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", stacked=True, ax=ax, alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Number of Failures")
    ax.set_title(title)
    ax.legend(title="Failure Type", loc="upper right")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved failure summary to %s", save_path)

    return fig


def plot_no_rag_comparison(
    comparison_df: pd.DataFrame,
    title: str = "RAG vs No-RAG: Retrieval Impact",
    figsize: tuple[int, int] = (14, 7),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart showing RAG vs no-RAG scores per metric.

    Args:
        comparison_df: DataFrame with columns: llm, metric, rag_score, no_rag_score, delta.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    set_style()

    llms = comparison_df["llm"].unique()
    n_llms = len(llms)

    fig, axes = plt.subplots(1, max(n_llms, 1), figsize=figsize, squeeze=False)

    for i, llm in enumerate(llms):
        ax = axes[0][i]
        subset = comparison_df[comparison_df["llm"] == llm]
        metrics = subset["metric"].values
        x = np.arange(len(metrics))
        width = 0.35

        bars_rag = ax.bar(x - width / 2, subset["rag_score"], width, label="RAG", alpha=0.85, color="#4a90d9")
        bars_norag = ax.bar(x + width / 2, subset["no_rag_score"], width, label="No-RAG", alpha=0.85, color="#e57373")

        for bar, delta in zip(bars_rag, subset["delta"]):
            if not pd.isna(delta) and delta != 0:
                sign = "+" if delta > 0 else ""
                ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.02,
                        f"{sign}{delta:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(llm, fontsize=11)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved no-RAG comparison to %s", save_path)

    return fig


def _embed_image(path: Path) -> str:
    """Encode a PNG file as a base64 data URI."""
    import base64

    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _optional_section(heading: str, content: str) -> str:
    """Return an HTML section if content is non-empty, else empty string."""
    if not content.strip():
        return ""
    return f"<h2>{heading}</h2>\n{content}"


def _optional_plot(plots: dict[str, Path] | None, key: str) -> str:
    """Return an embedded image tag if the plot exists, else empty string."""
    if not plots or key not in plots:
        return ""
    path = plots[key]
    if not Path(path).exists():
        return ""
    label = key.replace("_", " ").replace(".png", "").title()
    return f"""<div class="plot"><h3>{label}</h3><img src="{_embed_image(path)}" alt="{label}" /></div>"""


def generate_html_report(
    ragas_df: pd.DataFrame,
    det_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    best: dict,
    plots: dict[str, Path],
    output_path: str | Path,
    title: str = "RAG Benchmark Report",
    # Feature 1: Category breakdown
    category_by_type_df: pd.DataFrame | None = None,
    category_by_difficulty_df: pd.DataFrame | None = None,
    category_plots: dict[str, Path] | None = None,
    # Feature 2: Statistical robustness
    ragas_stats_df: pd.DataFrame | None = None,
    det_stats_df: pd.DataFrame | None = None,
    stats_plots: dict[str, Path] | None = None,
    # Feature 3: Failure analysis
    failure_df: pd.DataFrame | None = None,
    failure_plots: dict[str, Path] | None = None,
    # Feature 4: No-RAG comparison
    no_rag_comparison_df: pd.DataFrame | None = None,
    no_rag_plots: dict[str, Path] | None = None,
) -> Path:
    """Generate a self-contained HTML report with embedded plots.

    Embeds PNG images as base64 so the report is a single shareable file.

    Args:
        ragas_df: RAGAS summary results (one row per config).
        det_df: Deterministic summary results (one row per config).
        corr_df: Correlation analysis results.
        best: Dict with 'best_ragas', 'best_deterministic', 'agreement'.
        plots: Dict mapping plot names to PNG file paths.
        output_path: Where to write the HTML file.
        title: Report title.
        category_by_type_df: Category breakdown by question type.
        category_by_difficulty_df: Category breakdown by difficulty.
        category_plots: Category breakdown plot paths.
        ragas_stats_df: RAGAS statistical summary.
        det_stats_df: Deterministic statistical summary.
        stats_plots: Boxplot paths.
        failure_df: Failure analysis details.
        failure_plots: Failure analysis plot paths.
        no_rag_comparison_df: RAG vs no-RAG comparison.
        no_rag_plots: No-RAG comparison plot paths.

    Returns:
        Path to the generated HTML file.
    """
    from datetime import datetime, timezone

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    agreement_cls = "agreement-yes" if best.get("agreement") else "agreement-no"
    agreement_text = "YES" if best.get("agreement") else "NO"

    plot_sections = ""
    for name, path in plots.items():
        if Path(path).exists():
            label = name.replace("_", " ").replace(".png", "").title()
            plot_sections += f"""
            <div class="plot">
                <h3>{label}</h3>
                <img src="{_embed_image(path)}" alt="{label}" />
            </div>"""

    ragas_table = ragas_df.to_html(index=False, float_format="%.4f", classes="data-table")
    det_table = det_df.to_html(index=False, float_format="%.4f", classes="data-table")
    corr_table = corr_df.to_html(index=False, float_format="%.4f", classes="data-table")

    # --- Feature 2: Statistical Robustness ---
    stats_html = ""
    if ragas_stats_df is not None and not ragas_stats_df.empty:
        stats_html += "<h3>RAGAS Metric Distributions</h3>\n"
        stats_html += ragas_stats_df.to_html(index=False, float_format="%.4f", classes="data-table")
    if det_stats_df is not None and not det_stats_df.empty:
        stats_html += "<h3>Deterministic Metric Distributions</h3>\n"
        stats_html += det_stats_df.to_html(index=False, float_format="%.4f", classes="data-table")
    if stats_plots:
        for key in ("ragas_boxplots", "deterministic_boxplots"):
            stats_html += _optional_plot(stats_plots, key)

    # --- Feature 1: Category Breakdown ---
    category_html = ""
    if category_by_type_df is not None and not category_by_type_df.empty:
        category_html += "<h3>Performance by Question Type</h3>\n"
        category_html += category_by_type_df.to_html(index=False, float_format="%.4f", classes="data-table")
    if category_by_difficulty_df is not None and not category_by_difficulty_df.empty:
        category_html += "<h3>Performance by Difficulty</h3>\n"
        category_html += category_by_difficulty_df.to_html(index=False, float_format="%.4f", classes="data-table")
    if category_plots:
        for key in ("by_type", "by_difficulty"):
            category_html += _optional_plot(category_plots, key)

    # --- Feature 3: Failure Analysis ---
    failure_html = ""
    if failure_df is not None and not failure_df.empty:
        failure_html += "<p>Questions with key metrics below threshold (0.3).</p>\n"
        if failure_plots:
            failure_html += _optional_plot(failure_plots, "failure_summary")
        display_cols = ["question", "config", "type", "difficulty", "failure_types"]
        avail_cols = [c for c in display_cols if c in failure_df.columns]
        failure_html += failure_df[avail_cols].head(50).to_html(index=False, classes="data-table")
        if len(failure_df) > 50:
            failure_html += f"<p><em>Showing 50 of {len(failure_df)} failure cases.</em></p>"
    elif failure_plots:
        failure_html += _optional_plot(failure_plots, "failure_summary")

    # --- Feature 4: No-RAG Comparison ---
    no_rag_html = ""
    if no_rag_comparison_df is not None and not no_rag_comparison_df.empty:
        no_rag_html += "<p>Shows how much retrieval improves answer quality over parametric knowledge alone.</p>\n"
        no_rag_html += no_rag_comparison_df.to_html(index=False, float_format="%.4f", classes="data-table")
    if no_rag_plots:
        no_rag_html += _optional_plot(no_rag_plots, "no_rag_comparison")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; color: #333; }}
  h1 {{ border-bottom: 2px solid #4a90d9; padding-bottom: 8px; }}
  h2 {{ color: #4a90d9; margin-top: 30px; }}
  .summary {{ background: #e8f5e9; padding: 15px 20px; border-radius: 8px;
              margin: 20px 0; }}
  .agreement-yes {{ color: #2e7d32; font-weight: bold; }}
  .agreement-no {{ color: #c62828; font-weight: bold; }}
  table.data-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }}
  table.data-table th {{ background: #4a90d9; color: #fff; padding: 8px 12px;
                         text-align: left; }}
  table.data-table td {{ border: 1px solid #ddd; padding: 6px 12px; }}
  table.data-table tr:nth-child(even) {{ background: #f5f5f5; }}
  .plot {{ text-align: center; margin: 20px 0; }}
  .plot img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .timestamp {{ color: #888; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="timestamp">Generated: {timestamp}</p>

<div class="summary">
  <h2 style="margin-top:0">Best Configuration</h2>
  <p><strong>By RAGAS (LLM-as-judge):</strong> {best.get('best_ragas', 'N/A')}</p>
  <p><strong>By Deterministic NLP metrics:</strong> {best.get('best_deterministic', 'N/A')}</p>
  <p><strong>Agreement:</strong> <span class="{agreement_cls}">{agreement_text}</span></p>
</div>

<h2>RAGAS Results (LLM-as-Judge)</h2>
{ragas_table}

<h2>Deterministic NLP Metrics</h2>
{det_table}

<h2>Correlation Analysis (Spearman)</h2>
{corr_table}

{_optional_section("Statistical Robustness", stats_html)}
{_optional_section("QA Category Breakdown", category_html)}
{_optional_section("Failure Analysis", failure_html)}
{_optional_section("No-RAG vs RAG Comparison", no_rag_html)}

<h2>Visualizations</h2>
{plot_sections}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", output_path)
    return output_path
