"""Compare RAGAS (LLM-as-judge) vs deterministic NLP metrics.

Loads results from both benchmark frameworks, computes Spearman and Pearson
correlations, and produces visualizations answering whether both approaches
agree on the best RAG configuration.

Usage:
    python3 scripts/compare_frameworks.py
    python3 scripts/compare_frameworks.py --results-dir results
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
DETERMINISTIC_METRICS = ["bleu", "rouge_l", "meteor", "bertscore_f1", "semantic_similarity"]


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RAGAS and deterministic benchmark results."""
    ragas_path = results_dir / "ragas_benchmark_results.csv"
    det_path = results_dir / "deterministic_benchmark_results.csv"

    if not ragas_path.exists():
        logger.error("RAGAS results not found: %s", ragas_path)
        sys.exit(1)
    if not det_path.exists():
        logger.error("Deterministic results not found: %s", det_path)
        sys.exit(1)

    ragas_df = pd.read_csv(ragas_path)
    det_df = pd.read_csv(det_path)

    logger.info("RAGAS configs: %s", list(ragas_df["config"]))
    logger.info("Deterministic configs: %s", list(det_df["config"]))

    return ragas_df, det_df


def compute_correlations(ragas_df: pd.DataFrame, det_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations between RAGAS and deterministic metrics."""
    from scipy.stats import pearsonr, spearmanr

    merged = ragas_df.merge(det_df, on="config", suffixes=("_ragas", "_det"))
    if merged.empty:
        logger.error("No matching configs found between RAGAS and deterministic results.")
        sys.exit(1)

    rows = []
    for rm in RAGAS_METRICS:
        col_r = f"{rm}_ragas" if f"{rm}_ragas" in merged.columns else rm
        if col_r not in merged.columns:
            continue
        for dm in DETERMINISTIC_METRICS:
            col_d = f"{dm}_det" if f"{dm}_det" in merged.columns else dm
            if col_d not in merged.columns:
                continue

            vals_r = merged[col_r].dropna()
            vals_d = merged[col_d].dropna()

            # Align indices after dropna
            common = vals_r.index.intersection(vals_d.index)
            if len(common) < 3:
                continue

            r_vals = merged.loc[common, col_r].values
            d_vals = merged.loc[common, col_d].values

            spearman_r, spearman_p = spearmanr(r_vals, d_vals)
            pearson_r, pearson_p = pearsonr(r_vals, d_vals)

            rows.append({
                "ragas_metric": rm,
                "deterministic_metric": dm,
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "pearson_r": round(pearson_r, 4),
                "pearson_p": round(pearson_p, 4),
            })

    return pd.DataFrame(rows)


def find_best_configs(ragas_df: pd.DataFrame, det_df: pd.DataFrame) -> dict:
    """Find the best config according to each framework."""
    ragas_cols = [c for c in RAGAS_METRICS if c in ragas_df.columns]
    det_cols = [c for c in DETERMINISTIC_METRICS if c in det_df.columns]

    ragas_df = ragas_df.copy()
    det_df = det_df.copy()

    ragas_df["ragas_mean"] = ragas_df[ragas_cols].mean(axis=1)
    det_df["det_mean"] = det_df[det_cols].mean(axis=1)

    best_ragas = ragas_df.loc[ragas_df["ragas_mean"].idxmax(), "config"]
    best_det = det_df.loc[det_df["det_mean"].idxmax(), "config"]

    return {
        "best_ragas": best_ragas,
        "best_deterministic": best_det,
        "agreement": best_ragas == best_det,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAGAS vs deterministic metrics")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None, help="Directory for plots (default: results/plots)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    ragas_df, det_df = load_results(results_dir)

    # Merge for analysis
    merged = ragas_df.merge(det_df, on="config", suffixes=("_ragas", "_det"))

    # Compute correlations
    corr_df = compute_correlations(ragas_df, det_df)
    corr_path = results_dir / "correlation_analysis.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info("Correlation analysis saved to %s", corr_path)

    # Find best configs
    best = find_best_configs(ragas_df, det_df)
    print("\n" + "=" * 70)
    print("FRAMEWORK AGREEMENT ANALYSIS")
    print("=" * 70)
    print(f"Best config by RAGAS mean:         {best['best_ragas']}")
    print(f"Best config by deterministic mean: {best['best_deterministic']}")
    print(f"Agreement: {'YES' if best['agreement'] else 'NO'}")
    print("=" * 70)

    # Print correlation table
    print("\nCorrelation Analysis:")
    print(corr_df.to_string(index=False))

    # Generate visualizations
    try:
        from src.evaluation.visualization import (
            plot_correlation_heatmap,
            plot_ranking_comparison,
        )

        plot_correlation_heatmap(corr_df, save_path=output_dir / "correlation_heatmap.png")
        plot_ranking_comparison(ragas_df, det_df, save_path=output_dir / "ranking_comparison.png")
        logger.info("Plots saved to %s", output_dir)
    except Exception as e:
        logger.warning("Visualization failed: %s", e)


if __name__ == "__main__":
    main()
