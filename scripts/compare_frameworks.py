"""Compare RAGAS (LLM-as-judge) and deterministic metric frameworks."""

import logging
import sys

import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

DETERMINISTIC_METRICS = [
    "bleu",
    "rouge_l",
    "meteor",
    "bertscore_f1",
    "semantic_similarity",
]


def compute_correlations(
    ragas_df: pd.DataFrame, det_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations between RAGAS and deterministic metrics."""
    merged = ragas_df.merge(det_df, on="config", suffixes=("_ragas", "_det"))

    if merged.empty:
        logger.error(
            "No matching configs found between RAGAS and deterministic results."
        )
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


def find_best_configs(
    ragas_df: pd.DataFrame, det_df: pd.DataFrame
) -> dict:
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


def load_results(results_dir: str = "results") -> tuple:
    """Load RAGAS and deterministic benchmark CSVs."""
    ragas_path = f"{results_dir}/ragas_benchmark_results.csv"
    det_path = f"{results_dir}/deterministic_benchmark_results.csv"

    ragas_df = pd.read_csv(ragas_path) if __import__("os").path.exists(ragas_path) else None
    det_df = pd.read_csv(det_path) if __import__("os").path.exists(det_path) else None

    return ragas_df, det_df


def main() -> None:
    ragas_df, det_df = load_results()

    if ragas_df is not None and det_df is not None:
        corr_df = compute_correlations(ragas_df, det_df)
        print(corr_df.to_string(index=False))

        best = find_best_configs(ragas_df, det_df)
        print(f"\nBest by RAGAS:         {best['best_ragas']}")
        print(f"Best by deterministic: {best['best_deterministic']}")
        print(f"Agreement: {'YES' if best['agreement'] else 'NO'}")
    else:
        print("Missing results files.")


if __name__ == "__main__":
    main()
