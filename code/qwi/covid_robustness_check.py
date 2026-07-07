"""
covid_robustness_check.py — Check whether 2020-2021 pandemic disruption distorts
the QWI seasonal index.

The national seasonal index (seasonal_index.py) pools 2000-2023. Mass pandemic-era
layoffs in industries already ranked highly seasonal (accommodation/food, arts &
entertainment) could inflate their "excess separation" estimates for reasons that
have nothing to do with normal annual seasonality. This script recomputes the index
restricted to 2000-2019 (pre-COVID) using the same methodology, and compares it
against the full-sample index already saved by seasonal_index.py.

Reuses seasonal_index.py's own pipeline functions (load_qwi, compute_sep_rates,
compute_excess_recurrence_by_quarter, build_seasonal_index) so the pre-COVID
index is computed identically to the full-sample one, differing only in the
year range of the underlying data.

Outputs:
  output/tables/covid_robustness_naics6.csv
  output/figures/covid_robustness_scatter.pdf

Usage:
    python code/qwi/covid_robustness_check.py
    python code/qwi/covid_robustness_check.py --cutoff-year 2020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, TABLES_DIR

from seasonal_index import (
    load_qwi,
    compute_sep_rates,
    compute_excess_recurrence_by_quarter,
    build_seasonal_index,
)

# Sectors of particular interest: hit hardest by pandemic-era shutdowns, and
# already ranked as highly seasonal in the full-sample index.
WATCH_SECTORS = {
    "11": "Agriculture", "23": "Construction", "61": "Education",
    "71": "Arts/entertainment", "72": "Accommodation/food",
}


def build_precovid_index(cutoff_year: int, qwi_path: Path | None = None) -> pd.DataFrame:
    """Recompute the seasonal index using only years < cutoff_year."""
    qwi = load_qwi(path=qwi_path)
    qwi = qwi[qwi["year"] < cutoff_year].copy()
    print(f"Restricted to years < {cutoff_year}: {len(qwi):,} rows, "
          f"years {qwi['year'].min():.0f}-{qwi['year'].max():.0f}")

    rates = compute_sep_rates(qwi)
    excess = compute_excess_recurrence_by_quarter(rates)
    return build_seasonal_index(excess)


def compare_indices(full: pd.DataFrame, precovid: pd.DataFrame) -> pd.DataFrame:
    """Merge full-sample and pre-COVID indices and compute rank/value shifts."""
    full = full[["naics_code", "sector_2d", "sector_label", "seasonal_index",
                 "peak_excess", "peak_quarter", "n_excess_obs"]].copy()
    pre = precovid[["naics_code", "seasonal_index", "peak_excess", "peak_quarter",
                     "n_excess_obs"]].copy()

    full = full.dropna(subset=["seasonal_index"])
    pre = pre.dropna(subset=["seasonal_index"])

    full["rank_full"] = full["seasonal_index"].rank(ascending=False, method="min")
    pre["rank_precovid"] = pre["seasonal_index"].rank(ascending=False, method="min")

    merged = full.merge(
        pre, on="naics_code", how="inner", suffixes=("_full", "_precovid")
    )
    merged["seasonal_index_delta"] = (
        merged["seasonal_index_precovid"] - merged["seasonal_index_full"]
    )
    merged["rank_delta"] = merged["rank_full"] - merged["rank_precovid"]
    merged["peak_quarter_changed"] = (
        merged["peak_quarter_full"] != merged["peak_quarter_precovid"]
    )

    return merged.sort_values("seasonal_index_delta")


def print_summary(merged: pd.DataFrame) -> None:
    n = len(merged)
    corr_pearson = merged[["seasonal_index_full", "seasonal_index_precovid"]].corr().iloc[0, 1]
    corr_spearman = merged[["rank_full", "rank_precovid"]].corr(method="spearman").iloc[0, 1]

    print(f"\n{n} industries with a seasonal_index in both samples.")
    print(f"Pearson correlation (seasonal_index, full vs. pre-COVID):  {corr_pearson:.3f}")
    print(f"Spearman correlation (rank, full vs. pre-COVID):           {corr_spearman:.3f}")
    print(f"Industries whose peak quarter changed: "
          f"{merged['peak_quarter_changed'].sum()} / {n} "
          f"({merged['peak_quarter_changed'].mean():.1%})")

    print("\nBiggest rank movers (|rank_delta|, i.e. moved most once COVID years are excluded):")
    movers = merged.reindex(merged["rank_delta"].abs().sort_values(ascending=False).index)
    print(movers.head(10)[
        ["naics_code", "sector_label", "rank_full", "rank_precovid", "rank_delta",
         "seasonal_index_full", "seasonal_index_precovid"]
    ].to_string(index=False))

    print("\nWatch sectors (pandemic-exposed, already ranked highly seasonal):")
    watch = merged[merged["sector_2d"].isin(WATCH_SECTORS)]
    sector_summary = (
        watch.groupby("sector_2d")[["seasonal_index_full", "seasonal_index_precovid"]]
        .mean()
        .rename(index=WATCH_SECTORS)
    )
    sector_summary["delta"] = (
        sector_summary["seasonal_index_precovid"] - sector_summary["seasonal_index_full"]
    )
    print(sector_summary.round(4).to_string())


def plot_comparison(merged: pd.DataFrame) -> plt.Figure:
    """Scatter of full-sample vs. pre-COVID seasonal_index, one point per industry."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = np.where(merged["sector_2d"].isin(WATCH_SECTORS), "#d62728", "#1f77b4")
    ax.scatter(
        merged["seasonal_index_full"], merged["seasonal_index_precovid"],
        s=10, alpha=0.5, c=colors,
    )
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel("Seasonal index, full sample (2000-2023)")
    ax.set_ylabel("Seasonal index, pre-COVID (2000-2019)")
    ax.set_title("Does excluding 2020-2021 change industry seasonality rankings?")
    ax.legend(["y = x", "watch sectors (agri/constr/educ/arts/accom.)"], fontsize=8)
    fig.tight_layout()
    return fig


def run(cutoff_year: int = 2020, qwi_path: Path | None = None) -> pd.DataFrame:
    full_path = QWI_CLEAN / "seasonal_index_naics6.csv"
    if not full_path.exists():
        raise FileNotFoundError(
            f"{full_path} not found. Run seasonal_index.py first to build the "
            "full-sample index."
        )
    full = pd.read_csv(full_path, dtype={"naics_code": str, "sector_2d": str})

    precovid = build_precovid_index(cutoff_year=cutoff_year, qwi_path=qwi_path)

    merged = compare_indices(full, precovid)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "covid_robustness_naics6.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved comparison table to {out_path}")

    fig = plot_comparison(merged)
    fig_path = FIGURES_DIR / "covid_robustness_scatter.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    print_summary(merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the full-sample seasonal index to a pre-COVID-only version."
    )
    parser.add_argument(
        "--cutoff-year", type=int, default=2020,
        help="Pre-COVID sample is years strictly before this (default: 2020, i.e. 2000-2019).",
    )
    parser.add_argument("--qwi-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(cutoff_year=args.cutoff_year, qwi_path=args.qwi_file)
