"""
seasonality_trend_check.py — Has industry seasonality changed over time?

Splits the QWI sample into an early period (2000-2010) and a late period
(2011-2023) and recomputes the national seasonal index separately for each,
using the same methodology as seasonal_index.py. Comparing the two tests
whether industries have gotten more or less seasonal over the sample, rather
than (as in covid_robustness_check.py) whether a specific event distorts the
pooled index.

The MIN_EXCESS_OBS=5 threshold used everywhere else sets the natural floor
for this split: 2000-2010 is 11 years and 2011-2023 is 13, both comfortably
above the 5-year minimum needed for an industry to register a valid
excess-by-quarter estimate in either half (unlike a 3- or 4-way split, which
would push many industries below that floor).

Reuses seasonal_index.py's own pipeline functions (load_qwi, compute_sep_rates,
compute_excess_recurrence_by_quarter, build_seasonal_index) so both periods
are computed identically, differing only in the year range of the underlying
data.

Outputs:
  output/tables/seasonality_trend_naics6.csv       (every industry, both periods)
  output/tables/seasonality_trend_by_sector.csv     (sector-level mean shift)
  output/figures/seasonality_trend_scatter.pdf
  output/figures/seasonality_trend_by_sector.pdf

Usage:
    python code/qwi/seasonality_trend_check.py
    python code/qwi/seasonality_trend_check.py --split-year 2011
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
from config import FIGURES_DIR, TABLES_DIR

from seasonal_index import (
    load_qwi,
    compute_sep_rates,
    compute_excess_recurrence_by_quarter,
    build_seasonal_index,
)


def build_period_index(
    year_start: int, year_end: int, qwi: pd.DataFrame
) -> pd.DataFrame:
    """Recompute the seasonal index using only year_start <= year <= year_end."""
    period_qwi = qwi[(qwi["year"] >= year_start) & (qwi["year"] <= year_end)].copy()
    n_years = year_end - year_start + 1
    print(f"Period {year_start}-{year_end} ({n_years} years): {len(period_qwi):,} rows")

    rates = compute_sep_rates(period_qwi)
    excess = compute_excess_recurrence_by_quarter(rates)
    return build_seasonal_index(excess)


def compare_periods(
    early: pd.DataFrame, late: pd.DataFrame, early_label: str, late_label: str
) -> pd.DataFrame:
    """Merge the two periods' indices and compute value/rank shifts."""
    e = early[["naics_code", "sector_2d", "sector_label", "seasonal_index",
               "peak_excess", "peak_quarter", "n_excess_obs"]].dropna(subset=["seasonal_index"])
    l = late[["naics_code", "seasonal_index", "peak_excess", "peak_quarter",
              "n_excess_obs"]].dropna(subset=["seasonal_index"])

    e = e.copy()
    l = l.copy()
    e["rank_early"] = e["seasonal_index"].rank(ascending=False, method="min")
    l["rank_late"] = l["seasonal_index"].rank(ascending=False, method="min")

    merged = e.merge(l, on="naics_code", how="inner", suffixes=("_early", "_late"))
    merged["seasonal_index_delta"] = merged["seasonal_index_late"] - merged["seasonal_index_early"]
    merged["rank_delta"] = merged["rank_early"] - merged["rank_late"]
    merged["peak_quarter_changed"] = merged["peak_quarter_early"] != merged["peak_quarter_late"]
    merged.attrs["early_label"] = early_label
    merged.attrs["late_label"] = late_label

    return merged.sort_values("seasonal_index_delta")


def sector_trend_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Mean seasonal_index by sector in each period, sorted by the size of the shift."""
    summary = (
        merged.groupby("sector_label")[["seasonal_index_early", "seasonal_index_late"]]
        .agg(["mean", "count"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.rename(columns={
        "seasonal_index_early_mean": "mean_early", "seasonal_index_early_count": "n_early",
        "seasonal_index_late_mean": "mean_late", "seasonal_index_late_count": "n_late",
    })[["mean_early", "mean_late", "n_early", "n_late"]]
    summary["delta"] = summary["mean_late"] - summary["mean_early"]
    return summary.sort_values("delta")


def print_summary(merged: pd.DataFrame, n_early_total: int, n_late_total: int) -> None:
    n = len(merged)
    early_label = merged.attrs.get("early_label", "early")
    late_label = merged.attrs.get("late_label", "late")

    corr_pearson = merged[["seasonal_index_early", "seasonal_index_late"]].corr().iloc[0, 1]
    corr_spearman = merged[["rank_early", "rank_late"]].corr(method="spearman").iloc[0, 1]

    print(f"\n{n} industries with a valid seasonal_index in both periods "
          f"({n_early_total} qualified in {early_label}, {n_late_total} in {late_label} "
          f"before requiring both).")
    print(f"Pearson correlation (seasonal_index, {early_label} vs. {late_label}):  {corr_pearson:.3f}")
    print(f"Spearman correlation (rank, {early_label} vs. {late_label}):           {corr_spearman:.3f}")
    print(f"Industries whose peak quarter changed: "
          f"{merged['peak_quarter_changed'].sum()} / {n} "
          f"({merged['peak_quarter_changed'].mean():.1%})")

    mean_early = merged["seasonal_index_early"].mean()
    mean_late = merged["seasonal_index_late"].mean()
    print(f"\nEconomy-wide mean seasonal_index: {mean_early:.4f} ({early_label}) "
          f"-> {mean_late:.4f} ({late_label})  [delta {mean_late - mean_early:+.4f}]")

    print(f"\nBiggest movers (|seasonal_index_delta|, {early_label} -> {late_label}):")
    movers = merged.reindex(merged["seasonal_index_delta"].abs().sort_values(ascending=False).index)
    print(movers.head(15)[
        ["naics_code", "sector_label", "seasonal_index_early", "seasonal_index_late",
         "seasonal_index_delta", "rank_delta"]
    ].to_string(index=False))

    print(f"\nSector-level mean seasonal_index, {early_label} vs. {late_label} "
          f"(sorted by shift; needs >=3 industries in the sector to show):")
    sectors = sector_trend_summary(merged)
    sectors = sectors[(sectors["n_early"] >= 3) & (sectors["n_late"] >= 3)]
    print(sectors.round(4).to_string())


def plot_comparison(merged: pd.DataFrame) -> plt.Figure:
    """Scatter of early- vs. late-period seasonal_index, one point per industry."""
    early_label = merged.attrs.get("early_label", "early")
    late_label = merged.attrs.get("late_label", "late")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        merged["seasonal_index_early"], merged["seasonal_index_late"],
        s=10, alpha=0.4, c="#1f77b4",
    )
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x (no change)")
    ax.set_xlabel(f"Seasonal index, {early_label}")
    ax.set_ylabel(f"Seasonal index, {late_label}")
    ax.set_title("Has industry seasonality changed over time?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sector_trend(merged: pd.DataFrame) -> plt.Figure:
    """Bar chart of sector-level mean seasonal_index shift, early vs. late."""
    early_label = merged.attrs.get("early_label", "early")
    late_label = merged.attrs.get("late_label", "late")

    sectors = sector_trend_summary(merged)
    sectors = sectors[(sectors["n_early"] >= 3) & (sectors["n_late"] >= 3)].sort_values("delta")

    fig, ax = plt.subplots(figsize=(8, max(4, len(sectors) * 0.35)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in sectors["delta"]]
    ax.barh(sectors.index, sectors["delta"], color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"Change in mean seasonal_index ({early_label} -> {late_label})")
    ax.set_title("Which sectors got more (green) or less (red) seasonal?")
    fig.tight_layout()
    return fig


def run(split_year: int = 2011, qwi_path: Path | None = None) -> pd.DataFrame:
    qwi = load_qwi(path=qwi_path)
    year_min, year_max = int(qwi["year"].min()), int(qwi["year"].max())

    early_label = f"{year_min}-{split_year - 1}"
    late_label = f"{split_year}-{year_max}"

    early = build_period_index(year_min, split_year - 1, qwi)
    late = build_period_index(split_year, year_max, qwi)
    n_early_total = early["seasonal_index"].notna().sum()
    n_late_total = late["seasonal_index"].notna().sum()

    merged = compare_periods(early, late, early_label, late_label)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = TABLES_DIR / "seasonality_trend_naics6.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved comparison table to {out_path}")

    sector_path = TABLES_DIR / "seasonality_trend_by_sector.csv"
    sector_trend_summary(merged).to_csv(sector_path)
    print(f"Saved sector-level table to {sector_path}")

    fig = plot_comparison(merged)
    fig.savefig(FIGURES_DIR / "seasonality_trend_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    fig2 = plot_sector_trend(merged)
    fig2.savefig(FIGURES_DIR / "seasonality_trend_by_sector.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figures to {FIGURES_DIR}")

    print_summary(merged, n_early_total, n_late_total)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the seasonal index between an early and late period to test for a trend."
    )
    parser.add_argument(
        "--split-year", type=int, default=2011,
        help="Late period starts here (default: 2011, i.e. 2000-2010 vs 2011-2023).",
    )
    parser.add_argument("--qwi-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(split_year=args.split_year, qwi_path=args.qwi_file)
