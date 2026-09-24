"""
agriculture_exclusion_check.py — Does excluding agriculture change the
headline seasonality findings?

Feedback received: QWI/LEHD relies on state UI wage records, and
agricultural employment has historically had weaker/partial UI coverage
than most other industries (some states exempt small agricultural
employers, and coverage was extended to agriculture later than to other
sectors in several states). If agriculture's QWI-covered establishments are
a non-representative (more seasonal, more UI-compliant) slice of the true
agricultural workforce, the sector could look more seasonal in this index
than agricultural employment as a whole actually is. This script asks: how
much of the headline ranking depends on agriculture (NAICS sector 11 --
crop production, animal production, forestry/logging, fishing/hunting/
trapping, and support activities for agriculture and forestry)?

Unlike covid_robustness_check.py / seasonality_trend_check.py, this does
NOT need to reload raw QWI or recompute excess-by-quarter: each industry's
own seasonal_amplitude / peak_excess is computed independently of every
other industry, so dropping agriculture rows from the already-committed
clean CSVs is methodologically identical to excluding agriculture from the
QWI input before the pipeline runs. The one thing that DOES need
recomputing is seasonal_index itself, since it's built by winsorizing
seasonal_amplitude at the 99th percentile of whatever pool of industries is
included -- removing agriculture (some of the largest amplitudes in the
national data) shifts that threshold for everyone else.

Outputs:
  output/tables/agriculture_exclusion_sector_summary.csv
  output/tables/agriculture_exclusion_top_industries.csv
  output/tables/agriculture_exclusion_state_ranking.csv
  output/figures/agriculture_exclusion_state_comparison.pdf

Usage:
    python code/qwi/agriculture_exclusion_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_CLEAN, FIGURES_DIR, TABLES_DIR

AG_SECTOR = "11"


def rewinsorize(df: pd.DataFrame, amplitude_col: str, index_col: str) -> pd.DataFrame:
    """Recompute a 0-1 index by 99th-pct winsorization of amplitude_col,
    matching build_seasonal_index()'s normalization in seasonal_index.py --
    but over whatever subset of rows is passed in (e.g. post agriculture
    exclusion), not the full national pool."""
    df = df.copy()
    p99 = df[amplitude_col].quantile(0.99)
    df[index_col] = (df[amplitude_col] / p99).clip(0, 1)
    return df


def national_sector_comparison(full: pd.DataFrame) -> pd.DataFrame:
    """Sector-level mean seasonal_index / peak_excess, with vs. without
    agriculture, re-winsorizing seasonal_index each time so the comparison
    isn't distorted by a stale winsorization threshold."""
    ex_ag = full[full["sector_2d"] != AG_SECTOR].copy()
    ex_ag = rewinsorize(ex_ag, "seasonal_amplitude", "seasonal_index")

    def sector_table(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby("sector_label")
            .agg(
                seasonal_index=("seasonal_index", "mean"),
                peak_excess=("peak_excess", "mean"),
                peak_quarter=("peak_quarter", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
                n_industries=("naics_code", "count"),
            )
            .sort_values("seasonal_index", ascending=False)
        )

    full_sectors = sector_table(full).add_suffix("_full")
    exag_sectors = sector_table(ex_ag).add_suffix("_exag")

    merged = full_sectors.join(exag_sectors, how="outer")
    merged["rank_full"] = merged["seasonal_index_full"].rank(ascending=False, method="min")
    merged["rank_exag"] = merged["seasonal_index_exag"].rank(ascending=False, method="min")
    merged["rank_delta"] = merged["rank_full"] - merged["rank_exag"]
    return merged.sort_values("rank_exag")


def national_top_industries(full: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N most seasonal individual industries, full sample vs. ex-agriculture,
    re-winsorized. Shows what fills the National Seasonality table once
    agriculture is dropped."""
    ex_ag = full[full["sector_2d"] != AG_SECTOR].copy()
    ex_ag = rewinsorize(ex_ag, "seasonal_amplitude", "seasonal_index")

    cols = ["naics_code", "national_industry_title", "sector_label",
            "seasonal_index", "peak_excess", "peak_quarter", "n_excess_obs"]
    cols = [c for c in cols if c in full.columns]

    top_full = full.sort_values("seasonal_index", ascending=False).head(n)[cols]
    top_exag = ex_ag.sort_values("seasonal_index", ascending=False).head(n)[cols]

    top_full = top_full.reset_index(drop=True)
    top_exag = top_exag.reset_index(drop=True)
    top_full.columns = [f"{c}_full" for c in top_full.columns]
    top_exag.columns = [f"{c}_exag" for c in top_exag.columns]
    return pd.concat([top_full, top_exag], axis=1)


def state_ranking_comparison(state_si: pd.DataFrame) -> pd.DataFrame:
    """Mean peak_excess by state, full sample vs. ex-agriculture. This is
    the exact quantity behind the 'most/least seasonal state' headline
    (geo_state_mean_seasonality figure) -- peak_excess is unaffected by
    winsorization, so no re-winsorizing is needed here, just dropping rows."""
    ex_ag = state_si[state_si["sector_2d"] != AG_SECTOR]

    full_avg = state_si.groupby("state_name")["peak_excess"].mean().rename("mean_peak_excess_full")
    exag_avg = ex_ag.groupby("state_name")["peak_excess"].mean().rename("mean_peak_excess_exag")

    merged = pd.concat([full_avg, exag_avg], axis=1)
    merged["rank_full"] = merged["mean_peak_excess_full"].rank(ascending=False, method="min")
    merged["rank_exag"] = merged["mean_peak_excess_exag"].rank(ascending=False, method="min")
    merged["rank_delta"] = merged["rank_full"] - merged["rank_exag"]
    return merged.sort_values("rank_exag")


def plot_state_comparison(state_rank: pd.DataFrame) -> plt.Figure:
    """Vertical bar chart, states sorted by ex-agriculture mean peak excess
    (descending), with the full-sample value overlaid for comparison --
    mirrors geo_state_mean_seasonality.png's layout/orientation."""
    df = state_rank.sort_values("mean_peak_excess_exag", ascending=False)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.2, df["mean_peak_excess_full"] * 100, width=0.4,
           color="#1f77b4", label="Full sample (incl. agriculture)")
    ax.bar(x + 0.2, df["mean_peak_excess_exag"] * 100, width=0.4,
           color="#d62728", label="Excluding agriculture")
    ax.set_ylabel("Mean peak quarter excess separation rate (p.p.)")
    ax.set_title("State-level seasonality, with vs. without agriculture\n"
                  "(mean peak excess across 6-digit NAICS industries)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(df) - 0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def print_summary(sector_cmp: pd.DataFrame, top_cmp: pd.DataFrame, state_cmp: pd.DataFrame) -> None:
    print("\n=== SECTOR RANKING: full sample vs. excluding agriculture ===")
    print(sector_cmp[["seasonal_index_full", "seasonal_index_exag",
                       "rank_full", "rank_exag", "rank_delta"]].round(3).to_string())

    print("\n=== TOP INDUSTRIES: full sample vs. excluding agriculture ===")
    print(top_cmp.to_string(index=False))

    print("\n=== STATE RANKING: full sample vs. excluding agriculture ===")
    n = len(state_cmp)
    corr_pearson = state_cmp[["mean_peak_excess_full", "mean_peak_excess_exag"]].corr().iloc[0, 1]
    corr_spearman = state_cmp[["rank_full", "rank_exag"]].corr(method="spearman").iloc[0, 1]
    print(f"{n} states. Pearson correlation: {corr_pearson:.3f}. Spearman (rank): {corr_spearman:.3f}")

    full_top = state_cmp["mean_peak_excess_full"].idxmax()
    exag_top = state_cmp["mean_peak_excess_exag"].idxmax()
    full_bot = state_cmp["mean_peak_excess_full"].idxmin()
    exag_bot = state_cmp["mean_peak_excess_exag"].idxmin()
    print(f"\nMost seasonal state,  full sample:        {full_top} "
          f"({state_cmp.loc[full_top, 'mean_peak_excess_full']*100:.2f} p.p.)")
    print(f"Most seasonal state,  excluding agriculture: {exag_top} "
          f"({state_cmp.loc[exag_top, 'mean_peak_excess_exag']*100:.2f} p.p.)")
    print(f"Least seasonal state, full sample:        {full_bot} "
          f"({state_cmp.loc[full_bot, 'mean_peak_excess_full']*100:.2f} p.p.)")
    print(f"Least seasonal state, excluding agriculture: {exag_bot} "
          f"({state_cmp.loc[exag_bot, 'mean_peak_excess_exag']*100:.2f} p.p.)")

    ratio_full = state_cmp["mean_peak_excess_full"].max() / state_cmp["mean_peak_excess_full"].min()
    ratio_exag = state_cmp["mean_peak_excess_exag"].max() / state_cmp["mean_peak_excess_exag"].min()
    print(f"\nMax/min ratio, full sample:        {ratio_full:.1f}x")
    print(f"Max/min ratio, excluding agriculture: {ratio_exag:.1f}x")

    print("\nBiggest state rank movers once agriculture is dropped "
          "(states that relied most on agriculture for their seasonality rank):")
    movers = state_cmp.reindex(state_cmp["rank_delta"].abs().sort_values(ascending=False).index)
    print(movers.head(10)[["mean_peak_excess_full", "mean_peak_excess_exag",
                            "rank_full", "rank_exag", "rank_delta"]].round(3).to_string())


def run() -> None:
    nat_path = QWI_CLEAN / "seasonal_index_naics6.csv"
    state_path = QWI_CLEAN / "seasonal_index_naics6_by_state.csv"
    if not nat_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            f"Expected {nat_path} and {state_path}. Run seasonal_index.py and "
            "geographic_analysis.py first."
        )

    full = pd.read_csv(nat_path, dtype={"naics_code": str, "sector_2d": str})
    state_si = pd.read_csv(state_path, dtype={"naics_code": str, "sector_2d": str, "state": str})

    full = full.dropna(subset=["seasonal_index", "peak_excess"])
    state_si = state_si.dropna(subset=["peak_excess"])

    sector_cmp = national_sector_comparison(full)
    top_cmp = national_top_industries(full)
    state_cmp = state_ranking_comparison(state_si)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sector_cmp.to_csv(TABLES_DIR / "agriculture_exclusion_sector_summary.csv")
    top_cmp.to_csv(TABLES_DIR / "agriculture_exclusion_top_industries.csv", index=False)
    state_cmp.to_csv(TABLES_DIR / "agriculture_exclusion_state_ranking.csv")

    fig = plot_state_comparison(state_cmp)
    fig.savefig(FIGURES_DIR / "agriculture_exclusion_state_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "agriculture_exclusion_state_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  {TABLES_DIR / 'agriculture_exclusion_sector_summary.csv'}")
    print(f"  {TABLES_DIR / 'agriculture_exclusion_top_industries.csv'}")
    print(f"  {TABLES_DIR / 'agriculture_exclusion_state_ranking.csv'}")
    print(f"  {FIGURES_DIR / 'agriculture_exclusion_state_comparison.pdf/png'}")

    print_summary(sector_cmp, top_cmp, state_cmp)


if __name__ == "__main__":
    run()
