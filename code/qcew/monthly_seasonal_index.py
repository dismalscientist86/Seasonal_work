"""
monthly_seasonal_index.py -- Monthly (QCEW) vs. quarterly (QWI) stock-based
seasonal index: does monthly resolution change the seasonality picture?

QCEW has no separations/hires field, so it cannot replicate this project's
flow-based seasonal_index (excess separation rate). What it *can* do is
sharpen the timing resolution of the existing *stock* measure
(seasonal_index_emp in seasonal_index.py, built from quarterly employment
shares) from quarterly to monthly -- the same construction, applied to
QCEW's month1_emplvl/month2_emplvl/month3_emplvl instead of QWI's quarterly
EmpEnd:

    emp_share_{j,y,m} = emplvl_{j,y,m} / mean_m(emplvl_{j,y,*})
    Excess(m)_j = mean_y[ share(m,y) - (share(m-1,y) + share(m+1,y)) / 2 ]

for each calendar month m in 1..12, cyclically (December's neighbors are
November and next January), using excess_utils.cyclical_excess_by_month()
-- the same shared calculation as every other measure in this project,
generalized from 4 periods to 12.

Outputs:
  data/qcew_clean/monthly_seasonal_index_naics6.csv
  output/tables/monthly_vs_quarterly_stock_comparison.csv
  output/figures/monthly_vs_quarterly_stock_scatter.pdf

Usage:
    python code/qcew/monthly_seasonal_index.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # code/  (config)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwi"))  # code/qwi/  (excess_utils, seasonal_index)
from config import QCEW_RAW, QCEW_CLEAN, QWI_CLEAN, FIGURES_DIR, TABLES_DIR, MIN_EXCESS_OBS, XWALK
from excess_utils import cyclical_excess_by_month
from seasonal_index import NAICS_SECTOR_LABELS, load_naics_titles, add_naics_titles

warnings.filterwarnings("ignore")

NATIONAL_AGGLVL = "18"   # national, NAICS6, by ownership (see fetch_qcew.py)


def load_qcew(path: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all fetched QCEW year files, national rows only."""
    raw_dir = path or QCEW_RAW
    files = sorted(raw_dir.glob("qcew_state_naics6_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No QCEW files found in {raw_dir}. Run fetch_qcew.py first.")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["agglvl_code"] == NATIONAL_AGGLVL].copy()

    # Suppressed cells report 0, not NaN -- treat disclosure_code == 'N' as missing
    for c in ["month1_emplvl", "month2_emplvl", "month3_emplvl"]:
        df.loc[df["disclosure_code"] == "N", c] = np.nan

    print(f"Loaded QCEW: {len(df):,} national industry-quarter rows, "
          f"{df['industry_code'].nunique()} industries, "
          f"years {df['year'].min():.0f}-{df['year'].max():.0f}")
    return df


def to_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Melt month1/2/3_emplvl into one row per (industry, year, calendar month)."""
    records = []
    for month_in_qtr, col in enumerate(["month1_emplvl", "month2_emplvl", "month3_emplvl"], start=1):
        sub = df[["industry_code", "year", "qtr", col]].rename(columns={col: "emplvl"})
        sub["month"] = (sub["qtr"] - 1) * 3 + month_in_qtr
        records.append(sub[["industry_code", "year", "month", "emplvl"]])
    long = pd.concat(records, ignore_index=True)
    return long.rename(columns={"industry_code": "industry"})


def compute_monthly_employment_shares(monthly: pd.DataFrame) -> pd.DataFrame:
    """emp_share_{j,y,m} = emplvl / mean_m(emplvl in that industry-year)."""
    d = monthly.copy()
    d["year_mean_emp"] = d.groupby(["industry", "year"])["emplvl"].transform("mean")
    d["emp_share"] = d["emplvl"] / d["year_mean_emp"]
    return d[["industry", "year", "month", "emp_share"]]


def compute_monthly_excess(emp_shares: pd.DataFrame, min_obs: int = MIN_EXCESS_OBS) -> pd.DataFrame:
    """Per-industry cyclical excess by calendar month -- the monthly analog of
    compute_employment_excess_by_quarter() in seasonal_index.py."""
    records = []
    for ind, grp in emp_shares.groupby("industry"):
        pivot = grp.pivot(index="year", columns="month", values="emp_share")
        mean_shares = {
            f"emp_share_M{m}": (pivot[m].mean() if m in pivot.columns else np.nan)
            for m in range(1, 13)
        }
        excess, _ = cyclical_excess_by_month(pivot, min_obs, "emp_excess")
        records.append({
            "naics_code": ind,
            "n_years": len(grp["year"].unique()),
            **excess,
            **mean_shares,
        })
    return pd.DataFrame(records)


def compute_common_calendar_effect(excess_df: pd.DataFrame) -> dict:
    """
    The cross-industry mean emp_excessM* pattern -- the monthly analog of
    earnings_index.py's compute_common_calendar_effect().

    42% of industries have their *raw* monthly_seasonal_index peak in
    December, 22% in January (verified directly) -- QCEW/QWI data is not
    seasonally adjusted, and non-seasonally-adjusted U.S. employment data
    has a well-documented broad, economy-wide December build-up/January
    pull-back that isn't industry-specific seasonality (it's the same
    phenomenon BLS's own seasonal adjustment exists to remove). Left in,
    this swamps genuine industry-specific timing for any industry whose own
    signal is weak, the same way earnings_index.py found a common Q4-bonus
    effect dominating weakly-seasonal industries' apparent earnings peak.
    """
    cols = [f"emp_excessM{m}" for m in range(1, 13)]
    return {c: excess_df[c].mean() for c in cols}


def add_idiosyncratic_excess(excess_df: pd.DataFrame, common_effect: dict) -> pd.DataFrame:
    """Subtract the common calendar effect from each industry's raw monthly
    excess, isolating industry-specific timing -- mirrors
    earnings_index.py's add_idiosyncratic_excess()."""
    df = excess_df.copy()
    for m in range(1, 13):
        col = f"emp_excessM{m}"
        df[f"idio_{col}"] = df[col] - common_effect[col]
    return df


def compute_placebo_amplitude(
    emp_shares: pd.DataFrame,
    common_effect: dict,
    min_obs: int = MIN_EXCESS_OBS,
    n_perms: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate, per industry, the *idiosyncratic* monthly amplitude expected
    under the null of no true industry-specific seasonal pattern, by
    permuting month labels within each year independently and recomputing
    the cyclical-excess amplitude net of the common calendar effect.

    This corrects for a real statistical artifact, not a bug: max(x) - min(x)
    over 12 monthly draws is mechanically larger than over 4 quarterly draws
    for the same underlying noise process, since the expected range of an
    i.i.d. sample grows with sample size. Comparing raw monthly amplitude to
    raw quarterly amplitude therefore overstates how much *real* seasonality
    monthly resolution reveals -- some of the gap is just "more chances for
    an extreme month." Permuting within each year preserves each industry's
    actual year-to-year variance while destroying any genuine across-year
    seasonal timing, giving a same-industry, same-scale null to compare
    against. `common_effect` is fixed (estimated once from the real,
    unpermuted data) and subtracted from every permutation's recomputed
    excess too, so the placebo distribution is on the same idiosyncratic
    scale as the real signal it's compared against.

    Returns naics_code, placebo_amplitude (mean over n_perms), n_perms_used.
    """
    rng = np.random.default_rng(seed)
    common_vec = np.array([common_effect[f"emp_excessM{m}"] for m in range(1, 13)])
    records = []
    for ind, grp in emp_shares.groupby("industry"):
        pivot = grp.pivot(index="year", columns="month", values="emp_share")
        if pivot.shape[1] < 12 or pivot.dropna().shape[0] < min_obs:
            continue
        reps = []
        for _ in range(n_perms):
            p = pivot.copy()
            for yr in p.index:
                row = p.loc[yr].to_numpy()
                if not pd.isna(row).any():
                    p.loc[yr] = rng.permutation(row)
            excess_p, _ = cyclical_excess_by_month(p, min_obs, "e")
            vals = [excess_p[f"eM{m}"] - common_vec[m - 1]
                    for m in range(1, 13) if pd.notna(excess_p[f"eM{m}"])]
            if len(vals) >= 2:
                reps.append(max(vals) - min(vals))
        if reps:
            records.append({
                "naics_code": ind,
                "placebo_amplitude": float(np.mean(reps)),
                "n_perms_used": len(reps),
            })
    return pd.DataFrame(records)


def build_monthly_seasonal_index(
    excess_df: pd.DataFrame, in_prefix: str = "emp_excess", out_prefix: str = "monthly"
) -> pd.DataFrame:
    """Construct the 0-1 monthly stock seasonal index, mirroring
    build_employment_seasonal_index()'s quarterly construction exactly,
    generalized from 4 to 12 periods. `in_prefix` selects which excess
    columns to build from -- "emp_excess" (raw) or "idio_emp_excess"
    (net of the common calendar effect, see compute_common_calendar_effect);
    `out_prefix` names the resulting columns so both versions can coexist
    on the same DataFrame (e.g. "monthly" -> monthly_seasonal_index,
    "idio_monthly" -> idio_monthly_seasonal_index)."""
    df = excess_df.copy()
    excess_cols = [f"{in_prefix}M{m}" for m in range(1, 13)]
    available = [c for c in excess_cols if c in df.columns]
    p = out_prefix

    # Same degenerate-period guard as the quarterly measures: <2 non-missing
    # months makes max-min trivially 0, a data-thinness artifact.
    enough_months = df[available].notna().sum(axis=1) >= 2

    df[f"{p}_seasonal_amplitude"] = df[available].max(axis=1) - df[available].min(axis=1)
    df.loc[~enough_months, f"{p}_seasonal_amplitude"] = np.nan

    _peak = df[available].apply(lambda row: row.idxmax() if row.notna().any() else pd.NA, axis=1)
    df[f"{p}_peak_month"] = (
        _peak.str.replace(f"{in_prefix}M", "", regex=False).astype("Int64")
    )
    df.loc[~enough_months, f"{p}_peak_month"] = pd.NA
    df[f"{p}_peak_quarter_from_month"] = ((df[f"{p}_peak_month"] - 1) // 3 + 1).astype("Int64")

    df[f"{p}_peak_excess_month"] = df[available].max(axis=1)
    df.loc[~enough_months, f"{p}_peak_excess_month"] = np.nan

    # Sample-count columns (n_emp_excessM*) exist only for the raw excess --
    # idio_emp_excessM* is just emp_excessM* shifted by a fixed constant, so
    # it shares the same underlying year-counts. Always look counts up under
    # the raw "emp_excess" prefix regardless of in_prefix.
    df[f"{p}_n_excess_obs"] = df.apply(
        lambda row: row[f"n_emp_excessM{int(row[f'{p}_peak_month'])}"]
        if pd.notna(row[f"{p}_peak_month"])
           and f"n_emp_excessM{int(row[f'{p}_peak_month'])}" in df.columns
        else pd.NA,
        axis=1,
    ).astype("Int64")

    p99 = df[f"{p}_seasonal_amplitude"].quantile(0.99)
    df[f"{p}_seasonal_index"] = (df[f"{p}_seasonal_amplitude"] / p99).clip(0, 1)

    df["sector_2d"] = df["naics_code"].astype(str).str[:2]
    df["sector_label"] = df["sector_2d"].map(NAICS_SECTOR_LABELS).fillna("Other")

    return df[[
        "naics_code", "sector_2d", "sector_label", "n_years",
        f"{p}_seasonal_amplitude", f"{p}_peak_month", f"{p}_peak_quarter_from_month",
        f"{p}_peak_excess_month", f"{p}_n_excess_obs", f"{p}_seasonal_index",
    ]]


def add_placebo_correction(monthly_idx: pd.DataFrame, placebo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge in the placebo (permutation-null) amplitude -- estimated on the
    idiosyncratic scale, see compute_placebo_amplitude() -- and derive the
    final bias-corrected signal_amplitude / signal_index. This is the most
    defensible measure in the file: net of both (1) the common calendar
    effect shared by most industries and (2) the mechanical range inflation
    from comparing a 12-point max-min range to a 4-point one, so
    signal_index isolates "how much MORE seasonal is this industry, at
    monthly resolution, than pure noise plus the broad economy-wide
    December/January pattern would produce."
    """
    df = monthly_idx.merge(placebo_df, on="naics_code", how="left")
    df["signal_amplitude"] = (df["idio_monthly_seasonal_amplitude"] - df["placebo_amplitude"]).clip(lower=0)
    p99_signal = df["signal_amplitude"].quantile(0.99)
    df["signal_index"] = (df["signal_amplitude"] / p99_signal).clip(0, 1)
    return df


def compare_to_quarterly_stock(monthly_idx: pd.DataFrame) -> pd.DataFrame:
    """Merge onto the existing quarterly stock index (seasonal_index_emp,
    from seasonal_index.py) to see whether monthly resolution changes the
    seasonality picture."""
    q_path = QWI_CLEAN / "seasonal_index_naics6.csv"
    if not q_path.exists():
        raise FileNotFoundError(f"{q_path} not found. Run seasonal_index.py first.")
    quarterly = pd.read_csv(q_path, dtype={"naics_code": str})
    quarterly = quarterly[[
        "naics_code", "national_industry_title", "seasonal_index_emp",
        "peak_quarter_emp", "peak_excess_emp", "n_emp_excess_obs",
    ]]

    merged = monthly_idx.merge(quarterly, on="naics_code", how="inner")
    merged = merged.dropna(subset=["monthly_seasonal_index", "seasonal_index_emp"])
    merged["peak_quarter_concordant"] = (
        merged["idio_monthly_peak_quarter_from_month"] == merged["peak_quarter_emp"]
    )
    return merged.sort_values("monthly_seasonal_index", ascending=False)


def plot_comparison(merged: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    ax = axes[0]
    ax.scatter(merged["seasonal_index_emp"], merged["monthly_seasonal_index"],
               s=10, alpha=0.5, c="#1f77b4")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel("Seasonal index (stock), quarterly (QWI)")
    ax.set_ylabel("Raw seasonal index (stock), monthly (QCEW)")
    ax.set_title("Raw comparison\n(conflates real seasonality with two confounds)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(merged["seasonal_index_emp"], merged["signal_index"],
               s=10, alpha=0.5, c="#d62728")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel("Seasonal index (stock), quarterly (QWI)")
    ax.set_ylabel("Bias-corrected signal index, monthly (QCEW)")
    ax.set_title("Bias-corrected comparison\n(net of common calendar effect + sample-size artifact)")
    ax.legend(fontsize=8)

    fig.suptitle("Does monthly resolution change the stock-seasonality picture?")
    fig.tight_layout()
    return fig


def print_summary(merged: pd.DataFrame) -> None:
    n = len(merged)
    pearson = merged[["seasonal_index_emp", "monthly_seasonal_index"]].corr().iloc[0, 1]
    spearman = merged[["seasonal_index_emp", "monthly_seasonal_index"]].corr(method="spearman").iloc[0, 1]
    pearson_sig = merged[["seasonal_index_emp", "signal_index"]].corr().iloc[0, 1]
    spearman_sig = merged[["seasonal_index_emp", "signal_index"]].corr(method="spearman").iloc[0, 1]
    concordance = merged["peak_quarter_concordant"].mean()

    print(f"\n{n} industries with both a quarterly (QWI) and monthly (QCEW) stock index.")
    print(f"Raw:               Pearson {pearson:.3f}, Spearman {spearman:.3f}")
    print(f"Bias-corrected:    Pearson {pearson_sig:.3f}, Spearman {spearman_sig:.3f}")
    print(f"Peak-quarter concordance (QCEW idiosyncratic peak month's quarter "
          f"vs. QWI peak quarter): {concordance:.1%}")

    raw_peak_month_share = merged["monthly_peak_month"].value_counts(normalize=True)
    print(f"\nCommon calendar effect check: {raw_peak_month_share.get(12, 0):.1%} of "
          f"industries have their RAW peak in December, {raw_peak_month_share.get(1, 0):.1%} "
          f"in January (vs. ~8.3% each if peaks were spread evenly across 12 months) -- "
          f"consistent with a broad, non-seasonally-adjusted economy-wide pattern, not "
          f"industry-specific timing. Netted out via idio_monthly_* columns before placebo correction.")

    frac_above = (merged["idio_monthly_seasonal_amplitude"] > merged["placebo_amplitude"]).mean()
    mean_real = merged["idio_monthly_seasonal_amplitude"].mean()
    mean_placebo = merged["placebo_amplitude"].mean()
    print(f"\nPlacebo check: idiosyncratic monthly amplitude exceeds its own industry's "
          f"placebo (permuted-null) amplitude in {frac_above:.1%} of industries.")
    print(f"Mean idiosyncratic amplitude: {mean_real:.4f}   Mean placebo amplitude: {mean_placebo:.4f}"
          f"   (placebo is {mean_placebo/mean_real:.1%} of the idiosyncratic amplitude -- this "
          f"share reflects the mechanical range inflation from 12 vs. 4 sampling points, not "
          f"real seasonality)")

    print("\nTop 10 by bias-corrected signal index:")
    cols = ["naics_code", "national_industry_title", "signal_index",
            "monthly_seasonal_index", "idio_monthly_peak_month",
            "seasonal_index_emp", "peak_quarter_emp"]
    print(merged.sort_values("signal_index", ascending=False)[cols].head(10).to_string(index=False))

    print("\nBiggest divergences (bias-corrected signal index vs. quarterly stock index):")
    merged["divergence"] = merged["signal_index"] - merged["seasonal_index_emp"]
    movers = merged.reindex(merged["divergence"].abs().sort_values(ascending=False).index)
    print(movers.head(10)[cols + ["divergence"]].to_string(index=False))


def run() -> None:
    QCEW_CLEAN.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    qcew = load_qcew()
    monthly = to_monthly_panel(qcew)
    shares = compute_monthly_employment_shares(monthly)
    excess = compute_monthly_excess(shares)

    print("\nComputing and netting out the common cross-industry calendar effect...")
    common_effect = compute_common_calendar_effect(excess)
    idio_excess = add_idiosyncratic_excess(excess, common_effect)

    raw_idx = build_monthly_seasonal_index(idio_excess, in_prefix="emp_excess", out_prefix="monthly")
    idio_idx = build_monthly_seasonal_index(idio_excess, in_prefix="idio_emp_excess", out_prefix="idio_monthly")
    monthly_idx = raw_idx.merge(
        idio_idx.drop(columns=["sector_2d", "sector_label", "n_years"]),
        on="naics_code", how="outer",
    )

    print("Running placebo (within-year month-permutation) correction "
          "-- this takes a few minutes...")
    placebo = compute_placebo_amplitude(shares, common_effect)
    monthly_idx = add_placebo_correction(monthly_idx, placebo)

    # Compare against the quarterly (QWI) stock index before attaching NAICS
    # titles to monthly_idx -- both sides would otherwise carry their own
    # national_industry_title column and pandas would suffix them (_x/_y)
    # rather than merge cleanly.
    merged = compare_to_quarterly_stock(monthly_idx)

    titles = load_naics_titles(XWALK / "naics6_2022_titles.csv")
    monthly_idx = add_naics_titles(monthly_idx, titles)

    out_path = QCEW_CLEAN / "monthly_seasonal_index_naics6.csv"
    monthly_idx.to_csv(out_path, index=False)
    print(f"Saved monthly index to {out_path}")
    merged.to_csv(TABLES_DIR / "monthly_vs_quarterly_stock_comparison.csv", index=False)

    if merged.empty:
        print("\n[warn] No industries have both a monthly and quarterly stock "
              "index yet -- likely still mid-fetch (not enough years for the "
              f"{MIN_EXCESS_OBS}-year minimum). Skipping comparison plot/summary.")
        return

    fig = plot_comparison(merged)
    fig.savefig(FIGURES_DIR / "monthly_vs_quarterly_stock_scatter.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "monthly_vs_quarterly_stock_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print_summary(merged)


if __name__ == "__main__":
    run()
