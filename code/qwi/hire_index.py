"""
hire_index.py — Build a 6-digit NAICS *hiring* seasonality index from QWI
accessions data (HirA), and compare its timing to the existing separations
(flow) index: when an industry hires seasonally, how many quarters later
does it separate seasonally?

Requires a full 51-state refetch with "HirA" in QWI_VARS (added to
fetch_qwi.py 2026-09) — the existing qwi_state_*.parquet files predate this
variable and don't have the column. HirA ("Hires All: Counts (Accessions)")
was verified live against the Census API before adding it: populates with
sensible magnitudes on both a large (722511) and thin (111336) cell, with a
"good" status flag (sHirA=1) — unlike the Payroll field, which looked valid
in the schema but returned null for every cell tested (see fetch_qwi.py).

Methodology mirrors the existing flow-side (separations) index almost
exactly, via the shared excess_utils.cyclical_excess_by_quarter():
    hire_rate_{j,t,q} = HirA_{j,t,q} / EmpEnd_{j,t,q}
    HireExcess(q)_j = mean_t[ hire_rate(q,t) - (hire_rate(q-1,t) + hire_rate(q+1,t))/2 ]
normalized to hire_seasonal_index in [0,1] via 99th-pct winsorization, with
the same >= MIN_EXCESS_OBS coverage filter used everywhere else.

Unlike sep_rate, hire_rate is NOT truncated at >=1: a quarter's hires can
plausibly exceed end-of-quarter headcount for a rapidly ramping-up seasonal
operation (e.g. a resort hiring 300 workers for a season that nets out at
150 after also losing some early), so an apparent >100% rate isn't
automatically a disclosure-suppression artifact the way it is for
separations. Extreme values are instead handled by the same 99th-percentile
winsorization already used for every other index in this project.

The key output is not the hire index alone, but its *timing* relative to
each industry's own separation-peak quarter (from seasonal_index.py): does
a hiring surge precede the separation surge by one quarter (e.g. retail:
hire Q4, separate Q1), by more than one (e.g. a industry that staffs up
mid-season and lets go at season's end), or coincide with it (high churn —
the same jobs filled by different workers each cycle, rather than genuine
seasonal net job creation/destruction)?

Outputs:
  data/qwi_clean/hire_index_naics6.csv
  output/tables/hire_vs_separation_timing.csv
  output/figures/hire_index_naics6.pdf
  output/figures/hire_vs_separation_scatter.pdf
  output/figures/hire_separation_lag_histogram.pdf

Usage:
    python code/qwi/hire_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, TABLES_DIR, MIN_EXCESS_OBS, XWALK

from seasonal_index import load_qwi, NAICS_SECTOR_LABELS, load_naics_titles, add_naics_titles
from excess_utils import cyclical_excess_by_quarter


#  Hire rate / excess computation (mirrors compute_sep_rates /
#  compute_excess_recurrence_by_quarter in seasonal_index.py, but for HirA)

def compute_hire_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarterly hire rates: hire_rate = HirA / EmpEnd, aggregated to
    national (industry, year, quarter) cells the same way compute_sep_rates()
    aggregates separations.

    Unlike compute_sep_rates(), does not truncate hire_rate >= 1 to NaN —
    see module docstring for why that "impossible rate" heuristic doesn't
    transfer cleanly from separations to hires.
    """
    d = df.copy()
    for c in ["HirA", "EmpEnd", "Emp"]:
        if c in d.columns:
            d.loc[d[c] < 0, c] = np.nan

    agg_cols = [c for c in ["HirA", "EmpEnd", "Emp"] if c in d.columns]
    groupcols = ["industry", "year", "quarter"]
    national = (
        d.groupby(groupcols)[agg_cols]
        .agg(lambda x: x.dropna().sum() if x.dropna().sum() > 0 else np.nan)
        .reset_index()
    )

    denom = "EmpEnd" if "EmpEnd" in national.columns else "Emp"
    national["hire_rate"] = national["HirA"] / national[denom]

    # A zero hire_rate in a quarter with real employment is a legitimate,
    # if unusual, observation (an industry that simply didn't hire that
    # quarter) -- unlike sep_rate, there's no natural "must be positive"
    # floor, so only drop the denominator-is-degenerate case.
    national.loc[national[denom] <= 0, "hire_rate"] = np.nan

    return national


def compute_hire_excess_by_quarter(
    rates: pd.DataFrame, min_obs: int = MIN_EXCESS_OBS
) -> pd.DataFrame:
    """
    Analog of compute_excess_recurrence_by_quarter(), but for hire rates.

    Returns a DataFrame indexed by industry with columns:
        hire_excessQ1..Q4, n_hire_excessQ1..Q4, hire_rate_Q1..Q4, n_years
    """
    rates = rates.sort_values(["industry", "year", "quarter"]).copy()
    records = []

    for ind, grp in rates.groupby("industry"):
        pivot = grp.pivot(index="year", columns="quarter", values="hire_rate")

        mean_rates = {
            f"hire_rate_Q{q}": (pivot[q].mean() if q in pivot.columns else np.nan)
            for q in [1, 2, 3, 4]
        }
        excess, _ = cyclical_excess_by_quarter(pivot, min_obs, "hire_excess")

        records.append({
            "industry": ind,
            "n_years": len(grp["year"].unique()),
            **excess,
            **mean_rates,
        })

    return pd.DataFrame(records)


#  Hire seasonal index construction (mirrors build_seasonal_index)

def build_hire_seasonal_index(hire_excess_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a 0-1 hire seasonal index, mirroring build_seasonal_index() so
    peak timing (not just amplitude) can be compared to the separation-based
    index.

    Returns: naics_code, hire_seasonal_amplitude, peak_quarter_hire,
    peak_excess_hire, n_hire_excess_obs, hire_seasonal_index
    """
    df = hire_excess_df.copy()
    excess_cols = ["hire_excessQ1", "hire_excessQ2", "hire_excessQ3", "hire_excessQ4"]
    available = [c for c in excess_cols if c in df.columns]

    # A meaningful amplitude needs >=2 valid quarters -- see
    # build_seasonal_index() in seasonal_index.py for why (with only 1,
    # max-min collapses trivially to 0, a data-thinness artifact).
    enough_quarters = df[available].notna().sum(axis=1) >= 2

    df["hire_seasonal_amplitude"] = df[available].max(axis=1) - df[available].min(axis=1)
    df.loc[~enough_quarters, "hire_seasonal_amplitude"] = np.nan

    _peak = df[available].apply(lambda row: row.idxmax() if row.notna().any() else pd.NA, axis=1)
    df["peak_quarter_hire"] = (
        _peak.str.replace("hire_excess", "", regex=False).str.replace("Q", "", regex=False).astype("Int64")
    )
    df.loc[~enough_quarters, "peak_quarter_hire"] = pd.NA
    df["peak_excess_hire"] = df[available].max(axis=1)
    df.loc[~enough_quarters, "peak_excess_hire"] = np.nan

    n_cols = {c.replace("hire_excessQ", "n_hire_excessQ"): c.replace("hire_excessQ", "Q")
              for c in available if c.replace("hire_excessQ", "n_hire_excessQ") in df.columns}
    if n_cols:
        df["n_hire_excess_obs"] = df.apply(
            lambda row: row[f"n_hire_excessQ{int(row['peak_quarter_hire'])}"]
            if pd.notna(row["peak_quarter_hire"])
               and f"n_hire_excessQ{int(row['peak_quarter_hire'])}" in df.columns
            else pd.NA,
            axis=1,
        ).astype("Int64")

    p99 = df["hire_seasonal_amplitude"].quantile(0.99)
    df["hire_seasonal_index"] = (df["hire_seasonal_amplitude"] / p99).clip(0, 1)

    return df.rename(columns={"industry": "naics_code"})[
        ["naics_code", "hire_seasonal_amplitude", "peak_quarter_hire",
         "peak_excess_hire", "n_hire_excess_obs", "hire_seasonal_index"]
    ]


#  Hire vs. separation timing comparison

def compare_hire_vs_separation_timing(
    hire_index: pd.DataFrame,
    sep_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the hire index onto each industry's separation (flow) index and
    compute the timing lag between them.

    Parameters
    ----------
    hire_index : output of build_hire_seasonal_index()
    sep_index  : national index from seasonal_index.py (needs naics_code,
                 seasonal_index, peak_quarter, and — if present —
                 seasonal_index_emp for the churn flag below)

    Returns a merged DataFrame with:
        hire_to_sep_lag_quarters : (peak_quarter - peak_quarter_hire) % 4,
            in {0,1,2,3}. 0 = hire and separation peak in the same quarter
            (high churn -- same jobs, different workers, each cycle); 1-3 =
            separation follows the hiring peak by that many quarters
            cyclically (e.g. 1 = hire Q4/separate Q1, retail's holiday
            pattern; 2-3 = a longer hire-to-season's-end gap, e.g. a
            spring-hire/fall-separate agricultural or tourism operation).
        likely_churn : True where both hire_seasonal_index and seasonal_index
            (flow) are seasonal (each > 0.3) but seasonal_index_emp (stock,
            i.e. headcount itself) is not (< 0.3) -- same job count, filled
            by different workers each cycle -- only computed when
            seasonal_index_emp is present in sep_index.
    """
    keep_sep_cols = ["naics_code", "sector_2d", "sector_label", "seasonal_index", "peak_quarter"]
    if "seasonal_index_emp" in sep_index.columns:
        keep_sep_cols.append("seasonal_index_emp")
    sep = sep_index[[c for c in keep_sep_cols if c in sep_index.columns]].copy()

    merged = hire_index.merge(sep, on="naics_code", how="inner")
    valid_peaks = merged.dropna(subset=["peak_quarter_hire", "peak_quarter"]).index
    merged["hire_to_sep_lag_quarters"] = pd.array([pd.NA] * len(merged), dtype="Int64")
    merged.loc[valid_peaks, "hire_to_sep_lag_quarters"] = (
        (merged.loc[valid_peaks, "peak_quarter"].astype(int)
         - merged.loc[valid_peaks, "peak_quarter_hire"].astype(int)) % 4
    ).astype("Int64")

    if "seasonal_index_emp" in merged.columns:
        merged["likely_churn"] = (
            (merged["hire_seasonal_index"] > 0.3)
            & (merged["seasonal_index"] > 0.3)
            & (merged["seasonal_index_emp"] < 0.3)
        )

    return merged.sort_values("hire_seasonal_index", ascending=False)


def print_timing_summary(merged: pd.DataFrame) -> None:
    """Human-readable summary of compare_hire_vs_separation_timing()'s output."""
    both = merged.dropna(subset=["hire_seasonal_index", "seasonal_index"])
    print(f"\nHire vs. separation seasonal index comparison")
    print(f"  n = {len(both)}")
    print(f"  Pearson r  = {both[['hire_seasonal_index', 'seasonal_index']].corr().iloc[0, 1]:.3f}")
    print(f"  Spearman r = {both[['hire_seasonal_index', 'seasonal_index']].corr(method='spearman').iloc[0, 1]:.3f}")

    lags = merged.dropna(subset=["hire_to_sep_lag_quarters"])
    if len(lags):
        print(f"\nHire-to-separation lag distribution (n={len(lags)}, quarters):")
        print((lags["hire_to_sep_lag_quarters"].value_counts(normalize=True).sort_index() * 100).round(1))

    if "likely_churn" in merged.columns:
        n_churn = int(merged["likely_churn"].sum())
        print(f"\n{n_churn} industries flagged as likely high-churn "
              f"(seasonal hiring AND separations, but not seasonal headcount)")


#  Figures

def plot_hire_index(index: pd.DataFrame, top_n: int = 40) -> plt.Figure:
    """Bar chart of the top-N most hire-seasonal industries (mirrors plot_seasonal_index)."""
    df = index.dropna(subset=["hire_seasonal_index"]).sort_values(
        "hire_seasonal_index", ascending=False
    ).head(top_n).sort_values("hire_seasonal_index")

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.28)))
    colors = plt.cm.RdYlGn_r(df["hire_seasonal_index"].values)
    label_col = "national_industry_title" if "national_industry_title" in df.columns else "sector_label"
    ax.barh(df["naics_code"].astype(str) + " " + df[label_col].fillna(""),
            df["hire_seasonal_index"], color=colors)
    ax.set_xlabel("Hire Seasonal Index (0-1)")
    ax.set_title(f"Top {top_n} Most Hire-Seasonal Industries")
    fig.tight_layout()
    return fig


def plot_hire_vs_separation_scatter(merged: pd.DataFrame) -> plt.Figure:
    """Scatter of hire_seasonal_index (y) vs. seasonal_index (x, separations)."""
    d = merged.dropna(subset=["hire_seasonal_index", "seasonal_index"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(d["seasonal_index"], d["hire_seasonal_index"], s=10, alpha=0.5, c="#1f77b4")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel("Seasonal index -- separations")
    ax.set_ylabel("Seasonal index -- hires")
    ax.set_title("Hire vs. separation seasonal index")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_lag_histogram(merged: pd.DataFrame) -> plt.Figure:
    """Bar chart of the hire-to-separation lag distribution (0-3 quarters)."""
    lags = merged.dropna(subset=["hire_to_sep_lag_quarters"])
    counts = lags["hire_to_sep_lag_quarters"].value_counts().reindex([0, 1, 2, 3], fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [
        "0\n(same quarter --\nlikely churn)",
        "1", "2", "3",
    ]
    ax.bar(labels, counts.values, color="#1f77b4")
    ax.set_xlabel("Quarters from hire-peak to separation-peak")
    ax.set_ylabel("Number of industries")
    ax.set_title("Hire-to-separation timing lag")
    fig.tight_layout()
    return fig


#  Main

def run_hire_index(
    qwi_path: Path | None = None,
    separation_index_path: Path | None = None,
    naics_titles_path: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: load QWI -> hire rates -> excess -> index -> compare to separations -> save."""
    QWI_CLEAN.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    qwi = load_qwi(path=qwi_path)
    if "HirA" not in qwi.columns or qwi["HirA"].notna().sum() == 0:
        raise RuntimeError(
            "No HirA data found. HirA was added to QWI_VARS in fetch_qwi.py "
            "2026-09 -- re-run fetch_qwi.py (full 51-state refetch, ~17 hrs) "
            "before building the hire index; existing qwi_state_*.parquet "
            "files predate this variable."
        )

    rates = compute_hire_rates(qwi)
    excess = compute_hire_excess_by_quarter(rates)
    print(f"Hire excess computed for {len(excess):,} industries")

    index = build_hire_seasonal_index(excess)
    index["sector_2d"] = index["naics_code"].astype(str).str[:2]
    index["sector_label"] = index["sector_2d"].map(NAICS_SECTOR_LABELS).fillna("Other")
    index = index.sort_values("hire_seasonal_index", ascending=False)

    naics_titles_path = naics_titles_path or (XWALK / "naics6_2022_titles.csv")
    index = add_naics_titles(index, load_naics_titles(naics_titles_path))

    out_path = QWI_CLEAN / "hire_index_naics6.csv"
    index.to_csv(out_path, index=False)
    print(f"Hire index saved to {out_path}")

    top15_cols = ["naics_code"]
    if "national_industry_title" in index.columns:
        top15_cols.append("national_industry_title")
    top15_cols += ["sector_label", "hire_seasonal_index", "peak_excess_hire", "peak_quarter_hire", "n_hire_excess_obs"]
    print(f"\nTop 15 most hire-seasonal industries:")
    print(index.head(15)[top15_cols].to_string(index=False))

    fig = plot_hire_index(index, top_n=40)
    fig.savefig(FIGURES_DIR / "hire_index_naics6.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── The key comparison: hire vs. separation timing ──
    sep_path = separation_index_path or (QWI_CLEAN / "seasonal_index_naics6.csv")
    if sep_path.exists():
        sep_index = pd.read_csv(sep_path, dtype={"naics_code": str, "sector_2d": str})
        merged = compare_hire_vs_separation_timing(index, sep_index)

        out_timing_path = TABLES_DIR / "hire_vs_separation_timing.csv"
        merged.to_csv(out_timing_path, index=False)
        print(f"Saved: {out_timing_path}")

        print_timing_summary(merged)

        fig2 = plot_hire_vs_separation_scatter(merged)
        fig2.savefig(FIGURES_DIR / "hire_vs_separation_scatter.pdf", bbox_inches="tight")
        plt.close(fig2)

        fig3 = plot_lag_histogram(merged)
        fig3.savefig(FIGURES_DIR / "hire_separation_lag_histogram.pdf", bbox_inches="tight")
        plt.close(fig3)
    else:
        print(f"\n[warn] {sep_path} not found; skipping the hire-vs-separation timing comparison.")

    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the QWI hire (accessions) seasonal index")
    parser.add_argument("--qwi-file", type=Path, default=None)
    parser.add_argument("--separation-index", type=Path, default=None)
    parser.add_argument("--naics-titles", type=Path, default=None)
    args = parser.parse_args()

    run_hire_index(
        qwi_path=args.qwi_file,
        separation_index_path=args.separation_index,
        naics_titles_path=args.naics_titles,
    )
