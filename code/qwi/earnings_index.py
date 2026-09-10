"""
earnings_index.py — Build a 6-digit NAICS *income seasonality* index from QWI
average-earnings data, and test the paper's central hypothesis directly:
does income dip in an industry's own off-season?

Uses EarnBeg (End-of-Quarter Employment: Average Monthly Earnings), not the
"Payroll" field fetch_qwi.py originally requested — Payroll returns null from
the Census API for every cell tested (verified live), while EarnBeg is the
working equivalent. See fetch_qwi.py's QWI_VARS comment.

Methodology mirrors the existing stock-side (employment) index exactly:
    earnings_share_{j,t,q} = EarnBeg_{j,t,q} / mean_year(EarnBeg_{j,t})
    ExcessEarnings(q)_j = mean_t[ share(q,t) - (share(q-1,t) + share(q+1,t))/2 ]
normalized to earnings_seasonal_index in [0,1] via 99th-pct winsorization,
with the same >= MIN_EXCESS_OBS coverage filter used everywhere else.

The key output is not the earnings index alone, but its value merged onto
each industry's *separation*-based peak_quarter (from seasonal_index.py):
does income (via ExcessEarnings) fall in the same quarter separations spike?
That is the direct QWI-based analog of the paper's income-in-the-off-season
story, at the industry level rather than the worker level.

Outputs:
  data/qwi_clean/earnings_index_naics6.csv
  output/tables/earnings_at_separation_peak.csv
  output/figures/earnings_seasonal_index.pdf
  output/figures/earnings_at_separation_peak_hist.pdf

Usage:
    python code/qwi/earnings_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, TABLES_DIR, MIN_EXCESS_OBS

from seasonal_index import load_qwi, NAICS_SECTOR_LABELS
from excess_utils import cyclical_excess_by_quarter


#  Earnings share / excess computation (mirrors compute_employment_shares /
#  compute_employment_excess_by_quarter in seasonal_index.py, but for EarnBeg)

def compute_earnings_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build national earnings 'shares' s_{industry,year,quarter} =
    EarnBeg / mean(EarnBeg in year) — average monthly earnings normalized to
    each industry's own annual mean, so levels are comparable across
    industries regardless of absolute dollar scale.
    """
    d = df.copy()
    d.loc[d["EarnBeg"] <= 0, "EarnBeg"] = np.nan

    nat = (
        d.groupby(["industry", "year", "quarter"])["EarnBeg"]
        .mean()
        .reset_index()
        .rename(columns={"EarnBeg": "earnings_level"})
    )

    nat["year_mean_earnings"] = nat.groupby(["industry", "year"])["earnings_level"].transform("mean")
    nat["earnings_share"] = nat["earnings_level"] / nat["year_mean_earnings"]

    return nat[["industry", "year", "quarter", "earnings_share"]]


def compute_earnings_excess_by_quarter(
    earnings_shares: pd.DataFrame, min_obs: int = MIN_EXCESS_OBS
) -> pd.DataFrame:
    """
    Analog of compute_employment_excess_by_quarter(), but for earnings shares.

    Returns a DataFrame indexed by industry with columns:
        earnings_excessQ1..Q4, n_earnings_excessQ1..Q4,
        earnings_share_Q1..Q4, n_years
    """
    earnings_shares = earnings_shares.sort_values(["industry", "year", "quarter"]).copy()
    records = []

    for ind, grp in earnings_shares.groupby("industry"):
        pivot = grp.pivot(index="year", columns="quarter", values="earnings_share")

        mean_shares = {
            f"earnings_share_Q{q}": (pivot[q].mean() if q in pivot.columns else np.nan)
            for q in [1, 2, 3, 4]
        }
        excess, _ = cyclical_excess_by_quarter(pivot, min_obs, "earnings_excess")

        records.append({
            "industry": ind,
            "n_years": len(grp["year"].unique()),
            **excess,
            **mean_shares,
        })

    return pd.DataFrame(records)


def build_earnings_seasonal_index(earnings_excess_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a 0-1 earnings seasonal index, mirroring build_seasonal_index()
    so peak timing (not just amplitude) can be compared to the separation-based
    index.

    Returns: naics_code, earnings_seasonal_amplitude, peak_quarter_earnings,
    peak_excess_earnings, n_earnings_excess_obs, earnings_seasonal_index,
    trough_quarter_earnings, trough_excess_earnings
    (trough = the quarter with the *lowest* excess, i.e. the biggest income dip)
    """
    df = earnings_excess_df.copy()
    excess_cols = ["earnings_excessQ1", "earnings_excessQ2", "earnings_excessQ3", "earnings_excessQ4"]
    available = [c for c in excess_cols if c in df.columns]

    # A meaningful amplitude needs >=2 valid quarters -- with only 1,
    # max-min (and peak==trough) collapses trivially, a data-thinness
    # artifact rather than genuine flatness (see build_seasonal_index() in
    # seasonal_index.py for the separations-side version of this fix).
    enough_quarters = df[available].notna().sum(axis=1) >= 2

    df["earnings_seasonal_amplitude"] = df[available].max(axis=1) - df[available].min(axis=1)
    df.loc[~enough_quarters, "earnings_seasonal_amplitude"] = np.nan

    _peak = df[available].apply(lambda row: row.idxmax() if row.notna().any() else pd.NA, axis=1)
    df["peak_quarter_earnings"] = (
        _peak.str.replace("earnings_excess", "", regex=False).str.replace("Q", "", regex=False).astype("Int64")
    )
    df.loc[~enough_quarters, "peak_quarter_earnings"] = pd.NA
    df["peak_excess_earnings"] = df[available].max(axis=1)
    df.loc[~enough_quarters, "peak_excess_earnings"] = np.nan

    _trough = df[available].apply(lambda row: row.idxmin() if row.notna().any() else pd.NA, axis=1)
    df["trough_quarter_earnings"] = (
        _trough.str.replace("earnings_excess", "", regex=False).str.replace("Q", "", regex=False).astype("Int64")
    )
    df.loc[~enough_quarters, "trough_quarter_earnings"] = pd.NA
    df["trough_excess_earnings"] = df[available].min(axis=1)
    df.loc[~enough_quarters, "trough_excess_earnings"] = np.nan

    df["n_earnings_excess_obs"] = df.apply(
        lambda row: row[f"n_earnings_excessQ{int(row['peak_quarter_earnings'])}"]
        if pd.notna(row["peak_quarter_earnings"])
        else pd.NA,
        axis=1,
    ).astype("Int64")

    p99 = df["earnings_seasonal_amplitude"].quantile(0.99)
    df["earnings_seasonal_index"] = (df["earnings_seasonal_amplitude"] / p99).clip(0, 1)

    return df.rename(columns={"industry": "naics_code"})[
        ["naics_code", "earnings_seasonal_amplitude", "peak_quarter_earnings",
         "peak_excess_earnings", "trough_quarter_earnings", "trough_excess_earnings",
         "n_earnings_excess_obs", "earnings_seasonal_index"]
    ]


def compute_common_calendar_effect(excess_df: pd.DataFrame) -> dict:
    """
    The cross-industry mean earnings_excessQ* pattern.

    ~83% of all 1,011 industries have their *raw* earnings_seasonal_index
    peak in Q4 (verified directly) — a broad, economy-wide year-end
    bonus/holiday-pay effect, not industry-specific seasonality. Left in, it
    swamps any industry-specific "income in the off-season" signal: whether
    an industry's *own* separation-peak quarter happens to coincide with the
    common Q4 bump almost entirely determines whether its earnings look like
    they "dip" at that quarter, regardless of that industry's actual
    behavior. This computes that common factor so it can be netted out.
    """
    cols = [f"earnings_excessQ{q}" for q in [1, 2, 3, 4]]
    return {c: excess_df[c].mean() for c in cols}


def add_idiosyncratic_excess(excess_df: pd.DataFrame, common_effect: dict) -> pd.DataFrame:
    """
    Subtract the common calendar effect from each industry's raw quarterly
    excess, isolating industry-specific earnings timing from the broad
    Q4-bonus pattern most industries share.
    """
    df = excess_df.copy()
    for q in [1, 2, 3, 4]:
        col = f"earnings_excessQ{q}"
        df[f"idio_{col}"] = df[col] - common_effect[col]
    return df


def run_earnings_index(
    qwi_path: Path | None = None,
    separation_index_path: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: load QWI -> earnings shares -> excess -> index -> test -> save."""
    QWI_CLEAN.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    qwi = load_qwi(path=qwi_path)
    if "EarnBeg" not in qwi.columns or qwi["EarnBeg"].notna().sum() == 0:
        raise RuntimeError(
            "No EarnBeg data found. Run fetch_qwi.py (current QWI_VARS includes "
            "EarnBeg) before building the earnings index."
        )

    shares = compute_earnings_shares(qwi)
    excess = compute_earnings_excess_by_quarter(shares)
    print(f"Earnings excess computed for {len(excess):,} industries")

    index = build_earnings_seasonal_index(excess)
    index["sector_2d"] = index["naics_code"].astype(str).str[:2]
    index["sector_label"] = index["sector_2d"].map(NAICS_SECTOR_LABELS).fillna("Other")
    index = index.sort_values("earnings_seasonal_index", ascending=False)

    # Idiosyncratic (de-meaned) version: nets out the common cross-industry
    # calendar effect (see compute_common_calendar_effect) so the index and
    # the income-dip test below reflect industry-specific timing, not the
    # broad Q4-bonus pattern ~83% of industries share.
    common_effect = compute_common_calendar_effect(excess)
    print(f"\nCommon calendar effect (cross-industry mean earnings excess by quarter):")
    for q in [1, 2, 3, 4]:
        print(f"  Q{q}: {common_effect[f'earnings_excessQ{q}']:+.4f}")

    excess_idio = add_idiosyncratic_excess(excess, common_effect)
    idio_input = excess_idio[
        ["industry", "n_years"] + [f"idio_earnings_excessQ{q}" for q in [1, 2, 3, 4]]
        + [f"n_earnings_excessQ{q}" for q in [1, 2, 3, 4]]
    ].rename(columns={f"idio_earnings_excessQ{q}": f"earnings_excessQ{q}" for q in [1, 2, 3, 4]})
    index_idio = build_earnings_seasonal_index(idio_input).rename(
        columns={c: f"idio_{c}" for c in
                 ["earnings_seasonal_amplitude", "peak_quarter_earnings", "peak_excess_earnings",
                  "trough_quarter_earnings", "trough_excess_earnings", "n_earnings_excess_obs",
                  "earnings_seasonal_index"]}
    )
    index = index.merge(index_idio, on="naics_code", how="left")
    index = index.sort_values("earnings_seasonal_index", ascending=False)

    out_path = QWI_CLEAN / "earnings_index_naics6.csv"
    index.to_csv(out_path, index=False)
    print(f"Earnings index saved to {out_path}")

    print(f"\nTop 15 most income-seasonal industries (raw, Q4-bonus effect included):")
    print(index.head(15)[
        ["naics_code", "sector_label", "earnings_seasonal_index",
         "peak_quarter_earnings", "trough_quarter_earnings", "n_earnings_excess_obs"]
    ].to_string(index=False))

    print(f"\nTop 15 by idiosyncratic (de-meaned) earnings seasonality:")
    print(index.sort_values("idio_earnings_seasonal_index", ascending=False).head(15)[
        ["naics_code", "sector_label", "idio_earnings_seasonal_index",
         "idio_peak_quarter_earnings", "idio_trough_quarter_earnings"]
    ].to_string(index=False))

    # ── The key test: income dip at the separation-peak quarter ──
    sep_path = separation_index_path or (QWI_CLEAN / "seasonal_index_naics6.csv")
    if sep_path.exists():
        sep_index = pd.read_csv(sep_path, dtype={"naics_code": str, "sector_2d": str})
        excess_full = excess_idio.rename(columns={"industry": "naics_code"})

        merged = sep_index[["naics_code", "sector_label", "seasonal_index", "peak_quarter", "n_excess_obs"]].merge(
            excess_full, on="naics_code", how="inner"
        )
        merged = merged.dropna(subset=["peak_quarter"])

        def _excess_at_peak(row, prefix):
            col = f"{prefix}earnings_excessQ{int(row['peak_quarter'])}"
            return row.get(col, np.nan)

        merged["earnings_excess_at_sep_peak"] = merged.apply(lambda r: _excess_at_peak(r, ""), axis=1)
        merged["idio_earnings_excess_at_sep_peak"] = merged.apply(lambda r: _excess_at_peak(r, "idio_"), axis=1)
        merged["income_dips_in_off_season"] = merged["earnings_excess_at_sep_peak"] < 0
        merged["income_dips_in_off_season_idio"] = merged["idio_earnings_excess_at_sep_peak"] < 0

        valid = merged.dropna(subset=["earnings_excess_at_sep_peak"])
        valid_idio = merged.dropna(subset=["idio_earnings_excess_at_sep_peak"])
        print(f"\n=== Income in the off-season: does earnings dip when separations peak? ===")
        print(f"Raw (confounded by the common Q4-bonus effect above):")
        print(f"  {len(valid):,} industries; dip rate {valid['income_dips_in_off_season'].mean():.1%}; "
              f"mean excess {valid['earnings_excess_at_sep_peak'].mean():+.4f}")
        print(f"Idiosyncratic (common calendar effect removed) — the credible answer:")
        print(f"  {len(valid_idio):,} industries; dip rate {valid_idio['income_dips_in_off_season_idio'].mean():.1%}; "
              f"mean excess {valid_idio['idio_earnings_excess_at_sep_peak'].mean():+.4f}")

        out_test_path = TABLES_DIR / "earnings_at_separation_peak.csv"
        merged.to_csv(out_test_path, index=False)
        print(f"Saved: {out_test_path}")

        # Figure: distribution of the idiosyncratic earnings excess at the
        # separation-peak quarter (the de-confounded version of the test)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(valid_idio["idio_earnings_excess_at_sep_peak"], bins=40, color="steelblue", edgecolor="white")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("Idiosyncratic earnings excess at the industry's separation-peak quarter")
        ax.set_ylabel("Count of industries")
        ax.set_title("Income in the off-season (common calendar effect removed)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "earnings_at_separation_peak_hist.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"\n[warn] {sep_path} not found; skipping the separation-peak income-dip test.")

    # Figure: top-30 by idiosyncratic earnings seasonal index (the raw index
    # is ~83% dominated by the common Q4-bonus effect, so it mostly just
    # ranks industries by bonus size, not industry-specific income timing)
    top = index.sort_values("idio_earnings_seasonal_index", ascending=False).head(30).sort_values("idio_earnings_seasonal_index")
    fig, ax = plt.subplots(figsize=(9, max(4, 30 * 0.28)))
    colors = plt.cm.RdYlGn_r(top["idio_earnings_seasonal_index"].values)
    ax.barh(top["naics_code"].astype(str) + " " + top["sector_label"], top["idio_earnings_seasonal_index"], color=colors)
    ax.set_xlabel("Idiosyncratic Earnings Seasonal Index (0-1)")
    ax.set_title("Top 30 Most Income-Seasonal Industries\n(common Q4-bonus effect removed)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "earnings_seasonal_index.pdf", bbox_inches="tight")
    plt.close(fig)

    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the QWI earnings/income seasonal index")
    parser.add_argument("--qwi-file", type=Path, default=None)
    parser.add_argument("--separation-index", type=Path, default=None)
    args = parser.parse_args()

    run_earnings_index(qwi_path=args.qwi_file, separation_index_path=args.separation_index)
