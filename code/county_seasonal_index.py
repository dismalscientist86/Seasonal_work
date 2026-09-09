"""
county_seasonal_index.py — County-level seasonal-work exposure index (CBP x QWI).

Weights the state x NAICS6 seasonal index (geographic_analysis.py's output) by
each county's own industry mix (establishment/employment counts from County
Business Patterns) to produce a single "seasonal exposure" score per county:
how seasonal is the average job in this county, given what industries are
actually here?

    county_exposure_c = sum_j( EMP_{c,j} * seasonal_index_j ) / sum_j( EMP_{c,j} )

Currently scoped to a single state at a time (CBP has only been fetched for
California so far; see fetch_cbp.py). Three-tier seasonal_index lookup per
NAICS6 industry, falling back when a code is missing at a finer level:
  1. State x NAICS6 (geographic_analysis.py) — most locally accurate
  2. National NAICS6 (seasonal_index.py) — when the state-specific cell is
     too thin (< MIN_EXCESS_OBS years) to have a state-level estimate
  3. NAICS4 aggregate (simple mean of the national NAICS6 index within that
     4-digit group) — when CBP itself suppresses the county's NAICS6 cell but
     still reports a NAICS4 cell

Outputs:
  output/tables/county_seasonal_exposure_{state}.csv       (one row per county)
  output/tables/county_seasonal_exposure_{state}_detail.csv (one row per
      county x industry, with the matched seasonal_index and which tier
      matched — useful for auditing what's driving a given county's score)
  output/figures/county_seasonal_exposure_{state}.pdf

Usage:
    python code/county_seasonal_index.py --state 06 --highlight 06113
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QWI_CLEAN, FIGURES_DIR, TABLES_DIR, CBP_RAW, REPO_ROOT, STATE_NAMES


def load_cbp(state: str, naics_level: int) -> pd.DataFrame:
    """Load county x NAICS CBP data for one state and NAICS digit level."""
    path = CBP_RAW / f"cbp_county_naics{naics_level}_state_{state}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python code/fetch_cbp.py --state {state} "
            f"--naics-level {naics_level}"
        )
    df = pd.read_parquet(path)
    df["naics_code"] = df["naics_code"].astype(str)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["EMP"] = pd.to_numeric(df["EMP"], errors="coerce")
    df["ESTAB"] = pd.to_numeric(df["ESTAB"], errors="coerce")
    return df


def build_naics4_index(national_index: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the national NAICS6 seasonal index up to NAICS4, as a fallback
    for CBP cells suppressed at 6-digit but present at 4-digit. Simple
    unweighted mean across the NAICS6 codes within each NAICS4 group (no
    national employment-share weights are readily available industry-side
    to do better).
    """
    df = national_index.copy()
    df["naics4"] = df["naics_code"].str[:4]
    return (
        df.groupby("naics4")["seasonal_index"]
        .mean()
        .reset_index()
        .rename(columns={"seasonal_index": "seasonal_index_naics4"})
    )


def match_seasonal_index(
    cbp6: pd.DataFrame,
    cbp4: pd.DataFrame,
    state_index: pd.DataFrame,
    national_index: pd.DataFrame,
    naics4_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach a seasonal_index + match_tier to every county x NAICS6 CBP row,
    falling back to national NAICS6 then NAICS4 as needed. Also appends
    NAICS4-only CBP rows (cells suppressed at 6-digit) matched via the
    NAICS4 fallback, so county employment isn't silently dropped just
    because its finest-grain cell was suppressed.
    """
    state_map = state_index.set_index("naics_code")["seasonal_index"].to_dict()
    national_map = national_index.set_index("naics_code")["seasonal_index"].to_dict()
    naics4_map = naics4_index.set_index("naics4")["seasonal_index_naics4"].to_dict()

    d6 = cbp6.copy()
    d6["seasonal_index"] = d6["naics_code"].map(state_map)
    d6["match_tier"] = np.where(d6["seasonal_index"].notna(), "state_naics6", None)

    still_missing = d6["seasonal_index"].isna()
    d6.loc[still_missing, "seasonal_index"] = d6.loc[still_missing, "naics_code"].map(national_map)
    d6.loc[still_missing & d6["seasonal_index"].notna(), "match_tier"] = "national_naics6"

    still_missing = d6["seasonal_index"].isna()
    d6.loc[still_missing, "seasonal_index"] = d6.loc[still_missing, "naics_code"].str[:4].map(naics4_map)
    d6.loc[still_missing & d6["seasonal_index"].notna(), "match_tier"] = "national_naics4_from_naics6_row"

    # NAICS6 codes CBP suppresses entirely for this county, but which still
    # show up as a NAICS4 cell: recover that employment via the NAICS4 fallback
    # rather than losing it.
    covered_naics6 = set(zip(d6["county"], d6["naics_code"]))
    d4 = cbp4.copy()
    d4["seasonal_index"] = d4["naics_code"].map(naics4_map)
    d4["match_tier"] = "national_naics4_direct"
    # Only keep a NAICS4 row if this (county, naics4-as-naics6-prefix) combo
    # isn't already covered by a NAICS6 row for that county (avoids double
    # counting employment already captured at the finer level).
    naics4_covered_by_6 = {
        (county, code[:4]) for county, code in covered_naics6
    }
    is_covered = pd.MultiIndex.from_frame(d4[["county", "naics_code"]]).isin(naics4_covered_by_6)
    d4 = d4[~is_covered]

    combined = pd.concat([d6, d4], ignore_index=True, sort=False)
    return combined


def compute_county_exposure(matched: pd.DataFrame) -> pd.DataFrame:
    """Employment-weighted mean seasonal_index per county, plus coverage diagnostics."""
    valid = matched.dropna(subset=["seasonal_index", "EMP"])
    valid = valid[valid["EMP"] > 0]

    def _agg(g: pd.DataFrame) -> pd.Series:
        total_emp = g["EMP"].sum()
        exposure = (g["EMP"] * g["seasonal_index"]).sum() / total_emp
        return pd.Series({
            "seasonal_exposure": exposure,
            "matched_emp": total_emp,
            "n_industries_matched": g["naics_code"].nunique(),
        })

    county_agg = valid.groupby(["county", "county_name"]).apply(_agg, include_groups=False).reset_index()

    # Coverage: matched employment vs. total CBP employment reported for the county
    total_emp_by_county = matched.groupby("county")["EMP"].sum().rename("total_emp")
    county_agg = county_agg.merge(total_emp_by_county, on="county", how="left")
    county_agg["coverage_share"] = county_agg["matched_emp"] / county_agg["total_emp"]

    return county_agg.sort_values("seasonal_exposure", ascending=False)


def top_industries_for_county(matched: pd.DataFrame, county: str, top_n: int = 8) -> pd.DataFrame:
    """Industries driving a given county's exposure score, by EMP x seasonal_index contribution."""
    sub = matched[(matched["county"] == county)].dropna(subset=["seasonal_index", "EMP"])
    sub = sub[sub["EMP"] > 0].copy()
    sub["contribution"] = sub["EMP"] * sub["seasonal_index"]
    cols = ["naics_code", "NAICS2017_LABEL", "EMP", "seasonal_index", "match_tier", "contribution"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].sort_values("contribution", ascending=False).head(top_n)


def plot_county_ranking(county_agg: pd.DataFrame, highlight: str | None, state: str) -> plt.Figure:
    df = county_agg.sort_values("seasonal_exposure")
    colors = ["#d62728" if c == highlight else "#1f77b4" for c in df["county"]]
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.22)))
    ax.barh(df["county_name"], df["seasonal_exposure"], color=colors)
    ax.set_xlabel("Employment-weighted seasonal exposure (0-1 index)")
    ax.set_title(f"County seasonal-work exposure, state {state}\n(industry mix from CBP, seasonality from QWI)")
    fig.tight_layout()
    return fig


def run(state: str = "06", highlight: str | None = None) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cbp6 = load_cbp(state, 6)
    cbp4 = load_cbp(state, 4)

    county_names = cbp6[["county", "NAME"]].drop_duplicates().rename(columns={"NAME": "county_name"})
    # CBP's NAME field is "<county>, <state>" (e.g. "Yolo County, California") —
    # strip the state suffix using this state's actual name, so this works for
    # any state once CBP is fetched nationally, not just the California pilot.
    state_name = STATE_NAMES.get(state)
    if state_name:
        county_names["county_name"] = county_names["county_name"].str.replace(
            rf", {re.escape(state_name)}$", "", regex=True
        )

    national_index = pd.read_csv(QWI_CLEAN / "seasonal_index_naics6.csv", dtype={"naics_code": str})
    state_index_all = pd.read_csv(
        QWI_CLEAN / "seasonal_index_naics6_by_state.csv", dtype={"naics_code": str, "state": str}
    )
    state_index = state_index_all[state_index_all["state"] == state]
    naics4_index = build_naics4_index(national_index)

    matched = match_seasonal_index(cbp6, cbp4, state_index, national_index, naics4_index)
    matched = matched.merge(county_names, on="county", how="left")

    county_agg = compute_county_exposure(matched)
    county_agg["rank"] = county_agg["seasonal_exposure"].rank(ascending=False, method="min").astype(int)
    county_agg["n_counties"] = len(county_agg)
    county_agg["percentile"] = 1 - (county_agg["rank"] - 1) / (len(county_agg) - 1)

    out_path = TABLES_DIR / f"county_seasonal_exposure_{state}.csv"
    county_agg.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    detail_path = TABLES_DIR / f"county_seasonal_exposure_{state}_detail.csv"
    matched.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    print(f"\n{len(county_agg)} counties ranked by employment-weighted seasonal exposure:")
    print(county_agg[["rank", "county_name", "seasonal_exposure", "coverage_share", "matched_emp"]]
          .to_string(index=False))

    fig = plot_county_ranking(county_agg, highlight=highlight, state=state)
    fig_path = FIGURES_DIR / f"county_seasonal_exposure_{state}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    if highlight:
        row = county_agg[county_agg["county"] == highlight]
        if len(row):
            r = row.iloc[0]
            print(f"\n=== {r['county_name']} ===")
            print(f"Seasonal exposure: {r['seasonal_exposure']:.4f}")
            print(f"Rank: {int(r['rank'])} of {int(r['n_counties'])} "
                  f"({r['percentile']:.0%} percentile, 100% = most seasonal)")
            print(f"Coverage: {r['coverage_share']:.1%} of county employment matched to a seasonal_index")
            print(f"\nTop industries driving this county's score:")
            top = top_industries_for_county(matched, highlight)
            print(top.to_string(index=False))
        else:
            print(f"\n[warn] county {highlight} not found in results")

    return county_agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="County-level seasonal-work exposure index (CBP x QWI)")
    parser.add_argument("--state", type=str, default="06", help="State FIPS code (default: 06, California)")
    parser.add_argument("--highlight", type=str, default=None, help="County FIPS code to highlight/report on, e.g. 06113 (Yolo)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    highlight_county = args.highlight[-3:] if args.highlight else None
    run(state=args.state, highlight=highlight_county)
