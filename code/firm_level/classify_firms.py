"""
classify_firms.py — Classify firms as seasonal using two complementary methods.

Method 1 (Industry lookup): Merge firm NAICS6 to the QWI seasonal index.
    - Fast, requires only NAICS code and the pre-computed seasonal_index CSV.
    - No individual worker data needed at this step.

Method 2 (Direct estimation): Estimate firm-level excess Q4 separation rates
    from the UI wage records panel using the event study structure.
    - Requires building the separation events panel first.
    - More data-intensive but captures firm-specific seasonality.

Both methods produce a firm-level 'seasonal_score' (0–1) and a binary
'is_seasonal' flag based on a configurable threshold.

Usage on the other machine:
    1. Copy code/firm_level/ directory
    2. Copy data/qwi_clean/seasonal_index_naics6.csv
    3. Run this script with your UI wage records path

    python classify_firms.py --ui-data path/to/ui_records.parquet
                             --qwi-index path/to/seasonal_index_naics6.csv
                             --output path/to/firm_seasonality.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from code.config import FIRM_DATA, TABLES_DIR
    from code.firm_level.build_events import (
        load_ui_records,
        build_quarterly_employment,
        build_separation_events_fast,
    )
except ImportError:
    # Standalone mode: running on another machine without the full repo
    FIRM_DATA  = Path(".")
    TABLES_DIR = Path("output")


# ── Method 1: Industry-lookup classification ──────────────────────────────────

def classify_by_industry(
    firms_df: pd.DataFrame,
    qwi_index: pd.DataFrame,
    naics_col: str = "naics6",
    firm_id_col: str = "employer_id",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Assign a seasonal score to each firm by merging its NAICS6 code to the
    QWI seasonal index.

    Parameters
    ----------
    firms_df   : DataFrame with at minimum (firm_id_col, naics_col).
    qwi_index  : DataFrame from seasonal_index.py (naics_code, seasonal_index, …).
    naics_col  : Column in firms_df containing 6-digit NAICS code.
    threshold  : seasonal_index cutoff for is_seasonal flag (default 0.5).

    Returns
    -------
    firms_df with added columns:
        seasonal_index : QWI seasonal score for this firm's industry
        peak_quarter   : quarter with the highest separation excess
        is_seasonal    : 1 if seasonal_index > threshold
    """
    # Standardize NAICS code format (string, zero-padded to 6 digits)
    firms = firms_df.copy()
    firms["naics6_str"] = firms[naics_col].astype(str).str.strip().str.zfill(6)

    qwi = qwi_index.copy()
    qwi["naics_code_str"] = qwi["naics_code"].astype(str).str.strip().str.zfill(6)

    # Try exact 6-digit match
    merged = firms.merge(
        qwi[["naics_code_str", "seasonal_index", "peak_quarter",
             "excessQ4", "seasonal_amplitude", "sector_label"]].rename(
            columns={"naics_code_str": "naics6_str"}
        ),
        on="naics6_str",
        how="left",
    )

    # For firms with no 6-digit match, try 4-digit
    missing_mask = merged["seasonal_index"].isna()
    if missing_mask.any():
        firms["naics4_str"] = firms["naics6_str"].str[:4]
        qwi4_agg = (
            qwi.assign(naics4=qwi["naics_code_str"].str[:4])
            .groupby("naics4")[["seasonal_index", "excessQ4", "seasonal_amplitude"]]
            .mean()
            .reset_index()
            .rename(columns={"naics4": "naics4_str"})
        )
        fill = firms[missing_mask][["naics4_str"]].merge(
            qwi4_agg, on="naics4_str", how="left"
        )
        merged.loc[missing_mask, "seasonal_index"]    = fill["seasonal_index"].values
        merged.loc[missing_mask, "excessQ4"]          = fill["excessQ4"].values
        merged.loc[missing_mask, "seasonal_amplitude"] = fill["seasonal_amplitude"].values

    merged["is_seasonal"] = (merged["seasonal_index"] > threshold).astype(int)
    merged["classification_method"] = "qwi_industry_lookup"

    n_classified = merged["seasonal_index"].notna().sum()
    n_total      = len(merged)
    pct_seasonal = merged["is_seasonal"].mean()
    print(
        f"Industry lookup: {n_classified:,}/{n_total:,} firms classified "
        f"({n_classified/n_total:.1%}); {pct_seasonal:.1%} flagged as seasonal."
    )

    return merged.drop(columns=["naics6_str"], errors="ignore")


# ── Method 2: Direct estimation from UI records ───────────────────────────────

def estimate_firm_excess_q4(
    events: pd.DataFrame,
    min_seps: int = 20,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    For each firm, estimate the excess Q4 separation rate using the event panel.

    Model:
        sep_at_k4 = α + Σ_q β_q · I(quarter_sep = q) + ε
        (where k=4 is a one-year follow-up, in quarterly data)

    Excess Q4 = β_Q4 - (β_Q3 + β_Q1) / 2  [analogous to CP's excess recurrence]

    For firms with too few events (< min_seps), the estimate is set to NaN.

    Parameters
    ----------
    events  : Separation events panel from build_separation_events_fast().
    min_seps: Minimum focal separations per firm for direct estimation.
    weight_col: Optional column for analytic weights.

    Returns
    -------
    DataFrame with employer_id and estimated excess_q4.
    """
    # We need the k=4 follow-up and the quarter of the focal separation
    # k=4 in quarterly data = one year after the focal separation
    k4 = events[events["k"] == 4].copy()

    if "sep_at_k" not in k4.columns:
        raise ValueError("sep_at_k column not found in events DataFrame.")

    k4 = k4.dropna(subset=["sep_at_k", "quarter_sep"])
    k4["sep_at_k"] = k4["sep_at_k"].astype(float)

    # Create quarter dummies
    for q in [1, 2, 3, 4]:
        k4[f"q{q}"] = (k4["quarter_sep"] == q).astype(float)

    rows = []
    for eid, grp in k4.groupby("employer_id"):
        if len(grp) < min_seps:
            rows.append({
                "employer_id":  eid,
                "excess_q4":    np.nan,
                "se_excess_q4": np.nan,
                "n_seps":       len(grp),
                "method":       "direct_insufficient_data",
            })
            continue

        try:
            formula = "sep_at_k ~ q1 + q2 + q3 + q4 - 1"
            if weight_col and weight_col in grp.columns:
                mod = smf.wls(formula, data=grp, weights=grp[weight_col]).fit()
            else:
                mod = smf.ols(formula, data=grp).fit()

            b1 = mod.params.get("q1", 0.0)
            b3 = mod.params.get("q3", 0.0)
            b4 = mod.params.get("q4", 0.0)

            excess_q4 = b4 - (b1 + b3) / 2

            # Delta method SE
            cov = mod.cov_params()
            var_e = (
                  cov.loc["q4", "q4"]
                + 0.25 * cov.get("q1", pd.Series({"q1": 0})).get("q1", 0)
                + 0.25 * cov.get("q3", pd.Series({"q3": 0})).get("q3", 0)
                - cov.loc["q4", "q1"] if "q1" in cov.index else 0
                - cov.loc["q4", "q3"] if "q3" in cov.index else 0
            )
            se_e = np.sqrt(max(var_e, 0))

            rows.append({
                "employer_id":  eid,
                "excess_q4":    excess_q4,
                "se_excess_q4": se_e,
                "n_seps":       len(grp),
                "method":       "direct",
            })
        except Exception as err:
            rows.append({
                "employer_id":  eid,
                "excess_q4":    np.nan,
                "se_excess_q4": np.nan,
                "n_seps":       len(grp),
                "method":       f"direct_failed: {err}",
            })

    result = pd.DataFrame(rows)
    n_ok = (result["method"] == "direct").sum()
    print(f"Direct estimation: {n_ok:,}/{len(result):,} firms estimated "
          f"({n_ok/len(result):.1%} had ≥{min_seps} separations)")
    return result


def classify_by_direct_estimation(
    direct_estimates: pd.DataFrame,
    threshold_pct: float = 90.0,
) -> pd.DataFrame:
    """
    Flag the top percentile of directly-estimated excess_q4 firms as seasonal.

    Parameters
    ----------
    direct_estimates: Output of estimate_firm_excess_q4().
    threshold_pct   : Percentile cutoff (default 90th percentile = top decile).
    """
    df = direct_estimates.copy()
    valid = df["excess_q4"].notna()

    cutoff = df.loc[valid, "excess_q4"].quantile(threshold_pct / 100)
    df["is_seasonal"]          = (df["excess_q4"] >= cutoff).astype(int)
    df["seasonal_score"]       = df["excess_q4"]   # continuous version
    df["classification_method"] = "direct_estimation"

    print(f"Top {100 - threshold_pct:.0f}% excess-Q4 threshold: {cutoff:.4f}")
    print(f"Firms flagged seasonal: {df['is_seasonal'].sum():,} "
          f"({df['is_seasonal'].mean():.1%})")
    return df


# ── Combined classifier ───────────────────────────────────────────────────────

def compute_firm_seasonality(
    ui_path: Path | pd.DataFrame,
    qwi_index_path: Path,
    output_path: Path | None = None,
    direct_min_seps: int = 20,
    industry_threshold: float = 0.5,
    direct_threshold_pct: float = 90.0,
    naics_col: str = "naics6",
) -> pd.DataFrame:
    """
    Full pipeline: load UI data → build events → classify firms.

    Produces two seasonal scores per firm:
        industry_seasonal_index : from QWI lookup (always available if NAICS present)
        direct_excess_q4        : from firm's own separation pattern (needs enough data)

    When direct estimation has insufficient data, falls back to industry lookup.

    Parameters
    ----------
    ui_path             : Path to UI wage records parquet/CSV, or DataFrame directly.
    qwi_index_path      : Path to seasonal_index_naics6.csv (from seasonal_index.py).
    output_path         : Where to save results (CSV).
    direct_min_seps     : Min separations per firm for direct estimation.
    industry_threshold  : Cutoff on QWI seasonal_index (0–1) for is_seasonal flag.
    direct_threshold_pct: Percentile cutoff for direct-estimation flag.
    naics_col           : Column name for NAICS code in UI records.
    """
    # Load QWI index
    qwi_index = pd.read_csv(qwi_index_path, dtype={"naics_code": str})
    print(f"QWI index loaded: {len(qwi_index):,} industries")

    # Load UI data
    if isinstance(ui_path, pd.DataFrame):
        ui = ui_path
    else:
        ui = load_ui_records(ui_path, naics_col=naics_col)

    # Build employment panel
    panel = build_quarterly_employment(ui)

    # Build separation events
    events = build_separation_events_fast(panel, max_horizon=8)

    # ── Method 1: Industry lookup ────────────────────────────────────────────
    # Firm list with NAICS code
    firm_naics = (
        ui.groupby("employer_id")[naics_col]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={naics_col: "naics6"})
    )
    ind_class = classify_by_industry(
        firm_naics, qwi_index,
        threshold=industry_threshold,
    )
    ind_class = ind_class.rename(columns={
        "seasonal_index": "industry_seasonal_index",
        "is_seasonal":    "is_seasonal_industry",
        "peak_quarter":   "industry_peak_quarter",
    })

    # ── Method 2: Direct estimation ──────────────────────────────────────────
    direct = estimate_firm_excess_q4(events, min_seps=direct_min_seps)
    direct_class = classify_by_direct_estimation(
        direct, threshold_pct=direct_threshold_pct
    )
    direct_class = direct_class.rename(columns={
        "excess_q4":  "direct_excess_q4",
        "is_seasonal": "is_seasonal_direct",
        "n_seps":      "n_focal_seps",
    })

    # ── Merge both methods ───────────────────────────────────────────────────
    result = ind_class.merge(
        direct_class[["employer_id", "direct_excess_q4", "is_seasonal_direct",
                       "n_focal_seps", "se_excess_q4"]],
        on="employer_id",
        how="left",
    )

    # Combined flag: use direct if available, else industry lookup
    result["is_seasonal"] = np.where(
        result["n_focal_seps"] >= direct_min_seps,
        result["is_seasonal_direct"],
        result["is_seasonal_industry"],
    )
    result["seasonal_method"] = np.where(
        result["n_focal_seps"] >= direct_min_seps,
        "direct",
        "industry_lookup",
    )

    # Summary
    print(f"\n=== Final firm classification ===")
    print(f"  Firms classified by direct estimation: "
          f"{(result['seasonal_method']=='direct').sum():,}")
    print(f"  Firms classified by industry lookup:   "
          f"{(result['seasonal_method']=='industry_lookup').sum():,}")
    print(f"  Seasonal firms (combined): {result['is_seasonal'].sum():,} "
          f"({result['is_seasonal'].mean():.1%})")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"  Results saved to {output_path}")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify firms as seasonal using UI wage records and QWI index"
    )
    parser.add_argument("--ui-data",    type=Path, required=True,
                        help="Path to UI wage records (parquet or CSV)")
    parser.add_argument("--qwi-index",  type=Path, required=True,
                        help="Path to seasonal_index_naics6.csv")
    parser.add_argument("--output",     type=Path,
                        default=Path("output/firm_seasonality.csv"),
                        help="Output CSV path")
    parser.add_argument("--naics-col",  default="naics6",
                        help="Column name for NAICS code in UI records")
    parser.add_argument("--min-seps",   type=int, default=20,
                        help="Min separations per firm for direct estimation")
    parser.add_argument("--ind-threshold", type=float, default=0.5,
                        help="QWI seasonal_index threshold for is_seasonal (0–1)")
    parser.add_argument("--direct-pct",    type=float, default=90.0,
                        help="Percentile cutoff for direct-estimation flag")
    args = parser.parse_args()

    result = compute_firm_seasonality(
        ui_path=args.ui_data,
        qwi_index_path=args.qwi_index,
        output_path=args.output,
        naics_col=args.naics_col,
        direct_min_seps=args.min_seps,
        industry_threshold=args.ind_threshold,
        direct_threshold_pct=args.direct_pct,
    )
    print(result.describe())
