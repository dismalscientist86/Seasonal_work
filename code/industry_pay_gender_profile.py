"""
industry_pay_gender_profile.py — Build one profile per Census industry code
combining: realized annual pay (ACS PUMS `WAGP`), gender composition (ACS
PUMS `SEX`), and seasonality (QWI `seasonal_index`, aggregated up from
6-digit NAICS to match PUMS' coarser industry classification).

This exists specifically to fix a problem with an earlier, more ad hoc pass
at this comparison: that pass used QWI's `EarnBeg` (average monthly pay
among people employed that quarter) annualized by x12 as a stand-in for
"annual pay" — which silently assumes year-round work, overstating realized
income for anyone in a seasonal job who doesn't actually work all four
quarters. It also pulled gender composition from a *different*, coarser
BLS table than the NAICS6-level seasonality figure it was compared to,
mixing resolutions without saying so. This script fixes both: PUMS' `WAGP`
is what a person actually received in the past 12 months (whatever that
was), and pay/gender/seasonality are all computed at the same Census
industry-code resolution (see build_indp_naics_crosswalk.py for why that
resolution, not 6-digit NAICS, is the ceiling here).

Pipeline:
  1. Load ACS PUMS person records (fetch_acs_pums.py) -> per Census
     industry code: weighted median realized annual wage/salary income,
     weighted %female, weighted mean weeks worked (a direct check on
     "how much of the year do this industry's workers actually work").
  2. Load QWI's national seasonal_index_naics6.csv -> per NAICS6: the
     quarterly excess-separation values (excessQ1..Q4).
  3. Weight each NAICS6 industry's excess-by-quarter by its average QWI
     employment level, map it to its Census industry code via
     build_indp_naics_crosswalk.py, and take an employment-weighted mean
     within each code -- then rebuild amplitude/peak-quarter/seasonal_index
     from *those* aggregated quarters, the same way seasonal_index.py does
     at the NAICS6 level (see build_seasonal_index()).
  4. Merge (1) and (3) on the Census industry code.

Outputs:
  output/tables/industry_pay_gender_seasonality_profile.csv

Usage:
    python code/build_indp_naics_crosswalk.py   # once
    python code/fetch_acs_pums.py               # once, ~20-25 min
    python code/industry_pay_gender_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QWI_RAW, QWI_CLEAN, TABLES_DIR, REPO_ROOT

from build_indp_naics_crosswalk import match_naics6_to_indp

ACS_RAW = REPO_ROOT / "data" / "acs_raw"

# Civilian employed (at work, or with a job but not at work). Excludes
# unemployed, armed forces, and not-in-labor-force -- matches this project's
# other QWI/CPS work, which is likewise civilian-employment-based.
EMPLOYED_ESR_CODES = {1, 2}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median via the standard cumulative-weight-crossing-50% method."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    cutoff = cum[-1] / 2.0
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def load_pums(path: Path | None = None) -> pd.DataFrame:
    """Load and lightly filter the combined PUMS person file."""
    path = path or (ACS_RAW / "pums_all.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python code/fetch_acs_pums.py")
    df = pd.read_parquet(path)
    df = df[df["ESR"].isin(EMPLOYED_ESR_CODES)].copy()
    df = df[df["INDP"].str.len() == 4]  # drops 'N' (not in universe)
    df = df[df["WAGP"] >= 0]
    return df


def compute_pay_gender_by_indp(pums: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """
    Per Census industry code: weighted median realized annual wage/salary
    income, weighted %female, weighted mean weeks worked, and the
    unweighted sample size (for a thin-cell caveat -- PUMS is a ~1% sample,
    so a numerically small industry can have very few responding people
    even though it employs many nationally).
    """
    records = []
    for indp, grp in pums.groupby("INDP"):
        w = grp["PWGTP"].to_numpy(dtype=float)
        wage = grp["WAGP"].to_numpy(dtype=float)
        female = (grp["SEX"] == 2).to_numpy(dtype=float)
        weeks = grp["WKWN"].to_numpy(dtype=float)

        records.append({
            "indp_code": indp,
            "n_pums_records": len(grp),
            "median_annual_wage": _weighted_median(wage, w),
            "pct_female": float(np.average(female, weights=w)),
            "mean_weeks_worked": float(np.average(weeks[~np.isnan(weeks)], weights=w[~np.isnan(weeks)]))
                                 if np.any(~np.isnan(weeks)) else np.nan,
        })

    out = pd.DataFrame(records)
    out["thin_pums_sample"] = out["n_pums_records"] < min_n
    return out


def load_qwi_employment_weights(qwi_path: Path | None = None) -> pd.Series:
    """Average EmpEnd (2019+) by NAICS6 -- the weight used to aggregate
    NAICS6-level seasonality up to the Census-industry-code level."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "qwi"))
    from seasonal_index import load_qwi  # local import: adds code/qwi to sys.path

    qwi = load_qwi(path=qwi_path)
    recent = qwi[(qwi["year"] >= 2019) & (qwi["EmpEnd"] > 0)]
    return recent.groupby("industry")["EmpEnd"].mean()


def aggregate_seasonality_to_indp(
    seasonal_index_path: Path,
    emp_weights: pd.Series,
    crosswalk: pd.DataFrame,
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Map each NAICS6 industry to its Census industry code, then compute an
    employment-weighted mean of excessQ1..Q4 within each code and rebuild
    seasonal_amplitude / peak_quarter / seasonal_index from that aggregate
    -- the same construction seasonal_index.py's build_seasonal_index() uses
    at the NAICS6 level, just one level up.
    """
    sea = pd.read_csv(seasonal_index_path, dtype={"naics_code": str, "sector_2d": str})
    sea["emp_weight"] = sea["naics_code"].map(emp_weights)
    sea["indp_code"] = sea["naics_code"].map(lambda c: match_naics6_to_indp(c, crosswalk))
    sea = sea.dropna(subset=["indp_code", "emp_weight"])

    excess_cols = ["excessQ1", "excessQ2", "excessQ3", "excessQ4"]
    records = []
    for indp, grp in sea.groupby("indp_code"):
        w = grp["emp_weight"].to_numpy(dtype=float)
        agg = {}
        for col in excess_cols:
            vals = grp[col].to_numpy(dtype=float)
            mask = ~np.isnan(vals)
            agg[col] = float(np.average(vals[mask], weights=w[mask])) if mask.sum() >= min_obs else np.nan

        available = [c for c in excess_cols if not np.isnan(agg[c])]
        record = {
            "indp_code": indp,
            "n_naics6_mapped": len(grp),
            "total_qwi_emp": w.sum(),
            "member_naics6": ";".join(sorted(grp["naics_code"])),
            **agg,
        }
        if len(available) >= 2:
            vals = {c: agg[c] for c in available}
            peak_col = max(vals, key=vals.get)
            record["indp_seasonal_amplitude"] = max(vals.values()) - min(vals.values())
            record["indp_peak_quarter"] = int(peak_col[-1])
            record["indp_peak_excess"] = vals[peak_col]
        else:
            record["indp_seasonal_amplitude"] = np.nan
            record["indp_peak_quarter"] = pd.NA
            record["indp_peak_excess"] = np.nan
        records.append(record)

    out = pd.DataFrame(records)
    p99 = out["indp_seasonal_amplitude"].quantile(0.99)
    out["indp_seasonal_index"] = (out["indp_seasonal_amplitude"] / p99).clip(0, 1)
    return out


def build_profile(
    pums_path: Path | None = None,
    seasonal_index_path: Path | None = None,
    crosswalk_path: Path | None = None,
) -> pd.DataFrame:
    crosswalk_path = crosswalk_path or (ACS_RAW / "indp_naics_crosswalk.csv")
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"{crosswalk_path} not found. Run: python code/build_indp_naics_crosswalk.py"
        )
    crosswalk = pd.read_csv(crosswalk_path, dtype=str)
    descriptions = crosswalk.set_index("indp_code")["description"]

    print("Loading ACS PUMS ...")
    pums = load_pums(pums_path)
    print(f"  {len(pums):,} employed-civilian person records across "
          f"{pums['INDP'].nunique()} industry codes")
    pay_gender = compute_pay_gender_by_indp(pums)

    print("Aggregating QWI seasonality to the Census-industry-code level ...")
    seasonal_index_path = seasonal_index_path or (QWI_CLEAN / "seasonal_index_naics6.csv")
    emp_weights = load_qwi_employment_weights()
    seasonality = aggregate_seasonality_to_indp(seasonal_index_path, emp_weights, crosswalk)

    profile = seasonality.merge(pay_gender, on="indp_code", how="inner")
    profile["description"] = profile["indp_code"].map(descriptions)
    profile = profile.sort_values("indp_seasonal_index", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "industry_pay_gender_seasonality_profile.csv"
    cols = [
        "indp_code", "description", "indp_seasonal_index", "indp_peak_quarter",
        "indp_peak_excess", "median_annual_wage", "pct_female", "mean_weeks_worked",
        "n_naics6_mapped", "total_qwi_emp", "n_pums_records", "thin_pums_sample",
        "member_naics6",
    ]
    profile[[c for c in cols if c in profile.columns]].to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(profile)} industry codes)")

    print("\nTop 15 by seasonality:")
    print(profile.head(15)[["indp_code", "description", "indp_seasonal_index",
                             "median_annual_wage", "pct_female"]].to_string(index=False))
    return profile


if __name__ == "__main__":
    build_profile()
