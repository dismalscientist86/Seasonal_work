"""
fetch_qwi.py — Download Quarterly Workforce Indicators (QWI) from the Census API.

The QWI provides quarterly employment, hiring, separation, and earnings statistics
by state × industry.  We want separation rates by 6-digit NAICS (or the finest
available granularity) to build a seasonal index.

Census QWI API documentation:
  https://lehd.ces.census.gov/data/schema/latest/LEHD_QWI_API.html
  https://api.census.gov/data/timeseries/qwi/se

Key variables we fetch:
  Emp     : employment (end of quarter)
  Sep     : separations (outflow from employment)
  SepBeg  : separations from beginning-of-quarter employment
  SepS    : stable separations (not returning within a year)

Usage:
    python code/qwi/fetch_qwi.py          # fetch all states, all available quarters
    python code/qwi/fetch_qwi.py --state CA  # single state
"""

import sys
import time
import json
from pathlib import Path
from typing import Iterator

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CENSUS_API_KEY,
    QWI_RAW,
    QWI_YEARS,
    QWI_NAICS_LEVEL,
    STATE_FIPS,
    STATE_NAMES,
)


# Census QWI API base URL
QWI_BASE = "https://api.census.gov/data/timeseries/qwi/se"

# Variables to fetch.
# NOTE: "Payroll" (Total Quarterly Payroll: Sum) was dropped 2026-07 — verified
# live against the Census API that it returns null for every cell tested
# (including large industries where Emp/EmpEnd populate fine; status flag is
# always 5, unlike EarnBeg/EarnS which return flag 1 = good). "EarnBeg"
# (End-of-Quarter Employment: Average Monthly Earnings) is the working
# equivalent and is what code/qwi/earnings_index.py is built on.
QWI_VARS = ["Emp", "EmpEnd", "Sep", "SepBeg", "EmpS", "EarnBeg"]


#  API helpers 

def _qwi_quarters(years: list[int]) -> list[str]:
    """Generate quarter strings like '2010Q1', '2010Q2', … for the given years."""
    quarters = []
    for y in years:
        for q in range(1, 5):
            quarters.append(f"{y}Q{q}")
    return quarters


def _build_url(
    variables: list[str],
    state_fips: str,
    industry_code: str,
    year: int,
    quarter: int,
    api_key: str = "",
) -> str:
    get = ",".join(variables)
    params = {
        "get":      get,
        "for":      f"state:{state_fips}",
        "industry": industry_code,
        "time":     f"{year}Q{quarter}",
    }
    if api_key:
        params["key"] = api_key

    base = f"{QWI_BASE}?"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return base + query


def _fetch_one(
    state: str,
    industry: str,
    years: list[int],
    api_key: str,
    retry_delay: float = 1.0,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """
    Fetch QWI data for one state × industry combination across all years/quarters.

    Returns a DataFrame or None if the query fails after retries.
    """
    # The API supports a time range in a single call
    time_range = f"{min(years)}Q1 to {max(years)}Q4"
    get_vars   = ",".join(QWI_VARS)

    params = {
        "get":      get_vars,
        "for":      f"state:{state}",
        "industry": industry,
        "time":     f"from {time_range}",
    }
    if api_key:
        params["key"] = api_key

    for attempt in range(max_retries):
        try:
            r = requests.get(QWI_BASE, params=params, timeout=30)
            if r.status_code == 204:
                return None   # No data for this cell
            r.raise_for_status()
            data = r.json()
            if len(data) < 2:
                return None
            cols = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=cols)
            # Convert numeric columns
            for c in QWI_VARS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["state"]    = state
            df["industry"] = industry
            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 204:
                return None   # No data for this cell
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"    HTTP error for state={state}, ind={industry}: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"    Error for state={state}, ind={industry}: {e}")
                return None

    return None


#  Industry list helper 

def get_available_industries(naics_level: int = 6, api_key: str = "") -> list[str]:
    """
    Query the Census API to discover which NAICS codes are available
    at the requested digit level.

    The Census QWI schema is documented at:
    https://lehd.ces.census.gov/data/schema/latest/variables_qwipu.xlsx

    For 6-digit NAICS, many codes may be suppressed due to disclosure avoidance;
    we fetch what is available.
    """
    # Try to get the list of industry codes from the Census variables endpoint
    schema_url = "https://api.census.gov/data/timeseries/qwi/se/variables.json"
    try:
        r = requests.get(schema_url, timeout=30)
        r.raise_for_status()
        # Parse variables for the industry dimension
        # (In practice the schema doesn't enumerate industry codes; use a lookup table)
    except Exception:
        pass

    # Use NAICS 2022 codes at the requested digit level
    # For a complete list, see https://www.census.gov/naics/
    # Here we load from a bundled NAICS table if available, else return placeholder
    naics_file = Path(__file__).parent / "naics_codes.csv"
    if naics_file.exists():
        naics_df = pd.read_csv(naics_file, dtype=str)
        col = f"naics{naics_level}"
        if col in naics_df.columns:
            return naics_df[col].dropna().unique().tolist()

    # Fallback: return 2-digit NAICS codes (always available)
    two_digit = [f"{i:02d}" for i in [
        11, 21, 22, 23, 31, 32, 33, 42, 44, 45, 48, 49,
        51, 52, 53, 54, 55, 56, 61, 62, 71, 72, 81, 92,
    ]]
    print(
        f"WARNING: NAICS code list for level {naics_level} not found. "
        f"Falling back to 2-digit codes.\n"
        f"To use finer granularity, provide code/qwi/naics_codes.csv "
        f"with a 'naics{naics_level}' column."
    )
    return two_digit


#  Main fetch function 

def fetch_qwi(
    states: list[str] | None = None,
    naics_level: int | None = None,
    years: list[int] | None = None,
    api_key: str | None = None,
    output_dir: Path | None = None,
    national_only: bool = False,
) -> Path:
    """
    Fetch QWI separation data and save to parquet files.

    Parameters
    ----------
    states       : List of 2-digit FIPS codes. Defaults to all states.
    naics_level  : Industry digit level (2, 3, 4, or 6). Defaults to config.
    years        : List of years. Defaults to config QWI_YEARS.
    api_key      : Census API key. Defaults to CENSUS_API_KEY from config/.env.
    output_dir   : Where to save files. Defaults to QWI_RAW.
    national_only: If True, aggregate across all states before saving.

    Returns
    -------
    Path to the saved parquet file.
    """
    states     = states     or STATE_FIPS
    naics_level = naics_level or QWI_NAICS_LEVEL
    years      = years      or QWI_YEARS
    api_key    = api_key    or CENSUS_API_KEY
    output_dir = output_dir or QWI_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        print(
            "WARNING: No Census API key set. Requests will be rate-limited to "
            "500/day without a key. Set CENSUS_API_KEY in your .env file.\n"
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )

    industries = get_available_industries(naics_level=naics_level, api_key=api_key)
    print(
        f"\nFetching QWI: {len(states)} states × {len(industries)} NAICS-{naics_level} "
        f"industries × {len(years)} years"
    )

    all_frames = []
    n_cells    = len(states) * len(industries)

    with tqdm(total=n_cells, desc="API calls") as pbar:
        for state in states:
            out_path = output_dir / f"qwi_state_{state}.parquet"

            # Resume support: skip a state if it was already fetched with the
            # current QWI_VARS (checked via a variable that must have real,
            # non-null data if this file reflects the current fetch — guards
            # against silently reusing a file saved under an older QWI_VARS
            # list that didn't include that variable at all).
            if out_path.exists():
                cached = pd.read_parquet(out_path)
                check_var = QWI_VARS[-1]
                if check_var in cached.columns and cached[check_var].notna().any():
                    all_frames.append(cached)
                    pbar.update(len(industries))
                    continue

            state_frames = []
            for ind in industries:
                df = _fetch_one(state, ind, years, api_key)
                if df is not None and len(df) > 0:
                    state_frames.append(df)
                pbar.update(1)
                time.sleep(0.05)  # gentle rate limiting

            if state_frames:
                state_df = pd.concat(state_frames, ignore_index=True)
                all_frames.append(state_df)

                # Save per-state file incrementally (enables the resume check above)
                state_df.to_parquet(out_path, index=False)

    if not all_frames:
        raise RuntimeError("No QWI data fetched. Check your API key and internet connection.")

    full_df = pd.concat(all_frames, ignore_index=True)

    # Parse time column: "2010-Q1" → year and quarter
    full_df["year"]    = full_df["time"].str[:4].astype(int)
    full_df["quarter"] = full_df["time"].str[-1].astype(int)

    # Optionally aggregate to national
    if national_only:
        agg_cols = [c for c in QWI_VARS if c in full_df.columns]
        full_df = (
            full_df
            .groupby(["industry", "year", "quarter"])[agg_cols]
            .sum()
            .reset_index()
        )

    # Only overwrite the shared "_all" combined file (used by seasonal_index.py
    # and geographic_analysis.py as the default input) when this run actually
    # covered every state. A partial/test run (e.g. --state 06) would otherwise
    # silently clobber that file with a small subset — this happened once
    # (2026-07) and corrupted downstream results until caught.
    if set(states) == set(STATE_FIPS):
        out_path = output_dir / f"qwi_naics{naics_level}_all.parquet"
    else:
        state_tag = "-".join(sorted(states))
        out_path = output_dir / f"qwi_naics{naics_level}_subset_{state_tag}.parquet"
        print(f"\nPartial run ({len(states)} state(s)): NOT overwriting the "
              f"combined qwi_naics{naics_level}_all.parquet. Saving to {out_path} instead.")

    full_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(full_df):,} rows to {out_path}")
    return out_path


#  Entry point 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch QWI data from Census API")
    parser.add_argument("--state", nargs="+", default=None,
                        help="State FIPS code(s), e.g. 06 48")
    parser.add_argument("--naics-level", type=int, default=QWI_NAICS_LEVEL,
                        help=f"NAICS digit level (default {QWI_NAICS_LEVEL})")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Years to fetch (default from config)")
    parser.add_argument("--national", action="store_true",
                        help="Aggregate to national totals")
    args = parser.parse_args()

    out = fetch_qwi(
        states=args.state,
        naics_level=args.naics_level,
        years=args.years,
        national_only=args.national,
    )
    print(f"\nOutput: {out}")
