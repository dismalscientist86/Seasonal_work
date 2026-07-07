"""
fetch_cbp.py — Download 2023 County Business Patterns (CBP) data from the Census API
at the county level, for building a county x NAICS seasonality index.

This is the county-level counterpart to code/qwi/fetch_qwi.py: where QWI gives
state x NAICS6 separation rates, CBP gives county x NAICS establishment counts.
The intended use is to merge counties onto the state x NAICS6 seasonal index
(geographic_analysis.py's output) using each county's industry mix as weights.

Census CBP API documentation:
  https://www.census.gov/data/developers/data-sets/cbp-zbp/cbp-api.html
  https://api.census.gov/data/2023/cbp/variables.html

Key facts about the 2023 CBP API (verified directly, since vintage-specific
Census variables can shift):
  - 2023 CBP still classifies industries using NAICS2017, not NAICS2022.
    Most 6-digit codes are unchanged between vintages, but a handful differ;
    codes that don't exist in NAICS2017 simply return no data (HTTP 204)
    and are skipped automatically — no special handling needed here.
  - Geography: `for=county:*&in=state:XX` returns every county in one state
    for a single NAICS code in ONE API call (the 'in' clause only accepts a
    single state per call, so we still loop over states one at a time).
  - LFO (legal form of organization) is "default displayed": if you don't
    pass it as a FILTER (not just in `get=`), the API returns a separate row
    for every LFO category instead of just the total. We filter to LFO=001
    ("Total for all establishments") by default.
  - County-level NAICS6 cells are suppressed far more heavily than QWI's
    state-level NAICS6 cells. This script also fetches 4-digit (industry
    group) codes as a coarser fallback with much better county coverage.

Usage:
    python code/cbp/fetch_cbp.py --state 06          # test: California only
    python code/cbp/fetch_cbp.py                     # all states, both NAICS levels
    python code/cbp/fetch_cbp.py --naics-level 6      # 6-digit only
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CENSUS_API_KEY,
    CBP_RAW,
    CBP_YEAR,
    CBP_NAICS_LEVELS,
    CBP_LFO_TOTAL,
    STATE_FIPS,
    STATE_NAMES,
)

CBP_BASE = "https://api.census.gov/data/{year}/cbp"

# Variables to fetch, plus their suppression-flag companions (xxx_F).
# A flag value other than "G" (good) indicates the estimate was modified
# for disclosure avoidance; that distinction is left for a later cleaning
# step, not handled here.
CBP_VARS = [
    "ESTAB", "ESTAB_F",
    "EMP", "EMP_F",
    "PAYANN", "PAYANN_F",
    "PAYQTR1", "PAYQTR1_F",
    "NAICS2017_LABEL",
    "NAME",
]

NUMERIC_VARS = ["ESTAB", "EMP", "PAYANN", "PAYQTR1"]


#  NAICS code list helpers 

def load_naics_codes(level: int, naics_file: Path | None = None) -> list[str]:
    """
    Load the project's NAICS code list at `level` digits.

    Reuses code/qwi/naics_codes.csv by default (the same file fetch_qwi.py
    uses). If that file has a 'naics{level}' column, use it directly;
    otherwise derive the level by truncating the 6-digit code list.

    NOTE: per the project README, naics_codes.csv is NAICS2022 vintage,
    while CBP 2023 uses NAICS2017. Most NAICS6 codes are unchanged between
    the two vintages — codes that don't exist in NAICS2017 will simply
    return no data (HTTP 204) from the API and are skipped, so this vintage
    mismatch should not require any special-casing here.
    """
    if naics_file is None:
        naics_file = Path(__file__).resolve().parents[1] / "qwi" / "naics_codes.csv"

    if not naics_file.exists():
        raise FileNotFoundError(
            f"Could not find {naics_file}. This script reuses the NAICS code "
            "list from the qwi/ fetcher; pass --naics-file to point elsewhere."
        )

    df = pd.read_csv(naics_file, dtype=str)
    col = f"naics{level}"
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())

    if "naics6" not in df.columns:
        raise ValueError(f"{naics_file} has neither a '{col}' nor a 'naics6' column.")

    print(f"  No '{col}' column in {naics_file.name}; deriving NAICS-{level} "
          f"codes by truncating the 6-digit list.")
    codes6 = df["naics6"].dropna().unique().tolist()
    return sorted({c[:level] for c in codes6 if len(c) >= level})


#  API helpers 

def _fetch_state_industry(
    state: str,
    naics_code: str,
    year: int,
    api_key: str,
    lfo: str = CBP_LFO_TOTAL,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> pd.DataFrame | None:
    """
    Fetch every county in `state` for a single NAICS2017 code (one API call).

    Returns a DataFrame or None if the cell has no data (suppressed, the
    code doesn't exist in NAICS2017, or the request failed after retries).
    """
    url = CBP_BASE.format(year=year)
    params = {
        "get": ",".join(CBP_VARS),
        "for": "county:*",
        "in": f"state:{state}",
        "NAICS2017": naics_code,
        "LFO": lfo,
    }
    if api_key:
        params["key"] = api_key

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 204:
                return None  # no data for this cell
            r.raise_for_status()
            data = r.json()
            if len(data) < 2:
                return None
            cols, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=cols)
            for c in NUMERIC_VARS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["naics_code"] = naics_code
            df["naics_level"] = len(naics_code)
            df["year"] = year
            return df

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 204:
                return None
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"    HTTP error for state={state}, naics={naics_code}: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"    Error for state={state}, naics={naics_code}: {e}")
                return None

    return None


#  Main fetch function 

def fetch_cbp_counties(
    states: list[str] | None = None,
    naics_level: int = 6,
    year: int = CBP_YEAR,
    lfo: str = CBP_LFO_TOTAL,
    api_key: str | None = None,
    output_dir: Path | None = None,
    naics_codes: list[str] | None = None,
) -> Path:
    """
    Fetch CBP county-level data for one NAICS digit level across all requested
    states, and save to parquet. Resumable: if a per-state file already
    exists for this naics_level, that state is skipped (so an interrupted
    run — these can be large — can simply be re-launched).

    Parameters
    ----------
    states      : List of 2-digit FIPS codes. Defaults to all 50 states + DC.
    naics_level : 6 or 4 (digit length of the NAICS codes to fetch).
    year        : CBP reference year. Defaults to config.CBP_YEAR (2023).
    lfo         : Legal form of organization filter. Defaults to "001" (total).
    api_key     : Census API key. Defaults to CENSUS_API_KEY from config/.env.
    output_dir  : Where to save files. Defaults to config.CBP_RAW.
    naics_codes : Explicit code list, bypassing load_naics_codes().

    Returns
    -------
    Path to the combined parquet file for this naics_level.
    """
    states      = states or STATE_FIPS
    api_key     = api_key or CENSUS_API_KEY
    output_dir  = output_dir or CBP_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        print(
            "WARNING: No Census API key set. Requests will be rate-limited to "
            "500/day without a key. Set CENSUS_API_KEY in your .env file.\n"
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )

    if naics_codes is None:
        naics_codes = load_naics_codes(naics_level)

    n_cells = len(states) * len(naics_codes)
    print(
        f"\nFetching CBP {year}: {len(states)} states x {len(naics_codes)} "
        f"NAICS-{naics_level} codes (LFO={lfo}); {n_cells:,} API calls total"
    )

    with tqdm(total=n_cells, desc=f"API calls (NAICS-{naics_level})") as pbar:
        for state in states:
            out_path = output_dir / f"cbp_county_naics{naics_level}_state_{state}.parquet"
            if out_path.exists():
                pbar.update(len(naics_codes))
                continue  # resume: this state already fetched at this NAICS level

            state_frames = []
            for naics_code in naics_codes:
                df = _fetch_state_industry(state, naics_code, year, api_key, lfo=lfo)
                if df is not None and len(df) > 0:
                    state_frames.append(df)
                pbar.update(1)
                time.sleep(0.05)  # gentle rate limiting

            if state_frames:
                state_df = pd.concat(state_frames, ignore_index=True)
            else:
                state_df = pd.DataFrame(columns=CBP_VARS)
            state_df.to_parquet(out_path, index=False)

    # Combine whatever is on disk for this naics_level (works for resumed runs too)
    state_files = sorted(output_dir.glob(f"cbp_county_naics{naics_level}_state_*.parquet"))
    combined = pd.concat(
        [pd.read_parquet(f) for f in state_files], ignore_index=True
    ) if state_files else pd.DataFrame(columns=CBP_VARS)

    if "state" in combined.columns:
        combined["state_name"] = combined["state"].map(STATE_NAMES)

    combined_path = output_dir / f"cbp_county_naics{naics_level}_all.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nSaved {len(combined):,} county x industry rows to {combined_path}")
    return combined_path


#  Entry point 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch 2023 CBP county-level data from the Census API")
    parser.add_argument(
        "--state", nargs="+", default=None,
        help="State FIPS code(s) to fetch, e.g. --state 06 for California. "
             "Recommended for a first test run before fetching all states.",
    )
    parser.add_argument(
        "--naics-level", nargs="+", type=int, default=CBP_NAICS_LEVELS, choices=[4, 6],
        help=f"NAICS digit level(s) to fetch (default: {CBP_NAICS_LEVELS}, i.e. both).",
    )
    parser.add_argument("--year", type=int, default=CBP_YEAR)
    parser.add_argument(
        "--lfo", type=str, default=CBP_LFO_TOTAL,
        help="Legal form of organization code to filter to (default 001 = total).",
    )
    parser.add_argument(
        "--naics-file", type=Path, default=None,
        help="Optional override for the NAICS code list CSV (default: code/qwi/naics_codes.csv).",
    )
    args = parser.parse_args()

    for level in args.naics_level:
        codes = load_naics_codes(level, naics_file=args.naics_file) if args.naics_file else None
        out = fetch_cbp_counties(
            states=args.state,
            naics_level=level,
            year=args.year,
            lfo=args.lfo,
            naics_codes=codes,
        )
        print(f"NAICS-{level} output: {out}")
