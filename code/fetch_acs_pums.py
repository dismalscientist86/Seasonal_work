"""
fetch_acs_pums.py — Download ACS 1-year Public Use Microdata Sample (PUMS)
person records from the Census API.

Why: QWI's `EarnBeg` (used throughout code/qwi/) is a *pay rate* — average
monthly earnings among people employed that quarter — not a worker's actual
annual income. Annualizing it (rate x 12) silently assumes year-round work,
which overstates realized income for anyone in a seasonal job who doesn't
actually work all four quarters — exactly the population this project's
seasonal-index work is about. ACS PUMS asks each respondent for their real
wage/salary income received in the past 12 months (`WAGP`), whatever that
was, so someone who only worked part of the year reports their actual
(lower) earnings. It also carries `SEX` and `WKWN` (weeks worked in the past
12 months) on the same person records, so pay, gender composition, and an
independent check on "how much of the year did this industry's workers
actually work" all come from one consistent source.

Industry classification note: PUMS' `INDP` variable uses the Census industry
code system (2022 vintage here), which is *coarser* than 6-digit NAICS for
many service industries. code/build_indp_naics_crosswalk.py builds the
NAICS6 -> INDP lookup used to join this data onto the QWI seasonal index.

Variables fetched: SEX, WAGP (wage/salary income, past 12 months), PWGTP
(person weight), WKWN (weeks worked, past 12 months), WKHP (usual hours/week),
ESR (employment status recode), INDP (detailed industry), AGEP (age).

Usage:
    python code/fetch_acs_pums.py                 # all states, ~20-25 min
    python code/fetch_acs_pums.py --state 06 48    # a subset, for testing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CENSUS_API_KEY, REPO_ROOT, STATE_FIPS

ACS_YEAR = 2023
ACS_BASE = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs1/pums"

PUMS_VARS = ["SEX", "WAGP", "PWGTP", "WKWN", "WKHP", "ESR", "INDP", "AGEP"]

RAW_DIR = REPO_ROOT / "data" / "acs_raw"


def _fetch_one_state(state: str, api_key: str, max_retries: int = 3, retry_delay: float = 2.0) -> pd.DataFrame | None:
    """Fetch every PUMS person record for one state (one API call)."""
    params = {"get": ",".join(PUMS_VARS), "for": f"state:{state}"}
    if api_key:
        params["key"] = api_key

    for attempt in range(max_retries):
        try:
            r = requests.get(ACS_BASE, params=params, timeout=120)
            if r.status_code == 204:
                return None
            r.raise_for_status()
            data = r.json()
            if len(data) < 2:
                return None
            cols, rows = data[0], data[1:]
            df = pd.DataFrame(rows, columns=cols)
            for c in ["SEX", "WAGP", "PWGTP", "WKWN", "WKHP", "ESR", "AGEP"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 204:
                return None
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"    HTTP error for state={state}: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"    Error for state={state}: {e}")
                return None
    return None


def fetch_pums(
    states: list[str] | None = None,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Fetch ACS 1-year PUMS person records for each state (resumable: skips a
    state whose output file already exists), then save a combined parquet.

    Returns the path to the combined parquet.
    """
    states = states or STATE_FIPS
    api_key = api_key or CENSUS_API_KEY
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not api_key:
        print(
            "WARNING: No Census API key set. Requests will be rate-limited to "
            "500/day without a key. Set CENSUS_API_KEY in your .env file."
        )

    frames = []
    for state in tqdm(states, desc="States"):
        out_path = output_dir / f"pums_state_{state}.parquet"
        if out_path.exists():
            frames.append(pd.read_parquet(out_path))
            continue

        df = _fetch_one_state(state, api_key)
        if df is not None and len(df) > 0:
            df.to_parquet(out_path, index=False)
            frames.append(df)
        time.sleep(0.1)

    if not frames:
        raise RuntimeError("No PUMS data fetched. Check your API key and internet connection.")

    combined = pd.concat(frames, ignore_index=True)

    # Only overwrite the shared combined file when every requested state
    # actually has data, mirroring fetch_qwi.py's guard against a partial
    # test run silently clobbering the full-coverage file.
    if set(states) == set(STATE_FIPS):
        out_path = output_dir / "pums_all.parquet"
    else:
        state_tag = "-".join(sorted(states))
        out_path = output_dir / f"pums_subset_{state_tag}.parquet"
        print(f"\nPartial run ({len(states)} state(s)): NOT overwriting the "
              f"combined pums_all.parquet. Saving to {out_path} instead.")

    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined):,} person records to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch ACS 1-year PUMS person records from the Census API")
    parser.add_argument("--state", nargs="+", default=None, help="State FIPS code(s), e.g. 06 48")
    args = parser.parse_args()

    out = fetch_pums(states=args.state)
    print(f"\nOutput: {out}")
