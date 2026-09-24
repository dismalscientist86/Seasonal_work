"""
fetch_qcew.py -- Download QCEW (Quarterly Census of Employment and Wages)
data: state x 6-digit NAICS x quarter, with true monthly employment levels.

Unlike QWI (a Census API, one HTTP call per state x industry x year range),
QCEW publishes one bulk "singlefile" CSV per year covering every area,
industry, and ownership sector for all 4 quarters at once:

    https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip

Each year's file is large (~300MB compressed; the uncompressed CSV is
mostly county-level rows we don't need), so this fetcher streams the zip
member directly out of memory in chunks, keeping only:
  - agglvl_code in {"18", "58"}: national or statewide, NAICS 6-digit,
    by ownership sector (see bls.gov/cew/classifications/aggregation)
  - own_code == "5": Private ownership -- QCEW's headline, most-complete-
    coverage ownership sector, and the standard choice for this kind of
    analysis (excludes federal/state/local government employment)

No API key is required. Only two required fields matter for our purposes
beyond identification: month1_emplvl, month2_emplvl, month3_emplvl -- the
true monthly employment counts within each quarter. QCEW has no
separations/hires field at all, so it cannot replicate CP's flow-based
excess recurrence measure -- see monthly_seasonal_index.py, which builds a
monthly analog of our existing *stock* measure (seasonal_index_emp)
instead.

Usage:
    python code/qcew/fetch_qcew.py                  # all QCEW_YEARS
    python code/qcew/fetch_qcew.py --years 2022 2023 # specific years
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QCEW_RAW, QCEW_YEARS, STATE_FIPS

BASE_URL = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"

KEEP_AGGLVL = {"18", "58"}   # national / statewide, NAICS6, by ownership
KEEP_OWN = "5"               # Private

USE_COLS = [
    "area_fips", "own_code", "industry_code", "agglvl_code",
    "year", "qtr", "disclosure_code", "qtrly_estabs",
    "month1_emplvl", "month2_emplvl", "month3_emplvl",
    "total_qtrly_wages", "avg_wkly_wage",
]

VALID_AREA_FIPS = {f"{fips}000" for fips in STATE_FIPS} | {"US000"}


def fetch_year(year: int, out_dir: Path, max_retries: int = 3, retry_delay: float = 5.0) -> Path | None:
    out_path = out_dir / f"qcew_state_naics6_{year}.parquet"
    if out_path.exists():
        print(f"{year}: already fetched, skipping ({out_path})")
        return out_path

    url = BASE_URL.format(year=year)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"{year}: downloading {url} (attempt {attempt}/{max_retries})...")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  [warn] {e}")
            if attempt == max_retries:
                print(f"{year}: FAILED after {max_retries} attempts, skipping")
                return None
            time.sleep(retry_delay)

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        print(f"{year}: no CSV found inside zip, skipping")
        return None
    csv_name = csv_names[0]

    print(f"{year}: filtering {csv_name} ({r.headers.get('Content-Length', '?')} bytes compressed)...")
    chunks = []
    with zf.open(csv_name) as f:
        for chunk in pd.read_csv(f, usecols=USE_COLS, dtype=str, chunksize=500_000, low_memory=False):
            filt = chunk[
                chunk["agglvl_code"].isin(KEEP_AGGLVL)
                & (chunk["own_code"] == KEEP_OWN)
                & chunk["area_fips"].isin(VALID_AREA_FIPS)
            ]
            if len(filt):
                chunks.append(filt)

    if not chunks:
        print(f"{year}: no matching rows found -- check agglvl/own_code filters")
        return None

    df = pd.concat(chunks, ignore_index=True)
    for c in ["qtrly_estabs", "month1_emplvl", "month2_emplvl", "month3_emplvl",
              "total_qtrly_wages", "avg_wkly_wage", "year", "qtr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{year}: saved {len(df):,} rows to {out_path}")
    return out_path


def main(years: list[int]) -> None:
    QCEW_RAW.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for year in years:
        result = fetch_year(year, QCEW_RAW)
        (done if result else failed).append(year)

    print(f"\nDone. {len(done)}/{len(years)} years fetched.")
    if failed:
        print(f"Failed years (retry these): {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch QCEW state x NAICS6 monthly employment data")
    parser.add_argument("--years", type=int, nargs="+", default=QCEW_YEARS,
                         help="Years to fetch (default: all QCEW_YEARS)")
    args = parser.parse_args()
    main(args.years)
