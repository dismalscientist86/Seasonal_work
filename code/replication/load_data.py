"""
load_data.py — Extract the replication zip and load pre-cleaned Stata datasets.

The replication package (238636-V1.zip) contains pre-cleaned .dta.gz files
so we can skip the full data-build pipeline and jump straight to analysis.

Usage:
    python code/replication/load_data.py   # extracts zip to data/replication/
    from code.replication.load_data import load_cps_separators, load_sipp_separators
"""

import sys
import gzip
import zipfile
import tempfile
from pathlib import Path
import pandas as pd
import pyreadstat
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from code.config import REPKG_ZIP, REPKG_DIR, DATA_CLEAN


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_replication_zip(force: bool = False) -> Path:
    """
    Extract 238636-V1.zip into data/replication/.

    Skips extraction if the destination already exists, unless force=True.
    The zip is ~1.8 GB; extraction produces ~2+ GB of data files.

    Returns the path to the extracted 'replication' directory.
    """
    dest = REPKG_DIR
    flag = dest / "replication" / ".extracted"

    if flag.exists() and not force:
        print(f"Already extracted at {dest}. Pass force=True to re-extract.")
        return dest / "replication"

    if not REPKG_ZIP.exists():
        raise FileNotFoundError(
            f"Replication zip not found at {REPKG_ZIP}.\n"
            "Make sure you have the file 238636-V1.zip in replication_package/."
        )

    print(f"Extracting {REPKG_ZIP} → {dest} (this may take a few minutes) …")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(REPKG_ZIP, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, dest)

    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    print("Extraction complete.")
    return dest / "replication"


# ── Generic loader ────────────────────────────────────────────────────────────

def load_dta_gz(
    path: Path,
    usecols: list[str] | None = None,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    Load a gzip-compressed Stata .dta file.

    The .dta.gz files were saved with Stata's gzsave command.  We decompress
    to a temp file, then read with pyreadstat (which handles Stata value labels
    and date formats better than pandas.read_stata).

    Parameters
    ----------
    path : Path to the .dta.gz file.
    usecols : If given, only load these columns (saves memory).
    filters : Dict of {column: value} filters applied after loading.
              Useful for large files (e.g., {'earn_sample': 1}).

    Returns
    -------
    pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run extract_replication_zip() first."
        )

    print(f"Loading {path.name} …", end=" ", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Decompress
        with gzip.open(path, "rb") as gz_in, open(tmp_path, "wb") as dta_out:
            while chunk := gz_in.read(1 << 20):  # 1 MB chunks
                dta_out.write(chunk)

        # Read Stata file
        if usecols:
            df, meta = pyreadstat.read_dta(
                str(tmp_path),
                usecols=usecols,
                apply_value_formats=False,
            )
        else:
            df, meta = pyreadstat.read_dta(
                str(tmp_path),
                apply_value_formats=False,
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Convert Stata date columns (monthly dates stored as integers)
    # Stata's monthly date: tm = months since Jan 1960
    if "tm" in df.columns:
        df["tm"] = pd.to_datetime("1960-01-01") + pd.to_timedelta(
            df["tm"].astype(int), unit="MS"
        )
    if "tm0" in df.columns:
        df["tm0"] = pd.to_datetime("1960-01-01") + pd.to_timedelta(
            df["tm0"].astype(int), unit="MS"
        )

    # Apply row filters
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]

    print(f"{len(df):,} rows × {df.shape[1]} cols")
    return df.reset_index(drop=True)


def load_dta(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load an uncompressed .dta file (e.g., gbt_predictions_sipp.dta)."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    print(f"Loading {path.name} …", end=" ", flush=True)
    df, _ = pyreadstat.read_dta(str(path), usecols=usecols, apply_value_formats=False)
    # Stata date conversion
    for col in ("tm", "tm0"):
        if col in df.columns:
            df[col] = pd.to_datetime("1960-01-01") + pd.to_timedelta(
                df[col].astype(int), unit="MS"
            )
    print(f"{len(df):,} rows × {df.shape[1]} cols")
    return df.reset_index(drop=True)


# ── Dataset-specific loaders ─────────────────────────────────────────────────

def load_cps_separators(
    horizons: list[int] | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the CPS separators panel (19 MB compressed).

    Each row is a (separation event × event-time horizon) observation.
    Key columns:
        rid         : replicant ID (unique per focal separation)
        k           : event time in months (relative to focal separation)
        tm          : calendar date
        pid         : person ID
        wtfnl       : survey weight
        sep         : 1 if worker separates at horizon k, else 0
        weeks_elapsed: weeks since last CPS reference week (length-of-interval control)
        base_ind    : 1-digit industry at separation (1–15)
        base_occ    : 14-category occupation at separation
        base_female, base_age, base_educ, base_wbho : demographics
        base_type   : 1 = E→U, 2 = E→N
        base_whyunemp : reason for unemployment
        month       : calendar month (1–12) at event time
    """
    path = DATA_CLEAN / "cps_separators.dta.gz"

    default_cols = [
        "rid", "k", "tm", "pid", "wtfnl", "sep", "weeks_elapsed",
        "base_ind", "base_occ", "base_female", "base_age", "base_educ",
        "base_wbho", "base_type", "base_whyunemp", "base_temp",
        "month", "emp", "ump", "nlf",
    ]
    cols = usecols or default_cols

    df = load_dta_gz(path, usecols=cols)

    # Cast key columns to correct types
    df["k"]   = df["k"].astype(int)
    df["sep"] = df["sep"].astype(float)
    df["rid"] = df["rid"].astype(int)
    df["pid"] = df["pid"].astype(int)

    if horizons is not None:
        df = df[df["k"].isin(horizons)].copy()

    return df


def load_sipp_separators(
    earn_sample_only: bool = True,
    horizons: list[int] | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the SIPP separators panel (97 MB compressed).

    Compared to the CPS, this dataset has:
        earn_sample : 1 if personal earnings ≥ $450/month at baseline
        pred_decile : decile of GBT-predicted excess recurrence (1–10)
        pred_excess : predicted excess recurrence (continuous)
        base_pp_ern : baseline personal earnings
        base_hh_inc : baseline household income
        srefmon     : 1 if this obs is a SIPP reference month (seam indicator)
        panel       : SIPP panel year
        five        : 1 if this is a 5-week calendar month
        base_ind    : 1-digit industry (uses sipp-based coding)
    """
    path = DATA_CLEAN / "sipp_separators.dta.gz"

    default_cols = [
        "rid", "k", "tm", "pid", "wtfnl", "sep", "srefmon",
        "base_ind", "base_occ", "base_female", "base_age", "base_educ",
        "base_wbho", "base_type", "base_pp_ern", "base_hh_inc",
        "earn_sample", "pred_decile", "pred_excess",
        "panel", "five", "month",
    ]
    cols = usecols or default_cols

    filters = {"earn_sample": 1} if earn_sample_only else None
    df = load_dta_gz(path, usecols=cols, filters=filters)

    df["k"]   = df["k"].astype(int)
    df["sep"] = df["sep"].astype(float)
    df["rid"] = df["rid"].astype(int)
    df["pid"] = df["pid"].astype(int)

    # Compute SIPP panel-structure controls (from recurrence.do)
    # first_seam: obs is a reference-month seam within the first 4 months post-sep
    # later_seam: obs is a reference-month seam after month 4
    if "srefmon" in df.columns:
        df["first_seam"] = (df["srefmon"] == 1) & (df["k"] <= 4)
        df["later_seam"] = (df["srefmon"] == 1) & (df["k"] >= 5)
        df["first_seam"] = df["first_seam"].astype(float)
        df["later_seam"] = df["later_seam"].astype(float)

    if horizons is not None:
        df = df[df["k"].isin(horizons)].copy()

    return df


def load_gbt_predictions() -> pd.DataFrame:
    """
    Load pre-computed GBT predictions for SIPP separators (5.6 MB, uncompressed).

    Columns: pid, tm, sep_11, sep_12, sep_13, pred_excess
    These were generated by the original Stata/LightGBM pipeline.
    """
    path = DATA_CLEAN / "gbt_predictions" / "v1" / "gbt_predictions_sipp.dta"
    return load_dta(path)


def load_cps_fullsample(usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Load the full CPS sample (379 MB compressed → ~1.5 GB in memory).
    Only load if you need to retrain the GBT classifier.
    """
    path = DATA_CLEAN / "cps_fullsample.dta.gz"
    return load_dta_gz(path, usecols=usecols)


def load_sipp_fullsample(usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the full SIPP sample (392 MB compressed)."""
    path = DATA_CLEAN / "sipp_fullsample.dta.gz"
    return load_dta_gz(path, usecols=usecols)


# ── Entry point (extract only) ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract replication zip")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if destination exists")
    args = parser.parse_args()

    out = extract_replication_zip(force=args.force)
    print(f"\nData ready at: {out}")
    print("\nKey files:")
    for f in sorted((out / "data" / "clean").glob("*.dta*")):
        print(f"  {f.name:50s}  {f.stat().st_size / 1e6:6.0f} MB")
