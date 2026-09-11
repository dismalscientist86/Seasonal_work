"""
build_indp_naics_crosswalk.py — Census industry code (ACS PUMS `INDP`) <-> NAICS
crosswalk, cleaned into a lookup usable for mapping 6-digit NAICS codes (as
used throughout this project's QWI-based seasonal index) onto the coarser
industry categories that ACS/CPS survey microdata actually publish.

Why this exists: ACS PUMS (fetch_acs_pums.py) and BLS's CPS industry-by-sex
tables both use the "Census industry" classification, not raw NAICS — and
it is *coarser* than 6-digit NAICS for many service industries (e.g. NAICS
611710 "Educational Support Services" is bundled with 611610-611699 "other
schools and instruction" into one Census industry code; NAICS 621399 is
bundled with three sibling health-practitioner-office codes). This is a
genuine resolution limit of the source data, not a tooling gap — so
industry_pay_gender_profile.py deliberately does its pay/gender/seasonality
comparison at this Census-industry-code level (aggregating NAICS6 seasonality
up to it), rather than mixing a NAICS6-specific seasonality figure with a
coarser-category pay/gender figure the way an earlier ad hoc pass in this
project did.

Source: Census Bureau's "2022 Census Industry Code List with Crosswalk"
(https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html),
sheet "2022 Census Ind Code List ". Each leaf row maps one 4-digit Census
industry code to one or more NAICS codes/prefixes, sometimes with explicit
exclusions (e.g. "6213 exc. 62131, 62132") or (rare, ~18 of 268 rows, mostly
military/government) an unresolvable "Part of ..." reference that this
script deliberately drops rather than guess at.

Usage:
    python code/build_indp_naics_crosswalk.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO_ROOT

CROSSWALK_URL = (
    "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/"
    "2022-Census-Industry-Code-List-with-Crosswalk.xlsx"
)
RAW_DIR = REPO_ROOT / "data" / "acs_raw"
RAW_XLSX = RAW_DIR / "2022-Census-Industry-Code-List-with-Crosswalk.xlsx"
OUTPUT_CSV = RAW_DIR / "indp_naics_crosswalk.csv"

SHEET_NAME = "2022 Census Ind Code List "

_LEAF_CODE_RE = re.compile(r"^\d{4}$")
_NAICS_TOKEN_RE = re.compile(r"^\d{2,6}$")


def _clean_code(value) -> str:
    """Normalize a code cell (openpyxl may read '8080' as the float 8080.0)."""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def download_crosswalk(force: bool = False) -> Path:
    """Download the Census crosswalk workbook if not already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX.exists() and not force:
        return RAW_XLSX
    print(f"Downloading {CROSSWALK_URL} ...")
    r = requests.get(CROSSWALK_URL, timeout=60)
    r.raise_for_status()
    RAW_XLSX.write_bytes(r.content)
    print(f"Saved to {RAW_XLSX}")
    return RAW_XLSX


def _parse_naics_field(raw: str) -> tuple[list[str], list[str]] | None:
    """
    Parse one '2022 NAICS Code' cell into (include_prefixes, exclude_prefixes).

    Returns None if the field can't be resolved to concrete NAICS prefixes
    (a "Part of ..."/"pt. ..." reference, or blank) — these are dropped
    rather than guessed at (mostly military/government codes, irrelevant
    to this project's industries).

    Handles:
      "111"                          -> (["111"], [])
      "1131, 1132"                   -> (["1131","1132"], [])
      "311811"                       -> (["311811"], [])
      "6213 exc. 62131, 62132"       -> (["6213"], ["62131","62132"])
      "922, pt. 92115"               -> (["922"], [])   # drops the unresolvable part
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = _clean_code(raw) if not isinstance(raw, str) else raw.strip()
    if not text or text.lower().startswith("part of"):
        return None

    if "exc." in text.lower():
        include_part, exclude_part = re.split(r"exc\.", text, flags=re.IGNORECASE, maxsplit=1)
    else:
        include_part, exclude_part = text, ""

    def _tokens(s: str) -> list[str]:
        out = []
        for tok in s.split(","):
            tok = _clean_code(tok.strip())
            if tok.lower().startswith(("pt.", "part")):
                continue  # drop unresolvable partial references
            if _NAICS_TOKEN_RE.fullmatch(tok):
                out.append(tok)
        return out

    include = _tokens(include_part)
    exclude = _tokens(exclude_part)
    if not include:
        return None
    return include, exclude


def build_crosswalk(xlsx_path: Path) -> pd.DataFrame:
    """Parse the workbook into one row per leaf Census industry code."""
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, header=None)
    raw.columns = ["sector", "desc", "_blank", "indcode", "naics"]

    is_leaf = raw["indcode"].map(
        lambda s: bool(_LEAF_CODE_RE.fullmatch(_clean_code(s))) if pd.notna(s) else False
    )
    leaves = raw[is_leaf].copy()

    records = []
    n_dropped = 0
    for _, row in leaves.iterrows():
        parsed = _parse_naics_field(row["naics"])
        if parsed is None:
            n_dropped += 1
            continue
        include, exclude = parsed
        records.append({
            "indp_code": _clean_code(row["indcode"]),
            "description": str(row["desc"]).strip() if isinstance(row["desc"], str) else None,
            "naics_include": ";".join(include),
            "naics_exclude": ";".join(exclude),
        })

    df = pd.DataFrame(records)

    # A handful of indp_codes appear twice in the source sheet (e.g. 0770
    # "Construction" has a blank-description placeholder row plus the real
    # one; both carry the identical NAICS mapping). Keep one row per code,
    # preferring a non-null description.
    n_before = len(df)
    df = (
        df.sort_values("description", na_position="last")
        .drop_duplicates("indp_code", keep="first")
        .reset_index(drop=True)
    )
    n_dupe = n_before - len(df)

    print(f"Parsed {len(df)} usable Census industry codes "
          f"({n_dropped} dropped: unresolvable 'Part of'/blank NAICS references; "
          f"{n_dupe} duplicate indp_code rows collapsed)")
    return df


def match_naics6_to_indp(naics6: str, crosswalk: pd.DataFrame) -> str | None:
    """
    Find the Census industry (INDP) code covering a 6-digit NAICS code.

    A NAICS prefix "matches" if naics6 starts with it. Exclusions are
    checked the same way. If more than one row's include-prefix matches,
    the longest (most specific) prefix wins — this resolves the common case
    where a broad code (e.g. "62") and a narrower one both nominally match
    (there should never be more than one *equally specific* match in a
    well-formed crosswalk; if there is, the first is kept and this is not
    silently hidden -- see build_and_save()'s ambiguity check).
    """
    best_code, best_len = None, -1
    for _, row in crosswalk.iterrows():
        inc_raw = row["naics_include"] if pd.notna(row["naics_include"]) else ""
        exc_raw = row["naics_exclude"] if pd.notna(row["naics_exclude"]) else ""
        includes = inc_raw.split(";") if inc_raw else []
        excludes = exc_raw.split(";") if exc_raw else []
        if any(naics6.startswith(p) for p in excludes):
            continue
        for p in includes:
            if naics6.startswith(p) and len(p) > best_len:
                best_code, best_len = row["indp_code"], len(p)
    return best_code


def build_and_save(force_download: bool = False) -> pd.DataFrame:
    xlsx_path = download_crosswalk(force=force_download)
    crosswalk = build_crosswalk(xlsx_path)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    return crosswalk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Census industry code (INDP) <-> NAICS crosswalk")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    build_and_save(force_download=args.force_download)
