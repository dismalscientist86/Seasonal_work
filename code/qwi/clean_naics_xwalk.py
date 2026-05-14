"""
clean_naics_xwalk.py - Clean Census 2022 NAICS structure to a NAICS6 lookup.

Input:
  data/naics_xwalk/2022_NAICS_Structure.xlsx

Output:
  data/naics_xwalk/naics6_2022_titles.csv

The output is merge-ready on `naics_code` and includes the 6-digit title plus
available hierarchy titles.

Usage:
    python code/qwi/clean_naics_xwalk.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPO_ROOT


DEFAULT_INPUT = REPO_ROOT / "data" / "naics_xwalk" / "2022_NAICS_Structure.xlsx"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "naics_xwalk" / "naics6_2022_titles.csv"

COMBINED_SECTOR_PREFIXES = {
    "31": "31-33",
    "32": "31-33",
    "33": "31-33",
    "44": "44-45",
    "45": "44-45",
    "48": "48-49",
    "49": "48-49",
}


def _clean_column_name(value: object) -> str:
    """Normalize Excel column names to snake case."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _clean_naics_code(value: object) -> str | None:
    """Normalize NAICS code values while preserving combined sectors like 31-33."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    text = text.replace(" ", "")

    if re.fullmatch(r"\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{2,6}", text):
        return text
    return None


def _clean_title(value: object) -> str | None:
    """Clean NAICS title text and remove visible Census footnote markers."""
    if pd.isna(value):
        return None
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    # The Census workbook stores some footnote markers as superscript rich text.
    # Common plain-text renderings are trailing "T" or "Tt" appended to titles.
    text = re.sub(r"(?:Tt|T)+$", "", text).strip()
    return text or None


def _find_column(columns: list[str], *needles: str) -> str:
    """Find the first column containing all requested substrings."""
    for col in columns:
        if all(needle in col for needle in needles):
            return col
    raise ValueError(f"Could not find column matching: {needles}")


def load_raw_structure(path: Path) -> pd.DataFrame:
    """Read and normalize the Census NAICS structure workbook."""
    raw = pd.read_excel(path, sheet_name=0, header=2, dtype=str)
    raw.columns = [_clean_column_name(c) for c in raw.columns]

    code_col = _find_column(list(raw.columns), "naics", "code")
    title_col = _find_column(list(raw.columns), "naics", "title")
    change_col = None
    try:
        change_col = _find_column(list(raw.columns), "change", "indicator")
    except ValueError:
        pass

    keep_cols = [code_col, title_col]
    if change_col:
        keep_cols.append(change_col)

    df = raw[keep_cols].rename(
        columns={
            code_col: "naics_code",
            title_col: "naics_title",
            **({change_col: "change_indicator"} if change_col else {}),
        }
    )
    df["naics_code"] = df["naics_code"].map(_clean_naics_code)
    df["naics_title"] = df["naics_title"].map(_clean_title)
    if "change_indicator" in df.columns:
        df["change_indicator"] = df["change_indicator"].fillna("").astype(str).str.strip()
    else:
        df["change_indicator"] = ""

    df = df.dropna(subset=["naics_code", "naics_title"]).drop_duplicates("naics_code")
    df["naics_level"] = df["naics_code"].str.replace("-", "", regex=False).str.len()
    return df


def build_naics6_lookup(structure: pd.DataFrame) -> pd.DataFrame:
    """Convert all-level NAICS structure rows to a 6-digit merge lookup."""
    title_by_code = structure.set_index("naics_code")["naics_title"].to_dict()
    change_by_code = structure.set_index("naics_code")["change_indicator"].to_dict()

    naics6 = structure[
        structure["naics_code"].str.fullmatch(r"\d{6}", na=False)
    ][["naics_code", "naics_title", "change_indicator"]].copy()

    naics6["sector_2d"] = naics6["naics_code"].str[:2]
    naics6["naics3"] = naics6["naics_code"].str[:3]
    naics6["naics4"] = naics6["naics_code"].str[:4]
    naics6["naics5"] = naics6["naics_code"].str[:5]
    naics6["naics_level"] = 6

    sector_key = naics6["sector_2d"].map(COMBINED_SECTOR_PREFIXES).fillna(naics6["sector_2d"])
    naics6["sector_title"] = sector_key.map(title_by_code)
    naics6["subsector_title"] = naics6["naics3"].map(title_by_code)
    naics6["industry_group_title"] = naics6["naics4"].map(title_by_code)
    naics6["naics_industry_title"] = naics6["naics5"].map(title_by_code)
    naics6 = naics6.rename(columns={"naics_title": "national_industry_title"})

    cols = [
        "naics_code", "national_industry_title", "change_indicator",
        "sector_2d", "sector_title", "naics3", "subsector_title",
        "naics4", "industry_group_title", "naics5", "naics_industry_title",
        "naics_level",
    ]
    return naics6[cols].sort_values("naics_code").reset_index(drop=True)


def clean_naics_xwalk(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Clean the raw workbook and write a NAICS6 title lookup CSV."""
    structure = load_raw_structure(input_path)
    lookup = build_naics6_lookup(structure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(output_path, index=False)

    print(f"Read {len(structure):,} NAICS structure rows from {input_path}")
    print(f"Wrote {len(lookup):,} NAICS6 rows to {output_path}")
    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Census 2022 NAICS structure workbook.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_naics_xwalk(input_path=args.input, output_path=args.output)
