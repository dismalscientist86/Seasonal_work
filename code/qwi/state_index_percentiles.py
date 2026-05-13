"""
state_index_percentiles.py - Analyze top/bottom state-specific NAICS seasonality.

Reads the reusable state x NAICS6 seasonal index from geographic_analysis.py and
identifies the most and least seasonal industries within each state.

Default outputs use the top and bottom 1 percent within each state and require
all four quarterly excess estimates to be non-missing. The complete-quarter
restriction avoids treating thin cells with mechanically zero amplitude as
"least seasonal."

The script also reads the national NAICS6 seasonal index and flags state-tail
industries that are also in the national top or bottom tail.

Outputs are written to output/tables:
  state_top1pct_naics6.csv
  state_bottom1pct_naics6.csv
  top1pct_naics_frequency.csv
  bottom1pct_naics_frequency.csv
  top1pct_sector_frequency.csv
  bottom1pct_sector_frequency.csv
  state_percentile_overlap_summary.csv
  state_percentile_pairwise_jaccard.csv

Usage:
    python code/qwi/state_index_percentiles.py
    python code/qwi/state_index_percentiles.py --percentile 0.05
    python code/qwi/state_index_percentiles.py --no-require-complete-quarters
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_CLEAN, TABLES_DIR


TAIL_LABELS = {
    "top": "top",
    "bottom": "bottom",
}


def percentile_to_label(percentile: float) -> str:
    """Convert a percentile share to a filename-safe label."""
    pct = percentile * 100
    if pct.is_integer():
        return f"{int(pct)}pct"
    return f"{pct:g}pct".replace(".", "p")


def load_state_index(path: Path, require_complete_quarters: bool = True) -> pd.DataFrame:
    """Load the state x NAICS6 index and optionally require all excessQ* fields."""
    df = pd.read_csv(path, dtype={"state": str, "naics_code": str, "sector_2d": str})

    numeric_cols = [
        "seasonal_index", "peak_excess", "seasonal_amplitude",
        "excessQ1", "excessQ2", "excessQ3", "excessQ4",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if require_complete_quarters:
        df = df.dropna(subset=["excessQ1", "excessQ2", "excessQ3", "excessQ4"])

    return df.dropna(subset=["state_name", "naics_code", "seasonal_index"])


def load_national_flags(
    path: Path,
    percentile: float,
    require_complete_quarters: bool = True,
) -> pd.DataFrame:
    """Load national index and flag NAICS codes in national top/bottom tails."""
    df = pd.read_csv(path, dtype={"naics_code": str, "sector_2d": str})

    numeric_cols = [
        "seasonal_index", "peak_excess", "seasonal_amplitude",
        "excessQ1", "excessQ2", "excessQ3", "excessQ4",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if require_complete_quarters:
        df = df.dropna(subset=["excessQ1", "excessQ2", "excessQ3", "excessQ4"])

    df = df.dropna(subset=["naics_code", "seasonal_index"]).copy()
    k = max(1, int(np.ceil(len(df) * percentile)))

    top_codes = set(
        df.sort_values(
            ["seasonal_index", "peak_excess", "naics_code"],
            ascending=[False, False, True],
        )
        .head(k)["naics_code"]
    )
    bottom_codes = set(
        df.sort_values(
            ["seasonal_index", "peak_excess", "naics_code"],
            ascending=[True, True, True],
        )
        .head(k)["naics_code"]
    )

    flags = pd.DataFrame({"naics_code": sorted(set(df["naics_code"]))})
    flags["in_national_top_percentile"] = flags["naics_code"].isin(top_codes)
    flags["in_national_bottom_percentile"] = flags["naics_code"].isin(bottom_codes)
    return flags


def add_national_flags(df: pd.DataFrame, national_flags: pd.DataFrame) -> pd.DataFrame:
    """Attach national top/bottom percentile flags to a state-index DataFrame."""
    out = df.merge(national_flags, on="naics_code", how="left")
    for col in ["in_national_top_percentile", "in_national_bottom_percentile"]:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def select_state_tail(
    df: pd.DataFrame,
    percentile: float,
    tail: str,
) -> pd.DataFrame:
    """Select top or bottom percentile of industries within each state."""
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1")
    if tail not in TAIL_LABELS:
        raise ValueError(f"tail must be one of {sorted(TAIL_LABELS)}")

    selected = []
    ascending = tail == "bottom"

    for _, grp in df.groupby("state_name", sort=True):
        n = len(grp)
        k = max(1, int(np.ceil(n * percentile)))
        ranked = grp.sort_values(
            ["seasonal_index", "peak_excess", "naics_code"],
            ascending=[ascending, ascending, True],
        ).head(k)
        ranked = ranked.copy()
        ranked["tail"] = tail
        ranked["rank_in_state"] = range(1, len(ranked) + 1)
        ranked["state_industries_ranked"] = n
        ranked["state_industries_selected"] = k
        selected.append(ranked)

    return pd.concat(selected, ignore_index=True)


def frequency_by_naics(selected: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """Count how often each NAICS appears in a selected state tail."""
    freq = (
        selected
        .groupby(["naics_code", "sector_2d", "sector_label"], dropna=False)
        .agg(
            n_states=("state_name", "nunique"),
            n_state_industry_slots=("state_name", "size"),
            mean_seasonal_index=("seasonal_index", "mean"),
            mean_peak_excess=("peak_excess", "mean"),
            median_peak_excess=("peak_excess", "median"),
            in_national_top_percentile=("in_national_top_percentile", "max"),
            in_national_bottom_percentile=("in_national_bottom_percentile", "max"),
        )
        .reset_index()
    )
    freq["share_states"] = freq["n_states"] / n_states
    return freq.sort_values(
        ["n_states", "mean_seasonal_index", "mean_peak_excess", "naics_code"],
        ascending=[False, False, False, True],
    )


def frequency_by_sector(selected: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """Count selected state-tail industries by broad sector."""
    freq = (
        selected
        .groupby(["sector_2d", "sector_label"], dropna=False)
        .agg(
            n_state_industry_slots=("state_name", "size"),
            n_states=("state_name", "nunique"),
            n_unique_naics=("naics_code", "nunique"),
            mean_seasonal_index=("seasonal_index", "mean"),
            mean_peak_excess=("peak_excess", "mean"),
            n_national_top_percentile=("in_national_top_percentile", "sum"),
            n_national_bottom_percentile=("in_national_bottom_percentile", "sum"),
        )
        .reset_index()
    )
    freq["share_states"] = freq["n_states"] / n_states
    return freq.sort_values(
        ["n_state_industry_slots", "n_states", "sector_2d"],
        ascending=[False, False, True],
    )


def pairwise_jaccard(selected: pd.DataFrame, tail: str) -> pd.DataFrame:
    """Compute pairwise Jaccard overlap of selected NAICS sets across states."""
    state_sets = {
        state: set(grp["naics_code"])
        for state, grp in selected.groupby("state_name")
    }

    records = []
    for state_a, state_b in itertools.combinations(sorted(state_sets), 2):
        set_a = state_sets[state_a]
        set_b = state_sets[state_b]
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        records.append({
            "tail": tail,
            "state_a": state_a,
            "state_b": state_b,
            "n_a": len(set_a),
            "n_b": len(set_b),
            "n_intersection": intersection,
            "n_union": union,
            "jaccard": intersection / union if union else np.nan,
        })

    return pd.DataFrame(records)


def overlap_summary(
    selected: pd.DataFrame,
    naics_freq: pd.DataFrame,
    jaccard: pd.DataFrame,
    tail: str,
) -> dict:
    """Summarize state-tail overlap in one row."""
    state_counts = selected.groupby("state_name")["naics_code"].nunique()
    return {
        "tail": tail,
        "n_states": selected["state_name"].nunique(),
        "n_state_industry_slots": len(selected),
        "mean_industries_per_state": state_counts.mean(),
        "min_industries_per_state": state_counts.min(),
        "max_industries_per_state": state_counts.max(),
        "n_unique_naics": selected["naics_code"].nunique(),
        "n_naics_appearing_once": int((naics_freq["n_states"] == 1).sum()),
        "n_naics_appearing_2plus": int((naics_freq["n_states"] >= 2).sum()),
        "n_naics_appearing_5plus": int((naics_freq["n_states"] >= 5).sum()),
        "n_naics_appearing_10plus": int((naics_freq["n_states"] >= 10).sum()),
        "top10_naics_slot_share": naics_freq.head(10)["n_state_industry_slots"].sum() / len(selected),
        "n_slots_in_national_top_percentile": int(selected["in_national_top_percentile"].sum()),
        "share_slots_in_national_top_percentile": selected["in_national_top_percentile"].mean(),
        "n_slots_in_national_bottom_percentile": int(selected["in_national_bottom_percentile"].sum()),
        "share_slots_in_national_bottom_percentile": selected["in_national_bottom_percentile"].mean(),
        "mean_pairwise_jaccard": jaccard["jaccard"].mean(),
        "min_pairwise_jaccard": jaccard["jaccard"].min(),
        "max_pairwise_jaccard": jaccard["jaccard"].max(),
    }


def write_tail_outputs(
    selected: pd.DataFrame,
    tail: str,
    percentile_label: str,
    n_states: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write selected rows and frequency tables for one tail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    naics_freq = frequency_by_naics(selected, n_states)
    sector_freq = frequency_by_sector(selected, n_states)
    jaccard = pairwise_jaccard(selected, tail)

    selected.to_csv(out_dir / f"state_{tail}{percentile_label}_naics6.csv", index=False)
    naics_freq.to_csv(out_dir / f"{tail}{percentile_label}_naics_frequency.csv", index=False)
    sector_freq.to_csv(out_dir / f"{tail}{percentile_label}_sector_frequency.csv", index=False)

    return naics_freq, sector_freq, jaccard


def run_analysis(
    input_path: Path,
    national_input_path: Path,
    percentile: float = 0.01,
    require_complete_quarters: bool = True,
    out_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Run top and bottom percentile overlap analysis and write CSV outputs."""
    df = load_state_index(input_path, require_complete_quarters=require_complete_quarters)
    national_flags = load_national_flags(
        national_input_path,
        percentile=percentile,
        require_complete_quarters=require_complete_quarters,
    )
    df = add_national_flags(df, national_flags)
    n_states = df["state_name"].nunique()
    percentile_label = percentile_to_label(percentile)

    summaries = []
    all_jaccard = []

    for tail in ["top", "bottom"]:
        selected = select_state_tail(df, percentile=percentile, tail=tail)
        naics_freq, _, jaccard = write_tail_outputs(
            selected=selected,
            tail=tail,
            percentile_label=percentile_label,
            n_states=n_states,
            out_dir=out_dir,
        )
        summaries.append(overlap_summary(selected, naics_freq, jaccard, tail))
        all_jaccard.append(jaccard)

    summary = pd.DataFrame(summaries)
    summary["percentile"] = percentile
    summary["require_complete_quarters"] = require_complete_quarters
    summary.to_csv(out_dir / "state_percentile_overlap_summary.csv", index=False)
    pd.concat(all_jaccard, ignore_index=True).to_csv(
        out_dir / "state_percentile_pairwise_jaccard.csv",
        index=False,
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze top/bottom state-specific NAICS6 seasonal-index percentiles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=QWI_CLEAN / "seasonal_index_naics6_by_state.csv",
        help="Path to state x NAICS6 seasonal index CSV.",
    )
    parser.add_argument(
        "--national-input",
        type=Path,
        default=QWI_CLEAN / "seasonal_index_naics6.csv",
        help="Path to national NAICS6 seasonal index CSV.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.01,
        help="Within-state tail share to select. Default: 0.01 for top/bottom 1 percent.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory for output tables.",
    )
    parser.add_argument(
        "--no-require-complete-quarters",
        action="store_true",
        help="Allow cells with missing quarter-specific excess estimates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_analysis(
        input_path=args.input,
        national_input_path=args.national_input,
        percentile=args.percentile,
        require_complete_quarters=not args.no_require_complete_quarters,
        out_dir=args.out_dir,
    )
    print(result.to_string(index=False))
    print(f"\nTables saved to {args.out_dir}")
