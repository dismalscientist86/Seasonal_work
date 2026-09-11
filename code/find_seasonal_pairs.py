"""
find_seasonal_pairs.py — Screen for industry pairs with similar realized pay,
very different seasonality, and a female-dominated workforce; report the
rent burden implied at each industry's typical pay.

Operates on output/tables/industry_pay_gender_seasonality_profile.csv
(from industry_pay_gender_profile.py), which already resolves the two
issues an earlier ad hoc pass at this had: pay is ACS PUMS' realized
annual wage/salary income (`WAGP`, what a person actually received in the
past 12 months, including anyone who didn't work all four quarters) rather
than a QWI pay-rate annualized by x12, and gender/pay/seasonality are all
computed at the same (Census-industry-code) resolution rather than mixing
a NAICS6-specific seasonality figure with a coarser-category gender share.

Usage:
    python code/find_seasonal_pairs.py
    python code/find_seasonal_pairs.py --min-female-share 0.6 --max-pay-gap-pct 0.10
    python code/find_seasonal_pairs.py --monthly-rent 1600
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TABLES_DIR

# Census ACS 2020-2024 5-year national median gross rent (all renter-occupied
# units) -- see https://www.census.gov/quickfacts. This is a blended 5-year
# estimate, so it trails current asking rents somewhat; a rent-burden
# conclusion at this benchmark is if anything conservative.
DEFAULT_MONTHLY_RENT = 1413.0


def load_profile(path: Path | None = None) -> pd.DataFrame:
    path = path or (TABLES_DIR / "industry_pay_gender_seasonality_profile.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python code/industry_pay_gender_profile.py"
        )
    return pd.read_csv(path, dtype={"indp_code": str})


def screen_candidates(
    profile: pd.DataFrame,
    min_female_share: float = 0.55,
    min_pums_records: int = 200,
    min_seasonal_gap: float = 0.25,
) -> pd.DataFrame:
    """Restrict to female-dominated, adequately-sampled, seasonality-scored industries."""
    df = profile.dropna(subset=["indp_seasonal_index", "median_annual_wage", "pct_female"]).copy()
    df = df[
        (df["pct_female"] >= min_female_share)
        & (df["n_pums_records"] >= min_pums_records)
    ]
    return df


def find_pairs(
    candidates: pd.DataFrame,
    max_pay_gap_pct: float = 0.15,
    min_seasonal_gap: float = 0.25,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    All pairs among `candidates` with similar realized pay and a large
    seasonality gap, ranked by seasonality gap (highest first) among those
    passing the pay-similarity filter.
    """
    rows = candidates.to_dict("records")
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        wage_a, wage_b = a["median_annual_wage"], b["median_annual_wage"]
        if wage_a <= 0 or wage_b <= 0:
            continue
        pay_gap_pct = abs(wage_a - wage_b) / ((wage_a + wage_b) / 2)
        seasonal_gap = abs(a["indp_seasonal_index"] - b["indp_seasonal_index"])
        if pay_gap_pct > max_pay_gap_pct or seasonal_gap < min_seasonal_gap:
            continue

        seasonal_row, flat_row = (a, b) if a["indp_seasonal_index"] >= b["indp_seasonal_index"] else (b, a)
        pairs.append({
            "seasonal_industry": seasonal_row["description"],
            "seasonal_index": seasonal_row["indp_seasonal_index"],
            "seasonal_peak_quarter": seasonal_row.get("indp_peak_quarter"),
            "seasonal_pay": seasonal_row["median_annual_wage"],
            "seasonal_pct_female": seasonal_row["pct_female"],
            "seasonal_mean_weeks_worked": seasonal_row.get("mean_weeks_worked"),
            "flat_industry": flat_row["description"],
            "flat_index": flat_row["indp_seasonal_index"],
            "flat_pay": flat_row["median_annual_wage"],
            "flat_pct_female": flat_row["pct_female"],
            "flat_mean_weeks_worked": flat_row.get("mean_weeks_worked"),
            "pay_gap_pct": pay_gap_pct,
            "seasonal_gap": seasonal_gap,
        })

    out = pd.DataFrame(pairs)
    if len(out) == 0:
        return out
    return out.sort_values(["seasonal_gap", "pay_gap_pct"], ascending=[False, True]).head(top_n)


def add_rent_burden(pairs: pd.DataFrame, monthly_rent: float) -> pd.DataFrame:
    """Attach rent-burden columns (rent / annual pay, monthly) for both sides of each pair."""
    annual_rent = monthly_rent * 12
    pairs = pairs.copy()
    pairs["seasonal_rent_burden_pct"] = annual_rent / pairs["seasonal_pay"] * 100
    pairs["flat_rent_burden_pct"] = annual_rent / pairs["flat_pay"] * 100
    pairs["monthly_rent_used"] = monthly_rent
    return pairs


def print_pairs(pairs: pd.DataFrame) -> None:
    if len(pairs) == 0:
        print("No candidate pairs found at the current thresholds -- try "
              "loosening --max-pay-gap-pct, --min-seasonal-gap, or --min-female-share.")
        return
    for _, r in pairs.iterrows():
        print(f"\n{'='*90}")
        print(f"SEASONAL: {r['seasonal_industry']}  "
              f"(seasonal_index={r['seasonal_index']:.2f}, peak Q{r['seasonal_peak_quarter']})")
        print(f"  realized annual pay ${r['seasonal_pay']:,.0f} | {r['seasonal_pct_female']:.0%} female "
              f"| mean weeks worked/yr: {r['seasonal_mean_weeks_worked']:.1f} "
              f"| rent burden at ${r['monthly_rent_used']:,.0f}/mo: {r['seasonal_rent_burden_pct']:.1f}%")
        print(f"FLAT:     {r['flat_industry']}  (seasonal_index={r['flat_index']:.2f})")
        print(f"  realized annual pay ${r['flat_pay']:,.0f} | {r['flat_pct_female']:.0%} female "
              f"| mean weeks worked/yr: {r['flat_mean_weeks_worked']:.1f} "
              f"| rent burden: {r['flat_rent_burden_pct']:.1f}%")
        print(f"  pay gap: {r['pay_gap_pct']:.1%}  |  seasonality gap: {r['seasonal_gap']:.2f}")


def run(
    min_female_share: float = 0.55,
    min_pums_records: int = 200,
    max_pay_gap_pct: float = 0.15,
    min_seasonal_gap: float = 0.25,
    monthly_rent: float = DEFAULT_MONTHLY_RENT,
    top_n: int = 15,
) -> pd.DataFrame:
    profile = load_profile()
    candidates = screen_candidates(profile, min_female_share, min_pums_records)
    print(f"{len(candidates)} of {len(profile)} industry codes pass the female-share "
          f"(>={min_female_share:.0%}) and sample-size (>={min_pums_records} PUMS records) filters")

    pairs = find_pairs(candidates, max_pay_gap_pct, min_seasonal_gap, top_n)
    pairs = add_rent_burden(pairs, monthly_rent)

    out_path = TABLES_DIR / "seasonal_pair_candidates.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

    print_pairs(pairs)
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find same-pay, opposite-seasonality, female-dominated industry pairs")
    p.add_argument("--min-female-share", type=float, default=0.55)
    p.add_argument("--min-pums-records", type=int, default=200,
                    help="Minimum PUMS person-records for an industry code (thin-sample floor)")
    p.add_argument("--max-pay-gap-pct", type=float, default=0.15)
    p.add_argument("--min-seasonal-gap", type=float, default=0.25)
    p.add_argument("--monthly-rent", type=float, default=DEFAULT_MONTHLY_RENT)
    p.add_argument("--top-n", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        min_female_share=args.min_female_share,
        min_pums_records=args.min_pums_records,
        max_pay_gap_pct=args.max_pay_gap_pct,
        min_seasonal_gap=args.min_seasonal_gap,
        monthly_rent=args.monthly_rent,
        top_n=args.top_n,
    )
