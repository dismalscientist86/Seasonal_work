import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_CLEAN, FIGURES_DIR


DATA_DIR = QWI_CLEAN
FIG_DIR = FIGURES_DIR

nat_file = DATA_DIR / "seasonal_index_naics6.csv"
state_file = DATA_DIR / "seasonal_index_naics6_by_state.csv"
output_file = FIG_DIR / "scatter_naics_sorted_by_national_index.png"

AGRICULTURE_SECTOR = "11"


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    df_nat = pd.read_csv(nat_file, dtype={"naics_code": str})
    df_state = pd.read_csv(state_file, dtype={"naics_code": str})

    # ---------------------------------------------------------
    # 2. Clean and standardize NAICS codes (strings only)
    # ---------------------------------------------------------
    df_nat["naics_code"] = df_nat["naics_code"].astype(str).str.zfill(6)
    df_state["naics_code"] = df_state["naics_code"].astype(str).str.zfill(6)

    # Exclude agriculture (NAICS 11) and re-winsorize each series at the
    # 99th percentile of its own ex-agriculture pool -- same convention as the
    # rest of the project (QWI's UI coverage of agriculture is weaker/partial;
    # see agriculture_exclusion_check.py). The stored seasonal_index columns
    # were winsorized on the full sample, so they must be recomputed from
    # seasonal_amplitude rather than just filtered.
    for df in (df_nat, df_state):
        df["sector_2d"] = df["naics_code"].str.zfill(6).str[:2]
    df_nat = df_nat[df_nat["sector_2d"] != AGRICULTURE_SECTOR].dropna(subset=["seasonal_amplitude"]).copy()
    df_state = df_state[df_state["sector_2d"] != AGRICULTURE_SECTOR].dropna(subset=["seasonal_amplitude"]).copy()
    for df in (df_nat, df_state):
        p99 = df["seasonal_amplitude"].quantile(0.99)
        df["seasonal_index"] = (df["seasonal_amplitude"] / p99).clip(0, 1)

    # ---------------------------------------------------------
    # 3. Sort NAICS codes by national seasonal index
    # ---------------------------------------------------------
    df_nat_sorted = df_nat.sort_values("seasonal_index", ascending=True).reset_index(drop=True)

    # Assign each NAICS code a rank based on its national seasonal index
    df_nat_sorted["rank"] = df_nat_sorted.index

    # Build mapping from NAICS string → rank
    rank_map = dict(zip(df_nat_sorted["naics_code"], df_nat_sorted["rank"]))

    # Apply mapping to state-level data
    df_state["rank"] = df_state["naics_code"].map(rank_map)

    # Drop state rows for NAICS that didn't appear in national data (if any)
    df_state = df_state.dropna(subset=["rank"])

    # ---------------------------------------------------------
    # 4. Plot the scatter
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 6))

    # State dots (blue)
    plt.scatter(
        df_state["rank"],
        df_state["seasonal_index"],
        s=2,
        c="blue",
        alpha=0.3,
        label="State values"
    )

    # National dots (red)
    plt.scatter(
        df_nat_sorted["rank"],
        df_nat_sorted["seasonal_index"],
        s=20,
        c="red",
        alpha=0.8,
        label="National values"
    )

    # ---------------------------------------------------------
    # 5. Labels and formatting
    # ---------------------------------------------------------
    plt.xlabel("NAICS codes sorted by national seasonal index (rank)")
    plt.ylabel("Seasonal index")
    plt.title("State vs National Seasonal Index: NAICS sorted by national seasonality (excl. agriculture)")
    plt.legend(loc="upper left")
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
