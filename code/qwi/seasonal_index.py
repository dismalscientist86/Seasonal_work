"""
seasonal_index.py — Build a 6-digit NAICS seasonal index from QWI data.

Computes a quarterly analog of Coglianese & Price's excess recurrence measure:

    ExcessQ4_j = mean_t [ SepRate_{j,Q4,t} - (SepRate_{j,Q3,t} + SepRate_{j,Q1,t+1}) / 2 ]

where SepRate = Sep / Emp (quarterly separation rate).

This identifies industries where Q4 (Oct–Dec) separations are systematically
elevated relative to the adjacent quarters — the hallmark of winter seasonality.
We also compute analogous measures for Q1, Q2, Q3 to capture year-round patterns.

Outputs a CSV: data/qwi_clean/seasonal_index_naics<N>.csv
  Columns: naics_code, naics_level, sector_label,
           excessQ1, excessQ2, excessQ3, excessQ4,
           seasonal_amplitude, peak_quarter,
           excessQ4_norm, seasonal_index,
           seasonal_index_emp

The 'seasonal_index' is a 0–1 score suitable for classifying firms:
  0 = no seasonality, 1 = extreme seasonality.

Usage:
    python code/qwi/seasonal_index.py
    python code/qwi/seasonal_index.py --naics-file path/to/qwi_naics6_all.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    QWI_RAW,
    QWI_CLEAN,
    FIGURES_DIR,
    TABLES_DIR,
    QWI_NAICS_LEVEL,
    MIN_EXCESS_OBS,
    XWALK,
)

warnings.filterwarnings("ignore")

# NAICS sector labels (2-digit)
NAICS_SECTOR_LABELS = {
    "11": "Agriculture, forestry, fishing",
    "21": "Mining, quarrying, oil & gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing - food/textiles",
    "32": "Manufacturing - paper/chemicals",
    "33": "Manufacturing - metals/machinery",
    "42": "Wholesale trade",
    "44": "Retail trade - general",
    "45": "Retail trade - specialty",
    "48": "Transportation",
    "49": "Warehousing & couriers",
    "51": "Information",
    "52": "Finance & insurance",
    "53": "Real estate",
    "54": "Professional & technical services",
    "55": "Management of companies",
    "56": "Administrative & waste services",
    "61": "Educational services",
    "62": "Health care & social assistance",
    "71": "Arts, entertainment & recreation",
    "72": "Accommodation & food services",
    "81": "Other services",
    "92": "Public administration",
}


#  Data loading 

def load_qwi(path: Path | None = None, naics_level: int | None = None) -> pd.DataFrame:
    """Load QWI parquet file(s) from QWI_RAW directory."""
    naics_level = naics_level or QWI_NAICS_LEVEL

    if path is not None:
        df = pd.read_parquet(path)
    else:
        # Try combined file first
        combined = QWI_RAW / f"qwi_naics{naics_level}_all.parquet"
        if combined.exists():
            df = pd.read_parquet(combined)
        else:
            # Load per-state files
            files = sorted(QWI_RAW.glob("qwi_state_*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No QWI parquet files found in {QWI_RAW}.\n"
                    "Run fetch_qwi.py first."
                )
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Per-state files only have the raw "time" string (e.g. "2010-Q1") — the
    # year/quarter split normally happens once, on the combined file, inside
    # fetch_qwi(). Derive it here too so this fallback path works standalone.
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = df["time"].str[:4].astype(int)
        df["quarter"] = df["time"].str[-1].astype(int)

    # Ensure numeric columns
    for c in ["Emp", "EmpEnd", "Sep", "SepBeg", "EarnBeg", "year", "quarter"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"Loaded QWI: {len(df):,} rows, {df['industry'].nunique()} industries, "
          f"years {df['year'].min():.0f}–{df['year'].max():.0f}")
    return df


#  Separation rate calculation 

def compute_sep_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarterly separation rates: SepRate = Sep / Emp.

    Aggregates across states to get national averages, then computes
    the separation rate for each (industry, year, quarter).
    """
    # Suppress suppressed cells (Census sets to -9, -99, or similar)
    df = df.copy()
    for c in ["Sep", "Emp", "EmpEnd", "SepBeg"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # Aggregate to national level: sum Sep and Emp across states
    agg_cols = [c for c in ["Sep", "Emp", "EmpEnd", "SepBeg"] if c in df.columns]
    groupcols = ["industry", "year", "quarter"]

    national = (
        df.groupby(groupcols)[agg_cols]
        .agg(lambda x: x.dropna().sum() if x.dropna().sum() > 0 else np.nan)
        .reset_index()
    )

    # Separation rate: prefer EmpEnd as denominator (closer to flow timing)
    denom = "EmpEnd" if "EmpEnd" in national.columns else "Emp"
    national["sep_rate"] = national["Sep"] / national[denom]

    # Drop zero/extreme rates (likely disclosure suppression)
    national.loc[national["sep_rate"] <= 0,  "sep_rate"] = np.nan
    national.loc[national["sep_rate"] >= 1,  "sep_rate"] = np.nan  # >100% in a quarter is impossible

    return national

def compute_employment_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build national employment 'shares' s_{industry,year,quarter} = EmpEnd / mean(EmpEnd in year),
    where EmpEnd is preferred (time-aligned) and Emp is the fallback if EmpEnd is not present.
    Negative/suppressed values are treated as missing.
    """
    d = df.copy()
    for c in ["Emp", "EmpEnd"]:
        if c in d.columns:
            d.loc[d[c] < 0, c] = np.nan

    # National aggregation across states
    agg_cols = [c for c in ["Emp", "EmpEnd"] if c in d.columns]
    groupcols = ["industry", "year", "quarter"]
    nat = (
        d.groupby(groupcols)[agg_cols]
        .agg(lambda x: x.dropna().sum() if x.dropna().sum() > 0 else np.nan)
        .reset_index()
    )

    denom = "EmpEnd" if "EmpEnd" in nat.columns else "Emp"
    nat["emp_level"] = nat[denom]

    # Within-year normalization: share = level / mean(level in that industry-year)
    nat["year_mean_emp"] = nat.groupby(["industry", "year"])["emp_level"].transform("mean")
    nat["emp_share"] = nat["emp_level"] / nat["year_mean_emp"]

    # Sanity: if only one quarter exists in a year, shares=1 but no neighbors, so excess won't be computed
    return nat[["industry", "year", "quarter", "emp_share"]]


def compute_employment_excess_by_quarter(
    emp_shares: pd.DataFrame, min_obs: int = MIN_EXCESS_OBS
) -> pd.DataFrame:
    """
    Analog of compute_excess_recurrence_by_quarter(), but using employment shares s_{j,t,q}.

    Cells with fewer than `min_obs` contributing years are set to NaN, matching
    the threshold used for the separation-rate excess measure and for the
    state x industry index in geographic_analysis.py.

    Returns a DataFrame indexed by industry with columns:
        emp_excessQ1, emp_excessQ2, emp_excessQ3, emp_excessQ4,
        emp_share_Q1, emp_share_Q2, emp_share_Q3, emp_share_Q4,
        n_emp_excessQ1..Q4, n_years
    """
    emp_shares = emp_shares.sort_values(["industry", "year", "quarter"]).copy()
    records = []

    for ind, grp in emp_shares.groupby("industry"):
        pivot = grp.pivot(index="year", columns="quarter", values="emp_share")
        excess = {}
        mean_shares = {}

        for q in [1, 2, 3, 4]:
            q_prev = 4 if q == 1 else q - 1
            q_next = 1 if q == 4 else q + 1

            if q not in pivot.columns:
                excess[f"emp_excessQ{q}"]   = np.nan
                excess[f"n_emp_excessQ{q}"] = 0
                mean_shares[f"emp_share_Q{q}"] = np.nan
                continue

            mean_shares[f"emp_share_Q{q}"] = pivot[q].mean()

            # Need both adjacent quarters for the counterfactual
            if q_prev not in pivot.columns or q_next not in pivot.columns:
                excess[f"emp_excessQ{q}"]   = np.nan
                excess[f"n_emp_excessQ{q}"] = 0
                continue

            # Handle year wrap for Q1/Q4 neighbors
            if q == 1:
                s_curr = pivot[1]
                s_prev = pivot[4].shift(1)   # previous year's Q4
                s_next = pivot[2]
            elif q == 4:
                s_curr = pivot[4]
                s_prev = pivot[3]
                s_next = pivot[1].shift(-1)  # next year's Q1
            else:
                s_curr = pivot[q]
                s_prev = pivot[q_prev]
                s_next = pivot[q_next]

            cf = (s_prev + s_next) / 2
            exr_series = (s_curr - cf).dropna()

            excess[f"emp_excessQ{q}"]        = exr_series.mean() if len(exr_series) >= min_obs else np.nan
            excess[f"n_emp_excessQ{q}"]      = len(exr_series)

        records.append({
            "industry": ind,
            "n_years": len(grp["year"].unique()),
            **excess,
            **mean_shares,
        })

    return pd.DataFrame(records)


def build_employment_seasonal_index(emp_excess_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a 0-1 seasonal index from employment excess-by-quarter, mirroring
    build_seasonal_index() so the flow (separation-based) and stock
    (employment-based) indices carry the same kind of columns and are directly
    comparable — including *timing* (peak_quarter_emp), not just amplitude.

    Returns columns for merging:
        naics_code, emp_seasonal_amplitude, peak_quarter_emp, peak_excess_emp,
        n_emp_excess_obs, seasonal_index_emp
    """
    df = emp_excess_df.copy()
    excess_cols = ["emp_excessQ1", "emp_excessQ2", "emp_excessQ3", "emp_excessQ4"]
    available = [c for c in excess_cols if c in df.columns]

    # See build_seasonal_index() for why: <2 non-missing quarters makes
    # max-min trivially 0, a data-thinness artifact, not genuine flatness.
    enough_quarters = df[available].notna().sum(axis=1) >= 2

    df["emp_seasonal_amplitude"] = df[available].max(axis=1) - df[available].min(axis=1)
    df.loc[~enough_quarters, "emp_seasonal_amplitude"] = np.nan

    _peak = df[available].apply(
        lambda row: row.idxmax() if row.notna().any() else pd.NA, axis=1
    )
    df["peak_quarter_emp"] = (
        _peak
        .str.replace("emp_excess", "", regex=False)
        .str.replace("Q", "", regex=False)
        .astype("Int64")
    )
    df.loc[~enough_quarters, "peak_quarter_emp"] = pd.NA

    # peak_excess_emp: excess employment share at the industry's peak quarter
    df["peak_excess_emp"] = df[available].max(axis=1)
    df.loc[~enough_quarters, "peak_excess_emp"] = np.nan

    # n_emp_excess_obs: years contributing to the peak-quarter excess estimate
    # (the analog of n_excess_obs on the flow side — flags thin cells)
    n_cols = {c.replace("emp_excessQ", "n_emp_excessQ"): c.replace("emp_excessQ", "Q")
              for c in available if c.replace("emp_excessQ", "n_emp_excessQ") in df.columns}
    if n_cols:
        df["n_emp_excess_obs"] = df.apply(
            lambda row: row[f"n_emp_excessQ{int(row['peak_quarter_emp'])}"]
            if pd.notna(row["peak_quarter_emp"])
               and f"n_emp_excessQ{int(row['peak_quarter_emp'])}" in df.columns
            else pd.NA,
            axis=1,
        ).astype("Int64")

    # 99th-percentile winsorization for 0-1 scale
    p99_emp = df["emp_seasonal_amplitude"].quantile(0.99)
    df["seasonal_index_emp"] = (df["emp_seasonal_amplitude"] / p99_emp).clip(0, 1)

    return df.rename(columns={"industry": "naics_code"})[
        ["naics_code", "emp_seasonal_amplitude", "peak_quarter_emp",
         "peak_excess_emp", "n_emp_excess_obs", "seasonal_index_emp"]
    ]

#  Excess-Q4 computation 

def compute_excess_recurrence_by_quarter(
    rates: pd.DataFrame, min_obs: int = MIN_EXCESS_OBS
) -> pd.DataFrame:
    """
    For each industry and focal quarter Q ∈ {1,2,3,4}, compute the excess
    separation rate relative to the average of the two adjacent quarters.

    ExcessQ_j = mean_t[ SepRate(Q, t) - (SepRate(Q-1, t) + SepRate(Q+1, t)) / 2 ]

    where the average is taken over all years with non-missing adjacent quarters.

    Quarters are treated cyclically: Q4's neighbors are Q3 and Q1(next year).

    Cells with fewer than `min_obs` contributing years are set to NaN rather
    than reported as-is, to avoid thin cells masquerading as low-seasonality
    estimates (matches the min_obs filter used in geographic_analysis.py).

    Returns a DataFrame indexed by industry with columns:
        excessQ1, excessQ2, excessQ3, excessQ4,
        sep_rate_Q1, sep_rate_Q2, sep_rate_Q3, sep_rate_Q4
    """
    rates = rates.sort_values(["industry", "year", "quarter"]).copy()
    records = []

    for ind, grp in rates.groupby("industry"):
        grp = grp.sort_values(["year", "quarter"]).copy()

        # Pivot to (year × quarter) matrix
        pivot = grp.pivot(index="year", columns="quarter", values="sep_rate")

        excess = {}
        mean_rates = {}

        for q in [1, 2, 3, 4]:
            q_prev = 4 if q == 1 else q - 1
            q_next = 1 if q == 4 else q + 1

            if q not in pivot.columns:
                excess[f"excessQ{q}"]   = np.nan
                excess[f"n_excessQ{q}"] = 0
                mean_rates[f"sep_rate_Q{q}"] = np.nan
                continue

            mean_rates[f"sep_rate_Q{q}"] = pivot[q].mean()

            # Need both adjacent quarters to compute the counterfactual
            if q_prev not in pivot.columns or q_next not in pivot.columns:
                excess[f"excessQ{q}"]   = np.nan
                excess[f"n_excessQ{q}"] = 0
                continue

            # For cyclical neighbors, we need to handle year wrap
            if q == 1:
                # Q1(t)'s neighbors: Q4(t-1) and Q2(t)
                s_curr = pivot[1]
                s_prev = pivot[4].shift(1)   # previous year's Q4
                s_next = pivot[2]
            elif q == 4:
                # Q4(t)'s neighbors: Q3(t) and Q1(t+1)
                s_curr = pivot[4]
                s_prev = pivot[3]
                s_next = pivot[1].shift(-1)  # next year's Q1
            else:
                s_curr = pivot[q]
                s_prev = pivot[q_prev]
                s_next = pivot[q_next]

            cf = (s_prev + s_next) / 2
            exr_series = (s_curr - cf).dropna()

            excess[f"excessQ{q}"]        = exr_series.mean() if len(exr_series) >= min_obs else np.nan
            excess[f"n_excessQ{q}"]      = len(exr_series)

        # Data coverage: how many state × year cells contributed
        n_years = len(grp["year"].unique())

        records.append({
            "industry": ind,
            "n_years":  n_years,
            **excess,
            **mean_rates,
        })

    result = pd.DataFrame(records)
    return result


#  Seasonal index construction 

def build_seasonal_index(excess_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the four excess-quarter measures into a summary seasonal index.

    Metrics:
        seasonal_amplitude : max(excessQ*) − min(excessQ*)
            — overall within-year variation in separation rates
        peak_quarter       : quarter with highest excess recurrence
        seasonal_index     : 0–1 normalized score based on seasonal_amplitude
            — useful threshold: industries with seasonal_index > 0.5 are "seasonal"
    """
    df = excess_df.copy()

    excess_cols = ["excessQ1", "excessQ2", "excessQ3", "excessQ4"]
    available   = [c for c in excess_cols if c in df.columns]

    # A meaningful amplitude needs at least 2 non-missing quarters: with only
    # 1 (or 0), max-min collapses to exactly 0, producing a false
    # seasonal_index=0 for industries that are actually just data-thin
    # (verified: ~6 of 1,011 national industries have data in only one
    # quarter), not genuinely non-seasonal. NaN out amplitude/peak/index
    # for those rather than silently reporting a spurious zero.
    enough_quarters = df[available].notna().sum(axis=1) >= 2

    df["seasonal_amplitude"] = df[available].max(axis=1) - df[available].min(axis=1)
    df.loc[~enough_quarters, "seasonal_amplitude"] = np.nan

    _peak = df[available].apply(
        lambda row: row.idxmax() if row.notna().any() else pd.NA, axis=1
    )
    df["peak_quarter"] = (
        _peak
        .str.replace("excess", "", regex=False)
        .str.replace("Q", "", regex=False)
        .astype("Int64")
    )
    df.loc[~enough_quarters, "peak_quarter"] = pd.NA

    # peak_excess: excess separation rate at the peak quarter
    df["peak_excess"] = df[available].max(axis=1)
    df.loc[~enough_quarters, "peak_excess"] = np.nan

    # n_excess_obs: years contributing to the peak-quarter excess estimate
    n_cols = {c.replace("excessQ", "n_excessQ"): c.replace("excessQ", "Q")
              for c in available if c.replace("excessQ", "n_excessQ") in df.columns}
    if n_cols:
        df["n_excess_obs"] = df.apply(
            lambda row: row[f"n_excessQ{int(row['peak_quarter'])}"]
            if pd.notna(row["peak_quarter"])
               and f"n_excessQ{int(row['peak_quarter'])}" in df.columns
            else pd.NA,
            axis=1,
        ).astype("Int64")

    # Normalize seasonal_amplitude to [0, 1] using 99th-percentile winsorization
    p99 = df["seasonal_amplitude"].quantile(0.99)
    df["seasonal_index"] = (df["seasonal_amplitude"] / p99).clip(0, 1)

    # Add NAICS sector label (2-digit prefix)
    df["sector_2d"] = df["industry"].astype(str).str[:2]
    df["sector_label"] = df["sector_2d"].map(NAICS_SECTOR_LABELS).fillna("Other")

    # Infer NAICS level from the code length
    code_lengths = df["industry"].astype(str).str.len()
    df["naics_level"] = code_lengths

    return df.rename(columns={"industry": "naics_code"}).sort_values(
        "seasonal_index", ascending=False
    )


#  Validation: compare to CP's 1-digit results 

def validate_against_cp(
    seasonal_index: pd.DataFrame,
    cp_1digit_path: Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate our QWI-based seasonal index to 1-digit level and compare to
    CP's excess recurrence estimates by industry.

    CP's 1-digit mapping (approximate NAICS equivalents):
        1 Agr/fish/forest   → NAICS 11
        2 Mining            → NAICS 21
        3 Construction      → NAICS 23
        4 Manufacturing     → NAICS 31-33
        5 TCU               → NAICS 48-49, 22, 51
        6 Wholesale         → NAICS 42
        7 Retail            → NAICS 44-45
        8 FIRE              → NAICS 52-53
        9 Business services → NAICS 54-56
       10 Personal services → NAICS 81
       11 Entertainment     → NAICS 71-72
       12 Healthcare        → NAICS 62
       13 Education         → NAICS 61
       14 Social/prof svcs  → NAICS 54 (overlap with business)
       15 Public admin      → NAICS 92
    """
    cp_map = {
        "11": 1, "21": 2, "23": 3,
        "31": 4, "32": 4, "33": 4,
        "22": 5, "48": 5, "49": 5, "51": 5,
        "42": 6,
        "44": 7, "45": 7,
        "52": 8, "53": 8,
        "54": 9, "55": 9, "56": 9,
        "81": 10,
        "71": 11, "72": 11,
        "62": 12,
        "61": 13,
        "92": 15,
    }
    cp_labels = {
        1: "Agriculture/fishing/forestry", 2: "Mining", 3: "Construction",
        4: "Manufacturing", 5: "TCU", 6: "Wholesale", 7: "Retail",
        8: "FIRE", 9: "Business services", 10: "Personal services",
        11: "Entertainment/recreation", 12: "Healthcare",
        13: "Educational services", 14: "Social/prof. services",
        15: "Public admin./armed forces",
    }

    df = seasonal_index.copy()
    df["cp_ind"] = df["sector_2d"].map(cp_map)
    df_cp = df.dropna(subset=["cp_ind"])

    # Average seasonal index by CP 1-digit industry
    agg = (
        df_cp.groupby("cp_ind")[["seasonal_index", "excessQ4", "seasonal_amplitude"]]
        .mean()
        .reset_index()
    )
    agg["cp_label"] = agg["cp_ind"].map(cp_labels)
    agg = agg.sort_values("seasonal_index", ascending=False)

    print("\nQWI seasonal index vs. CP's 1-digit industry classification:")
    print(agg[["cp_label", "seasonal_index", "excessQ4"]].to_string(index=False))

    # Optionally merge in CP excess recurrence if the estimates table is available
    if cp_1digit_path and cp_1digit_path.exists():
        cp = pd.read_csv(cp_1digit_path)
        if "excess_recurrence" in cp.columns and "ind" in cp.columns:
            agg = agg.merge(
                cp[["ind", "excess_recurrence"]].rename(columns={"ind": "cp_ind"}),
                on="cp_ind", how="left",
            )
            print("\nCorrelation with CP's excess recurrence estimates:")
            corr = agg[["seasonal_index", "excess_recurrence"]].dropna().corr()
            print(f"  rho = {corr.loc['seasonal_index','excess_recurrence']:.3f}")

    return agg


#  Figures 

def plot_seasonal_index(
    seasonal_index: pd.DataFrame,
    top_n: int = 30,
    title: str = "QWI Seasonal Index by Industry",
) -> plt.Figure:
    """Bar chart of the top-N most seasonal industries."""
    df = seasonal_index.head(top_n).sort_values("seasonal_index")

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.28)))
    colors = plt.cm.RdYlGn_r(df["seasonal_index"].values)
    ax.barh(df["naics_code"].astype(str) + " " + df["sector_label"],
            df["seasonal_index"], color=colors)
    ax.set_xlabel("Seasonal Index (0–1)")
    ax.set_title(title)
    ax.axvline(0.5, color="black", lw=0.5, ls="--", label="0.5 threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_quarterly_profiles(
    seasonal_index: pd.DataFrame,
    naics_codes: list[str] | None = None,
) -> plt.Figure:
    """
    Plot the quarterly separation rate profile for selected industries,
    showing the seasonal pattern (excess by quarter).
    """
    if naics_codes is None:
        # Pick the 6 most seasonal industries
        naics_codes = seasonal_index.head(6)["naics_code"].astype(str).tolist()

    sub = seasonal_index[seasonal_index["naics_code"].astype(str).isin(naics_codes)]
    n = len(sub)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
    axes = axes.flatten()

    excess_cols = ["excessQ1", "excessQ2", "excessQ3", "excessQ4"]
    for i, (_, row) in enumerate(sub.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = [row.get(c, np.nan) for c in excess_cols]
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.bar(quarters, vals, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{row['naics_code']} {row['sector_label'][:20]}", fontsize=8)
        ax.set_ylabel("Excess sep. rate", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Quarterly excess separation rates by industry")
    fig.tight_layout()
    return fig


#  Flow vs. stock comparison

def compare_flow_vs_stock(
    df: pd.DataFrame,
    group_col: str | None = "sector_label",
) -> dict:
    """
    Compare the flow (separation-based, seasonal_index/peak_quarter) and stock
    (employment-based, seasonal_index_emp/peak_quarter_emp) seasonal indices on
    the same dataframe — works for either the national index (one row per
    naics_code) or the state x NAICS6 index from geographic_analysis.py (one
    row per state x naics_code), as long as both sets of columns are present.

    Returns a dict:
        overall    : dict of n, pearson_r, spearman_r, peak_concordance_rate
        by_group   : DataFrame with the same stats computed within each value
                     of `group_col` (e.g. sector_label nationally, or
                     state_name for the state-level index) — None if group_col
                     isn't in df
        divergence : df with a 'divergence' column (seasonal_index -
                     seasonal_index_emp) added, sorted ascending. Very negative
                     = "stock without flow" (headcount swings without a
                     separation spike); very positive = "flow without stock"
                     (separations spike without headcount visibly moving).
    """
    d = df.dropna(subset=["seasonal_index", "seasonal_index_emp"]).copy()
    d["divergence"] = d["seasonal_index"] - d["seasonal_index_emp"]

    has_peak_cols = "peak_quarter" in d.columns and "peak_quarter_emp" in d.columns

    def _stats(sub: pd.DataFrame) -> dict:
        row = {
            "n": len(sub),
            "pearson_r": sub[["seasonal_index", "seasonal_index_emp"]].corr().iloc[0, 1],
            "spearman_r": sub[["seasonal_index", "seasonal_index_emp"]].corr(method="spearman").iloc[0, 1],
        }
        if has_peak_cols:
            pv = sub.dropna(subset=["peak_quarter", "peak_quarter_emp"])
            row["peak_concordance_rate"] = (
                (pv["peak_quarter"] == pv["peak_quarter_emp"]).mean() if len(pv) else np.nan
            )
            row["n_peak_compared"] = len(pv)
        return row

    overall = _stats(d)

    by_group = None
    if group_col and group_col in d.columns:
        rows = []
        for g, sub in d.groupby(group_col):
            if len(sub) < 3:
                continue
            row = {group_col: g, "mean_divergence": sub["divergence"].mean(), **_stats(sub)}
            rows.append(row)
        by_group = pd.DataFrame(rows).sort_values("n", ascending=False) if rows else None

    return {
        "overall": overall,
        "by_group": by_group,
        "divergence": d.sort_values("divergence"),
    }


def print_flow_vs_stock_summary(
    result: dict, top_n: int = 10, label_cols: list[str] | None = None,
) -> None:
    """Print a human-readable summary of compare_flow_vs_stock()'s output."""
    overall = result["overall"]
    print(f"\nFlow (separations) vs. stock (employment) seasonal index comparison")
    print(f"  n = {overall['n']}")
    print(f"  Pearson r  = {overall['pearson_r']:.3f}")
    print(f"  Spearman r = {overall['spearman_r']:.3f}")
    if "peak_concordance_rate" in overall:
        print(f"  Peak-quarter concordance = {overall['peak_concordance_rate']:.1%} "
              f"({overall['n_peak_compared']} cells with both peak quarters)")

    if result["by_group"] is not None:
        print("\nBy group:")
        print(result["by_group"].round(3).to_string(index=False))

    divergence = result["divergence"]
    label_cols = label_cols or [
        c for c in ["state_name", "naics_code", "sector_label"] if c in divergence.columns
    ]
    cols = label_cols + ["seasonal_index", "seasonal_index_emp", "divergence"]

    print(f"\nTop {top_n} 'flow without stock' (separations spike, headcount doesn't):")
    print(divergence.tail(top_n)[cols].sort_values("divergence", ascending=False).to_string(index=False))

    print(f"\nTop {top_n} 'stock without flow' (headcount swings, separations don't spike):")
    print(divergence.head(top_n)[cols].to_string(index=False))


def plot_flow_vs_stock(
    df: pd.DataFrame,
    group_col: str = "sector_2d",
    title: str = "Flow vs. stock seasonal index",
) -> plt.Figure:
    """Scatter of seasonal_index (flow, x-axis) vs. seasonal_index_emp (stock, y-axis)."""
    d = df.dropna(subset=["seasonal_index", "seasonal_index_emp"]).copy()

    fig, ax = plt.subplots(figsize=(7, 7))
    if group_col in d.columns:
        groups = sorted(d[group_col].astype(str).unique())
        cmap = plt.colormaps["tab20"].resampled(max(len(groups), 1))
        color_map = {g: cmap(i) for i, g in enumerate(groups)}
        ax.scatter(
            d["seasonal_index"], d["seasonal_index_emp"],
            s=10, alpha=0.5, c=d[group_col].astype(str).map(color_map),
        )
    else:
        ax.scatter(d["seasonal_index"], d["seasonal_index_emp"], s=10, alpha=0.5, c="#1f77b4")

    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel("Seasonal index — flow (separations)")
    ax.set_ylabel("Seasonal index — stock (employment)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run_flow_vs_stock(
    level: str = "national",
    national_index: pd.DataFrame | None = None,
) -> dict:
    """
    Run the flow-vs-stock comparison and save tables/figure.

    level : "national", "state", or "both"
        "state" reads data/qwi_clean/seasonal_index_naics6_by_state.csv, which
        must already exist (run code/qwi/geographic_analysis.py first).
    """
    results = {}

    if level in ("national", "both"):
        if national_index is None:
            nat_path = QWI_CLEAN / "seasonal_index_naics6.csv"
            if not nat_path.exists():
                raise FileNotFoundError(f"{nat_path} not found. Run seasonal_index.py first.")
            national_index = pd.read_csv(nat_path, dtype={"naics_code": str, "sector_2d": str})

        result = compare_flow_vs_stock(national_index, group_col="sector_label")
        print("\n=== National: flow vs. stock comparison ===")
        print_flow_vs_stock_summary(result)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        result["divergence"].to_csv(TABLES_DIR / "flow_vs_stock_national.csv", index=False)
        if result["by_group"] is not None:
            result["by_group"].to_csv(TABLES_DIR / "flow_vs_stock_national_by_sector.csv", index=False)

        fig = plot_flow_vs_stock(
            national_index, group_col="sector_2d",
            title="National: flow (separations) vs. stock (employment) seasonal index",
        )
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "flow_vs_stock_national_scatter.pdf", bbox_inches="tight")
        plt.close(fig)
        results["national"] = result

    if level in ("state", "both"):
        state_path = QWI_CLEAN / "seasonal_index_naics6_by_state.csv"
        if not state_path.exists():
            raise FileNotFoundError(
                f"{state_path} not found. Run code/qwi/geographic_analysis.py first."
            )
        state_index = pd.read_csv(
            state_path, dtype={"naics_code": str, "sector_2d": str, "state": str}
        )

        result = compare_flow_vs_stock(state_index, group_col="state_name")
        print("\n=== State x NAICS6: flow vs. stock comparison ===")
        print_flow_vs_stock_summary(
            result, label_cols=["state_name", "naics_code", "sector_label"]
        )

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        result["divergence"].to_csv(TABLES_DIR / "flow_vs_stock_state.csv", index=False)
        if result["by_group"] is not None:
            result["by_group"].to_csv(TABLES_DIR / "flow_vs_stock_state_by_state.csv", index=False)

        fig = plot_flow_vs_stock(
            state_index, group_col="sector_2d",
            title="State x NAICS6: flow (separations) vs. stock (employment) seasonal index",
        )
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "flow_vs_stock_state_scatter.pdf", bbox_inches="tight")
        plt.close(fig)
        results["state"] = result

    return results


#  Main

def load_naics_titles(path: Path | None) -> pd.DataFrame | None:
    """Load the optional cleaned NAICS6 title lookup (clean_naics_xwalk.py's output)."""
    if path is None or not path.exists():
        return None
    titles = pd.read_csv(path, dtype={"naics_code": str, "sector_2d": str})
    keep_cols = [
        "naics_code", "national_industry_title", "sector_title",
        "subsector_title", "industry_group_title", "naics_industry_title",
    ]
    keep_cols = [c for c in keep_cols if c in titles.columns]
    return titles[keep_cols].drop_duplicates("naics_code")


def add_naics_titles(df: pd.DataFrame, titles: pd.DataFrame | None) -> pd.DataFrame:
    """Attach NAICS title fields when a cleaned lookup is available; no-op otherwise."""
    if titles is None:
        return df
    return df.merge(titles, on="naics_code", how="left")


def build_and_save_index(
    qwi_path: Path | None = None,
    naics_level: int | None = None,
    save: bool = True,
    naics_titles_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load QWI → compute sep rates → compute excess → build index → save.
    """
    naics_level = naics_level or QWI_NAICS_LEVEL
    QWI_CLEAN.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load QWI data
    qwi = load_qwi(path=qwi_path, naics_level=naics_level)

    # Compute separation rates
    rates = compute_sep_rates(qwi)
    print(f"\nSeparation rates: {len(rates):,} industry × year × quarter obs")

    # Compute excess recurrence by quarter
    excess = compute_excess_recurrence_by_quarter(rates)
    print(f"Excess recurrence computed for {len(excess):,} industries")

    # Build seasonal index
    index = build_seasonal_index(excess)
    
    
    # Employment-only seasonal index
    emp_shares = compute_employment_shares(qwi)
    emp_excess = compute_employment_excess_by_quarter(emp_shares)
    index_emp  = build_employment_seasonal_index(emp_excess)

    # Merge employment index into the main index
    index = index.merge(index_emp, on="naics_code", how="left")

    # Attach human-readable NAICS6 industry titles when the cleaned crosswalk
    # is available (code/qwi/clean_naics_xwalk.py); harmless no-op otherwise.
    naics_titles_path = naics_titles_path or (XWALK / "naics6_2022_titles.csv")
    index = add_naics_titles(index, load_naics_titles(naics_titles_path))

    top15_cols = ["naics_code"]
    if "national_industry_title" in index.columns:
        top15_cols.append("national_industry_title")
    top15_cols += ["sector_label", "seasonal_index", "peak_excess", "peak_quarter", "n_excess_obs"]

    print(f"\nTop 15 most seasonal industries (NAICS-{naics_level}):")
    print(index.head(15)[top15_cols].to_string(index=False))

    if save:
        out_path = QWI_CLEAN / f"seasonal_index_naics{naics_level}.csv"
        index.to_csv(out_path, index=False)
        print(f"\nIndex saved to {out_path}")

        # Figures
        fig = plot_seasonal_index(index, top_n=40,
                                  title=f"QWI Seasonal Index (NAICS-{naics_level})")
        fig.savefig(FIGURES_DIR / f"seasonal_index_naics{naics_level}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        fig2 = plot_quarterly_profiles(index)
        fig2.savefig(FIGURES_DIR / "seasonal_quarterly_profiles.pdf",
                     bbox_inches="tight")
        plt.close(fig2)

    # Validate against CP 1-digit results (if estimates exist)
    cp_path = TABLES_DIR / "recurrence_by_industry_cps.csv"
    validate_against_cp(index, cp_1digit_path=cp_path)

    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build QWI seasonal index")
    parser.add_argument("--naics-file", type=Path, default=None,
                        help="Path to the QWI parquet file")
    parser.add_argument("--naics-level", type=int, default=QWI_NAICS_LEVEL)
    parser.add_argument(
        "--naics-titles", type=Path, default=None,
        help="Optional cleaned NAICS6 title lookup created by clean_naics_xwalk.py "
             f"(default: {XWALK / 'naics6_2022_titles.csv'}).",
    )
    parser.add_argument(
        "--compare-flow-stock", choices=["none", "national", "state", "both"], default="none",
        help="After building the national index, also compare the flow "
             "(separations) vs. stock (employment) seasonal indices at this "
             "level. 'state'/'both' require geographic_analysis.py to have "
             "already been run (reads seasonal_index_naics6_by_state.csv).",
    )
    args = parser.parse_args()

    index = build_and_save_index(
        qwi_path=args.naics_file,
        naics_level=args.naics_level,
        naics_titles_path=args.naics_titles,
    )

    if args.compare_flow_stock != "none":
        run_flow_vs_stock(level=args.compare_flow_stock, national_index=index)

