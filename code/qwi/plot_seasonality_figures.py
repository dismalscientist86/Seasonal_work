import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, XWALK


def load_naics_titles(path: Path | None) -> pd.DataFrame | None:
    """Load the optional cleaned NAICS6 title lookup (clean_naics_xwalk.py's output)."""
    if path is None or not path.exists():
        return None
    titles = pd.read_csv(path, dtype={"naics_code": str})
    keep_cols = [c for c in ["naics_code", "national_industry_title"] if c in titles.columns]
    return titles[keep_cols].drop_duplicates("naics_code")


# -------------------------------------------------------------------
# Load employment time series (national aggregation)
# -------------------------------------------------------------------

def load_employment(qwi_file: Path) -> pd.DataFrame:
    """
    Load QWI data and aggregate to national-level employment:
        employment_{j,t,q} = sum_s EmpEnd_{s,j,t,q}

    Returns a DataFrame with:
        industry, year, quarter, employment, date
    """
    if qwi_file.exists():
        df = pd.read_parquet(qwi_file)
    else:
        # Fall back to concatenating per-state files (mirrors load_qwi() in
        # seasonal_index.py) rather than failing outright if the combined
        # file hasn't been (re)built yet.
        state_files = sorted(qwi_file.parent.glob("qwi_state_*.parquet"))
        if not state_files:
            raise FileNotFoundError(
                f"Neither {qwi_file} nor any qwi_state_*.parquet files found "
                f"in {qwi_file.parent}. Run fetch_qwi.py first."
            )
        print(f"Combined file missing; loading {len(state_files)} per-state files...")
        df = pd.concat([pd.read_parquet(f) for f in state_files], ignore_index=True)

    # Per-state files only have the raw "time" string (e.g. "2010-Q1") -- the
    # year/quarter split normally happens once, on the combined file, inside
    # fetch_qwi.py. Derive it here too so this fallback path works standalone.
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = df["time"].str[:4].astype(int)
        df["quarter"] = df["time"].str[-1].astype(int)

    # Ensure numeric
    for c in ["Emp", "EmpEnd", "year", "quarter"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Make codes strings (avoid 722511 → 722511.0 issues)
    df["industry"] = df["industry"].astype(str)

    
    # Drop suppressed negative values
    df.loc[df["EmpEnd"] < 0, "EmpEnd"] = np.nan
    df.loc[df["Emp"] < 0, "Emp"] = np.nan
    
    # National aggregation
    agg = (
        df.groupby(["industry", "year", "quarter"])[["EmpEnd", "Emp"]]
        .sum(min_count=1)
        .reset_index()
    )

    # Choose EmpEnd if available; otherwise fallback to Emp
    agg["employment"] = np.where(
        agg["EmpEnd"].notna(), agg["EmpEnd"], agg["Emp"]
    )
    agg["industry"] = agg["industry"].astype(str)

    # Convert to timestamp (quarter → first day of quarter)
    period_strings = agg["year"].astype(int).astype(str) + "Q" + agg["quarter"].astype(int).astype(str)
    agg["date"] = pd.PeriodIndex(period_strings, freq="Q").to_timestamp()

    return agg

def select_industries_for_plot(seasonal_index: pd.DataFrame,
                               n_seasonal: int = 3,
                               n_nonseasonal: int = 3,
                               min_years: int = 8,
                               min_excess_obs: int = 10,
                               naics_digits: int = 6):
    """
    Return lists of NAICS codes (strings) for most- and least-seasonal industries,
    filtered to a specific NAICS digit length and minimum coverage.

    Parameters
    ----------
    seasonal_index : DataFrame from build_seasonal_index()
                     Must contain columns: naics_code, seasonal_index,
                     seasonal_amplitude, n_excess_obs (optional), n_years (optional)
    n_seasonal : number of highly seasonal industries to return
    n_nonseasonal : number of least seasonal industries to return
    min_years : minimum distinct years required
    min_excess_obs : minimum year-adjacent comparisons contributing to excessQ
    naics_digits : enforce code length (6 for 6-digit NAICS)

    Returns
    -------
    most_seasonal, least_seasonal : lists of NAICS strings
    filtered_df : the filtered table used for selection (for diagnostics)
    """
    df = seasonal_index.copy()
    # Normalize types
    df["naics_code"] = df["naics_code"].astype(str)

    # Filter to exact NAICS length
    df = df[df["naics_code"].str.len() == naics_digits]

    # Coverage filters (use columns if present; otherwise skip)
    if "n_years" in df.columns:
        df = df[df["n_years"] >= min_years]
    if "n_excess_obs" in df.columns:
        df = df[df["n_excess_obs"] >= min_excess_obs]

    # Drop NA or pathological values (seasonal_index.py now NaNs these out at
    # the source when an industry has <2 non-missing quarters, so this alone
    # already excludes the worst thin-cell cases -- e.g. an industry with
    # data in only one quarter, where max-min trivially collapses to 0 and
    # would otherwise masquerade as "not seasonal")
    df = df.dropna(subset=["seasonal_index", "seasonal_amplitude"])

    # Belt-and-suspenders for this illustrative figure specifically: require
    # ALL FOUR quarters to have a real excess estimate, not just enough to
    # clear the n_excess_obs floor above (which only checks the peak
    # quarter). Otherwise a "least seasonal" pick could still be built from
    # e.g. 2 of 4 quarters rather than a genuinely flat year-round profile.
    q_obs_cols = [c for c in ["n_excessQ1", "n_excessQ2", "n_excessQ3", "n_excessQ4"] if c in df.columns]
    if q_obs_cols:
        df = df[(df[q_obs_cols] > 0).all(axis=1)]

    # Select extremes within filtered set
    most = (
        df.sort_values("seasonal_index", ascending=False)
        .head(n_seasonal)["naics_code"].tolist()
    )
    least = (
        df.sort_values("seasonal_index", ascending=True)
        .head(n_nonseasonal)["naics_code"].tolist()
    )

    # Print diagnostics (top/bottom selections with coverage)
    cols_to_show = [c for c in ["naics_code", "seasonal_index",
                                "seasonal_amplitude", "n_excess_obs", "n_years"]
                    if c in df.columns]
    print("\n[Diagnostics] Selection universe after filters:")
    print(df[cols_to_show].sort_values("seasonal_index").head(10).to_string(index=False))
    print(df[cols_to_show].sort_values("seasonal_index", ascending=False).head(10).to_string(index=False))

    return most, least, df


# -------------------------------------------------------------------
# Load seasonal index table
# -------------------------------------------------------------------

def load_seasonal_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"naics_code": str})
    return df


# -------------------------------------------------------------------
# Plot 1: Employment over time for seasonal vs non-seasonal industries
# -------------------------------------------------------------------

def plot_employment_timeseries(emp: pd.DataFrame,
                               seasonal_index: pd.DataFrame,
                               n_seasonal: int = 3,
                               n_nonseasonal: int = 3,
                               naics_titles_path: Path | None = None):
    """
    Plot normalized employment time series for seasonal vs non-seasonal industries,
    strictly using 6-digit NAICS and reasonable coverage thresholds.
    """
    # Normalize code types once (avoid silent mismatches)
    emp = emp.copy()
    emp["industry"] = emp["industry"].astype(str)

    seasonal_index = seasonal_index.copy()
    seasonal_index["naics_code"] = seasonal_index["naics_code"].astype(str)

    # Attach NAICS titles if not already present in the seasonal index CSV
    if "national_industry_title" not in seasonal_index.columns:
        naics_titles_path = naics_titles_path or (XWALK / "naics6_2022_titles.csv")
        titles = load_naics_titles(naics_titles_path)
        if titles is not None:
            seasonal_index = seasonal_index.merge(titles, on="naics_code", how="left")

    # Use the robust selector
    most_seasonal, least_seasonal, filtered_df = select_industries_for_plot(
        seasonal_index,
        n_seasonal=n_seasonal,
        n_nonseasonal=n_nonseasonal,
        min_years=8,
        min_excess_obs=10,
        naics_digits=6,
    )

    selected = most_seasonal + least_seasonal

    # Informative legend labels with index value and industry title
    idx_map = filtered_df.set_index("naics_code")["seasonal_index"].to_dict()
    title_map = (
        filtered_df.set_index("naics_code")["national_industry_title"].to_dict()
        if "national_industry_title" in filtered_df.columns else {}
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    for ind in selected:
        sub = emp[emp["industry"] == ind].sort_values("date")
        if sub.empty:
            print(f"  [warn] No employment series found for NAICS {ind}")
            continue

        # Normalize to first observation = 100
        base = sub["employment"].iloc[0]
        sub = sub.assign(emp_index = 100 * sub["employment"] / base)

        label_class = ("seasonal" if ind in most_seasonal else "non-seasonal")
        idx_val = idx_map.get(ind, np.nan)
        title = title_map.get(ind)
        name = f"{ind} {title}" if isinstance(title, str) else f"NAICS {ind}"
        label = f"{name} ({label_class}, idx={idx_val:.2f})"

        ax.plot(sub["date"], sub["emp_index"], label=label, linewidth=1.8)

    ax.set_title("Normalized Employment Over Time\n6-digit NAICS — Seasonal vs Non-Seasonal")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Employment Index (Base = 100)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig



# -------------------------------------------------------------------
# Plot 2: Seasonal amplitude distribution
# -------------------------------------------------------------------

def plot_seasonal_amplitude_distribution(seasonal_index: pd.DataFrame):
    """
    Histogram of seasonal amplitudes across industries.
    """
    vals = seasonal_index["seasonal_amplitude"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)

    ax.set_title("Distribution of Seasonal Amplitudes Across Industries")
    ax.set_xlabel("Seasonal Amplitude (max ExcessQ - min ExcessQ)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    # Optional KDE if seaborn is installed and compatible with the installed
    # pandas version (older seaborn releases call pandas internals that have
    # since been removed, e.g. the 'mode.use_inf_as_na' option).
    try:
        import seaborn as sns
        sns.kdeplot(vals, ax=ax, color="darkred", linewidth=1.6)
    except Exception as e:
        print(f"  [warn] Skipping KDE overlay (seaborn/pandas incompatibility): {e}")

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(qwi_file: Path, seasonal_file: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading QWI employment data…")
    emp = load_employment(qwi_file)

    print("Loading seasonal index…")
    seasonal_index = load_seasonal_index(seasonal_file)

    print("Plotting employment time series…")
    fig1 = plot_employment_timeseries(emp, seasonal_index)
    fig1.savefig(outdir / "employment_timeseries_examples.pdf",
                 bbox_inches="tight")
    plt.close(fig1)

    print("Plotting seasonal amplitude distribution…")
    fig2 = plot_seasonal_amplitude_distribution(seasonal_index)
    fig2.savefig(outdir / "seasonal_amplitude_distribution.pdf",
                 bbox_inches="tight")
    plt.close(fig2)

    print(f"Done. Figures saved in: {outdir}")


# -------------------------------------------------------------------
# Default paths for easy running in Spyder
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwi-file",
        type=Path,
        default=QWI_RAW / "qwi_naics6_all.parquet",
        help="Path to the aggregated QWI parquet file"
    )
    parser.add_argument(
        "--seasonal-file",
        type=Path,
        default=QWI_CLEAN / "seasonal_index_naics6.csv",
        help="Path to the seasonal index CSV"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save output figures"
    )

    args = parser.parse_args()

    print("Using QWI file:       ", args.qwi_file)
    print("Using seasonal file:  ", args.seasonal_file)
    print("Saving figures to:    ", args.outdir)

    main(args.qwi_file, args.seasonal_file, args.outdir)