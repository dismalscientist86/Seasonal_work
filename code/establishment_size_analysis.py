"""
establishment_size_analysis.py — Do seasonal jobs concentrate in small,
independent businesses or in large ones?

Merges the national CBP establishment-size distribution (from
`fetch_cbp.py --by-size`) onto the national QWI separation-based seasonal
index (`seasonal_index.py`), and asks, industry by industry: is a more
seasonal industry more likely to be made up of small establishments?

Why national, not county:
  - The question is industry-level, and
  - county x NAICS6 x size cells are almost entirely EMP-suppressed, whereas
    national x NAICS6 x size has EMP fully populated.

Coverage caveat: CBP does not cover NAICS 111/112 (crop and animal farm
production) at all, so those industries — the most seasonal in the QWI
index — drop out of the merge entirely. This analysis is therefore
*conditional on the CBP universe*: construction, tourism/lodging,
food service, seasonal retail, agricultural *support* services (NAICS 115),
seasonal manufacturing, etc. Findings should be read as "among the
industries CBP covers, ..." not "across all seasonal work."

Size metrics per NAICS6 (from the EMPSZES buckets, lower bound parsed from
each bucket's label so this doesn't depend on hard-coded EMPSZES codes):
  emp_share_lt20   : share of the industry's national employment in
                     establishments with < 20 employees
  emp_share_ge100  : share in establishments with >= 100 employees
  estab_share_lt20 : share of establishment *counts* with < 20 employees
  avg_estab_size   : total EMP / total ESTAB (the "001" all-establishments row)

Outputs:
  output/tables/establishment_size_naics6.csv
  output/tables/size_vs_seasonality_summary.csv
  output/figures/size_vs_seasonality_scatter.pdf
  output/figures/size_by_seasonality_quartile.pdf

Usage:
    python code/fetch_cbp.py --by-size          # ~20 min, one-time
    python code/establishment_size_analysis.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QWI_CLEAN, CBP_RAW, FIGURES_DIR, TABLES_DIR, CBP_EMPSZES_TOTAL


# 2-digit NAICS sector labels (same scheme as seasonal_index.py)
SECTOR_LABELS = {
    "11": "Agriculture, forestry, fishing", "21": "Mining", "22": "Utilities",
    "23": "Construction", "31": "Manufacturing", "32": "Manufacturing",
    "33": "Manufacturing", "42": "Wholesale trade", "44": "Retail trade",
    "45": "Retail trade", "48": "Transportation", "49": "Transportation",
    "51": "Information", "52": "Finance & insurance", "53": "Real estate",
    "54": "Professional & technical", "55": "Management of companies",
    "56": "Administrative & waste", "61": "Educational services",
    "62": "Health care & social assistance", "71": "Arts, entertainment & recreation",
    "72": "Accommodation & food services", "81": "Other services",
    "92": "Public administration",
}


def _size_lower_bound(label: str) -> float | None:
    """Parse the lower employee-count bound from an EMPSZES label."""
    if not isinstance(label, str):
        return None
    m = re.search(r"less than (\d+)", label)
    if m:
        return 0.0
    m = re.search(r"(\d[\d,]*)\s+to\s+(\d[\d,]*)", label)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"(\d[\d,]*)\s+or more", label)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def load_cbp_size(path: Path) -> pd.DataFrame:
    """Load the national CBP-by-size parquet and clean types."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python code/fetch_cbp.py --by-size"
        )
    df = pd.read_parquet(path)
    df["naics_code"] = df["naics_code"].astype(str)
    for c in ["ESTAB", "EMP", "PAYANN"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["EMPSZES"] = df["EMPSZES"].astype(str)
    df["size_lb"] = df["EMPSZES_LABEL"].map(_size_lower_bound)
    return df


def compute_size_metrics(cbp_size: pd.DataFrame) -> pd.DataFrame:
    """
    One row per NAICS6 industry with its establishment-size summary metrics.

    Uses only the breakdown rows (EMPSZES != "001") for the share
    calculations, and the "001" all-establishments row for avg_estab_size /
    industry totals.
    """
    total = cbp_size[cbp_size["EMPSZES"] == CBP_EMPSZES_TOTAL].set_index("naics_code")
    breakdown = cbp_size[cbp_size["EMPSZES"] != CBP_EMPSZES_TOTAL].copy()
    breakdown = breakdown.dropna(subset=["size_lb"])

    records = []
    for naics, grp in breakdown.groupby("naics_code"):
        emp_tot = grp["EMP"].sum(min_count=1)
        estab_tot = grp["ESTAB"].sum(min_count=1)
        lt20 = grp[grp["size_lb"] < 20]
        ge100 = grp[grp["size_lb"] >= 100]

        row = {
            "naics_code": naics,
            "naics_label": total.loc[naics, "NAICS2017_LABEL"] if naics in total.index else None,
            "cbp_emp_total": total.loc[naics, "EMP"] if naics in total.index else emp_tot,
            "cbp_estab_total": total.loc[naics, "ESTAB"] if naics in total.index else estab_tot,
            "n_size_classes": len(grp),
            "emp_share_lt20": (lt20["EMP"].sum(min_count=1) / emp_tot) if emp_tot else np.nan,
            "emp_share_ge100": (ge100["EMP"].sum(min_count=1) / emp_tot) if emp_tot else np.nan,
            "estab_share_lt20": (lt20["ESTAB"].sum(min_count=1) / estab_tot) if estab_tot else np.nan,
        }
        row["avg_estab_size"] = (
            row["cbp_emp_total"] / row["cbp_estab_total"]
            if row["cbp_estab_total"] else np.nan
        )
        records.append(row)

    out = pd.DataFrame(records)
    out["sector_2d"] = out["naics_code"].str[:2]
    out["sector_label"] = out["sector_2d"].map(SECTOR_LABELS).fillna("Other")
    return out


def merge_with_seasonal_index(size_df: pd.DataFrame, seasonal_path: Path) -> pd.DataFrame:
    """Inner-join the size metrics onto the national QWI seasonal index."""
    if not seasonal_path.exists():
        raise FileNotFoundError(
            f"{seasonal_path} not found. Run code/qwi/seasonal_index.py first."
        )
    sea = pd.read_csv(seasonal_path, dtype={"naics_code": str, "sector_2d": str})
    keep = [c for c in ["naics_code", "seasonal_index", "peak_excess", "peak_quarter",
                        "seasonal_index_emp", "n_excess_obs"] if c in sea.columns]
    merged = size_df.merge(sea[keep], on="naics_code", how="inner")
    return merged.dropna(subset=["seasonal_index", "emp_share_lt20"])


def analyze(merged: pd.DataFrame) -> dict:
    """Correlations + seasonality-quartile size composition + by-sector table."""
    metrics = ["emp_share_lt20", "emp_share_ge100", "estab_share_lt20", "avg_estab_size"]
    corr_rows = []
    for m in metrics:
        d = merged.dropna(subset=["seasonal_index", m])
        corr_rows.append({
            "metric": m,
            "n": len(d),
            "pearson_r": d[["seasonal_index", m]].corr().iloc[0, 1],
            "spearman_r": d[["seasonal_index", m]].corr(method="spearman").iloc[0, 1],
        })
    corr = pd.DataFrame(corr_rows)

    q = merged.copy()
    q["seasonality_quartile"] = pd.qcut(
        q["seasonal_index"], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"]
    )
    by_quartile = (
        q.groupby("seasonality_quartile", observed=True)
        .agg(
            n=("naics_code", "size"),
            mean_seasonal_index=("seasonal_index", "mean"),
            mean_emp_share_lt20=("emp_share_lt20", "mean"),
            mean_emp_share_ge100=("emp_share_ge100", "mean"),
            median_avg_estab_size=("avg_estab_size", "median"),
        )
        .reset_index()
    )

    by_sector = (
        merged.groupby("sector_label")
        .agg(
            n=("naics_code", "size"),
            mean_seasonal_index=("seasonal_index", "mean"),
            mean_emp_share_lt20=("emp_share_lt20", "mean"),
            median_avg_estab_size=("avg_estab_size", "median"),
        )
        .reset_index()
        .sort_values("mean_seasonal_index", ascending=False)
    )

    return {"corr": corr, "by_quartile": by_quartile, "by_sector": by_sector}


def print_summary(merged: pd.DataFrame, res: dict) -> None:
    print(f"\n=== Establishment size vs. seasonality "
          f"({len(merged):,} industries in the CBP x QWI merge) ===")
    print("\nCorrelation of seasonal_index with each size metric:")
    print(res["corr"].round(3).to_string(index=False))

    print("\nSize composition by seasonality quartile:")
    print(res["by_quartile"].round(3).to_string(index=False))

    print("\nBy sector (sorted by mean seasonal_index):")
    print(res["by_sector"].round(3).to_string(index=False))

    print("\nMost seasonal CBP industries and their size composition:")
    cols = ["naics_code", "naics_label", "seasonal_index", "emp_share_lt20",
            "avg_estab_size", "cbp_emp_total"]
    top = merged.sort_values("seasonal_index", ascending=False).head(15)
    print(top[[c for c in cols if c in top.columns]].round(3).to_string(index=False))


def plot_scatter(merged: pd.DataFrame) -> plt.Figure:
    d = merged.dropna(subset=["seasonal_index", "emp_share_lt20"]).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sectors = sorted(d["sector_label"].unique())
    cmap = plt.colormaps["tab20"].resampled(max(len(sectors), 1))
    color_map = {s: cmap(i) for i, s in enumerate(sectors)}
    sizes = 8 + 40 * (d["cbp_emp_total"] / d["cbp_emp_total"].max()).clip(0, 1) ** 0.4
    ax.scatter(d["seasonal_index"], d["emp_share_lt20"],
               s=sizes, alpha=0.4, c=d["sector_label"].map(color_map))

    # Binned-mean overlay so the (noisy) relationship is legible
    d["_bin"] = pd.qcut(d["seasonal_index"], 10, duplicates="drop")
    binned = d.groupby("_bin", observed=True).agg(
        x=("seasonal_index", "mean"), y=("emp_share_lt20", "mean")
    )
    ax.plot(binned["x"], binned["y"], "o-", color="black", lw=1.8, ms=5,
            label="decile mean")

    ax.set_xlabel("Seasonal index (separations)")
    ax.set_ylabel("Share of employment in establishments with < 20 employees")
    ax.set_title("Do seasonal industries skew toward small establishments?\n"
                 "(CBP universe — excludes NAICS 111/112 farm production)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_by_quartile(by_quartile: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(by_quartile["seasonality_quartile"].astype(str),
                by_quartile["mean_emp_share_lt20"] * 100, color="#1f77b4")
    axes[0].set_ylabel("Mean employment share in <20-employee establishments (%)")
    axes[0].set_title("Small-establishment employment share\nby seasonality quartile")

    axes[1].bar(by_quartile["seasonality_quartile"].astype(str),
                by_quartile["median_avg_estab_size"], color="#d62728")
    axes[1].set_ylabel("Median average establishment size (employees)")
    axes[1].set_title("Average establishment size\nby seasonality quartile")
    fig.tight_layout()
    return fig


def run(
    cbp_size_path: Path | None = None,
    seasonal_path: Path | None = None,
) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cbp_size_path = cbp_size_path or (CBP_RAW / "cbp_national_naics6_by_size.parquet")
    seasonal_path = seasonal_path or (QWI_CLEAN / "seasonal_index_naics6.csv")

    cbp_size = load_cbp_size(cbp_size_path)
    size_df = compute_size_metrics(cbp_size)
    print(f"Computed size metrics for {len(size_df):,} CBP industries")

    merged = merge_with_seasonal_index(size_df, seasonal_path)

    out_path = TABLES_DIR / "establishment_size_naics6.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    res = analyze(merged)
    pd.concat(
        [res["corr"].assign(table="correlation"),
         res["by_quartile"].assign(table="by_quartile"),
         res["by_sector"].assign(table="by_sector")],
        ignore_index=True,
    ).to_csv(TABLES_DIR / "size_vs_seasonality_summary.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'size_vs_seasonality_summary.csv'}")

    fig1 = plot_scatter(merged)
    fig1.savefig(FIGURES_DIR / "size_vs_seasonality_scatter.pdf", bbox_inches="tight")
    plt.close(fig1)
    fig2 = plot_by_quartile(res["by_quartile"])
    fig2.savefig(FIGURES_DIR / "size_by_seasonality_quartile.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figures to {FIGURES_DIR}")

    print_summary(merged, res)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Establishment size vs. seasonality (national CBP x QWI seasonal index)"
    )
    parser.add_argument("--cbp-size-file", type=Path, default=None)
    parser.add_argument("--seasonal-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(cbp_size_path=args.cbp_size_file, seasonal_path=args.seasonal_file)
