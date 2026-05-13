"""
geographic_analysis.py — State-level geographic variation in the QWI seasonal index.

Computes a reusable state x NAICS6 seasonal index, then produces:
  1. Bar chart: mean state seasonality (across industries)
  2. Bar chart: sector geographic variation (cross-state SD)
  3. Two-panel: construction and agriculture by state
  4. Heatmap: state x sector for top variable sectors

Outputs:
  data/qwi_clean/seasonal_index_naics6_by_state.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, TABLES_DIR

STATE_NAMES = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas","06":"California",
    "08":"Colorado","09":"Connecticut","10":"Delaware","11":"DC","12":"Florida",
    "13":"Georgia","15":"Hawaii","16":"Idaho","17":"Illinois","18":"Indiana",
    "19":"Iowa","20":"Kansas","21":"Kentucky","22":"Louisiana","23":"Maine",
    "24":"Maryland","25":"Massachusetts","26":"Michigan","27":"Minnesota",
    "28":"Mississippi","29":"Missouri","30":"Montana","31":"Nebraska","32":"Nevada",
    "33":"New Hampshire","34":"New Jersey","35":"New Mexico","36":"New York",
    "37":"North Carolina","38":"North Dakota","39":"Ohio","40":"Oklahoma",
    "41":"Oregon","42":"Pennsylvania","44":"Rhode Island","45":"South Carolina",
    "46":"South Dakota","47":"Tennessee","48":"Texas","49":"Utah","50":"Vermont",
    "51":"Virginia","53":"Washington","54":"West Virginia","55":"Wisconsin","56":"Wyoming",
}

SECTOR_LABELS = {
    "11":"Agriculture","21":"Mining","22":"Utilities","23":"Construction",
    "31":"Mfg-food","32":"Mfg-chem","33":"Mfg-metals","42":"Wholesale",
    "44":"Retail-gen","45":"Retail-spec","48":"Transportation","49":"Warehousing",
    "51":"Information","52":"Finance","53":"Real estate","54":"Prof services",
    "55":"Management","56":"Admin/waste","61":"Education","62":"Healthcare",
    "71":"Arts/entertain","72":"Accommodation/food","81":"Other services","92":"Public admin",
}


def load_and_compute(min_obs: int = 5) -> pd.DataFrame:
    """Load QWI data and compute seasonal indexes for state x NAICS6 cells."""
    print("Loading combined QWI parquet...")
    df = pd.read_parquet(QWI_RAW / "qwi_naics6_all.parquet")

    for c in ["Sep", "Emp", "EmpEnd"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    denom = "EmpEnd" if "EmpEnd" in df.columns else "Emp"
    df["sep_rate"] = df["Sep"] / df[denom]
    df.loc[df["sep_rate"] <= 0, "sep_rate"] = np.nan
    df.loc[df["sep_rate"] >= 1, "sep_rate"] = np.nan
    print(f"Loaded: {len(df):,} rows, {df['state'].nunique()} states, {df['industry'].nunique()} industries")

    print("Computing state x industry seasonal index (all quarters)...")
    records = []
    for (state, ind), grp in df.groupby(["state", "industry"]):
        pivot = grp.pivot_table(index="year", columns="quarter", values="sep_rate", aggfunc="mean")

        excesses = {}
        record = {
            "state": state,
            "naics_code": ind,
            "sector_2d": str(ind)[:2],
            "naics_level": len(str(ind)),
            "n_years": len(grp["year"].unique()),
        }

        for q in [1, 2, 3, 4]:
            q_prev = 4 if q == 1 else q - 1
            q_next = 1 if q == 4 else q + 1
            record[f"sep_rate_Q{q}"] = pivot[q].mean() if q in pivot.columns else np.nan

            if q not in pivot.columns or q_prev not in pivot.columns or q_next not in pivot.columns:
                record[f"excessQ{q}"] = np.nan
                record[f"n_excessQ{q}"] = 0
                continue
            if q == 1:
                exr = (pivot[1] - (pivot[4].shift(1) + pivot[2]) / 2).dropna()
            elif q == 4:
                exr = (pivot[4] - (pivot[3] + pivot[1].shift(-1)) / 2).dropna()
            else:
                exr = (pivot[q] - (pivot[q - 1] + pivot[q + 1]) / 2).dropna()

            record[f"excessQ{q}"] = exr.mean() if len(exr) >= min_obs else np.nan
            record[f"n_excessQ{q}"] = len(exr)
            if len(exr) >= min_obs:
                excesses[q] = (exr.mean(), len(exr))

        if not excesses:
            continue
        peak_q   = max(excesses, key=lambda q: excesses[q][0])
        peak_val = excesses[peak_q][0]
        peak_n   = excesses[peak_q][1]
        amplitude = max(v for v, _ in excesses.values()) - min(v for v, _ in excesses.values())

        record.update({
            "seasonal_amplitude": amplitude,
            "peak_quarter": peak_q,
            "peak_excess": peak_val,
            "n_excess_obs": peak_n,
        })
        records.append(record)

    si = pd.DataFrame(records)
    p99 = si["seasonal_amplitude"].quantile(0.99)
    si["seasonal_index"] = (si["seasonal_amplitude"] / p99).clip(0, 1)
    si["state_name"] = si["state"].map(STATE_NAMES)
    si["sector_label"] = si["sector_2d"].map(SECTOR_LABELS)
    si = si.sort_values(["state_name", "seasonal_index"], ascending=[True, False])
    print(f"Computed {len(si):,} state x industry cells (>={min_obs} obs each)")
    return si


def save_state_naics_index(si: pd.DataFrame) -> Path:
    """Save the reusable state x NAICS6 seasonal index for firm lookup."""
    QWI_CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = QWI_CLEAN / "seasonal_index_naics6_by_state.csv"

    cols = [
        "state", "state_name", "naics_code", "naics_level", "sector_2d", "sector_label",
        "n_years",
        "excessQ1", "n_excessQ1", "excessQ2", "n_excessQ2",
        "excessQ3", "n_excessQ3", "excessQ4", "n_excessQ4",
        "sep_rate_Q1", "sep_rate_Q2", "sep_rate_Q3", "sep_rate_Q4",
        "seasonal_amplitude", "peak_quarter", "peak_excess", "n_excess_obs",
        "seasonal_index",
    ]
    si[cols].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def make_figures(si: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    state_avg = si.groupby("state_name")["peak_excess"].mean().sort_values(ascending=True)
    sector_var = (
        si.groupby("sector_label")["peak_excess"]
        .agg(["mean", "std"])
        .sort_values("std", ascending=True)
    )

    # ── Fig 1: Mean state seasonality ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 10))
    med = state_avg.median()
    colors = ["#d62728" if v > med else "#1f77b4" for v in state_avg]
    ax.barh(state_avg.index, state_avg.values * 100, color=colors)
    ax.axvline(med * 100, color="black", lw=0.8, ls="--", label=f"Median ({med*100:.2f} p.p.)")
    ax.set_xlabel("Mean peak quarter excess separation rate (p.p.)")
    ax.set_title("State-level seasonality\n(mean peak excess across 6-digit NAICS industries)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_state_mean_seasonality.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_state_mean_seasonality.pdf")

    # ── Fig 2: Sector geographic variation ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(sector_var.index, sector_var["std"] * 100, color="steelblue", label="Cross-state SD")
    ax.barh(sector_var.index, sector_var["mean"] * 100, color="orange", alpha=0.7, label="National mean")
    ax.set_xlabel("Peak quarter excess separation rate (p.p.)")
    ax.set_title("Geographic variation in seasonality by sector\n(cross-state standard deviation vs. national mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_sector_variation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_sector_variation.pdf")

    # ── Fig 3: Construction and Agriculture by state ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, sector_code, title in zip(axes, ["23", "11"], ["Construction", "Agriculture"]):
        sub = (
            si[si["sector_2d"] == sector_code]
            .groupby("state_name")["peak_excess"]
            .mean()
            .sort_values(ascending=True)
        )
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sub)))
        ax.barh(sub.index, sub.values * 100, color=colors)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Peak quarter excess separation rate (p.p.)", fontsize=9)
        ax.set_title(f"{title}: peak excess by state", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_construction_agriculture_by_state.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_construction_agriculture_by_state.pdf")

    # ── Fig 4: Heatmap — state x sector ───────────────────────────────────────
    top_sectors = si.groupby("sector_label")["peak_excess"].std().nlargest(8).index.tolist()
    state_order = (
        si.groupby("state_name")["peak_excess"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    heat = (
        si[si["sector_label"].isin(top_sectors)]
        .pivot_table(index="state_name", columns="sector_label", values="peak_excess", aggfunc="mean")
        .reindex(state_order)
    )
    vmax = float(heat.abs().quantile(0.95).max()) * 100

    fig, ax = plt.subplots(figsize=(10, 13))
    im = ax.imshow(heat.values * 100, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="Peak excess sep. rate (p.p.)")
    ax.set_title(
        "Peak excess separation rate: state x sector\n"
        "(states sorted by overall seasonality; top 8 sectors by geographic variation)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_heatmap_state_sector.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_heatmap_state_sector.pdf")

    return state_avg, sector_var


def print_key_numbers(si: pd.DataFrame, state_avg: pd.Series):
    print("\n=== KEY NUMBERS ===")
    print(f"Most seasonal:  {state_avg.idxmax()} ({state_avg.max()*100:.2f} p.p.)")
    print(f"Least seasonal: {state_avg.idxmin()} ({state_avg.min()*100:.2f} p.p.)")
    print(f"Ratio max/min:  {state_avg.max()/state_avg.min():.1f}x")

    constr = si[si["sector_2d"] == "23"].groupby("state_name")["peak_excess"].mean()
    mn = constr.get("Minnesota", float("nan"))
    fl = constr.get("Florida",   float("nan"))
    nd = constr.get("North Dakota", float("nan"))
    print(f"\nConstruction peak excess: MN={mn*100:.2f}, FL={fl*100:.2f}, ND={nd*100:.2f} p.p.")
    print("Construction top 5:")
    print(constr.nlargest(5).mul(100).round(2).to_string())
    print("Construction bottom 5:")
    print(constr.nsmallest(5).mul(100).round(2).to_string())

    agr = si[si["sector_2d"] == "11"].groupby("state_name")["peak_excess"].mean()
    print(f"\nAgriculture top 5:")
    print(agr.nlargest(5).mul(100).round(2).to_string())

    # Save state_avg table for Beamer
    out = si.groupby(["state_name","sector_label"])["peak_excess"].mean().reset_index()
    out.to_csv(str(TABLES_DIR / "geo_state_sector_peak_excess.csv"), index=False)
    state_avg.mul(100).round(4).rename("mean_peak_excess_pp").to_csv(
        str(TABLES_DIR / "geo_state_avg_seasonality.csv")
    )
    print("\nTables saved.")


if __name__ == "__main__":
    si = load_and_compute(min_obs=5)
    save_state_naics_index(si)
    state_avg, sector_var = make_figures(si)
    print_key_numbers(si, state_avg)
