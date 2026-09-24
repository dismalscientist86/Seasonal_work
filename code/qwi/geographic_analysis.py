"""
geographic_analysis.py — State-level geographic variation in the QWI seasonal index.

Computes a reusable state x NAICS6 seasonal index, then produces:
  1. Bar chart: mean state seasonality (across industries)
  2. Bar chart: sector geographic variation (cross-state SD)
  3. Two-panel: construction and agriculture by state
  4. State tile map: mean state seasonality
  5. Heatmap: state x sector for top variable sectors

Figures 1, 1b, 2, and 4 (the state-level "how seasonal is this state overall"
figures) exclude agriculture (NAICS sector 11) as of 2026-09 -- QWI/LEHD UI
wage-record coverage for agricultural employment is historically weaker/
partial relative to other industries (state-level UI exemptions for small
ag employers; later coverage extension in several states), so agriculture's
QWI-covered establishments may be a non-representative (more seasonal)
slice of the true agricultural workforce. See agriculture_exclusion_check.py
for the full sensitivity analysis: state-level rankings barely move either
way (Pearson r=0.991 vs. including agriculture), so this exclusion doesn't
rest on a knife's edge, but it is the more defensible headline number. Fig 3
(construction and agriculture by state) still uses the full sample, since
it is explicitly and only about agriculture's own within-sector variation.

Outputs:
  data/qwi_clean/seasonal_index_naics6_by_state.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import QWI_RAW, QWI_CLEAN, FIGURES_DIR, TABLES_DIR, MIN_EXCESS_OBS, STATE_NAMES, XWALK
from excess_utils import cyclical_excess_by_quarter

STATE_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","DC":"DC","Florida":"FL",
    "Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN",
    "Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME",
    "Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}

STATE_TILE_POSITIONS = {
    "AK": (0, 6), "ME": (11, 0), "VT": (10, 1), "NH": (11, 1),
    "WA": (1, 1), "ID": (2, 2), "MT": (3, 2), "ND": (4, 2), "MN": (5, 2),
    "IL": (6, 2), "WI": (6, 1), "MI": (7, 1), "NY": (9, 2),
    "MA": (10, 2), "RI": (11, 2), "OR": (1, 2), "NV": (2, 3),
    "WY": (3, 3), "SD": (4, 3), "IA": (5, 3), "IN": (6, 3),
    "OH": (7, 3), "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3),
    "CA": (1, 4), "UT": (2, 4), "CO": (3, 4), "NE": (4, 4),
    "MO": (5, 4), "KY": (6, 4), "WV": (7, 4), "VA": (8, 4),
    "MD": (9, 4), "DE": (10, 4), "AZ": (2, 5), "NM": (3, 5),
    "KS": (4, 5), "AR": (5, 5), "TN": (6, 5), "NC": (7, 5),
    "SC": (8, 5), "DC": (9, 5), "OK": (4, 6), "LA": (5, 6),
    "MS": (6, 6), "AL": (7, 6), "GA": (8, 6), "HI": (1, 7),
    "TX": (4, 7), "FL": (9, 7),
}

SECTOR_LABELS = {
    "11":"Agriculture","21":"Mining","22":"Utilities","23":"Construction",
    "31":"Mfg-food","32":"Mfg-chem","33":"Mfg-metals","42":"Wholesale",
    "44":"Retail-gen","45":"Retail-spec","48":"Transportation","49":"Warehousing",
    "51":"Information","52":"Finance","53":"Real estate","54":"Prof services",
    "55":"Management","56":"Admin/waste","61":"Education","62":"Healthcare",
    "71":"Arts/entertain","72":"Accommodation/food","81":"Other services","92":"Public admin",
}


def _excess_by_quarter(pivot: pd.DataFrame, min_obs: int, prefix: str) -> tuple[dict, dict]:
    """
    Shared cyclical-neighbor excess computation for one (state, industry)
    pivot table (year x quarter). Used for both the flow (sep_rate) and
    stock (emp_share) measures so the two are computed identically.

    Thin wrapper: the actual computation now lives in excess_utils.py, shared
    with seasonal_index.py and earnings_index.py (this was the first, most
    general implementation, so the others were consolidated onto it rather
    than the reverse). Kept as a local wrapper so call sites below don't
    need to change.
    """
    return cyclical_excess_by_quarter(pivot, min_obs, prefix)


def load_and_compute(min_obs: int = MIN_EXCESS_OBS) -> pd.DataFrame:
    """
    Load QWI data and compute both the flow (separation-based) and stock
    (employment-based) seasonal indexes for state x NAICS6 cells, so the two
    can be compared directly (see compare_flow_vs_stock() in seasonal_index.py).
    """
    combined_path = QWI_RAW / "qwi_naics6_all.parquet"
    if combined_path.exists():
        print("Loading combined QWI parquet...")
        df = pd.read_parquet(combined_path)
    else:
        # Fall back to concatenating per-state files (mirrors load_qwi() in
        # seasonal_index.py) rather than failing outright if the combined
        # file hasn't been (re)built yet.
        state_files = sorted(QWI_RAW.glob("qwi_state_*.parquet"))
        if not state_files:
            raise FileNotFoundError(
                f"Neither {combined_path} nor any qwi_state_*.parquet files "
                f"found in {QWI_RAW}. Run fetch_qwi.py first."
            )
        print(f"Combined file missing; loading {len(state_files)} per-state files...")
        df = pd.concat([pd.read_parquet(f) for f in state_files], ignore_index=True)

    # Per-state files only have the raw "time" string (e.g. "2010-Q1") — the
    # year/quarter split normally happens once, on the combined file, inside
    # fetch_qwi(). Derive it here too so this fallback path works standalone.
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = df["time"].str[:4].astype(int)
        df["quarter"] = df["time"].str[-1].astype(int)

    for c in ["Sep", "Emp", "EmpEnd"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    denom = "EmpEnd" if "EmpEnd" in df.columns else "Emp"
    df["sep_rate"] = df["Sep"] / df[denom]
    df.loc[df["sep_rate"] <= 0, "sep_rate"] = np.nan
    df.loc[df["sep_rate"] >= 1, "sep_rate"] = np.nan

    # Employment share: EmpEnd normalized to its own (state, industry, year)
    # mean, mirroring compute_employment_shares() in seasonal_index.py but
    # keyed on state as well so it can be computed per state x industry cell.
    df["year_state_ind_mean_emp"] = df.groupby(["state", "industry", "year"])[denom].transform("mean")
    df["emp_share"] = df[denom] / df["year_state_ind_mean_emp"]

    print(f"Loaded: {len(df):,} rows, {df['state'].nunique()} states, {df['industry'].nunique()} industries")

    print("Computing state x industry seasonal index (flow + stock, all quarters)...")
    records = []
    for (state, ind), grp in df.groupby(["state", "industry"]):
        pivot = grp.pivot_table(index="year", columns="quarter", values="sep_rate", aggfunc="mean")
        emp_pivot = grp.pivot_table(index="year", columns="quarter", values="emp_share", aggfunc="mean")

        record = {
            "state": state,
            "naics_code": ind,
            "sector_2d": str(ind)[:2],
            "naics_level": len(str(ind)),
            "n_years": len(grp["year"].unique()),
        }
        for q in [1, 2, 3, 4]:
            record[f"sep_rate_Q{q}"] = pivot[q].mean() if q in pivot.columns else np.nan
            record[f"emp_share_Q{q}"] = emp_pivot[q].mean() if q in emp_pivot.columns else np.nan

        flow_fields, flow_excesses = _excess_by_quarter(pivot, min_obs, "excess")
        stock_fields, stock_excesses = _excess_by_quarter(emp_pivot, min_obs, "emp_excess")
        record.update(flow_fields)
        record.update(stock_fields)

        if not flow_excesses and not stock_excesses:
            continue

        # A meaningful amplitude needs >=2 valid quarters -- with only 1,
        # max-min collapses to exactly 0, a data-thinness artifact rather
        # than genuine flatness (see build_seasonal_index() in
        # seasonal_index.py for the national-level version of this fix).
        if len(flow_excesses) >= 2:
            peak_q = max(flow_excesses, key=lambda q: flow_excesses[q][0])
            record.update({
                "seasonal_amplitude": max(v for v, _ in flow_excesses.values()) - min(v for v, _ in flow_excesses.values()),
                "peak_quarter": peak_q,
                "peak_excess": flow_excesses[peak_q][0],
                "n_excess_obs": flow_excesses[peak_q][1],
            })
        else:
            record.update({"seasonal_amplitude": np.nan, "peak_quarter": pd.NA,
                           "peak_excess": np.nan, "n_excess_obs": pd.NA})

        if len(stock_excesses) >= 2:
            peak_q_emp = max(stock_excesses, key=lambda q: stock_excesses[q][0])
            record.update({
                "emp_seasonal_amplitude": max(v for v, _ in stock_excesses.values()) - min(v for v, _ in stock_excesses.values()),
                "peak_quarter_emp": peak_q_emp,
                "peak_excess_emp": stock_excesses[peak_q_emp][0],
                "n_emp_excess_obs": stock_excesses[peak_q_emp][1],
            })
        else:
            record.update({"emp_seasonal_amplitude": np.nan, "peak_quarter_emp": pd.NA,
                           "peak_excess_emp": np.nan, "n_emp_excess_obs": pd.NA})

        records.append(record)

    si = pd.DataFrame(records)
    p99 = si["seasonal_amplitude"].quantile(0.99)
    si["seasonal_index"] = (si["seasonal_amplitude"] / p99).clip(0, 1)
    p99_emp = si["emp_seasonal_amplitude"].quantile(0.99)
    si["seasonal_index_emp"] = (si["emp_seasonal_amplitude"] / p99_emp).clip(0, 1)
    si["state_name"] = si["state"].map(STATE_NAMES)
    si["sector_label"] = si["sector_2d"].map(SECTOR_LABELS)
    si = si.sort_values(["state_name", "seasonal_index"], ascending=[True, False])
    print(f"Computed {len(si):,} state x industry cells (>={min_obs} obs each)")
    return si


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


def save_state_naics_index(si: pd.DataFrame, naics_titles_path: Path | None = None) -> Path:
    """Save the reusable state x NAICS6 seasonal index for firm lookup."""
    QWI_CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = QWI_CLEAN / "seasonal_index_naics6_by_state.csv"

    naics_titles_path = naics_titles_path or (XWALK / "naics6_2022_titles.csv")
    si = add_naics_titles(si, load_naics_titles(naics_titles_path))

    cols = [
        "state", "state_name", "naics_code", "naics_level", "sector_2d", "sector_label",
        "n_years",
        "excessQ1", "n_excessQ1", "excessQ2", "n_excessQ2",
        "excessQ3", "n_excessQ3", "excessQ4", "n_excessQ4",
        "sep_rate_Q1", "sep_rate_Q2", "sep_rate_Q3", "sep_rate_Q4",
        "seasonal_amplitude", "peak_quarter", "peak_excess", "n_excess_obs",
        "seasonal_index",
        "emp_excessQ1", "n_emp_excessQ1", "emp_excessQ2", "n_emp_excessQ2",
        "emp_excessQ3", "n_emp_excessQ3", "emp_excessQ4", "n_emp_excessQ4",
        "emp_share_Q1", "emp_share_Q2", "emp_share_Q3", "emp_share_Q4",
        "emp_seasonal_amplitude", "peak_quarter_emp", "peak_excess_emp", "n_emp_excess_obs",
        "seasonal_index_emp",
        "national_industry_title", "sector_title", "subsector_title",
        "industry_group_title", "naics_industry_title",
    ]
    cols = [c for c in cols if c in si.columns]
    si[cols].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def make_figures(si: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Primary analysis excludes agriculture -- see module docstring.
    si_primary = si[si["sector_2d"] != "11"].copy()

    state_avg = si_primary.groupby("state_name")["peak_excess"].mean().sort_values(ascending=True)
    sector_var = (
        si_primary.groupby("sector_label")["peak_excess"]
        .agg(["mean", "std"])
        .sort_values("std", ascending=True)
    )

    # ── Fig 1: Mean state seasonality ─────────────────────────────────────────
    # Vertical bars, most-to-least seasonal left to right -- 51 states read
    # more easily this way than as a very tall horizontal ranking.
    state_avg_desc = state_avg.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    med = state_avg_desc.median()
    colors = ["#d62728" if v > med else "#1f77b4" for v in state_avg_desc]
    ax.bar(state_avg_desc.index, state_avg_desc.values * 100, color=colors)
    ax.axhline(med * 100, color="black", lw=0.8, ls="--", label=f"Median ({med*100:.2f} p.p.)")
    ax.set_ylabel("Mean peak quarter excess separation rate (p.p.)")
    ax.set_title("State-level seasonality\n"
                 "(mean peak excess across 6-digit NAICS industries, excl. agriculture)")
    ax.set_xticks(range(len(state_avg_desc)))
    ax.set_xticklabels(state_avg_desc.index, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(state_avg_desc) - 0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_state_mean_seasonality.pdf", bbox_inches="tight")
    # Also save a .png -- used directly by output/slides/lehd_seasonality.qmd
    # (revealjs needs a raster image, unlike LaTeX). Saving it here, not just
    # the .pdf, means it can't go stale the way an unreproduced manual export
    # would the next time this script's numbers change.
    fig.savefig(FIGURES_DIR / "geo_state_mean_seasonality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_state_mean_seasonality.pdf/png")

    # Fig 1b: State tile map
    tile = state_avg.rename_axis("state_name").reset_index(name="mean_peak_excess")
    tile["abbr"] = tile["state_name"].map(STATE_ABBR)
    tile["x"] = tile["abbr"].map(lambda s: STATE_TILE_POSITIONS.get(s, (np.nan, np.nan))[0])
    tile["y"] = tile["abbr"].map(lambda s: STATE_TILE_POSITIONS.get(s, (np.nan, np.nan))[1])
    tile = tile.dropna(subset=["x", "y"])

    fig, ax = plt.subplots(figsize=(11, 7.2))
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(tile["mean_peak_excess"].min(), tile["mean_peak_excess"].max())
    for _, row in tile.iterrows():
        ax.add_patch(plt.Rectangle(
            (row["x"], row["y"]), 0.9, 0.9,
            facecolor=cmap(norm(row["mean_peak_excess"])),
            edgecolor="white", linewidth=1.5,
        ))
        ax.text(row["x"] + 0.45, row["y"] + 0.33, row["abbr"],
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(row["x"] + 0.45, row["y"] + 0.62, f"{row['mean_peak_excess']*100:.1f}",
                ha="center", va="center", fontsize=7)

    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(8.1, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Mean peak excess (p.p.)")
    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x*100:.1f}"))
    ax.set_title(
        "State-level seasonality\n"
        "Mean peak excess separation rate across 6-digit NAICS industries (excl. agriculture)",
        loc="left", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_state_mean_tile_map.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "geo_state_mean_tile_map.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_state_mean_tile_map.png/pdf")

    # ── Fig 2: Sector geographic variation ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(sector_var.index, sector_var["std"] * 100, color="steelblue", label="Cross-state SD")
    ax.barh(sector_var.index, sector_var["mean"] * 100, color="orange", alpha=0.7, label="National mean")
    ax.set_xlabel("Peak quarter excess separation rate (p.p.)")
    ax.set_title("Geographic variation in seasonality by sector\n"
                 "(cross-state standard deviation vs. national mean, excl. agriculture)")
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
    top_sectors = si_primary.groupby("sector_label")["peak_excess"].std().nlargest(8).index.tolist()
    state_order = (
        si_primary.groupby("state_name")["peak_excess"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    heat = (
        si_primary[si_primary["sector_label"].isin(top_sectors)]
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
        "(states sorted by overall seasonality; top 8 sectors by geographic "
        "variation, excl. agriculture)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geo_heatmap_state_sector.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: geo_heatmap_state_sector.pdf")

    return state_avg, sector_var


def print_key_numbers(si: pd.DataFrame, state_avg: pd.Series):
    print("\n=== KEY NUMBERS (excl. agriculture -- see agriculture_exclusion_check.py "
          "for the with-agriculture sensitivity numbers) ===")
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
    import argparse

    parser = argparse.ArgumentParser(description="State-level geographic variation in the QWI seasonal index")
    parser.add_argument(
        "--naics-titles", type=Path, default=None,
        help="Optional cleaned NAICS6 title lookup created by clean_naics_xwalk.py "
             f"(default: {XWALK / 'naics6_2022_titles.csv'}).",
    )
    args = parser.parse_args()

    si = load_and_compute()
    save_state_naics_index(si, naics_titles_path=args.naics_titles)
    state_avg, sector_var = make_figures(si)
    print_key_numbers(si, state_avg)
