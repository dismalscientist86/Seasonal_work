"""
recurrence.py — Python replication of Coglianese & Price's excess recurrence analysis.

Replicates the core results from recurrence.do using the pre-cleaned CPS and SIPP
separators datasets.

Key estimand:
    Excess Recurrence = P(sep at k=12) - [P(sep at k=11) + P(sep at k=13)] / 2

Usage:
    python code/replication/recurrence.py
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from code.config import (
    CPS_HORIZONS, SIPP_HORIZONS, BASE_MONTH_K,
    OUTPUT_DIR, FIGURES_DIR, TABLES_DIR,
)
from code.replication.load_data import load_cps_separators, load_sipp_separators

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Industry labels (matches CP's 15-category scheme) ────────────────────────

IND_LABELS = {
    0:  "Missing",
    1:  "Agriculture/fishing/forestry",
    2:  "Mining",
    3:  "Construction",
    4:  "Manufacturing",
    5:  "Transport./comm./utilities",
    6:  "Wholesale",
    7:  "Retail",
    8:  "FIRE",
    9:  "Business services",
    10: "Personal services",
    11: "Entertainment/recreation",
    12: "Healthcare",
    13: "Educational services",
    14: "Social/prof. services",
    15: "Public admin./armed forces",
}


# ── Core regression ───────────────────────────────────────────────────────────

def estimate_recurrence(
    df: pd.DataFrame,
    outcome: str = "sep",
    horizons: list[int] | None = None,
    controls: list[str] | None = None,
    weight_col: str = "wtfnl",
    cluster_col: str = "pid",
    base_k: int = BASE_MONTH_K,
    no_const: bool = False,
) -> dict:
    """
    Estimate a separation-profile regression and compute excess recurrence.

    The model is:
        outcome ~ Σ_{k ∈ horizons} β_k · I(horizon=k)  +  controls  [+ const]

    with analytic weights (weight_col) and clustered SEs (cluster_col).

    Excess recurrence = β_{base_k+1} - (β_{base_k} + β_{base_k+2}) / 2
    where base_k corresponds to the 12-month horizon (by default k=12 with
    adjacent counterfactual at k=11 and k=13).

    For CPS preferred spec:  no_const=True, controls=['weeks_elapsed']
    For SIPP preferred spec: no_const=False, controls=['five','first_seam','later_seam']
      and horizons excludes base_k (k=11 omitted, absorbed into constant).

    Parameters
    ----------
    df          : DataFrame with outcome, horizon indicator 'k', weight and cluster cols.
    outcome     : Dependent variable (default 'sep').
    horizons    : Horizons to include as dummies. If None, use all unique values of df['k'].
    controls    : Extra RHS variables.
    weight_col  : Column for analytic weights.
    cluster_col : Column for clustering.
    base_k      : The *counterfactual base month* for normalization (k=11 in the paper).
                  When no_const=False, base_k is omitted from the dummies and serves as
                  the reference category; β_{base_k} is set to 0 in excess recurrence.
    no_const    : If True, suppress the intercept (CPS style: 'nocons' in Stata).

    Returns
    -------
    dict with keys:
        params, bse, cov       : coefficient estimates, SEs, covariance matrix
        excess_recurrence      : point estimate (float)
        se_excess_recurrence   : standard error via delta method (float)
        t_excess               : t-statistic
        horizon_profile        : DataFrame of (k, beta, se, lo95, hi95)
        result                 : raw statsmodels RegressionResults object
    """
    horizons = horizons or sorted(df["k"].unique())

    # When using a constant, omit base_k from the dummies (it's the reference)
    dummy_ks = horizons if no_const else [k for k in horizons if k != base_k]

    data = df[df["k"].isin(horizons)].copy()

    # Create horizon dummies
    for k in dummy_ks:
        data[f"_k{k}"] = (data["k"] == k).astype(float)

    # Build formula
    k_terms = " + ".join(f"_k{k}" for k in dummy_ks)
    ctrl_terms = (" + " + " + ".join(controls)) if controls else ""
    intercept = "" if no_const else ""   # statsmodels includes intercept by default
    formula = f"{outcome} ~ {k_terms}{ctrl_terms}" + (" - 1" if no_const else "")

    # Drop rows with NaN in any variable used by the formula so that
    # the cluster array length matches the regression design matrix exactly.
    all_vars = [outcome, cluster_col, weight_col] + list(dummy_ks and []) + (controls or [])
    dummy_var_names = [f"_k{k}" for k in dummy_ks]
    all_vars = [outcome, cluster_col, weight_col] + dummy_var_names + (controls or [])
    data = data.dropna(subset=[c for c in all_vars if c in data.columns]).copy()

    # statsmodels needs a 0-indexed non-negative integer array for np.bincount
    cluster_ids = pd.factorize(data[cluster_col])[0]
    wls = smf.wls(formula, data=data, weights=data[weight_col])
    result = wls.fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_ids},
    )

    params = result.params
    cov    = result.cov_params()

    # Helper: get coefficient for horizon k (0 if it's the omitted base)
    def _b(k):
        key = f"_k{k}"
        return params.get(key, 0.0)

    def _cov(k1, k2):
        k1k = f"_k{k1}"
        k2k = f"_k{k2}"
        if k1k not in cov.index or k2k not in cov.index:
            return 0.0
        return cov.loc[k1k, k2k]

    # Excess recurrence: β_12 - (β_11 + β_13) / 2
    h  = SIPP_HORIZONS[-7]  # = 12 (annual horizon)
    h_lo = h - 1            # = 11
    h_hi = h + 1            # = 13

    exr = _b(h) - (_b(h_lo) + _b(h_hi)) / 2

    # Delta method variance: Var(β_h - (β_{h-1} + β_{h+1})/2)
    # gradient = [-1/2, 1, -1/2] over (β_{h-1}, β_h, β_{h+1})
    var_exr = (
          _cov(h,    h)
        + 0.25 * _cov(h_lo, h_lo)
        + 0.25 * _cov(h_hi, h_hi)
        - _cov(h,    h_lo)
        - _cov(h,    h_hi)
        + 0.5  * _cov(h_lo, h_hi)
    )
    se_exr = np.sqrt(max(var_exr, 0))
    t_exr  = exr / se_exr if se_exr > 0 else np.nan

    # Separation profile table
    profile_rows = []
    for k in horizons:
        b  = _b(k)
        se = result.bse.get(f"_k{k}", 0.0)
        profile_rows.append({
            "k": k,
            "beta": b,
            "se":   se,
            "lo95": b - 1.96 * se,
            "hi95": b + 1.96 * se,
        })
    profile = pd.DataFrame(profile_rows)

    return {
        "params":               params,
        "bse":                  result.bse,
        "cov":                  cov,
        "excess_recurrence":    exr,
        "se_excess_recurrence": se_exr,
        "t_excess":             t_exr,
        "horizon_profile":      profile,
        "result":               result,
        "nobs":                 result.nobs,
    }


# ── Convenience wrappers matching the paper's two preferred specs ─────────────

def estimate_cps(df: pd.DataFrame, **kwargs) -> dict:
    """
    CPS preferred specification:
        sep ~ k10 + k11 + k12 + k13 + k14 + weeks_elapsed  (no intercept)
    """
    return estimate_recurrence(
        df,
        horizons=CPS_HORIZONS,
        controls=["weeks_elapsed"],
        no_const=True,
        **kwargs,
    )


def estimate_sipp(df: pd.DataFrame, **kwargs) -> dict:
    """
    SIPP preferred specification (Appendix F.1 in paper):
        sep ~ k1 + ... + k10 + k12 + ... + k18 + five + first_seam + later_seam
              (k=11 omitted as base; intercept included)
    Gives ExR ~ 1.6 p.p.
    """
    return estimate_recurrence(
        df,
        horizons=SIPP_HORIZONS,
        controls=["five", "first_seam", "later_seam"],
        no_const=False,
        base_k=BASE_MONTH_K,
        **kwargs,
    )


def estimate_cps_raw(df: pd.DataFrame, **kwargs) -> dict:
    """
    CPS raw specification (Figure 2 in paper):
        sep ~ k10 + k11 + k12 + k13 + k14  (no intercept, no controls)
    Gives ExR ~ 1.4 p.p.
    """
    return estimate_recurrence(
        df,
        horizons=CPS_HORIZONS,
        controls=[],
        no_const=True,
        **kwargs,
    )


def estimate_sipp_raw(df: pd.DataFrame, **kwargs) -> dict:
    """
    SIPP raw specification (Figure 2 in paper):
        sep ~ k1 + ... + k18  (all dummies, no intercept, no controls)
    Gives ExR ~ 2.0 p.p.
    """
    return estimate_recurrence(
        df,
        horizons=SIPP_HORIZONS,
        controls=[],
        no_const=True,
        **kwargs,
    )


# ── Stratified analyses (by industry, month, demographics) ───────────────────

def run_by_industry(
    df: pd.DataFrame,
    dataset: str = "cps",
    ind_col: str = "base_ind",
) -> pd.DataFrame:
    """
    Run the recurrence regression separately for each 1-digit industry.

    Returns a DataFrame with columns: ind, label, excess_recurrence, se, t, n.
    """
    estimator = estimate_cps if dataset == "cps" else estimate_sipp
    rows = []
    for j in range(1, 16):
        sub = df[df[ind_col] == j]
        if len(sub) < 200:
            continue
        try:
            res = estimator(sub)
            rows.append({
                "ind":   j,
                "label": IND_LABELS.get(j, str(j)),
                "excess_recurrence": 100 * res["excess_recurrence"],
                "se":    100 * res["se_excess_recurrence"],
                "t":     res["t_excess"],
                "n":     int(res["nobs"]),
            })
        except Exception as e:
            print(f"  Industry {j} failed: {e}")
    return pd.DataFrame(rows).sort_values("excess_recurrence", ascending=False)


def run_by_month(
    df: pd.DataFrame,
    dataset: str = "cps",
) -> pd.DataFrame:
    """
    Run the recurrence regression separately for each calendar month.

    Replicates Stata's approach: filter the entire panel on `month == m`,
    then run the full event-study regression on that subset.
    At k=12, month==m means the focal separation was also in month m (since
    the worker is observed exactly 12 months later), so this correctly
    stratifies by month of the focal separation.
    """
    estimator = estimate_cps if dataset == "cps" else estimate_sipp

    rows = []
    for m in range(1, 13):
        sub = df[df["month"] == m]
        if len(sub) < 200:
            continue
        try:
            res = estimator(sub)
            rows.append({
                "month":             m,
                "excess_recurrence": 100 * res["excess_recurrence"],
                "se":                100 * res["se_excess_recurrence"],
                "t":                 res["t_excess"],
                "n":                 int(res["nobs"]),
            })
        except Exception as e:
            print(f"  Month {m} failed: {e}")
    return pd.DataFrame(rows).sort_values("month")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_separation_profile(
    results: dict,
    title: str = "",
    ax: plt.Axes | None = None,
    color: str = "#1f77b4",
    show_excess: bool = True,
) -> plt.Axes:
    """
    Plot the separation profile β_k against horizon k.

    Replicates Figure 1 of the paper (raw or regression-adjusted profiles).
    """
    profile = results["horizon_profile"]
    exr     = 100 * results["excess_recurrence"]
    se_exr  = 100 * results["se_excess_recurrence"]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ks = profile["k"].values
    ax.plot(ks, 100 * profile["beta"], color=color, marker="o", ms=4, lw=1.5)
    ax.fill_between(
        ks,
        100 * profile["lo95"],
        100 * profile["hi95"],
        alpha=0.15,
        color=color,
    )

    if show_excess and 11 in profile["k"].values and 13 in profile["k"].values:
        b11 = 100 * profile.loc[profile["k"] == 11, "beta"].values[0]
        b12 = 100 * profile.loc[profile["k"] == 12, "beta"].values[0]
        b13 = 100 * profile.loc[profile["k"] == 13, "beta"].values[0]
        cf  = (b11 + b13) / 2
        # Draw counterfactual dashed line
        ax.plot([11, 13], [b11, b13], "--", color="gray", lw=1)
        # Draw excess recurrence arrow
        ax.annotate(
            f"excess recurrence:\n{exr:.1f} p.p. (se {se_exr:.2f})",
            xy=(12, b12),
            xytext=(12.4, b12 + 0.5),
            color="#d62728",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
        )

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Months after initial separation")
    ax.set_ylabel("Probability of separation (%)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
    return ax


def plot_by_industry(by_ind: pd.DataFrame, title: str = "") -> plt.Figure:
    """Horizontal bar chart of excess recurrence by industry."""
    fig, ax = plt.subplots(figsize=(8, 6))
    by_ind_plot = by_ind.sort_values("excess_recurrence")
    colors = ["#d62728" if x > 0 else "#aec7e8" for x in by_ind_plot["excess_recurrence"]]
    ax.barh(by_ind_plot["label"], by_ind_plot["excess_recurrence"], color=colors)
    ax.errorbar(
        by_ind_plot["excess_recurrence"],
        by_ind_plot["label"],
        xerr=1.96 * by_ind_plot["se"],
        fmt="none",
        color="black",
        lw=0.8,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Excess recurrence (p.p.)")
    ax.set_title(title or "Excess recurrence by 1-digit industry")
    fig.tight_layout()
    return fig


# ── Main replication run ──────────────────────────────────────────────────────

def run_replication(save_outputs: bool = True) -> dict:
    """
    Run the full replication of Table 1 and key figures from CP (2025).

    Returns a dict with all results.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── CPS overall ──────────────────────────────────────────────────────────
    print("\n=== CPS: loading separators ===")
    cps = load_cps_separators(horizons=CPS_HORIZONS)
    print(f"  {cps['rid'].nunique():,} focal separations, {len(cps):,} obs")

    print("\n=== CPS: raw specification (Figure 2 in paper, target ~1.4 p.p.) ===")
    res_cps_raw = estimate_cps_raw(cps)
    exr_cps_raw = 100 * res_cps_raw["excess_recurrence"]
    se_cps_raw  = 100 * res_cps_raw["se_excess_recurrence"]
    print(f"  Excess recurrence: {exr_cps_raw:.2f} p.p.  (SE {se_cps_raw:.2f})  [paper: 1.4 (0.14)]")
    results["cps_raw"] = res_cps_raw

    print("\n=== CPS: preferred specification (Appendix F.1, target ~1.5 p.p.) ===")
    res_cps = estimate_cps(cps)
    exr_cps = 100 * res_cps["excess_recurrence"]
    se_cps  = 100 * res_cps["se_excess_recurrence"]
    print(f"  Excess recurrence: {exr_cps:.2f} p.p.  (SE {se_cps:.2f})  [paper: 1.5 (0.14)]")
    results["cps_preferred"] = res_cps

    # By industry (CPS)
    print("\n=== CPS: by 1-digit industry ===")
    cps_by_ind = run_by_industry(cps, dataset="cps")
    print(cps_by_ind[["label", "excess_recurrence", "se", "n"]].to_string(index=False))
    results["cps_by_industry"] = cps_by_ind

    # By month (CPS)
    print("\n=== CPS: by month of separation ===")
    cps_by_mth = run_by_month(cps, dataset="cps")
    print(cps_by_mth[["month", "excess_recurrence", "se", "n"]].to_string(index=False))
    results["cps_by_month"] = cps_by_mth

    # ── SIPP overall ─────────────────────────────────────────────────────────
    print("\n=== SIPP: loading separators ===")
    # The main recurrence analysis does NOT restrict to earn_sample==1
    # (that restriction is only for the income event studies and ML analysis)
    sipp = load_sipp_separators(earn_sample_only=False, horizons=SIPP_HORIZONS)
    print(f"  {sipp['rid'].nunique():,} focal separations, {len(sipp):,} obs")

    print("\n=== SIPP: raw specification (Figure 2 in paper, target ~2.0 p.p.) ===")
    res_sipp_raw = estimate_sipp_raw(sipp)
    exr_sipp_raw = 100 * res_sipp_raw["excess_recurrence"]
    se_sipp_raw  = 100 * res_sipp_raw["se_excess_recurrence"]
    print(f"  Excess recurrence: {exr_sipp_raw:.2f} p.p.  (SE {se_sipp_raw:.2f})  [paper: 2.0 (0.11)]")
    results["sipp_raw"] = res_sipp_raw

    print("\n=== SIPP: preferred specification (Appendix F.1, target ~1.6 p.p.) ===")
    res_sipp = estimate_sipp(sipp)
    exr_sipp = 100 * res_sipp["excess_recurrence"]
    se_sipp  = 100 * res_sipp["se_excess_recurrence"]
    print(f"  Excess recurrence: {exr_sipp:.2f} p.p.  (SE {se_sipp:.2f})  [paper: 1.6 (0.11)]")
    results["sipp_preferred"] = res_sipp

    # By industry (SIPP)
    print("\n=== SIPP: by 1-digit industry ===")
    sipp_by_ind = run_by_industry(sipp, dataset="sipp")
    print(sipp_by_ind[["label", "excess_recurrence", "se", "n"]].to_string(index=False))
    results["sipp_by_industry"] = sipp_by_ind

    # By month (SIPP)
    print("\n=== SIPP: by month of separation ===")
    sipp_by_mth = run_by_month(sipp, dataset="sipp")
    print(sipp_by_mth[["month", "excess_recurrence", "se", "n"]].to_string(index=False))
    results["sipp_by_month"] = sipp_by_mth

    # ── Save outputs ─────────────────────────────────────────────────────────
    if save_outputs:
        # Robustness table
        rob = pd.DataFrame([
            {"spec": "CPS: raw (Figure 2)",           "excess_recurrence": exr_cps_raw, "se": se_cps_raw,  "paper_target": "1.4 (0.14)"},
            {"spec": "CPS: preferred (Appendix F.1)", "excess_recurrence": exr_cps,     "se": se_cps,      "paper_target": "1.5 (0.14)"},
            {"spec": "SIPP: raw (Figure 2)",          "excess_recurrence": exr_sipp_raw,"se": se_sipp_raw, "paper_target": "2.0 (0.11)"},
            {"spec": "SIPP: preferred (App. F.1)",    "excess_recurrence": exr_sipp,    "se": se_sipp,     "paper_target": "1.6 (0.11)"},
        ])
        rob.to_csv(TABLES_DIR / "recurrence_overall.csv", index=False)
        cps_by_ind.to_csv(TABLES_DIR / "recurrence_by_industry_cps.csv", index=False)
        sipp_by_ind.to_csv(TABLES_DIR / "recurrence_by_industry_sipp.csv", index=False)
        cps_by_mth.to_csv(TABLES_DIR / "recurrence_by_month_cps.csv", index=False)
        sipp_by_mth.to_csv(TABLES_DIR / "recurrence_by_month_sipp.csv", index=False)

        # Separation profile figures — raw (Figure 2) and preferred (Appendix F.1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_separation_profile(res_cps_raw,  title="CPS (raw, Figure 2)",           ax=axes[0][0])
        plot_separation_profile(res_sipp_raw, title="SIPP (raw, Figure 2)",          ax=axes[0][1])
        plot_separation_profile(res_cps,      title="CPS (preferred, Appendix F.1)", ax=axes[1][0])
        plot_separation_profile(res_sipp,     title="SIPP (preferred, App. F.1)",    ax=axes[1][1])
        fig.suptitle("Separation probability profiles (Coglianese & Price replication)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "recurrence_profiles.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigures saved to {FIGURES_DIR}")

        # Industry figures
        for tag, by_ind in [("cps", cps_by_ind), ("sipp", sipp_by_ind)]:
            fig = plot_by_industry(by_ind, title=f"Excess recurrence by industry ({tag.upper()})")
            fig.savefig(FIGURES_DIR / f"recurrence_by_industry_{tag}.pdf", bbox_inches="tight")
            plt.close(fig)

    return results


if __name__ == "__main__":
    run_replication(save_outputs=True)
