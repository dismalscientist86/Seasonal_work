"""
build_events.py — Build quarterly separation and hire events from UI wage records.

Input:  UI wage records (worker_id × employer_id × quarter × earnings)
Output: panel of separation/hire events with firm and worker identifiers

A "separation" is a spell where the worker is observed at employer E in quarter t
but NOT in quarter t+1 (regardless of whether they go to another employer).

A "hire" (accession) is the reverse: present in t+1 but not in t.

We then build a separators panel: for each (worker, employer, sep_quarter),
track employment status for the following 8 quarters, to estimate excess Q4
recurrence at the firm and industry level.

Usage:
    from code.firm_level.build_events import build_separation_events
    events = build_separation_events(ui_df)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from code.config import FIRM_DATA


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ui_records(
    path: Path | str,
    worker_col: str = "worker_id",
    employer_col: str = "employer_id",
    year_col: str = "year",
    quarter_col: str = "quarter",
    earnings_col: str = "earnings",
    naics_col: str | None = "naics6",
    min_earnings: float = 1.0,
) -> pd.DataFrame:
    """
    Load UI wage records from parquet or CSV.

    Expected columns (rename using the col arguments if your file uses different names):
        worker_id   : unique person identifier (e.g., masked SSN)
        employer_id : unique firm identifier (e.g., EIN or UI account number)
        year        : calendar year
        quarter     : calendar quarter (1–4)
        earnings    : quarterly earnings (in dollars)
        naics6      : 6-digit NAICS code (optional but recommended)

    Rows with earnings below min_earnings are dropped (proxy for minimal attachment).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".gz"):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Rename to standard column names
    rename_map = {}
    for std, col in [
        ("worker_id",   worker_col),
        ("employer_id", employer_col),
        ("year",        year_col),
        ("quarter",     quarter_col),
        ("earnings",    earnings_col),
    ]:
        if col != std and col in df.columns:
            rename_map[col] = std
    if naics_col and naics_col != "naics6" and naics_col in df.columns:
        rename_map[naics_col] = "naics6"
    df = df.rename(columns=rename_map)

    # Basic cleaning
    df["earnings"] = pd.to_numeric(df["earnings"], errors="coerce")
    df = df[df["earnings"] >= min_earnings].copy()

    # Create a single time index: quarters since some base date
    df["yq"] = (df["year"].astype(int) - 2000) * 4 + df["quarter"].astype(int) - 1

    print(f"Loaded {len(df):,} UI records: "
          f"{df['worker_id'].nunique():,} workers, "
          f"{df['employer_id'].nunique():,} employers, "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


# ── Employment panel ──────────────────────────────────────────────────────────

def build_quarterly_employment(
    ui: pd.DataFrame,
    min_quarters: int = 2,
) -> pd.DataFrame:
    """
    Convert raw UI records to a balanced (worker × employer × quarter) panel.

    For each (worker, employer) pair, fills in 0 for quarters where the worker
    has no earnings at that employer, over the range they are in the dataset.

    Parameters
    ----------
    ui : Raw UI records from load_ui_records().
    min_quarters : Minimum quarters a worker must appear at an employer to be included.

    Returns
    -------
    DataFrame with columns:
        worker_id, employer_id, yq, year, quarter, employed, earnings, naics6
    """
    ui = ui.copy()
    ui["employed"] = 1

    # Determine the yq range to cover for each (worker, employer) pair
    pairs = (
        ui.groupby(["worker_id", "employer_id"])["yq"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    pairs.columns = ["worker_id", "employer_id", "yq_min", "yq_max", "n_qtrs"]
    pairs = pairs[pairs["n_qtrs"] >= min_quarters]

    # Expand to full quarterly panel for each pair
    expanded_rows = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Building panel"):
        qrange = range(int(row["yq_min"]), int(row["yq_max"]) + 1)
        for yq in qrange:
            expanded_rows.append({
                "worker_id":   row["worker_id"],
                "employer_id": row["employer_id"],
                "yq":          yq,
            })

    panel = pd.DataFrame(expanded_rows)
    panel["employed"] = 0

    # Merge in actual employment records
    panel = panel.merge(
        ui[["worker_id", "employer_id", "yq", "employed", "earnings"]],
        on=["worker_id", "employer_id", "yq"],
        how="left",
        suffixes=("", "_actual"),
    )
    panel["employed"] = panel[["employed", "employed_actual"]].max(axis=1).fillna(0).astype(int)
    if "employed_actual" in panel.columns:
        panel = panel.drop(columns=["employed_actual"])

    # Merge in NAICS code (from the most common code observed for the employer)
    if "naics6" in ui.columns:
        naics_map = (
            ui.groupby("employer_id")["naics6"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        )
        panel["naics6"] = panel["employer_id"].map(naics_map)

    # Recover year and quarter from yq
    panel["year"]    = (panel["yq"] // 4) + 2000
    panel["quarter"] = (panel["yq"] %  4) + 1

    panel = panel.sort_values(["worker_id", "employer_id", "yq"])
    return panel.reset_index(drop=True)


# ── Separation event identification ──────────────────────────────────────────

def identify_separations(
    panel: pd.DataFrame,
    employer_col: str = "employer_id",
) -> pd.DataFrame:
    """
    Flag quarterly separations and hires in the employment panel.

    A separation at (worker, employer, yq) means employed==1 in yq but employed==0 in yq+1.
    A hire (accession) means employed==0 in yq but employed==1 in yq+1.

    Returns
    -------
    panel with added columns:
        sep_next : 1 if worker separates from this employer next quarter
        hire     : 1 if this quarter is the start of an employment spell
        recall   : 1 if sep_next==1 AND the worker returns within 4 quarters
    """
    panel = panel.sort_values(["worker_id", employer_col, "yq"]).copy()

    grp = panel.groupby(["worker_id", employer_col])

    # Next-quarter employment status
    panel["emp_next"] = grp["employed"].shift(-1)

    # Separations and hires
    panel["sep_next"] = (
        (panel["employed"] == 1) & (panel["emp_next"] == 0)
    ).astype(int)
    panel["hire"] = (
        (panel["employed"] == 1) & (grp["employed"].shift(1).fillna(0) == 0)
    ).astype(int)

    return panel


def build_separation_events(
    panel: pd.DataFrame,
    max_horizon: int = 8,
) -> pd.DataFrame:
    """
    Build a separators panel: one row per (focal_sep × follow-up quarter).

    For each separation event, we record the employment status at the same
    employer for up to max_horizon quarters after the separation.

    The key outcome is sep_recurs: does the worker separate again from the
    SAME employer in quarter Q+4 (one year later)?

    Parameters
    ----------
    panel       : Employment panel from build_quarterly_employment().
    max_horizon : Number of follow-up quarters.

    Returns
    -------
    DataFrame with columns:
        worker_id, employer_id, yq_sep, year_sep, quarter_sep, naics6,
        k (horizon), employed_k, sep_at_k
    """
    panel = identify_separations(panel)

    # Extract all focal separations
    seps = panel[panel["sep_next"] == 1][
        ["worker_id", "employer_id", "yq", "year", "quarter"]
    ].copy()
    if "naics6" in panel.columns:
        naics_map = panel.drop_duplicates("employer_id").set_index("employer_id")["naics6"]
        seps["naics6"] = seps["employer_id"].map(naics_map)

    seps = seps.rename(columns={"yq": "yq_sep", "year": "year_sep", "quarter": "quarter_sep"})

    print(f"Building event panel: {len(seps):,} focal separations")

    # For each focal separation, look up employment status at k = 1 .. max_horizon
    event_rows = []
    for _, sep in tqdm(seps.iterrows(), total=len(seps), desc="Event panel"):
        wid = sep["worker_id"]
        eid = sep["employer_id"]
        yq0 = sep["yq_sep"]

        worker_emp = panel[
            (panel["worker_id"] == wid) & (panel["employer_id"] == eid)
        ].set_index("yq")[["employed", "sep_next"]]

        for k in range(1, max_horizon + 1):
            yq_k = yq0 + k
            emp_k = int(worker_emp.loc[yq_k, "employed"]) if yq_k in worker_emp.index else None
            sep_k = int(worker_emp.loc[yq_k, "sep_next"]) if yq_k in worker_emp.index else None

            event_rows.append({
                "worker_id":    wid,
                "employer_id":  eid,
                "yq_sep":       yq0,
                "year_sep":     sep["year_sep"],
                "quarter_sep":  sep["quarter_sep"],
                "naics6":       sep.get("naics6"),
                "k":            k,
                "employed_k":   emp_k,
                "sep_at_k":     sep_k,
            })

    events = pd.DataFrame(event_rows)
    events["employed_k"] = events["employed_k"].astype("Int64")
    events["sep_at_k"]   = events["sep_at_k"].astype("Int64")
    return events


# ── Faster vectorized separation event builder ────────────────────────────────

def build_separation_events_fast(
    panel: pd.DataFrame,
    max_horizon: int = 8,
) -> pd.DataFrame:
    """
    Vectorized version of build_separation_events for large datasets.

    Creates horizon dummies by shifting the employment column.
    Much faster than the row-by-row version for >100k separations.
    """
    panel = identify_separations(panel)
    panel = panel.sort_values(["worker_id", "employer_id", "yq"]).copy()

    # Create forward employment columns k=1..max_horizon
    grp = panel.groupby(["worker_id", "employer_id"])
    for k in range(1, max_horizon + 1):
        panel[f"emp_k{k}"]    = grp["employed"].shift(-k)
        panel[f"sep_at_k{k}"] = grp["sep_next"].shift(-k)

    # Keep only rows that are focal separations
    seps = panel[panel["sep_next"] == 1].copy()

    # Melt to long format
    id_vars = ["worker_id", "employer_id", "yq", "year", "quarter"]
    if "naics6" in seps.columns:
        id_vars.append("naics6")

    # Melt employed_k columns
    emp_long = seps.melt(
        id_vars=id_vars,
        value_vars=[f"emp_k{k}" for k in range(1, max_horizon + 1)],
        var_name="k_str",
        value_name="employed_k",
    )
    emp_long["k"] = emp_long["k_str"].str.extract(r"(\d+)$").astype(int)

    # Melt sep_at_k columns
    sep_long = seps.melt(
        id_vars=id_vars,
        value_vars=[f"sep_at_k{k}" for k in range(1, max_horizon + 1)],
        var_name="k_str",
        value_name="sep_at_k",
    )
    sep_long["k"] = sep_long["k_str"].str.extract(r"(\d+)$").astype(int)

    # Merge
    merge_cols = id_vars + ["k"]
    events = emp_long[merge_cols + ["employed_k"]].merge(
        sep_long[merge_cols + ["sep_at_k"]],
        on=merge_cols,
    )
    events = events.rename(columns={"yq": "yq_sep", "year": "year_sep", "quarter": "quarter_sep"})

    print(f"Event panel: {len(seps):,} separations × {max_horizon} horizons = {len(events):,} rows")
    return events


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    print("Generating synthetic UI wage records for testing …")
    rng = np.random.default_rng(42)
    n = 50_000
    synthetic = pd.DataFrame({
        "worker_id":   rng.integers(1, 5_000, n),
        "employer_id": rng.integers(1, 500, n),
        "year":        rng.integers(2010, 2020, n),
        "quarter":     rng.integers(1, 5, n),
        "earnings":    rng.exponential(10_000, n),
        "naics6":      rng.choice(["236220","238210","611110","722513","621111"], n),
    })
    synthetic = synthetic.drop_duplicates(["worker_id", "employer_id", "year", "quarter"])

    ui = load_ui_records(synthetic if isinstance(synthetic, Path) else synthetic)
    print("Done loading synthetic data.")
