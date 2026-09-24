"""
excess_utils.py — Shared "cyclical excess by quarter" computation.

Every QWI-based seasonal measure in this project — separations (flow),
employment levels (stock), earnings, and now hires — reduces to the same
calculation applied to a different underlying rate/share:

    Excess(q)_j = mean_t[ s(q,t) - (s(q-1,t) + s(q+1,t)) / 2 ]

for each quarter q in {1,2,3,4}, treating quarters cyclically (Q4's
neighbors are Q3 and Q1 of the *following* year), averaged over all years t
with non-missing adjacent quarters, and reported only when at least
`min_obs` years contribute (thin cells are set to NaN rather than reported
as a spuriously precise near-zero).

Before this module, that ~20-line block was copy-pasted four times:
seasonal_index.py's compute_excess_recurrence_by_quarter() (separations) and
compute_employment_excess_by_quarter() (employment shares),
earnings_index.py's compute_earnings_excess_by_quarter(), and
geographic_analysis.py's _excess_by_quarter() (state x industry, both
flow and stock). geographic_analysis.py's version was already the most
general (parameterized by a `prefix` for output column naming) — this module
just lifts that version out so every caller shares one implementation,
instead of one script's fix having to be manually repeated in the other three.

Each caller still owns its own groupby-loop wrapper (looping over industry,
or state x industry) and its own "build a 0-1 index from the excess table"
step, since those differ enough across flow/stock/earnings/hire (different
output column names, earnings' extra trough_quarter/trough_excess columns,
etc.) that forcing them into one generic function would cost more in
indirection than it saves in duplication.

`cyclical_excess_by_period()` generalizes the same computation to any cycle
length (quarters, or months for QCEW's monthly employment levels in
qcew/monthly_seasonal_index.py); `cyclical_excess_by_quarter()` is now a
thin wrapper over it, kept for backwards-compatible column naming
(verified byte-identical to the pre-generalization version).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cyclical_excess_by_period(
    pivot: pd.DataFrame, min_obs: int, prefix: str, n_periods: int, label: str
) -> tuple[dict, dict]:
    """
    Cyclical-neighbor excess computation for one (year x period) pivot
    table — one industry, or one (state, industry) cell, depending on the
    caller's groupby level. `n_periods` is 4 for quarters, 12 for months.

    Parameters
    ----------
    pivot     : DataFrame indexed by year, with integer columns 1..n_periods
                holding the rate/share value being tested for seasonality
                (e.g. sep_rate, emp_share, earnings_share, hire_rate).
    min_obs   : Minimum number of year-observations required for a period's
                excess estimate to be reported; fewer than this and it's NaN.
    prefix    : Used to name the output dict keys, matching each caller's
                existing convention — e.g. prefix="excess", label="Q" ->
                "excessQ1".."excessQ4"/"n_excessQ1".."n_excessQ4".
    n_periods : Cycle length (4 for quarters, 12 for months).
    label     : Single-letter tag distinguishing the period type in column
                names ("Q" for quarter, "M" for month).

    Returns
    -------
    record   : dict with f"{prefix}{label}{p}" (mean excess, or NaN) and
               f"n_{prefix}{label}{p}" (contributing year count) for
               p in 1..n_periods.
    excesses : dict of {period: (mean_excess, n_obs)}, but only for periods
               that met `min_obs` — convenience for callers that need the
               peak/trough period without re-filtering `record`.
    """
    record = {}
    excesses = {}
    for p in range(1, n_periods + 1):
        p_prev = n_periods if p == 1 else p - 1
        p_next = 1 if p == n_periods else p + 1

        if p not in pivot.columns or p_prev not in pivot.columns or p_next not in pivot.columns:
            record[f"{prefix}{label}{p}"] = np.nan
            record[f"n_{prefix}{label}{p}"] = 0
            continue

        if p == 1:
            exr = (pivot[1] - (pivot[n_periods].shift(1) + pivot[2]) / 2).dropna()
        elif p == n_periods:
            exr = (pivot[n_periods] - (pivot[n_periods - 1] + pivot[1].shift(-1)) / 2).dropna()
        else:
            exr = (pivot[p] - (pivot[p - 1] + pivot[p + 1]) / 2).dropna()

        record[f"{prefix}{label}{p}"] = exr.mean() if len(exr) >= min_obs else np.nan
        record[f"n_{prefix}{label}{p}"] = len(exr)
        if len(exr) >= min_obs:
            excesses[p] = (exr.mean(), len(exr))

    return record, excesses


def cyclical_excess_by_quarter(
    pivot: pd.DataFrame, min_obs: int, prefix: str
) -> tuple[dict, dict]:
    """Quarterly special case of cyclical_excess_by_period() -- see there
    for parameter docs. Column names: f"{prefix}Q{q}" / f"n_{prefix}Q{q}"."""
    return cyclical_excess_by_period(pivot, min_obs, prefix, n_periods=4, label="Q")


def cyclical_excess_by_month(
    pivot: pd.DataFrame, min_obs: int, prefix: str
) -> tuple[dict, dict]:
    """Monthly special case of cyclical_excess_by_period() -- see there for
    parameter docs. Column names: f"{prefix}M{m}" / f"n_{prefix}M{m}"."""
    return cyclical_excess_by_period(pivot, min_obs, prefix, n_periods=12, label="M")
