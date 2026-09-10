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
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cyclical_excess_by_quarter(
    pivot: pd.DataFrame, min_obs: int, prefix: str
) -> tuple[dict, dict]:
    """
    Cyclical-neighbor excess computation for one (year x quarter) pivot
    table — one industry, or one (state, industry) cell, depending on the
    caller's groupby level.

    Parameters
    ----------
    pivot   : DataFrame indexed by year, with integer columns 1-4 (quarter)
              holding the rate/share value being tested for seasonality
              (e.g. sep_rate, emp_share, earnings_share, hire_rate).
    min_obs : Minimum number of year-observations required for a quarter's
              excess estimate to be reported; fewer than this and it's NaN.
    prefix  : Used to name the output dict keys, matching each caller's
              existing convention — e.g. prefix="excess" -> "excessQ1"..
              "excessQ4"/"n_excessQ1".."n_excessQ4"; prefix="emp_excess" ->
              "emp_excessQ1".. ; prefix="hire_excess" -> "hire_excessQ1"..

    Returns
    -------
    record   : dict with f"{prefix}Q{q}" (mean excess, or NaN) and
               f"n_{prefix}Q{q}" (contributing year count) for q in 1..4.
    excesses : dict of {quarter: (mean_excess, n_obs)}, but only for
               quarters that met `min_obs` — convenience for callers that
               need the peak/trough quarter without re-filtering `record`.
    """
    record = {}
    excesses = {}
    for q in [1, 2, 3, 4]:
        q_prev = 4 if q == 1 else q - 1
        q_next = 1 if q == 4 else q + 1

        if q not in pivot.columns or q_prev not in pivot.columns or q_next not in pivot.columns:
            record[f"{prefix}Q{q}"] = np.nan
            record[f"n_{prefix}Q{q}"] = 0
            continue

        if q == 1:
            exr = (pivot[1] - (pivot[4].shift(1) + pivot[2]) / 2).dropna()
        elif q == 4:
            exr = (pivot[4] - (pivot[3] + pivot[1].shift(-1)) / 2).dropna()
        else:
            exr = (pivot[q] - (pivot[q - 1] + pivot[q + 1]) / 2).dropna()

        record[f"{prefix}Q{q}"] = exr.mean() if len(exr) >= min_obs else np.nan
        record[f"n_{prefix}Q{q}"] = len(exr)
        if len(exr) >= min_obs:
            excesses[q] = (exr.mean(), len(exr))

    return record, excesses
