# Codebook: seasonal_index_naics6.csv

## File

- **Path:** `data/qwi_clean/seasonal_index_naics6.csv`
- **Created by:** `code/qwi/seasonal_index.py` (`build_and_save_index()`)
- **Last regenerated:** 2026-07-11 (after the degenerate-quarter fix; see
  "Data-Quality Notes" below)
- **Rows:** 1,011
- **Columns:** 32
- **Unit of observation:** one 6-digit NAICS industry.

## Purpose

This file contains a national QWI-based seasonal index for 6-digit NAICS
industries, built **two ways in parallel**:

- **Flow** (`seasonal_index` and related fields): adapts the Coglianese and
  Price excess recurrence idea to quarterly QWI data by measuring whether an
  industry's *separation rate* is elevated in a focal quarter relative to the
  adjacent quarters.
- **Stock** (`seasonal_index_emp` and related fields): measures whether
  *headcount itself* (employment level, normalized to its own annual mean)
  swells or shrinks in a focal quarter. Related but distinct from flow — see
  `code/qwi/seasonal_index.py`'s `compare_flow_vs_stock()` and CLAUDE.md's
  "Flow vs. Stock Seasonal Index Comparison" section. Nationally, flow and
  stock correlate at Pearson r ≈ 0.78, but agree on *which quarter* drives
  seasonality only ~15% of the time.

The primary firm-classification fields are:

- `seasonal_index`: normalized flow-based seasonality score in [0, 1].
- `peak_excess`: excess separation rate in the industry's most seasonal quarter.
- `peak_quarter`: quarter in which `peak_excess` occurs.
- `n_excess_obs`: number of year observations used for the peak-quarter estimate.
- `national_industry_title`: human-readable industry name (see below).

## Source Data and Construction

The script loads QWI state-level NAICS data, aggregates it to national
industry-year-quarter cells, and computes:

```text
sep_rate = Sep / EmpEnd
```

If `EmpEnd` is unavailable in the source data, the code falls back to `Emp`.
Suppressed or invalid negative QWI cells are set to missing before aggregation.
Separation rates less than or equal to 0 or greater than or equal to 1 are also
set to missing.

For each industry and quarter q, the script computes:

```text
excessQq = mean_t[ sep_rate(q,t) - (adjacent previous quarter + adjacent next quarter) / 2 ]
```

Quarter adjacency is cyclical:

- Q1 is compared with Q4 of the previous year and Q2 of the same year.
- Q2 is compared with Q1 and Q3 of the same year.
- Q3 is compared with Q2 and Q4 of the same year.
- Q4 is compared with Q3 of the same year and Q1 of the next year.

The stock-side (`emp_*`/`*_emp` columns) is constructed the same way, but from
`emp_share = EmpEnd / mean_year(EmpEnd)` (employment normalized to its own
annual mean) in place of `sep_rate`.

Every `excessQ*`/`emp_excessQ*` cell requires **at least 5 contributing years**
(`MIN_EXCESS_OBS` in `code/config.py`) or it is set to missing — this floor
exists specifically to stop thin, disclosure-suppressed cells from
masquerading as precise estimates.

The file is sorted by `seasonal_index` in descending order.

## Data-Quality Notes (read before using amplitude/index fields)

1. **Degenerate single-quarter amplitude (fixed 2026-07-11).**
   `seasonal_amplitude = max(excessQ*) - min(excessQ*)` collapses to exactly 0
   when only one of the four quarters has a non-missing excess estimate — a
   data-thinness artifact, not genuine flatness, that used to be silently
   reported as `seasonal_index = 0.0` (the most extreme possible
   "non-seasonal" score). `build_seasonal_index()` now requires **≥2**
   non-missing quarters before reporting `seasonal_amplitude`/`peak_quarter`/
   `peak_excess`/`seasonal_index` (and the analogous stock-side fields); all
   four are set to missing otherwise. This affected 6 of 1,011 industries
   nationally (see CLAUDE.md for the full list and the much larger impact at
   the state level).
2. **Stale-threshold artifact (fixed 2026-07, earlier pass).** A prior version
   of this file (dated 2026-03) let 3 industries rank #1 nationally at
   `seasonal_index = 1.0` off just 2 COVID-distorted years of data, below the
   current 5-year minimum. Already excluded from this build.
3. `national_industry_title` (and the sector/subsector/industry-group
   hierarchy) come from a separate Census crosswalk
   (`data/naics_xwalk/naics6_2022_titles.csv`, built by
   `code/qwi/clean_naics_xwalk.py`) merged in on `naics_code` — not derived
   from the QWI data itself, and not present at all if that crosswalk file
   hasn't been built.

## Column Definitions

| Column | Type | Definition |
|---|---|---|
| `naics_code` | string | 6-digit NAICS industry code. Stored as a string to preserve any leading zeros if present. |
| `n_years` | integer | Number of calendar years with at least one usable national QWI observation for the industry. Maximum is 24 in this file. |
| `excessQ1` | float | Mean Q1 excess separation rate. Q1 is compared with previous-year Q4 and same-year Q2. Missing if fewer than 5 contributing years or fewer than 2 non-missing quarters overall (see Data-Quality Notes). |
| `n_excessQ1` | integer | Number of annual observations used to compute `excessQ1`. |
| `excessQ2` | float | Mean Q2 excess separation rate. Q2 is compared with Q1 and Q3 of the same year. |
| `n_excessQ2` | integer | Number of annual observations used to compute `excessQ2`. |
| `excessQ3` | float | Mean Q3 excess separation rate. Q3 is compared with Q2 and Q4 of the same year. |
| `n_excessQ3` | integer | Number of annual observations used to compute `excessQ3`. |
| `excessQ4` | float | Mean Q4 excess separation rate. Q4 is compared with same-year Q3 and next-year Q1. |
| `n_excessQ4` | integer | Number of annual observations used to compute `excessQ4`. |
| `sep_rate_Q1` | float | Mean Q1 separation rate for the industry across usable years. |
| `sep_rate_Q2` | float | Mean Q2 separation rate for the industry across usable years. |
| `sep_rate_Q3` | float | Mean Q3 separation rate for the industry across usable years. |
| `sep_rate_Q4` | float | Mean Q4 separation rate for the industry across usable years. |
| `seasonal_amplitude` | float | Difference between the maximum and minimum of `excessQ1` through `excessQ4`. Missing unless ≥2 quarters are non-missing (see Data-Quality Notes). |
| `peak_quarter` | integer | Quarter (flow) with the largest excess separation rate. Values are 1, 2, 3, or 4. Missing when no valid excess-quarter estimate is available. |
| `peak_excess` | float | Maximum of `excessQ1` through `excessQ4`, i.e. the excess separation rate at `peak_quarter`. |
| `n_excess_obs` | integer | Number of annual observations used to estimate the excess value in `peak_quarter`. Equals the corresponding `n_excessQ*` field. |
| `seasonal_index` | float | Normalized flow seasonality score: `seasonal_amplitude / p99(seasonal_amplitude)`, clipped to [0, 1]. In this file, the 99th-percentile denominator is approximately 0.558. |
| `sector_2d` | string | First two digits of `naics_code`; used to assign broad NAICS sector. |
| `sector_label` | string | Human-readable broad NAICS sector label mapped from `sector_2d` (project-internal labels, distinct from `sector_title` below). |
| `naics_level` | integer | Number of digits in `naics_code`. This file is NAICS level 6. |
| `emp_seasonal_amplitude` | float | Stock-side analog of `seasonal_amplitude`, from employment-share excess (`emp_share = EmpEnd / mean_year(EmpEnd)`) rather than separation rate. |
| `peak_quarter_emp` | integer | Quarter with the largest employment-share excess (stock). Frequently differs from `peak_quarter` (flow) — see Purpose. |
| `peak_excess_emp` | float | Employment-share excess at `peak_quarter_emp`. |
| `n_emp_excess_obs` | integer | Number of annual observations behind the `peak_quarter_emp` estimate. |
| `seasonal_index_emp` | float | Normalized stock seasonality score, same [0, 1] winsorized-amplitude construction as `seasonal_index` but on the employment side. 99th-percentile denominator ≈ 0.651. |
| `national_industry_title` | string | Human-readable NAICS6 industry title from the Census 2022 NAICS structure crosswalk. Present for all 1,011 rows in this file. |
| `sector_title` | string | Census NAICS sector (2-digit, or combined ranges like "31-33") title from the same crosswalk. |
| `subsector_title` | string | Census NAICS subsector (3-digit) title. |
| `industry_group_title` | string | Census NAICS industry group (4-digit) title. |
| `naics_industry_title` | string | Census NAICS industry (5-digit) title. |

## Units and Interpretation

Rate variables are proportions, not percentages. For example:

- `peak_excess = 0.10` means the peak quarter's separation rate is 10 percentage
  points higher than the average of its adjacent quarters.
- `sep_rate_Q4 = 0.18` means average Q4 separations equal 18 percent of
  end-of-quarter employment.
- `seasonal_index = 1` means the industry's `seasonal_amplitude` is at or above
  the 99th percentile used for normalization, not that all employment is seasonal.
- `emp_share_Q4 = 1.10` (on the state-level file's raw stock columns; not
  present in this national file, see below) would mean Q4 employment is 10%
  above the industry's own annual mean.

Positive `excessQ*`/`emp_excessQ*` values indicate the rate/share is above the
adjacent-quarter baseline in that quarter; negative values indicate below.

**Note:** this national file carries only the *summary* stock statistics
(`emp_seasonal_amplitude`, `peak_quarter_emp`, `peak_excess_emp`,
`n_emp_excess_obs`, `seasonal_index_emp`) — not the raw per-quarter
`emp_excessQ1-4`/`emp_share_Q1-4` values. Those raw quarter-level stock
columns are only in `seasonal_index_naics6_by_state.csv`.

## File-Level Checks (as of 2026-07-11 regeneration)

- `n_years` ranges from 1 to 24, mean 23.97.
- `seasonal_index` is missing for 14 industries (up from 5 in the original
  2026-03 file — this file now correctly flags more thin cells rather than
  reporting a false 0.0 for them; see Data-Quality Notes).
- `seasonal_index_emp` is missing for 3 industries.
- `seasonal_index` ranges from ~0.005 to 1 (11+ industries clipped at 1).
- `peak_quarter` (flow) distribution: Q1 = 24, Q2 = 58, Q3 = 469, Q4 = 446,
  missing = 14.
- `peak_quarter_emp` (stock) distribution: Q1 = 215, Q2 = 335, Q3 = 303,
  Q4 = 155, missing = 3 — visibly less concentrated in Q3/Q4 than the flow
  side, consistent with the low flow/stock peak-quarter concordance noted
  above.
- `national_industry_title`: 0 missing (100% match against the NAICS crosswalk).

## Recommended Use

For firm-level classification by industry lookup, merge firm NAICS6 codes to
`naics_code` and use:

- `seasonal_index` for a normalized ranking or threshold rule based on
  *separation/turnover* risk.
- `seasonal_index_emp` instead if the concern is *headcount/staffing swings*
  rather than turnover — the two are correlated (r ≈ 0.78) but not
  interchangeable, especially on timing.
- `peak_excess` when an interpretable percentage-point effect is preferred.
- `peak_quarter` (or `peak_quarter_emp`) to identify whether seasonality is
  winter, spring, summer, or fall driven — check both if timing matters, they
  often disagree.
- `n_excess_obs` / `n_emp_excess_obs` to flag thin estimates; use caution for
  low values, and treat missing `seasonal_index`/`seasonal_index_emp` as "not
  enough data," not "not seasonal."

When state is known, prefer `seasonal_index_naics6_by_state.csv` instead,
because seasonal intensity varies substantially by state (see its own
codebook).
