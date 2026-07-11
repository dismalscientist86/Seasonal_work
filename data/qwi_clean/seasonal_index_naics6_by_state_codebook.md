# Codebook: seasonal_index_naics6_by_state.csv

## File

- **Path:** `data/qwi_clean/seasonal_index_naics6_by_state.csv`
- **Created by:** `code/qwi/geographic_analysis.py` (`load_and_compute()`)
- **Last regenerated:** 2026-07-11
- **Rows:** 42,344
- **Columns:** 46
- **Unit of observation:** one state × 6-digit NAICS industry cell.

## Purpose

This is the state-level companion to `seasonal_index_naics6.csv`: the same
flow (separation-rate) and stock (employment-level) seasonal measures,
computed separately **within each state** rather than pooled nationally.
CLAUDE.md's "Geographic Variation" and "State-Level Percentile / Overlap
Analysis" sections are both built from this file, and it is the recommended
basis for firm classification whenever the firm's state is known — seasonal
intensity varies a lot by state (e.g. construction: MN 19.7 p.p. peak excess
vs. FL 2.6 p.p., an 8x gap; overall state means range from Alaska 8.0 p.p. to
Texas 3.3 p.p., a 2.4x gap).

Unlike the national file, this file **does** include the raw per-quarter
stock columns (`emp_excessQ1-4`, `n_emp_excessQ1-4`, `emp_share_Q1-4`), not
just their summary statistics.

## Source Data and Construction

Identical methodology to `seasonal_index_naics6.csv` (see that codebook for
the full excess-quarter formula and quarter-adjacency definition), applied
separately to each state's own QWI industry-year-quarter cells rather than
to the national aggregate. The same ≥5-year `MIN_EXCESS_OBS` floor and the
≥2-non-missing-quarters requirement (see Data-Quality Notes below and in the
national codebook) apply here too, cell by cell.

Only state × NAICS6 cells with at least one usable QWI observation appear as
a row; cells with no data at all are simply absent, not present with all
missing values.

## Data-Quality Notes

Same degenerate-quarter fix as the national file applies here — and matters
far more, since state × NAICS6 cells are much thinner than national ones.
Requiring ≥2 non-missing quarters before reporting `seasonal_amplitude`/
`peak_quarter`/`peak_excess`/`seasonal_index` (flow) and their stock-side
counterparts changed substantially more cells here than nationally (6 of
1,011 industries nationally; a much larger share of the 42,344 state ×
industry cells here, precisely because thin single-state cells are more
often reduced to 1 valid quarter). Treat missing `seasonal_index`/
`seasonal_index_emp` as "insufficient data for this state × industry cell,"
not as evidence of low seasonality.

## Column Definitions

| Column | Type | Definition |
|---|---|---|
| `state` | string | 2-digit state FIPS code. |
| `state_name` | string | Full state name corresponding to `state`. |
| `naics_code` | string | 6-digit NAICS industry code. |
| `naics_level` | integer | Number of digits in `naics_code`. This file is NAICS level 6. |
| `sector_2d` | string | First two digits of `naics_code`. |
| `sector_label` | string | Human-readable broad NAICS sector label mapped from `sector_2d` (project-internal labels). |
| `n_years` | integer | Number of calendar years with at least one usable QWI observation for this state × industry cell. Ranges 5–24 in this file (cells below the year floor needed for any excess estimate are dropped entirely, hence the floor above 1). |
| `excessQ1` | float | Mean Q1 excess separation rate for this state × industry cell. |
| `n_excessQ1` | integer | Number of annual observations behind `excessQ1`. |
| `excessQ2` | float | Mean Q2 excess separation rate. |
| `n_excessQ2` | integer | Number of annual observations behind `excessQ2`. |
| `excessQ3` | float | Mean Q3 excess separation rate. |
| `n_excessQ3` | integer | Number of annual observations behind `excessQ3`. |
| `excessQ4` | float | Mean Q4 excess separation rate. |
| `n_excessQ4` | integer | Number of annual observations behind `excessQ4`. |
| `sep_rate_Q1` | float | Mean Q1 separation rate for this state × industry cell. |
| `sep_rate_Q2` | float | Mean Q2 separation rate. |
| `sep_rate_Q3` | float | Mean Q3 separation rate. |
| `sep_rate_Q4` | float | Mean Q4 separation rate. |
| `seasonal_amplitude` | float | max(`excessQ1..4`) − min(`excessQ1..4`) for this cell. Missing unless ≥2 quarters are non-missing. |
| `peak_quarter` | integer | Quarter (flow) with the largest excess separation rate for this state × industry cell. 1–4, or missing. |
| `peak_excess` | float | Excess separation rate at `peak_quarter`. |
| `n_excess_obs` | integer | Annual observations behind the `peak_quarter` estimate. |
| `seasonal_index` | float | Flow seasonality score for this cell, normalized to [0, 1] using the same p99-amplitude denominator approach as the national file (computed over this file's own distribution). |
| `emp_excessQ1` | float | Mean Q1 employment-share excess (stock side), from `emp_share = EmpEnd / mean_year(EmpEnd)`. |
| `n_emp_excessQ1` | integer | Annual observations behind `emp_excessQ1`. |
| `emp_excessQ2` | float | Mean Q2 employment-share excess. |
| `n_emp_excessQ2` | integer | Annual observations behind `emp_excessQ2`. |
| `emp_excessQ3` | float | Mean Q3 employment-share excess. |
| `n_emp_excessQ3` | integer | Annual observations behind `emp_excessQ3`. |
| `emp_excessQ4` | float | Mean Q4 employment-share excess. |
| `n_emp_excessQ4` | integer | Annual observations behind `emp_excessQ4`. |
| `emp_share_Q1` | float | Mean Q1 employment level normalized to the cell's own annual mean (1.0 = exactly average). |
| `emp_share_Q2` | float | Mean Q2 normalized employment level. |
| `emp_share_Q3` | float | Mean Q3 normalized employment level. |
| `emp_share_Q4` | float | Mean Q4 normalized employment level. |
| `emp_seasonal_amplitude` | float | max(`emp_excessQ1..4`) − min(`emp_excessQ1..4`). Missing unless ≥2 quarters are non-missing. |
| `peak_quarter_emp` | integer | Quarter with the largest employment-share excess (stock) for this cell. |
| `peak_excess_emp` | float | Employment-share excess at `peak_quarter_emp`. |
| `n_emp_excess_obs` | integer | Annual observations behind the `peak_quarter_emp` estimate. |
| `seasonal_index_emp` | float | Stock seasonality score for this cell, normalized to [0, 1]. |
| `national_industry_title` | string | NAICS6 industry title from the Census 2022 NAICS crosswalk (same for a given `naics_code` regardless of state). |
| `sector_title` | string | Census NAICS sector title. |
| `subsector_title` | string | Census NAICS subsector title. |
| `industry_group_title` | string | Census NAICS industry group title. |
| `naics_industry_title` | string | Census NAICS industry (5-digit) title. |

## Units and Interpretation

Same conventions as `seasonal_index_naics6.csv`: rate/share variables are
proportions, not percentages, and `seasonal_index`/`seasonal_index_emp` are
normalized to [0, 1] within this file's own distribution (not directly
comparable in absolute terms to the national file's normalization, since the
p99 denominators differ). For cross-state comparisons of raw magnitude, use
`peak_excess`/`peak_excess_emp` (percentage-point excess) rather than the
normalized index.

## File-Level Checks (as of 2026-07-11 regeneration)

- 51 states represented (50 states + DC).
- `n_years` ranges from 5 to 24 across cells.
- `seasonal_index` is missing for 2,898 of 42,344 cells (~6.8%).
- `seasonal_index_emp` is missing for 155 of 42,344 cells (~0.4%).
- `national_industry_title`: 0 missing.

## Recommended Use

Prefer this file over the national index whenever the firm's or worker's
state is known — CLAUDE.md's state-percentile analysis found only ~11%
overlap between a state's most-seasonal industries and the national top 1%,
so a national-only index misses substantial state-specific heterogeneity.
Merge on (`state`, `naics_code`) for firm classification; fall back to the
national `seasonal_index_naics6.csv` when the state × industry cell is
missing here (too thin to estimate) or when state is unknown.
