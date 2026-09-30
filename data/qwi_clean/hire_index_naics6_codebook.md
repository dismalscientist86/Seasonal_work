# Codebook: hire_index_naics6.csv

## File

- **Path:** `data/qwi_clean/hire_index_naics6.csv`
- **Created by:** `code/qwi/hire_index.py`
- **Rows:** 1,011
- **Columns:** 13
- **Unit of observation:** one 6-digit NAICS industry (national level).

## Purpose

This file measures **hiring** seasonality — a third QWI-based flow measure
alongside `seasonal_index_naics6.csv`'s separations-based `seasonal_index`
(the primary flow measure) and `seasonal_index_emp` (the stock/headcount
measure). It uses QWI's `HirA` ("Hires All: Counts (Accessions)") the same
way the separations measure uses `Sep`, so it is the direct in-flow
counterpart to that out-flow measure.

**Motivating question**: does an industry's hiring surge and its separation
surge fall in the *same* quarter (a short season / high churn — the same
jobs refilled by different workers each cycle) or in *different* quarters (a
longer season — staffing up well before the eventual layoff)? This file
alone answers "when does this industry hire seasonally, and how much";
`output/tables/hire_vs_separation_timing.csv` (same repo, not itself
published — rebuild with `hire_index.py`) merges it onto the separations
index to answer the timing-lag question directly. See CLAUDE.md's "Hire
(Accessions) Seasonal Index" section for the full results and worked
examples.

## Source Data and Construction

```text
hire_rate(j, t, q) = HirA(j, t, q) / EmpEnd(j, t, q)
HireExcess(q)_j = mean_t[ hire_rate(q, t) - (hire_rate(q-1, t) + hire_rate(q+1, t)) / 2 ]
```

using `excess_utils.cyclical_excess_by_quarter()` — the same shared cyclical
calculation as every other quarterly measure in this project (quarters
treated cyclically: Q4 is compared against Q3 and next-year's Q1). Cells
with fewer than 5 contributing years (`MIN_EXCESS_OBS`) are set to missing.

Unlike `sep_rate`, `hire_rate` is **not** truncated at ≥1 before computing
excess: a quarter's hires can legitimately exceed end-of-quarter headcount
for a rapidly ramping-up seasonal operation (e.g. a resort hiring 300
workers for a season that nets out at 150 after also losing some early
starters), so an apparent >100% rate isn't automatically a
disclosure-suppression artifact the way it is for separations. Extreme
values are instead handled by the same 99th-percentile winsorization used
for every other index in this project.

## Data-Quality Notes (read before using `hire_seasonal_index`)

**Excludes agriculture from the winsorization threshold (2026-09), consistent
with every other national index in this project.** QWI's UI-based coverage
of agricultural employment is historically weaker/partial (see
`agriculture_exclusion_check.py` and CLAUDE.md's "Agriculture Exclusion
Check"), so agriculture's amplitudes may be a non-representative,
inflated slice. `hire_seasonal_index` is computed as
`hire_seasonal_amplitude / p99(hire_seasonal_amplitude among non-agriculture
industries)`, clipped to [0, 1] — **agriculture rows (`sector_2d == "11"`)
still get an index value in this file** (usually clipped at 1.0, since
agricultural amplitudes tend to exceed the ex-agriculture threshold), they
just don't get to set that threshold for everyone else. If you want the
full-sample (agriculture-inclusive) winsorization instead, recompute
`hire_seasonal_amplitude / hire_seasonal_amplitude.quantile(0.99)` over all
1,011 rows yourself.

**`hire_seasonal_amplitude` requires ≥2 non-missing quarters**, same
degenerate-single-quarter guard used everywhere else in this project — a
single populated quarter would otherwise trivially collapse max−min to 0,
a data-thinness artifact rather than genuine flatness.

## Column Definitions

| Column | Type | Definition |
|---|---|---|
| `naics_code` | string | 6-digit NAICS industry code. |
| `hire_seasonal_amplitude` | float | max(`HireExcess`) − min(`HireExcess`) across the four quarters. Missing unless ≥2 quarters are non-missing. |
| `peak_quarter_hire` | integer | Quarter (1–4) with the largest hire excess — when this industry's hiring is most concentrated. |
| `peak_excess_hire` | float | Hire excess at `peak_quarter_hire` (a proportion of end-of-quarter employment, e.g. 0.39 = hires that quarter run 39 percentage points above the adjacent-quarter average — not a percentage). |
| `n_hire_excess_obs` | integer | Year-observations contributing to the `peak_quarter_hire` estimate; flags thin cells. |
| `hire_seasonal_index` | float | Normalized [0,1] hiring-seasonality score: `hire_seasonal_amplitude` divided by the 99th percentile of `hire_seasonal_amplitude` **among non-agriculture industries** (see Data-Quality Notes), clipped to [0,1]. Missing for 3 of 1,011 industries. |
| `sector_2d` | string | First two digits of `naics_code`. |
| `sector_label` | string | Broad NAICS sector label mapped from `sector_2d` (same labels as `seasonal_index_naics6.csv`). |
| `national_industry_title` | string | Human-readable NAICS6 industry title from the Census 2022 NAICS structure crosswalk. 0 missing (100% match — this file uses QWI's own native NAICS6 codes, unlike the QCEW monthly index, which has a real vintage-mismatch gap; see that file's own codebook). |
| `sector_title` | string | Census NAICS sector (2-digit, or combined ranges like "31-33") title. |
| `subsector_title` | string | Census NAICS subsector (3-digit) title. |
| `industry_group_title` | string | Census NAICS industry group (4-digit) title. |
| `naics_industry_title` | string | Census NAICS industry (5-digit) title. |

## Units and Interpretation

`peak_excess_hire` is a proportion of employment (like `peak_excess` in the
separations index), not a percentage — `peak_excess_hire = 0.39` means
hires that quarter run 39 percentage points above the average of the
adjacent quarters, as a share of end-of-quarter employment.

`hire_seasonal_index = 1.0` means the industry's `hire_seasonal_amplitude`
is at or above the ex-agriculture 99th percentile, not that all hiring is
seasonal. As with every other 0–1 index in this project, several
industries tie at exactly 1.00 — use `hire_seasonal_amplitude` or
`peak_excess_hire` to rank *within* that tied group.

## File-Level Checks (2026-09, first real run after the HirA refetch)

- `peak_quarter_hire` distribution: Q1 = 30, Q2 = 501, Q3 = 271, Q4 = 206,
  missing = 3 — most industries' hiring peaks in Q2, consistent with
  staffing up ahead of a summer/fall season.
- `hire_seasonal_index` ranges from ~0.002 to 1.0 (multiple industries
  clipped at 1.0).
- Top of the ex-agriculture ranking: RV Parks and Campgrounds, Other
  Insurance Funds, Skiing Facilities, Political Organizations, Drive-In
  Motion Picture Theaters, Recreational and Vacation Camps.

## Recommended Use

- Compare against `seasonal_index_naics6.csv`'s `seasonal_index` and
  `peak_quarter` to see whether an industry hires and separates in the
  same quarter (short season / high churn) or in different quarters (a
  longer season) — rebuild `output/tables/hire_vs_separation_timing.csv`
  with `hire_index.py` for the merged, ready-to-use comparison, including
  `hire_to_sep_lag_quarters` and a `likely_churn` flag.
- Use `hire_seasonal_index` for a normalized ranking; `peak_excess_hire`
  when an interpretable percentage-point effect is preferred.
- Treat missing `hire_seasonal_index` as "not enough data" (fewer than 2
  non-missing quarters or below the 5-year minimum), not "not seasonal."
