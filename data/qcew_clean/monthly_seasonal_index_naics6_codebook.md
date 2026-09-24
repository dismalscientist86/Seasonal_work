# Codebook: monthly_seasonal_index_naics6.csv

## File

- **Path:** `data/qcew_clean/monthly_seasonal_index_naics6.csv`
- **Created by:** `code/qcew/monthly_seasonal_index.py`
- **Source data:** `code/qcew/fetch_qcew.py`, downloading BLS QCEW's public
  bulk annual "singlefile" CSVs (no API key needed)
- **Rows:** 1,364
- **Columns:** 25
- **Unit of observation:** one 6-digit NAICS industry (national level).

## Purpose

QCEW (Quarterly Census of Employment and Wages) reports true **monthly**
employment levels within each quarter (`month1_emplvl`/`month2_emplvl`/
`month3_emplvl`), unlike QWI, which only reports quarterly totals. This
file uses that extra timing resolution to build a **monthly** analog of
this project's quarterly **stock** seasonal index
(`seasonal_index_emp` in `data/qwi_clean/seasonal_index_naics6.csv`).

**QCEW has no separations/hires field at all** — it is a pure employment-
stock census. This file cannot replicate the project's primary flow-based
`seasonal_index` (excess separation rate); it only sharpens the *stock*
measure's timing from quarterly to monthly. See
`data/qwi_clean/seasonal_index_naics6_codebook.md` for the flow/stock
distinction.

**This file is not simply "the same index at finer resolution."** Two
real statistical confounds had to be corrected before any monthly-vs-
quarterly comparison was trustworthy — see Data-Quality Notes. The
headline column for most uses is **`signal_index`**, which nets out both.

## Source Data and Construction

`fetch_qcew.py` downloads one bulk CSV per year (2000–2023) from
`data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip` and
filters to national 6-digit-NAICS rows (`agglvl_code == "18"`), Private
ownership (`own_code == "5"`) — the standard, most-complete-coverage
choice, excluding federal/state/local government employment.

For each industry, month `m` (1–12), and year `y`:

```text
emp_share(j, y, m) = emplvl(j, y, m) / mean_m(emplvl(j, y, *))
```

i.e. each month's employment normalized to its own industry-year's mean —
the same construction as the quarterly stock measure's `emp_share`, just
at monthly instead of quarterly granularity. Then, cyclically (December's
neighbors are November and the *following* January):

```text
Excess(m)_j = mean_y[ emp_share(m, y) - (emp_share(m-1, y) + emp_share(m+1, y)) / 2 ]
```

using `excess_utils.cyclical_excess_by_month()` — the same shared
cyclical-excess calculation used by every other seasonal measure in this
project (separations, quarterly stock, earnings, hires), generalized from
4 periods to 12. Cells with fewer than 5 contributing years
(`MIN_EXCESS_OBS` in `code/config.py`) are set to missing, same floor used
everywhere else in the project.

Suppressed QCEW cells (`disclosure_code == "N"`) report a literal `0` in
the raw file, not a null — these are treated as missing before any
computation, the same handling as QWI's negative-sentinel suppression
codes.

## Data-Quality Notes (read before using any column in this file)

**1. Common calendar effect (netted out via `idio_monthly_*` columns).**
47.4% of industries have their *raw* monthly peak in December, 11.0% in
January — QCEW (like QWI) is **not seasonally adjusted**, and raw U.S.
employment data has a well-documented broad, economy-wide December
build-up / January pull-back that is not industry-specific seasonality
(the same phenomenon BLS's own seasonal adjustment exists to remove).
This is the same class of confound `earnings_index.py` found and
corrected for earnings (83% of industries peaked in Q4 there). Left in,
it swamps genuine industry-specific timing for any industry whose own
signal is weak. `compute_common_calendar_effect()` estimates this
economy-wide monthly pattern (the cross-industry mean excess in each
calendar month) and `add_idiosyncratic_excess()` subtracts it from each
industry's own raw excess before the `idio_monthly_*` columns are built.

**2. Sample-size artifact (netted out via `signal_index`).** A raw
comparison of monthly vs. quarterly amplitude suggested monthly
resolution reveals far more seasonality almost everywhere — but
`max(x) - min(x)` over 12 monthly draws is *mechanically* larger than
over 4 quarterly draws for the same underlying noise process, since the
expected range of a sample grows with its size even with zero true
seasonality. `compute_placebo_amplitude()` estimates each industry's
expected idiosyncratic amplitude under a null of "no true monthly
pattern" by permuting month labels within each year (10 repetitions) and
recomputing the same calculation. Averaged across all industries, the
placebo (null) amplitude is **89.7% of the real idiosyncratic amplitude**
— most of the apparent "extra" monthly seasonality is this artifact, not
real signal. `signal_amplitude`/`signal_index` subtract this out.

**3. `national_industry_title` (and the sector/subsector/industry-group
hierarchy) is missing for ~30% of rows (403 of 1,364) — a genuine
NAICS-vintage mismatch, not a bug.** QCEW's `industry_code` field
reflects whatever NAICS revision was current when each year's data was
originally published (NAICS is revised roughly every 5 years), while our
title crosswalk (`data/naics_xwalk/naics6_2022_titles.csv`) is a single,
fixed 2022-vintage lookup. Confirmed directly: code `211111` ("Crude
Petroleum and Natural Gas Extraction," a pre-2022 code) appears in QCEW's
2000 and 2010 files but not 2023; its 2022-revision replacement,
`211120`, appears in 2023 but not the earlier years. Both halves of this
kind of split show up as *separate rows* in this file with fewer years of
data each, and neither may match the fixed 2022 crosswalk if it predates
or postdates that vintage. Most affected sectors: Manufacturing (31-33,
208 of 403 missing rows), Retail trade (44-45, 73), Mining (21, 17). This
project does not currently reconcile pre-2022 and 2022-vintage NAICS
codes into a single continuous industry series — rows with a missing
title still have valid amplitude/index columns, just no human-readable
name and a `n_years` that may reflect only one side of a mid-panel code
change.

## Column Definitions

| Column | Type | Definition |
|---|---|---|
| `naics_code` | string | 6-digit NAICS industry code, as reported by QCEW for the years it appears in (see Data-Quality Note 3 on vintage mismatches). |
| `sector_2d` | string | First two digits of `naics_code`. |
| `sector_label` | string | Broad NAICS sector label mapped from `sector_2d` (same labels as `seasonal_index_naics6.csv`). |
| `n_years` | integer | Number of calendar years (2000–2023) with usable QCEW data for this exact `naics_code` — may be less than 24 if the code was introduced or retired mid-panel by a NAICS revision. |
| `monthly_seasonal_amplitude` | float | Raw: max − min of the 12 raw `emp_excessM*` values. **Confounded by both Data-Quality Notes 1 and 2 — not recommended for ranking industries.** Missing unless ≥2 of 12 months are non-missing. |
| `monthly_peak_month` | integer | Calendar month (1–12) of the raw peak. Heavily skewed toward December/January (see Note 1). |
| `monthly_peak_quarter_from_month` | integer | `monthly_peak_month` mapped to its containing quarter (1–4), for comparing against quarterly measures. |
| `monthly_peak_excess_month` | float | Raw excess value at `monthly_peak_month`. |
| `monthly_n_excess_obs` | integer | Year-observations contributing to the raw peak-month excess estimate. |
| `monthly_seasonal_index` | float | Raw amplitude normalized to [0,1] via 99th-percentile winsorization. **Confounded — see signal_index instead for ranking.** |
| `idio_monthly_seasonal_amplitude` | float | Amplitude of the *idiosyncratic* excess (raw excess minus the common calendar effect, Note 1) — corrects for the broad Dec/Jan pattern but not yet the sample-size artifact. |
| `idio_monthly_peak_month` | integer | Peak month of the idiosyncratic excess. The recommended "which month" answer once the common calendar effect is removed. |
| `idio_monthly_peak_quarter_from_month` | integer | `idio_monthly_peak_month` mapped to its containing quarter. |
| `idio_monthly_peak_excess_month` | float | Idiosyncratic excess value at `idio_monthly_peak_month`. |
| `idio_monthly_n_excess_obs` | integer | Year-observations contributing to the idiosyncratic peak-month estimate (same underlying counts as `monthly_n_excess_obs`, since the idiosyncratic transform is a fixed per-month shift). |
| `idio_monthly_seasonal_index` | float | `idio_monthly_seasonal_amplitude` normalized to [0,1] via 99th-percentile winsorization. Still confounded by the sample-size artifact (Note 2). |
| `placebo_amplitude` | float | Mean idiosyncratic amplitude expected under the null of no true industry-specific monthly pattern (10 within-year month-label permutations). This industry's own estimated noise floor. |
| `n_perms_used` | integer | Number of successful permutation repetitions behind `placebo_amplitude` (usually 10; fewer if some repetitions had insufficient data). |
| `signal_amplitude` | float | `max(idio_monthly_seasonal_amplitude - placebo_amplitude, 0)` — the bias-corrected amplitude, net of both confounds. |
| `signal_index` | float | **Recommended column for ranking.** `signal_amplitude` normalized to [0,1] via 99th-percentile winsorization. 660 of 1,364 industries (48.4%) have `signal_index == 0` — meaning their apparent monthly seasonality was entirely explained by the two confounds, not genuine industry-specific monthly-resolution signal. |
| `national_industry_title` | string | Human-readable NAICS6 industry title from the Census 2022 NAICS structure crosswalk. **Missing for 403 of 1,364 rows (~30%) — see Data-Quality Note 3.** |
| `sector_title` | string | Census NAICS sector (2-digit, or combined ranges like "31-33") title from the same crosswalk. Same missingness as `national_industry_title`. |
| `subsector_title` | string | Census NAICS subsector (3-digit) title. |
| `industry_group_title` | string | Census NAICS industry group (4-digit) title. |
| `naics_industry_title` | string | Census NAICS industry (5-digit) title. |

## Units and Interpretation

Amplitude/excess columns are proportions (shares of an industry's own
annual mean employment), not percentages — `idio_monthly_peak_excess_month
= 0.10` means that month's employment share is 10 percentage points above
what the common calendar effect and the industry's own annual average
would predict. `signal_index = 0` does not mean "not seasonal" — it means
"no monthly-specific signal beyond what pure noise and the broad
December/January pattern would produce," which is consistent with an
industry that IS seasonal at a coarser (quarterly-or-longer) timescale
already well captured by `seasonal_index_emp`.

## Recommended Use

- Use **`signal_index`** to rank industries by genuine monthly-specific
  seasonality; treat `monthly_seasonal_index` as a naive, confounded
  reference only.
- Use **`idio_monthly_peak_month`**, not `monthly_peak_month`, when asking
  "which month" — the raw version is biased toward December/January.
- Compare against `seasonal_index_emp`
  (`data/qwi_clean/seasonal_index_naics6.csv`) to see whether an
  industry's seasonality is already well captured at quarterly
  resolution (`signal_index` near 0 despite high `seasonal_index_emp`) or
  concentrated in a sharp, sub-quarter timing pattern quarterly data
  misses (high `signal_index` despite low `seasonal_index_emp` — e.g.
  School and Employee Bus Transportation, NAICS 485410).
- Treat a missing `national_industry_title` as "this NAICS code's title
  wasn't in the 2022-vintage crosswalk for the years it appears in this
  file," not as missing or low-quality index data — the amplitude/index
  columns are still valid for those rows.
- `n_years` below the low double digits is worth checking before trusting
  an individual row — it may reflect a NAICS revision mid-panel (Note 3)
  rather than genuine data thinness.
