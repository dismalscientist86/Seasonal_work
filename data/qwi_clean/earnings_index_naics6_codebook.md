# Codebook: earnings_index_naics6.csv

## File

- **Path:** `data/qwi_clean/earnings_index_naics6.csv`
- **Created by:** `code/qwi/earnings_index.py`
- **Last regenerated:** 2026-07-11 (after the full 51-state EarnBeg refetch)
- **Rows:** 1,011
- **Columns:** 17
- **Unit of observation:** one 6-digit NAICS industry.

## Purpose

This file is a third, independent seasonal measure alongside the flow
(separations) and stock (employment) measures in `seasonal_index_naics6.csv`:
**income seasonality**, built from QWI's `EarnBeg` (average monthly
earnings), directly testing the paper's title question — "income in the
off-season" — at the industry level instead of the individual-worker level.

It is used in CLAUDE.md's "Income/Earnings Seasonality Index" analysis to
test whether an industry's *average* earnings dip in its own separation-peak
quarter. That analysis, and the interpretation of this file, hinges on the
raw-vs-idiosyncratic distinction described below.

**Important:** unlike the other two `qwi_clean` files, this file does not
carry NAICS crosswalk title columns — only `naics_code`, `sector_2d`, and
`sector_label`. Merge on `naics_code` against `seasonal_index_naics6.csv` if
human-readable titles are needed.

## Source Data and Construction

1. Compute `earnings_share_{j,t,q} = EarnBeg_{j,t,q} / mean_year(EarnBeg_{j,t})`
   — average earnings in a given industry-year-quarter, normalized to that
   industry-year's own annual mean (analogous to the stock index's
   `emp_share`, but for earnings rather than headcount).
2. Apply the same cyclical excess-by-quarter computation used throughout the
   project (see `seasonal_index_naics6.csv`'s codebook for the formula and
   quarter-adjacency definition) to `earnings_share` in place of
   `sep_rate`/`emp_share`, producing `earnings_seasonal_amplitude`,
   `peak_quarter_earnings`, `peak_excess_earnings`,
   `earnings_seasonal_index`, etc. — this is the **raw** index.
3. **Confound correction.** 83% of all 1,011 industries have their *raw*
   earnings index peak in Q4 — a broad, economy-wide year-end bonus/holiday-
   pay effect common to nearly every industry, not industry-specific
   seasonality. Left uncorrected, this swamps genuine signal: whether an
   industry's own separation-peak quarter happens to coincide with the
   common Q4 bump almost entirely determines whether its earnings look like
   they "dip" there. `compute_common_calendar_effect()` estimates this
   common cross-industry calendar effect by quarter:

   | Quarter | Common calendar effect |
   |---|---|
   | Q1 | −0.031 |
   | Q2 | −0.005 |
   | Q3 | −0.048 |
   | Q4 | +0.085 |

   `add_idiosyncratic_excess()` then nets this out of each industry-quarter's
   raw excess before recomputing amplitude/peak/index — these are the
   **idiosyncratic (`idio_*`)** columns, and are the credible basis for any
   industry-specific interpretation. The raw columns are kept in this file
   for reference/diagnostic purposes but should not be used to argue an
   industry's earnings seasonality is "industry-specific" on their own.

## Data-Quality Notes

- Same ≥5-year `MIN_EXCESS_OBS` floor and ≥2-non-missing-quarters requirement
  as the other two files apply here, to both the raw and idiosyncratic
  amplitude/index columns independently.
- `n_earnings_excess_obs` and `idio_n_earnings_excess_obs` are typically equal
  since they are built from the same underlying earnings-share panel, but are
  tracked separately in case future changes decouple raw and idiosyncratic
  quarter-validity.
- This file went through a full-sample refetch after `fetch_qwi.py`'s
  originally-requested `Payroll` field was found to return `null` for every
  cell (a permanent Census API "not good" status flag) — `EarnBeg` is the
  working substitute now fetched by default. See CLAUDE.md's "Income/Earnings
  Seasonality Index" section for the refetch history, including a transient
  DNS blip that briefly left Ohio short ~45 industries (refetched cleanly).

## Column Definitions

| Column | Type | Definition |
|---|---|---|
| `naics_code` | string | 6-digit NAICS industry code. |
| `earnings_seasonal_amplitude` | float | Raw: max(quarterly earnings excess) − min(quarterly earnings excess), from uncorrected `earnings_share` excess values. Missing unless ≥2 quarters are non-missing. |
| `peak_quarter_earnings` | integer | Raw: quarter with the highest earnings-share excess. 1–4, or missing. Dominated economy-wide by Q4 (840 of 1,011 industries) — see Data-Quality/Purpose notes on the common calendar effect. |
| `peak_excess_earnings` | float | Raw: earnings-share excess at `peak_quarter_earnings`. |
| `trough_quarter_earnings` | integer | Raw: quarter with the lowest (most negative) earnings-share excess. |
| `trough_excess_earnings` | float | Raw: earnings-share excess at `trough_quarter_earnings`. |
| `n_earnings_excess_obs` | integer | Annual observations behind the raw `peak_quarter_earnings` estimate. |
| `earnings_seasonal_index` | float | Raw earnings seasonality score, normalized to [0, 1] via the same p99-amplitude approach used elsewhere. Confounded by the common calendar effect — see Purpose. |
| `sector_2d` | string | First two digits of `naics_code`. |
| `sector_label` | string | Human-readable broad NAICS sector label mapped from `sector_2d`. |
| `idio_earnings_seasonal_amplitude` | float | Idiosyncratic (de-confounded) analog of `earnings_seasonal_amplitude`, computed after subtracting the common cross-industry calendar effect from each quarter's excess. |
| `idio_peak_quarter_earnings` | integer | Idiosyncratic peak quarter — the credible measure of an industry's *own* earnings-timing pattern. |
| `idio_peak_excess_earnings` | float | Idiosyncratic earnings-share excess at `idio_peak_quarter_earnings`. |
| `idio_trough_quarter_earnings` | integer | Idiosyncratic trough quarter. |
| `idio_trough_excess_earnings` | float | Idiosyncratic earnings-share excess at `idio_trough_quarter_earnings`. |
| `idio_n_earnings_excess_obs` | integer | Annual observations behind the idiosyncratic `idio_peak_quarter_earnings` estimate. |
| `idio_earnings_seasonal_index` | float | Idiosyncratic earnings seasonality score, normalized to [0, 1]. **This is the recommended column for any industry-specific earnings-seasonality claim** — the raw `earnings_seasonal_index` should not be used for that purpose without also consulting this one. |

## Units and Interpretation

- All excess/amplitude/share values are proportions of the industry's own
  annual mean earnings, not percentages: `peak_excess_earnings = 0.10` means
  earnings in that quarter run 10% above the average of its adjacent
  quarters (raw, pre-de-confounding).
- Positive idiosyncratic excess at an industry's own separation-peak quarter
  means earnings are *higher*, not lower, when separations spike — this was
  the actual finding for Construction (mean idio excess +0.044 at its own
  separation peak, interpreted as end-of-season overtime pay ahead of winter
  layoffs).
- The overall credible (idiosyncratic) finding across all industries: 43.9%
  show a dip in earnings at their own separation-peak quarter (mean excess
  ≈ +0.0065, i.e. essentially flat/slightly positive) — no strong evidence
  of industry-average income dipping in the off-season. Contrast with the
  raw/naive test's 51.9%–52.3% dip rate, which is mostly a mechanical
  artifact of the Q4 confound (see CLAUDE.md for the full breakdown by which
  quarter the separation peak falls in).

## File-Level Checks (as of 2026-07-11 regeneration)

- `earnings_seasonal_index` missing for 1 of 1,011 industries.
- `idio_earnings_seasonal_index` missing for 1 of 1,011 industries.
- `peak_quarter_earnings` (raw) distribution: Q1 = 118, Q2 = 47, Q3 = 5,
  Q4 = 840, missing = 1 — the Q4 concentration is the common calendar effect,
  not industry-specific timing (see Purpose).
- Sample row (naics_code 541310, Professional & technical services):
  `earnings_seasonal_amplitude` = 0.573, `peak_quarter_earnings` = 4,
  `peak_excess_earnings` = 0.361, `earnings_seasonal_index` = 1.0;
  `idio_earnings_seasonal_amplitude` = 0.456, `idio_peak_quarter_earnings` = 4,
  `idio_peak_excess_earnings` = 0.276, `idio_earnings_seasonal_index` = 0.819
  — illustrates that de-confounding shrinks, but does not always eliminate,
  a Q4 peak; some industries genuinely do peak in Q4 beyond the common
  bonus effect.

## Recommended Use

- Use `idio_earnings_seasonal_index` / `idio_peak_quarter_earnings` /
  `idio_peak_excess_earnings` for any claim about an industry's own earnings
  seasonality or an off-season income-dip test. Do not use the raw `earnings_*`
  columns for that purpose in isolation — they are confounded by a common,
  economy-wide Q4 bonus effect (see Purpose).
- The raw columns remain useful as a diagnostic (e.g. to reproduce or audit
  the confound itself) and as an input to `compute_common_calendar_effect()`.
- To test the off-season income-dip hypothesis for a specific industry, merge
  this file's `idio_peak_excess_earnings` against that industry's own
  separation `peak_quarter` (flow) from `seasonal_index_naics6.csv`, matching
  on `naics_code` — this reproduces the logic behind
  `output/tables/earnings_at_separation_peak.csv`.
- Merge on `naics_code` against `seasonal_index_naics6.csv` for
  human-readable industry titles, since this file does not carry them.
