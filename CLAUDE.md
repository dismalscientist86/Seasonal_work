# Seasonal Work Project

## Project Overview

Replication and extension of Coglianese & Price (2025), "Income in the Off-Season:
Household Adaptation to Yearly Work Interruptions," *Industrial and Labor Relations
Review*.

### Goals

1. **Replicate** the core CP methodology in Python (original code is Stata-only).
2. **Extend** the seasonal industry classification to 6-digit NAICS using newly
   available QWI data from the Census Bureau.
3. **Firm-level code**: develop a standalone module to classify firms as seasonal
   using UI wage records (quarterly earnings by worker × employer).

---

## Methodology Summary

### Core Estimand: Excess Recurrence

For a worker who separates from employment in month t, define:

    Excess Recurrence = P(sep at t+12) - [P(sep at t+11) + P(sep at t+13)] / 2

The counterfactual is the average of adjacent lags (11 and 13 months), isolating
the *annual* spike that marks genuinely seasonal separations.

Estimated via event-study OLS on a panel centered around each focal separation.

### Data (Original Paper)

| Dataset | Size | Use |
|---|---|---|
| CPS 1984–2013 | 886 MB raw | Train GBT classifier; measure recurrence |
| SIPP (multiple panels) | ~766 MB | Income event studies; recurrence validation |
| NOAA climate divisions | small | State temperature as GBT feature |
| BLS LN | small | Aggregate unemployment validation |

Pre-cleaned Stata `.dta.gz` files are in the replication package zip.

### GBT Seasonal Classifier

Trained on CPS to predict `fwd_sep` (separation 12 months later), with features:
- `lag_ind`: 1-digit industry (15 categories, categorical)
- `lag_occ`: 14-category occupation (categorical)
- `mth_dim1`, `mth_dim2`: month encoded as sin/cos
- `jan_max_temp`: state January temperature

Applied to SIPP to assign each separator a `pred_excess` score.
Top decile = "most seasonal" workers, used in income event studies.

### QWI Extension

Census Bureau Quarterly Workforce Indicators provide quarterly separations by
industry at up to 6-digit NAICS. A quarterly analog to excess recurrence:

    ExcessQ4 = SepRate(Q4) - [SepRate(Q3) + SepRate(Q1_next)] / 2

This yields an industry-level seasonal index at 6-digit NAICS, useful for:
- Validating CP's 1-digit results
- Classifying firms by their NAICS industry seasonal intensity

**National index (completed Mar 2026, regenerated Jul 2026 — see robustness note below)**:
all 51 states, 2000–2023, 1,011 industries. Correlation with CP's 1-digit industry
rankings: ρ = 0.836. Saved to `data/qwi_clean/seasonal_index_naics6.csv`, which
now also carries `national_industry_title` and the full sector/subsector/
industry-group title hierarchy (via the same NAICS crosswalk lookup used in
`state_index_percentiles.py`) so 6-digit codes are human-readable without a
separate join.

Top sectors by seasonal_index (mean peak_excess):
- Agriculture (0.469, 14.8 p.p.) > Arts/entertainment (0.446, 12.1 p.p.) > Accommodation/food (0.323, 8.3 p.p.) > Education (0.273, 8.4 p.p.) > Construction (0.194, 5.9 p.p.)

**Data-quality fix (Jul 2026)**: `seasonal_index.py` was missing the `sys.path`
bootstrap present in every other `code/qwi/` script, so it silently depended on
the now-deleted shadowed `code/qwi/config.py` — fixed by adding the same
bootstrap line. Regenerating the index with the current `MIN_EXCESS_OBS=5`
threshold also fixed 3 industries (Tax Preparation Services, Cotton Ginning,
Strawberry Farming) whose old `seasonal_index_naics6.csv` (Mar 2026) ranked
them #1 nationally at `seasonal_index=1.0` off of just **2** contributing
years — below the 5-year minimum — driven by anomalous 2020/2022 Q2 separation
spikes, not genuine seasonality. The old file predated the current min-obs
standard. Fixing this also nudged `state_index_percentiles.py`'s
national-overlap figure (see below) from ~13% to ~11%.

Note: Education peaks in Q2 (end of school year), not Q4. All education industries have
positive peak_excess. Mathematical identity: the four quarter excesses sum to zero within
any year, so peak_excess ≥ 0 by construction (barring thin-data filtering that excludes
the true peak quarter).

The index is **quarter-flexible**: Excess(q) is computed for all four quarters;
the primary summary measures are:
- `peak_excess`: excess separation rate at the industry's peak quarter
- `seasonal_amplitude`: max(Excess) − min(Excess), normalized to `seasonal_index` ∈ [0,1]
- `peak_quarter`: which quarter drives seasonality (1–4)
- `n_excess_obs`: year-obs contributing to the peak quarter's estimate (flag thin cells)

**Census API notes**:
- Time format returned: `2000-Q1` (parse year with `str[:4]`, quarter with `str[-1]`)
- `industry=` is a regular query param; one code per call; ~18 min per state
- 6-digit NAICS only available at state level (no national wildcard)

### COVID-Era Robustness Check

`code/qwi/covid_robustness_check.py` recomputes the national index restricted
to 2000–2019 (reusing `seasonal_index.py`'s own pipeline functions) and compares
it to the full 2000–2023 sample, to check whether pandemic-era disruption
distorts industry rankings. Outputs `output/tables/covid_robustness_naics6.csv`
and `output/figures/covid_robustness_scatter.pdf`.

**Result: rankings are robust.** Pearson correlation between full-sample and
pre-COVID `seasonal_index` = 0.997; Spearman rank correlation = 0.986; only
6.3% of industries have a different peak quarter. The "watch sectors" most
exposed to pandemic shutdowns (agriculture, construction, education, arts/
entertainment, accommodation/food) all shift by ≤0.017 in `seasonal_index`.
The single largest mover is a transportation industry (likely transit —
NAICS 485111) whose seasonal_index roughly quintuples in the full sample
versus pre-COVID, plausibly a genuine ridership-collapse effect rather than
a data artifact. Running this check is what surfaced the stale national-index
bug described above — the three thin-cell artifacts it originally flagged as
the biggest "movers" turned out to be a min-obs threshold bug, not a COVID
effect, and are now fixed at the source.

### Has Seasonality Changed Over Time?

`code/qwi/seasonality_trend_check.py` splits the sample into an early period
(2000–2010, 11 years) and a late period (2011–2023, 13 years) and recomputes
the index separately for each — the same reused-pipeline approach as the
COVID check, but testing for a genuine secular trend rather than robustness
to a single disruptive event. The 5-year `MIN_EXCESS_OBS` floor is what makes
a clean two-way split possible (a three- or four-way split would push many
industries below it); both halves here have full or near-full coverage
(median 11/11 years early, 13/13 late), so the biggest movers below aren't
thin-cell artifacts. Outputs: `output/tables/seasonality_trend_naics6.csv`,
`seasonality_trend_by_sector.csv`, `output/figures/seasonality_trend_scatter.pdf`,
`seasonality_trend_by_sector.pdf`.

**Result: correlated but with real drift — a moderate, not extreme, amount of
change.** Pearson correlation between the two periods = 0.869, Spearman rank
correlation = 0.795 (both meaningfully lower than the COVID check's
0.997/0.986, as expected — an 11-year gap between period midpoints is a much
bigger ask than excluding 2 pandemic years). The economy-wide mean
`seasonal_index` is essentially flat (0.130 → 0.126), but that average masks
a real reshuffling underneath:
- **Agriculture got more seasonal** (+0.044, the largest sector-level
  increase, 0.418 → 0.462) — plausibly consistent with the well-documented
  growth of H-2A seasonal guest-worker visa usage over the 2010s, which
  would make farm employment more sharply cyclical (temporary workers who
  leave every season) rather than year-round. Not independently verified
  against H-2A program data here — a plausible hypothesis, not a confirmed
  mechanism.
- **Construction got less seasonal** (−0.046, the largest decrease,
  0.216 → 0.170) — the opposite direction, and the single biggest sector
  shift either way.
- Most other sectors show small declines (Accommodation/food, Manufacturing,
  Transportation, Professional services, Arts/entertainment all −0.01 to
  −0.03); a few besides Agriculture increased (Public administration +0.032,
  Finance & insurance +0.028).
- 30.5% of industries have a different peak quarter between periods — much
  higher than the COVID check's 6.3%, but this concentrates almost entirely
  in weakly-seasonal industries (43.8% for the bottom quartile by
  `seasonal_index` vs. 15.1% for the top quartile), consistent with ordinary
  noise in picking a "peak" when the underlying seasonal signal is weak to
  begin with, not genuine timing instability among strongly seasonal
  industries.

### Flow vs. Stock Seasonal Index Comparison

The seasonal index has always been built two ways in parallel: **flow**
(`seasonal_index`, from `Sep/EmpEnd` separation rates — does a quarter see an
excess of *separations*?) and **stock** (`seasonal_index_emp`, from `EmpEnd`
normalized to its own annual mean — does *headcount itself* swell or shrink
that quarter?). Until now the two were merged into the same CSV but never
formally compared, and the stock side had no peak-quarter or coverage tracking
at all, so timing couldn't be compared, only amplitude.

`code/qwi/seasonal_index.py` now provides `compare_flow_vs_stock()`,
`plot_flow_vs_stock()`, and a `--compare-flow-stock {national,state,both}` CLI
flag; `code/qwi/geographic_analysis.py`'s `load_and_compute()` now computes the
stock-side measure (`emp_excessQ1-4`, `seasonal_index_emp`, `peak_quarter_emp`,
`n_emp_excess_obs`) at the state × NAICS6 level too, so the comparison can run
at either level. Outputs: `output/tables/flow_vs_stock_{national,state}.csv`
(every cell, with a `divergence = seasonal_index - seasonal_index_emp` column),
`flow_vs_stock_{national,state}_by_sector/by_state.csv`, and
`flow_vs_stock_{national,state}_scatter.pdf`.

**Key findings** (complete 51-state run):
- **Correlated but far from identical**: national Pearson r = 0.653, Spearman
  r = 0.626 (n=1,004); state × NAICS6 Pearson r = 0.526, Spearman r = 0.585
  (n=40,239). Flow and stock seasonality are related but measure genuinely
  different things.
- **Peak-quarter concordance is low — ~15% nationally, ~14% by state**: even
  when both measures agree an industry is seasonal, they usually disagree on
  *which quarter* drives it. Timing, not just amplitude, differs between
  "separations spike" and "headcount swings."
- **Sector heterogeneity is the more interesting result**: Construction shows
  near-perfect flow/stock agreement (r ≈ 0.96-0.97) — a physically direct
  layoff-and-headcount relationship. Agriculture shows the *opposite* — high
  seasonal_index on both measures individually, but low correlation between
  them (r ≈ 0.21) — consistent with rapid within-quarter worker replacement
  (separating workers are immediately replaced by newly hired ones), which
  spikes the separation rate without much net headcount movement.
- **"Stock without flow" examples are economically sensible, not artifacts**:
  Tax Preparation Services and Cotton Ginning — the two industries whose
  *flow* index was wrongly inflated to 1.0 by the stale-threshold bug fixed
  above — now correctly show a low flow index, and instead show a genuinely
  high *stock* index (large seasonal hiring surges: tax season staffing,
  harvest-time ginning), a real, sensible econ story once the index bug was
  fixed rather than a leftover artifact.

### Income/Earnings Seasonality Index

`code/qwi/earnings_index.py` builds a third seasonal measure — **income**, not
separations or headcount — using `EarnBeg` (average monthly earnings),
directly testing the paper's title question at the industry level: does
income dip in an industry's own off-season?

**Data note**: `fetch_qwi.py` originally requested `Payroll` (Total Quarterly
Payroll: Sum), but that field returns `null` from the Census API for every
cell tested (confirmed live — even huge industries where `Emp` populates
fine), with a permanent "not good" status flag. `EarnBeg` is the working
substitute and is now what `QWI_VARS` fetches. This required a fresh ~15hr
QWI refetch (all 51 states); the refetch also caught two more scripts
missing the `sys.path` bootstrap (`fetch_qwi.py` itself), and a partial test
run silently overwrote the shared combined parquet (`fetch_qwi.py` now only
writes there when every state was actually fetched — see Code Structure
notes). One transient DNS blip during the big refetch left Ohio short ~45
industries; refetched cleanly on retry.

**Methodology**: `earnings_share_{j,t,q} = EarnBeg / mean_year(EarnBeg)`, then
the same cyclical excess-by-quarter computation used everywhere else, giving
`earnings_seasonal_index` (0–1) plus `peak_quarter_earnings` /
`trough_quarter_earnings`.

**Critical finding — the raw index is confounded, not industry-specific**:
83% of all 1,011 industries have their raw earnings index peak in Q4 — a
broad, economy-wide year-end bonus/holiday-pay effect, not industry-specific
seasonality. Left uncorrected, this swamps any genuine signal: whether an
industry's *own* separation-peak quarter happens to coincide with the common
Q4 bump almost entirely determines whether its earnings look like they "dip"
there. `earnings_index.py` computes this common cross-industry calendar
effect (`compute_common_calendar_effect()`: Q1 −0.031, Q2 −0.005, Q3 −0.048,
Q4 +0.085) and nets it out (`add_idiosyncratic_excess()`) to get an
**idiosyncratic** index isolating industry-specific timing — this, not the
raw index, is the credible basis for the hypothesis test below.

**The test: does income dip in the off-season?** Merges the earnings index
onto each industry's own separation `peak_quarter` (from `seasonal_index.py`)
and checks the earnings excess at that same quarter
(`output/tables/earnings_at_separation_peak.csv`).
- **Naive/raw test: 51.9% of industries "dip"** — but this is almost entirely
  mechanical (8.3% dip when the separation peak falls in Q4, since Q4 is
  everyone's earnings peak; 93.6% "dip" when it falls in Q3, since Q3 is
  rarely anyone's earnings peak) — an artifact of the Q4-bonus confound, not
  a behavioral finding.
- **Idiosyncratic/credible test: 43.9% dip, mean excess +0.0065** (essentially
  flat, slightly positive) — once the common calendar effect is removed,
  there is **no strong evidence of industry-average income dipping** when
  separations peak.
- **Construction is the sharpest counter-example**: dip rate only 3.2%, mean
  excess **+0.044** — earnings are *higher*, not lower, at construction's own
  separation-peak quarter, plausibly reflecting end-of-season overtime pay
  ahead of winter layoffs rather than income loss.
- **Interpretation**: this doesn't contradict CP's paper — their finding is
  about income loss for the *specific worker who separates*, a micro-level
  treatment effect. This test instead asks whether an industry's *average*
  earnings (across all its still-employed workers) dip that quarter, a
  different, aggregate/compositional question the QWI data can speak to but
  the paper's worker-level SIPP analysis does not directly address.

Outputs: `data/qwi_clean/earnings_index_naics6.csv`,
`output/tables/earnings_at_separation_peak.csv`,
`output/figures/earnings_seasonal_index.pdf` (idiosyncratic top-30),
`output/figures/earnings_at_separation_peak_hist.pdf`.

### Geographic Variation

`code/qwi/geographic_analysis.py` computes peak excess for each state × industry cell
(39k+ cells with ≥5 year-observations) and produces five figures:
- `geo_state_mean_seasonality.pdf`: mean peak excess by state
- `geo_state_mean_tile_map.png` / `.pdf`: state tile-grid map of mean peak excess
- `geo_sector_variation.pdf`: cross-state SD by sector
- `geo_construction_agriculture_by_state.pdf`: construction and agriculture by state
- `geo_heatmap_state_sector.pdf`: state × sector heatmap

`seasonal_index_naics6_by_state.csv` also carries `national_industry_title`
and the sector/subsector/industry-group title hierarchy now (same crosswalk
merge as the national index), 100% matched across all 42,344 state × industry
rows.

Key finding: 2.4× gap between most seasonal (Alaska, 8.0 p.p.) and least seasonal
(Texas, 3.3 p.p.) states. Construction: MN 19.7 p.p. vs FL 2.6 p.p. (8× gap).
Implication: use state × NAICS index for firm classification when state is known.

### NAICS Title Crosswalk

`code/qwi/clean_naics_xwalk.py` cleans the Census 2022 NAICS structure workbook
(`data/naics_xwalk/2022_NAICS_Structure.xlsx`) into a NAICS6 title lookup
(`data/naics_xwalk/naics6_2022_titles.csv`) with sector/subsector/industry-group
titles attached at every level. Used to attach human-readable industry names to
the seasonal index tables (e.g. in `state_index_percentiles.py` below) — purely
a labeling convenience, no effect on the underlying index.

### State-Level Percentile / Overlap Analysis

`code/qwi/state_index_percentiles.py` identifies, **within each state**, the
top and bottom 1% of NAICS6 industries by `seasonal_index` (restricted to
cells with all four quarterly excess estimates non-missing), then checks how
those state-specific tails relate to the national ranking and to each other.
Outputs (`output/tables/`): `state_top1pct_naics6.csv`, `state_bottom1pct_naics6.csv`,
`top1pct_naics_frequency.csv`, `bottom1pct_naics_frequency.csv`,
`top1pct_sector_frequency.csv`, `bottom1pct_sector_frequency.csv`,
`state_percentile_overlap_summary.csv`, `state_percentile_pairwise_jaccard.csv`.

Key findings:
- **Most-seasonal tail is sector-concentrated but state-specific**: Agriculture
  appears in the top 1% of 96% of states, Arts/entertainment in 86%, Accommodation/food
  in 41%, Construction in 33% — but only **133 unique NAICS6 codes** ever make a
  state's top-1% tail, and just 6 of those appear in 10+ states.
- **Least-seasonal tail**: dominated by Mfg-metals (84% of states), Wholesale (67%),
  Mfg-chemicals (57%), Finance (55%).
- **Only ~11% overlap with the national ranking** (47 of 413 state top-tail slots
  are also in the national top 1%): most industries that are "most seasonal" in a
  given state are not nationally seasonal, i.e. there's substantial state-specific
  heterogeneity a national-only index would miss.
- **Low cross-state overlap** (mean pairwise Jaccard ≈ 0.05 for top tails): beyond
  the common sectors above, which specific industries make a state's top-1% list
  varies a lot state to state.
- Implication: for firm classification, prefer the **state × NAICS** index
  (`geographic_analysis.py`'s output) over the national index when the firm's
  state is known — confirms and quantifies the geographic-variation finding above.

### County-Level Extension (CBP) — in progress

`code/fetch_cbp.py` downloads County Business Patterns (establishment counts by
county × NAICS) from the Census API, as a first step toward weighting the
state × NAICS6 seasonal index down to the county level by local industry mix.
NAICS-6 county cells are heavily suppressed; the script also fetches NAICS-4
as a fallback.

**Validated but not fully run nationally.** A single-state test (`--state 06
--naics-level 6`) succeeded after fixing a path bug (`load_naics_codes()`
resolved to the wrong directory for its default NAICS code list). The test
also revealed the real cost of a full 51-state run: ~1.15s/API call means
NAICS-6 alone is ~1,012 codes × 51 states ≈ **16.5 hours**, and NAICS-4
(~308 codes) adds another ~5 hours — a ~21–22 hour full run, vs. the QWI
fetcher's ~17 hours. The fetcher is resumable (skips a state if its output
file already exists), so this can run unattended whenever it's worth
committing the time nationally. Both NAICS-6 and NAICS-4 have since been
fetched for **California only**, to build the county-level index below.

### County-Level Seasonal Exposure Index (California pilot)

`code/county_seasonal_index.py` weights the state × NAICS6 seasonal index by
each county's own industry mix (CBP establishment/employment counts) to score
"how seasonal is the average job in this county, given what industries are
actually here":

    county_exposure_c = sum_j( EMP_{c,j} × seasonal_index_j ) / sum_j( EMP_{c,j} )

Since CBP suppresses most county × NAICS6 cells, each industry's seasonal
score is looked up through a three-tier fallback, in order: (1) state ×
NAICS6 (`geographic_analysis.py`), (2) national NAICS6
(`seasonal_index.py`), (3) national NAICS4 (simple mean of the national
NAICS6 index within that 4-digit group) — used both when a county's NAICS6
cell has no index match, and to recover employment from NAICS6 cells CBP
suppresses entirely but which still report at the NAICS4 level. In the
California run this reached **100% employment coverage in every one of the
58 counties** — no county's score rests on a partial industry match.

**Case study result (Yolo County, FIPS 06113)**: ranks **40th of 58** CA
counties (32nd percentile) — *less* exposed to seasonal work than roughly
two-thirds of California counties, despite being an agricultural county.
Driver: employment-weighting means the county's large, stable
service/logistics employers (limited-service restaurants, general
warehousing, couriers — all with low seasonal_index) outweigh its genuinely
extreme agricultural niches (crop harvesting scores the maximum 1.0, but
employs only ~250 people locally vs. thousands in restaurants/warehousing).
Central Valley farm counties top the ranking (Madera 0.179, Colusa 0.170 —
roughly 2× Yolo's 0.088); the rest of the Sacramento metro area (Sacramento,
Solano, Placer) clusters near Yolo at the low-seasonality end.

Outputs: `output/tables/county_seasonal_exposure_06.csv` (one row per
county), `..._06_detail.csv` (one row per county × industry, with the
matched seasonal_index and which fallback tier supplied it — for auditing
any county's score), `output/figures/county_seasonal_exposure_06.pdf`.

Usage: `python code/county_seasonal_index.py --state 06 --highlight 06113`
(any state/county FIPS works once that state's CBP data is fetched).

### Firm-Level Module

Input: UI wage records (worker_id, employer_id, quarter, earnings)
Output: firm-level seasonality score

Two approaches:
1. **Industry lookup**: merge firm NAICS6 to QWI seasonal index
2. **Direct estimation**: compute firm-level excess Q4 separation rate from UI records

---

## Code Structure

```
code/
├── config.py                        # Paths and parameters (REPO_ROOT auto-detected)
├── fetch_cbp.py                     # County Business Patterns fetcher (county x NAICS); CA done, other states [in progress]
├── county_seasonal_index.py         # County-level seasonal exposure (CBP industry mix x QWI seasonal index)
├── replication/
│   ├── load_data.py                 # Extract zip; load .dta.gz files
│   ├── recurrence.py                # Core excess recurrence estimation (main replication)
│   └── gbt_classifier.py           # LightGBM classifier: train on CPS, apply to SIPP
├── qwi/
│   ├── fetch_qwi.py                 # Download QWI data via Census API
│   ├── seasonal_index.py            # Build 6-digit NAICS seasonal index (flow + stock, peak-flexible); flow-vs-stock comparison
│   ├── geographic_analysis.py       # State-level variation (flow + stock): figures + tables
│   ├── state_index_percentiles.py   # Within-state top/bottom-1% seasonality + national/cross-state overlap
│   ├── covid_robustness_check.py    # Pre-COVID (2000-19) vs. full-sample index comparison
│   ├── seasonality_trend_check.py   # 2000-10 vs. 2011-23: has seasonality changed over time?
│   ├── earnings_index.py            # Income seasonality (EarnBeg): does income dip in the off-season?
│   ├── clean_naics_xwalk.py         # Census NAICS structure workbook -> naics6 title lookup
│   ├── national_state_scatterplot.py  # diagnostic: state vs. national seasonal index scatter
│   ├── plot_seasonality_figures.py    # diagnostic: employment time series + amplitude histogram
│   └── naics_codes.csv              # 1,012 six-digit NAICS codes (2022 vintage)
└── firm_level/
    ├── build_events.py              # Build separation events from UI wage records
    └── classify_firms.py           # Classify firms as seasonal
```

---

## Data Requirements

### Replication
- `replication_package/238636-V1.zip` — already in repo (gitignored due to size)
- Run `code/replication/load_data.py` to extract pre-cleaned `.dta.gz` files

### QWI Extension
- Census API key (free: https://api.census.gov/data/key_signup.html)
- Set `CENSUS_API_KEY` in a `.env` file at the repo root (see `.env.example`)

### NAICS Title Crosswalk (optional)
- `data/naics_xwalk/2022_NAICS_Structure.xlsx` — Census 2022 NAICS structure workbook
  (not in repo; download from census.gov and place here to run `clean_naics_xwalk.py`)

### County-Level Extension (CBP, in progress)
- Same Census API key as the QWI extension
- `code/qwi/naics_codes.csv` (reused for the CBP NAICS code list)

### Firm-level Module
- UI wage records in parquet or CSV format (see `code/firm_level/build_events.py`)
- QWI seasonal index CSV (generated by `code/qwi/seasonal_index.py`, shareable)

---

## Setup

```bash
pip install -r requirements.txt
```

`REPO_ROOT` in `code/config.py` is auto-detected from the file's location — no
editing needed. Set `CENSUS_API_KEY` in a `.env` file at the repo root instead
(copy `.env.example`).

Then run in order:
```bash
python code/replication/load_data.py        # extract data
python code/replication/recurrence.py       # replicate core results
python code/replication/gbt_classifier.py   # (optional) retrain GBT
python code/qwi/fetch_qwi.py               # download QWI data (all states, ~17 hrs)
python code/qwi/seasonal_index.py          # build national seasonal index
python code/qwi/geographic_analysis.py     # state-level variation + figures
python code/qwi/clean_naics_xwalk.py       # (optional) build NAICS6 title lookup for labeling
python code/qwi/state_index_percentiles.py # within-state top/bottom-1% seasonality + overlap
python code/qwi/covid_robustness_check.py  # pre-COVID vs. full-sample index comparison
python code/qwi/seasonality_trend_check.py # 2000-10 vs. 2011-23: has seasonality changed over time?
python code/qwi/seasonal_index.py --compare-flow-stock both  # flow vs. stock index comparison
python code/qwi/earnings_index.py          # income/earnings seasonality + off-season income-dip test
python code/fetch_cbp.py --state 06 --naics-level 6  # county x NAICS establishment counts (CA done; other states in progress)
python code/fetch_cbp.py --state 06 --naics-level 4  # NAICS-4 fallback for suppressed county cells
python code/county_seasonal_index.py --state 06 --highlight 06113  # county-level seasonal exposure (example: Yolo, CA)
```

---

## Progress

- [x] Project structure created
- [x] Python replication of excess recurrence (CPS and SIPP)
- [x] Python replication of GBT classifier
- [x] QWI data fetcher (6-digit NAICS separations)
- [x] QWI seasonal index builder
- [x] Firm-level separation event builder
- [x] Firm-level seasonal classification
- [x] Ran replication and validated against paper — all estimates match
- [x] Fetched QWI data — all 51 states, 2000–2023, 6-digit NAICS
- [x] Built national seasonal index — 1,011 industries, ρ = 0.836 vs CP rankings
- [x] Geographic variation analysis — state × industry peak excess; 5 figures + 2 tables
- [x] Beamer presentation — output/slides/slides.tex (15 slides); sector table uses peak_excess; education note corrected (peaks Q2, positive)
- [x] NAICS title crosswalk — Census 2022 structure workbook cleaned to a NAICS6 lookup
- [x] State-level percentile/overlap analysis — within-state top/bottom 1%; only ~11% overlap with national tail; 8 output tables
- [x] COVID-era robustness check — rankings hold (Pearson 0.997, Spearman 0.986) after excluding 2020-2021; fixed a stale-threshold bug in seasonal_index.py along the way (3 industries had wrongly ranked #1 off 2 years of data, below the 5-year minimum) and a missing sys.path bootstrap that broke the script entirely
- [x] Figures/tables/final slide deck now synced to GitHub (previously all of output/ was gitignored)
- [x] Ran exploratory QWI diagnostic scripts (`national_state_scatterplot.py`, `plot_seasonality_figures.py`); fixed a pandas 3.0 `PeriodIndex` API break and a seaborn/pandas incompatibility found along the way
- [x] Deduplicated `STATE_NAMES`/`STATE_FIPS` — `fetch_qwi.py` and `geographic_analysis.py` now import from `config.py` instead of redefining locally
- [x] Validated CBP fetcher with a single-state test (fixed a NAICS-code-list path bug); full 51-state run (~21–22 hrs) deferred by choice, not run yet
- [x] Flow (separations) vs. stock (employment) seasonal index comparison — `compare_flow_vs_stock()`/`plot_flow_vs_stock()`/`--compare-flow-stock` in seasonal_index.py, stock-side state x NAICS computation added to geographic_analysis.py. Correlated but distinct (national Pearson 0.653; peak-quarter concordance only ~15%); Construction near-perfectly aligned (r≈0.96), Agriculture surprisingly not (r≈0.21, likely rapid worker replacement). Also fixed: fetch_qwi.py missing sys.path bootstrap (broke the script entirely), a partial/test fetch silently overwriting the shared all-states combined parquet (now only writes there when every state was actually fetched), and load_qwi()'s per-state fallback missing year/quarter parsing.
- [x] Refetched QWI data with EarnBeg (earnings) in place of the broken Payroll field — complete 51-state run; rebuilt the national/state indices, flow-vs-stock comparison, and state percentiles on the final complete data (all numbers stable vs. the interim 50-state validation runs)
- [x] Income/earnings seasonality index (`earnings_index.py`) — found the raw index is ~83% dominated by a common Q4-bonus effect (not industry-specific), added a de-confounding step (`compute_common_calendar_effect`/`add_idiosyncratic_excess`). Credible test: only 43.9% of industries show an income dip at their own separation-peak quarter once de-confounded (mean excess ≈ 0, not clearly negative); Construction shows the opposite (earnings *higher*, not lower, likely overtime pay before layoffs). No strong evidence of industry-average income dipping in the off-season — consistent with CP's effect being about the separating worker specifically, not the industry-wide average
- [x] County-level seasonal exposure index (`county_seasonal_index.py`) — fetched CBP NAICS-6 and NAICS-4 for California, built a 3-tier fallback (state NAICS6 -> national NAICS6 -> national NAICS4) reaching 100% employment coverage in all 58 CA counties. Case study: Yolo County ranks 40th of 58 (32nd percentile) -- less seasonal than ~2/3 of CA counties despite being an ag county, because its employment is dominated by large stable service/logistics employers, not its small-but-extreme agricultural niches
- [ ] Fetch and merge County Business Patterns (CBP) data nationally for a 51-state county-level index (California pilot validated; full run not yet launched, ~21-22hrs)
- [x] Seasonality-over-time trend check (`seasonality_trend_check.py`) — 2000-2010 vs. 2011-2023. Economy-wide mean is essentially flat (0.130 -> 0.126), but masks real reshuffling: Agriculture got notably more seasonal (+0.044, plausibly H-2A guest-worker growth, not verified), Construction notably less (-0.046). Correlation (Pearson 0.869/Spearman 0.795) is meaningfully lower than the COVID check's, as expected for an 11-year period gap vs. excluding 2 years; the elevated peak-quarter-change rate (30.5%) concentrates in weakly-seasonal industries (noise), not strongly-seasonal ones
- [ ] Apply firm-level code on other machine

---

## Replication Results

The paper presents TWO sets of recurrence estimates:

| Spec | Where | CPS | SIPP |
|---|---|---|---|
| **Raw** (no controls) | Figure 2 (main text) | ~1.4 p.p. | ~2.0 p.p. |
| **Preferred** (with controls) | Appendix Figure F.1 | ~1.5 p.p. | ~1.6 p.p. |

**Our Python replication matches both:**

| Statistic | Paper value | Our value |
|---|---|---|
| CPS raw excess recurrence | 1.4 p.p. (SE 0.14) | 1.44 p.p. (SE 0.14) ✓ |
| CPS preferred (weeks_elapsed) | 1.5 p.p. (SE 0.14) | 1.42 p.p. (SE 0.14) ✓ |
| SIPP raw excess recurrence | 2.0 p.p. (SE 0.11) | 2.04 p.p. (SE 0.11) ✓ |
| SIPP preferred (seam+five) | 1.6 p.p. (SE 0.11) | 1.62 p.p. (SE 0.11) ✓ |

**Industry rankings** (both datasets): Agriculture > Education > Entertainment > Construction at top; Healthcare and FIRE at bottom.

**Seam coding note**: `srefmon` in the pre-cleaned data cycles 1→4 within each 4-month wave. srefmon=1 marks the first month of each wave (right after the seam boundary). Seam variables: `first_seam = (srefmon==1) & (k<=4)`, `later_seam = (srefmon==1) & (k>=5)`.

---

## Key Numbers to Validate Against Paper

| Statistic | Paper value | Our value |
|---|---|---|
| CPS raw (Figure 2) | 1.4 p.p. (SE 0.14) | 1.44 ✓ |
| SIPP raw (Figure 2) | 2.0 p.p. (SE 0.11) | 2.04 ✓ |
| CPS preferred (App. F.1) | 1.5 p.p. (SE 0.14) | 1.42 ✓ |
| SIPP preferred (App. F.1) | 1.6 p.p. (SE 0.11) | 1.62 ✓ |
| Construction excess recurrence | highest industry | Confirmed ✓ |
| Education excess recurrence | high | Confirmed ✓ |

---

## Notes

- Original code requires Stata 15+; our Python replication uses pandas + statsmodels + lightgbm.
- The pre-computed GBT predictions (`gbt_predictions_sipp.dta`) are in the replication zip
  and can be used directly, bypassing the need to retrain the model.
- QWI data is quarterly; CP data is monthly. The QWI seasonal index is therefore a
  coarser measure, but useful for industry-level classification.
- Firm-level code is designed to be self-contained: copy `code/firm_level/` and the
  QWI seasonal index CSV to the other machine.
