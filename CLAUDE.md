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
rankings: ρ = 0.836. Saved to `data/qwi_clean/seasonal_index_naics6.csv`.

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

### Geographic Variation

`code/qwi/geographic_analysis.py` computes peak excess for each state × industry cell
(39k+ cells with ≥5 year-observations) and produces five figures:
- `geo_state_mean_seasonality.pdf`: mean peak excess by state
- `geo_state_mean_tile_map.png` / `.pdf`: state tile-grid map of mean peak excess
- `geo_sector_variation.pdf`: cross-state SD by sector
- `geo_construction_agriculture_by_state.pdf`: construction and agriculture by state
- `geo_heatmap_state_sector.pdf`: state × sector heatmap

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
- **Only ~11% overlap with the national ranking** (45 of 413 state top-tail slots
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

**Validated but not fully run.** A single-state test (`--state 06 --naics-level 6`)
succeeded after fixing a path bug (`load_naics_codes()` resolved to the wrong
directory for its default NAICS code list). The test also revealed the real
cost of a full run: ~1.15s/API call means NAICS-6 alone is ~1,012 codes × 51
states ≈ **16.5 hours**, and NAICS-4 (~308 codes) adds another ~5 hours — a
~21–22 hour full run, vs. the QWI fetcher's ~17 hours. The fetcher is resumable
(skips a state if its output file already exists), so this can run unattended
whenever it's worth committing the time.

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
├── fetch_cbp.py                     # [in progress] County Business Patterns fetcher (county x NAICS)
├── replication/
│   ├── load_data.py                 # Extract zip; load .dta.gz files
│   ├── recurrence.py                # Core excess recurrence estimation (main replication)
│   └── gbt_classifier.py           # LightGBM classifier: train on CPS, apply to SIPP
├── qwi/
│   ├── fetch_qwi.py                 # Download QWI data via Census API
│   ├── seasonal_index.py            # Build 6-digit NAICS seasonal index (peak-flexible)
│   ├── geographic_analysis.py       # State-level variation: figures + tables
│   ├── state_index_percentiles.py   # Within-state top/bottom-1% seasonality + national/cross-state overlap
│   ├── covid_robustness_check.py    # Pre-COVID (2000-19) vs. full-sample index comparison
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
python code/fetch_cbp.py                   # (in progress) county x NAICS establishment counts
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
- [ ] Fetch and merge County Business Patterns (CBP) data for a county-level index (validated, ~21-22hr full run not yet launched)
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
