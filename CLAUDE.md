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

**National index (completed Mar 2026)**: all 51 states, 2000–2023, 1,011 industries.
Correlation with CP's 1-digit industry rankings: ρ = 0.826.
Saved to `data/qwi_clean/seasonal_index_naics6.csv`.

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

### Geographic Variation

`code/qwi/geographic_analysis.py` computes peak excess for each state × industry cell
(39k+ cells with ≥5 year-observations) and produces four figures:
- `geo_state_mean_seasonality.pdf`: mean peak excess by state
- `geo_sector_variation.pdf`: cross-state SD by sector
- `geo_construction_agriculture_by_state.pdf`: construction and agriculture by state
- `geo_heatmap_state_sector.pdf`: state × sector heatmap

Key finding: 2.4× gap between most seasonal (Alaska, 8.0 p.p.) and least seasonal
(Texas, 3.3 p.p.) states. Construction: MN 19.7 p.p. vs FL 2.6 p.p. (8× gap).
Implication: use state × NAICS index for firm classification when state is known.

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
├── config.py                    # Paths and parameters (edit before running)
├── replication/
│   ├── load_data.py             # Extract zip; load .dta.gz files
│   ├── recurrence.py            # Core excess recurrence estimation (main replication)
│   └── gbt_classifier.py       # LightGBM classifier: train on CPS, apply to SIPP
├── qwi/
│   ├── fetch_qwi.py             # Download QWI data via Census API
│   ├── seasonal_index.py        # Build 6-digit NAICS seasonal index (peak-flexible)
│   ├── geographic_analysis.py   # State-level variation: figures + tables
│   └── naics_codes.csv          # 1,012 six-digit NAICS codes (2022 vintage)
└── firm_level/
    ├── build_events.py          # Build separation events from UI wage records
    └── classify_firms.py       # Classify firms as seasonal
```

---

## Data Requirements

### Replication
- `replication_package/238636-V1.zip` — already in repo (gitignored due to size)
- Run `code/replication/load_data.py` to extract pre-cleaned `.dta.gz` files

### QWI Extension
- Census API key (free: https://api.census.gov/data/key_signup.html)
- Set `CENSUS_API_KEY` in `code/config.py`

### Firm-level Module
- UI wage records in parquet or CSV format (see `code/firm_level/build_events.py`)
- QWI seasonal index CSV (generated by `code/qwi/seasonal_index.py`, shareable)

---

## Setup

```bash
pip install -r requirements.txt
```

Edit `code/config.py` to set:
- `REPO_ROOT`: path to this directory
- `CENSUS_API_KEY`: your Census API key

Then run in order:
```bash
python code/replication/load_data.py        # extract data
python code/replication/recurrence.py       # replicate core results
python code/replication/gbt_classifier.py   # (optional) retrain GBT
python code/qwi/fetch_qwi.py               # download QWI data (all states, ~17 hrs)
python code/qwi/seasonal_index.py          # build national seasonal index
python code/qwi/geographic_analysis.py     # state-level variation + figures
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
- [x] Built national seasonal index — 1,011 industries, ρ = 0.826 vs CP rankings
- [x] Geographic variation analysis — state × industry peak excess; 4 figures + 2 tables
- [x] Beamer presentation — output/slides/slides.tex (15 slides)
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
