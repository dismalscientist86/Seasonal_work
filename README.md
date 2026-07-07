# Seasonal Work

Defining seasonal work at the industry and firm level. Replication and extension of
Coglianese & Price (2025), "Income in the Off-Season: Household Adaptation to Yearly
Work Interruptions," *Industrial and Labor Relations Review*.

See [CLAUDE.md](CLAUDE.md) for full methodology, findings, and setup details. This
file is a short orientation.

## What's here

- **Replication** (`code/replication/`): Python port of the paper's Stata-only excess
  recurrence methodology, validated against the paper's published estimates.
- **QWI extension** (`code/qwi/`): a 6-digit NAICS, quarter-flexible seasonal index
  built from Census QWI separations data — national, state-level, and within-state
  percentile/overlap analyses.
- **County-level extension** (`code/fetch_cbp.py`, in progress): County Business
  Patterns data for weighting the state-level index down to counties.
- **Firm-level module** (`code/firm_level/`): classifies individual firms as seasonal
  from UI wage records, either via industry lookup or direct estimation.

## Results

Figures and tables in [output/figures/](output/figures/) and [output/tables/](output/tables/)
are synced to this repo (the underlying `data/` is not, due to size). Headline finding:
seasonality varies enormously by state and by industry within a state — e.g. an 8x gap
in construction seasonality between Minnesota and Florida — so a national-only seasonal
index misses most of the state-specific pattern.

The Beamer slide deck summarizing results is at [output/slides/slides.pdf](output/slides/slides.pdf).

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `CENSUS_API_KEY` (free key:
https://api.census.gov/data/key_signup.html). See [CLAUDE.md](CLAUDE.md#setup) for
the full run order.
