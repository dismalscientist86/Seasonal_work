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

**National index (completed Mar 2026, regenerated Jul 2026 twice — see robustness
notes below)**: all 51 states, 2000–2023, 1,011 industries. Correlation with CP's
1-digit industry rankings: ρ = 0.829. Saved to `data/qwi_clean/seasonal_index_naics6.csv`,
which now also carries `national_industry_title` and the full sector/subsector/
industry-group title hierarchy (via the same NAICS crosswalk lookup used in
`state_index_percentiles.py`) so 6-digit codes are human-readable without a
separate join.

**Headline sector ranking excludes agriculture (2026-09)**: QWI's UI-based
coverage of agricultural employment is historically weaker/partial relative
to other industries, so agriculture's QWI-covered establishments may be a
non-representative (more seasonal) slice of the true agricultural
workforce — see "Agriculture Exclusion Check" below for the full
sensitivity analysis. With agriculture excluded and the index
re-winsorized:
- Arts/entertainment (0.551, 14.3 p.p.) > Accommodation/food (0.395, 10.3 p.p.) > Education (0.332, 8.4 p.p.) > Construction (0.236, 5.9 p.p.) > Real estate (0.193, 5.6 p.p.)

If agriculture is included, it ranks #1 (0.504, 15.7 p.p.) — see below.

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

**Second data-quality fix (Jul 2026)**: while building an illustrative "most
vs. least seasonal" employment figure (`plot_seasonality_figures.py`), a
handful of "least seasonal" picks turned out to have data in only **one** of
four quarters (e.g. Potato Farming: 24 years of Q2 data, zero in Q1/Q3/Q4).
`seasonal_amplitude = max(Excess) - min(Excess)` collapses to exactly 0 when
only one quarter has a real value — a data-thinness artifact, not genuine
flatness — and was silently reported as `seasonal_index = 0.0`, the most
extreme possible "non-seasonal" score. `build_seasonal_index()` (and its
stock-side, state-level, and earnings-index analogs) now requires **≥2**
non-missing quarters before reporting amplitude/peak/index, NaN otherwise.
Nationally this affected only 6 of 1,011 industries (incl. Tax Preparation
Services and Cotton Ginning, already flagged above for a *different* reason —
both are also thin on the flow side). At the **state** level the same bug was
far more consequential: **2,898 of 42,344 state × industry cells (6.8%)** had
this single-quarter problem, vs. 0.6% nationally — thin cells are much more
common once you slice by state. Regenerating the state index shifted real
headline numbers (see Geographic Variation and Flow vs. Stock below), since
small/sparse states were disproportionately affected. `plot_seasonality_figures.py`
also now labels its example industries by name (was NAICS code only) and adds
a belt-and-suspenders check requiring all four `n_excessQ*` counts to be
positive, not just the existing `n_excess_obs` (peak-quarter-only) floor.

One of the "least seasonal" picks that survived all of the above filters,
HMO Medical Centers (621491), still wasn't a clean illustration — its
employment series has a sharp 2015–2019 level shift (likely a merger or
reclassification effect) that has nothing to do with seasonality. Rather
than hand-edit the output, `select_industries_for_plot()` now takes an
`exclude_codes` list (default `DEFAULT_EXCLUDE_CODES`, currently just
621491) so this and any future exclusions are explicit and reproducible.
Artificial and Synthetic Fibers and Filaments Manufacturing (325220) fills
the slot instead — a clean flat/declining trend.

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
6.7% of industries have a different peak quarter. The "watch sectors" most
exposed to pandemic shutdowns (agriculture, construction, education, arts/
entertainment, accommodation/food) all shift by ≤0.017 in `seasonal_index`.
The single largest mover is a transportation industry (likely transit —
NAICS 485111) whose seasonal_index is roughly 4.7x higher in the full sample
versus pre-COVID, plausibly a genuine ridership-collapse effect rather than
a data artifact. Running this check is what surfaced the stale national-index
bug described above — the three thin-cell artifacts it originally flagged as
the biggest "movers" turned out to be a min-obs threshold bug, not a COVID
effect, and are now fixed at the source.

(Numbers refreshed 2026-09 after a full pipeline re-run; the 2020-2021
QWI data has apparently been revised slightly since this check was first
run, nudging the peak-quarter-changed share from 6.3% to 6.7% and the transit
ratio from ~5x to ~4.7x — same substantive conclusion, not a code change.)

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

**Result (revised 2026-09 — see correction note below): correlated but with
real drift.** Pearson correlation between the two periods = 0.931, Spearman
rank correlation = 0.826 (both meaningfully lower than the COVID check's
0.997/0.986, as expected — an 11-year gap between period midpoints is a much
bigger ask than excluding 2 pandemic years). The economy-wide mean
`seasonal_index` is essentially flat (0.131 → 0.122), but that average masks
a real reshuffling underneath:
- **Educational services fell most** (−0.053, 0.272 → 0.219) — now the
  single largest sector-level shift in either direction.
- **Construction kept falling** (−0.047, 0.216 → 0.169) — essentially
  unchanged from the original run of this check.
- **Agriculture also fell** (−0.037, 0.504 → 0.466) — see correction note.
- Most other sectors show small declines (Accommodation/food, Manufacturing,
  Transportation, Professional services, Arts/entertainment all −0.01 to
  −0.03); **Finance & insurance (+0.028) and Public administration (+0.032)**
  are the two sectors that rose the most — unchanged from the original run.
- 30.2% of industries have a different peak quarter between periods — much
  higher than the COVID check's 6.7%, but this concentrates almost entirely
  in weakly-seasonal industries (42.9% for the bottom quartile by
  `seasonal_index` vs. 15.0% for the top quartile), consistent with ordinary
  noise in picking a "peak" when the underlying seasonal signal is weak to
  begin with, not genuine timing instability among strongly seasonal
  industries.

**Correction (2026-09):** a full pipeline re-run (prompted by a request to
refresh all results and the slide deck) turned up a real, not cosmetic,
change here: this check had originally reported **Agriculture as the largest
increaser** (+0.044, 0.418 → 0.462), with a floated hypothesis that H-2A
guest-worker growth was driving it. Re-running against the current QWI raw
data (unchanged code) now shows Agriculture's *early*-period mean at 0.504 —
not 0.418 — meaning the underlying 2000-2010 QWI data for agricultural
industries has evidently been revised/expanded since that original run (most
likely as a side effect of the full 51-state EarnBeg refetch documented
below, which wasn't followed by a rebuild of this specific check — unlike
the national/state indices, flow-vs-stock, and state percentiles, which
*were* explicitly rebuilt post-refetch). The Finance/insurance and Public
administration numbers were unaffected (unchanged to 3 decimals), suggesting
the revision was concentrated in agriculture- and education-adjacent QWI
cells specifically. **Retract the H-2A hypothesis** — Agriculture now reads
as *less* seasonal in the later period, the opposite direction — and treat
any seasonality-trend finding as good only as of its last actual run against
current data.

### Agriculture Exclusion Check

Feedback received (2026-09): agricultural UI coverage in the QWI is
historically weaker/partial relative to most industries (some states
exempt small agricultural employers, and coverage was extended to
agriculture later than to other sectors in several states). If the
QWI-covered slice of agricultural establishments is a non-representative
(more seasonal, more UI-compliant) subset of the true agricultural
workforce, the sector could look more seasonal in this index than
agricultural employment as a whole actually is. `code/qwi/agriculture_exclusion_check.py`
asks how much of the headline ranking depends on agriculture (NAICS sector
11 -- crop production, animal production, forestry/logging, fishing/
hunting/trapping, and support activities for agriculture and forestry).

Unlike the COVID/trend checks, this doesn't reload raw QWI or recompute
excess-by-quarter: each industry's `seasonal_amplitude`/`peak_excess` is
computed independently of every other industry, so dropping agriculture's
rows from the already-committed clean CSVs is methodologically identical
to excluding it from the QWI input before the pipeline runs. The one thing
that does need recomputing is `seasonal_index` itself, since it's built by
winsorizing `seasonal_amplitude` at the 99th percentile of whichever pool
of industries is included -- removing agriculture (some of the largest
amplitudes nationally) shifts that threshold for everyone else, so the
script re-winsorizes after dropping agriculture rather than just deleting
them from the existing index column.

**Result: sector ranking reshuffles, but state-level rankings barely
move.** Agriculture (0.504) drops out; **Arts, entertainment & recreation
becomes the new #1 sector** (0.461 -> 0.551 after re-winsorization),
followed by Accommodation & food (0.395), Educational services (0.332),
Construction (0.236) -- every remaining sector's index rises somewhat
mechanically since the 99th-percentile threshold falls once agriculture's
large amplitudes are removed. The "most seasonal individual industries"
list changes completely without agriculture: Drive-In Motion Picture
Theaters, Recreational Goods Rental, Mobile Food Services, Health and
Welfare Funds, All Other Amusement/Recreation, Beet Sugar Manufacturing,
RV Parks & Campgrounds, Racetracks, Educational Support Services, and
Scenic/Sightseeing Water Transportation.

At the **state level**, full-sample and ex-agriculture rankings correlate
at Pearson r=0.991, Spearman r=0.989 -- agriculture is not doing much work
in the state-level headline finding. **Alaska remains the most seasonal
state either way** (9.58 -> 9.45 p.p., barely changed) and **Texas remains
least seasonal** (3.43 -> 2.99 p.p.); the max/min gap actually widens
slightly (2.8x -> 3.2x), since Texas's non-agricultural seasonality is even
lower relatively. This implies Alaska's high seasonality is driven mainly
by *non-agricultural* seasonal industries (plausibly tourism/fishing-
adjacent services), not farming specifically. The states most sensitive to
dropping agriculture: Oregon drops 8 ranks (18th -> 26th); Michigan and
Mississippi also drop several ranks; West Virginia and Utah rise.

**Decision (2026-09):** given how little the state-level story depends on
agriculture, and the genuine UI-coverage concern raised in feedback,
excluding agriculture is now the primary/default presentation throughout
this project -- `geographic_analysis.py`'s headline figures (state bar
chart, tile map, sector-variation chart, state x sector heatmap) and both
slide decks' National Seasonality / Sector Summary / Geographic Variation
content all exclude it. The full sample (agriculture included) is kept as
a documented sensitivity check, not deleted -- `data/qwi_clean/seasonal_index_naics6.csv`
and `seasonal_index_naics6_by_state.csv` are unchanged and still carry
agriculture's rows, and `agriculture_exclusion_check.py`'s outputs plus a
dedicated slide in both decks present the with-agriculture comparison.

Outputs: `output/tables/agriculture_exclusion_sector_summary.csv`,
`agriculture_exclusion_top_industries.csv`, `agriculture_exclusion_state_ranking.csv`,
`output/figures/agriculture_exclusion_state_comparison.pdf/png`.

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

**Key findings** (complete 51-state run, regenerated after the degenerate-quarter
fix above — this comparison moved more than most other outputs, since a false
flow=0 or stock=0 in either measure mechanically drags down their correlation):
- **More correlated than first reported**: national Pearson r = 0.784, Spearman
  r = 0.66 (n=997, up from 0.653/0.626 pre-fix); state × NAICS6 Pearson r = 0.75,
  Spearman r = 0.661 (n=39,446, up from 0.526/0.585). Flow and stock seasonality
  are more closely related than the pre-fix numbers suggested, though still not
  identical.
- **Peak-quarter concordance is low — ~15% nationally, ~13% by state — and this
  part didn't change**: even when both measures agree an industry is seasonal,
  they usually disagree on *which quarter* drives it. Timing, not just
  amplitude, differs between "separations spike" and "headcount swings."
- **Sector heterogeneity, revised**: Construction still shows near-perfect
  flow/stock agreement (r = 0.966, n=31, essentially unchanged) — a physically
  direct layoff-and-headcount relationship. Agriculture no longer looks
  decoupled: r = 0.642 (n=55), up sharply from an original 0.207. That 0.207
  was itself a symptom of the bug above — Potato Farming, Apple Orchards, and
  Sugar Beet Farming (all degenerate, single-quarter cells) were dragging the
  correlation down. Agriculture's flow/stock relationship is still weaker than
  Construction's, just not the "near-zero, rapid worker replacement" story
  originally reported — treat that hypothesis as retracted pending a fresh look
  at the corrected data.
- **"Stock without flow" examples, updated**: Tax Preparation Services no
  longer appears in this comparison at all — its flow index is now correctly
  NaN (it was one of the 6 degenerate national industries). Cotton Ginning's
  story holds (flow 0.009, stock 0.823) and pairs well with Political
  Organizations (flow 0.176, stock 0.826) — both show a genuine stock-side
  hiring/staffing surge (harvest-time ginning; election-cycle staffing) not
  matched by a comparable separation-rate spike.

### Monthly Resolution via QCEW: Does Quarterly Averaging Hide Seasonality?

Suggestion received (2026-09): use BLS's Quarterly Census of Employment and
Wages (QCEW) instead of / alongside QWI, since QCEW reports true monthly
employment (`month1_emplvl`/`month2_emplvl`/`month3_emplvl` per quarter),
closer to CP's month-level specification than our quarterly QWI analog.
**Important scope caveat, confirmed before building anything**: QCEW has no
separations/hires field at all -- it is a pure employment-*stock* census. It
cannot replicate this project's flow-based `seasonal_index` (excess
separation rate); it can only sharpen the *timing* resolution of the
existing **stock** measure (`seasonal_index_emp`) from quarterly to monthly.

`code/qcew/fetch_qcew.py` downloads QCEW's bulk annual "singlefile" CSVs
(`https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip`,
~300MB compressed each, no API key needed) and filters to national + state
6-digit-NAICS rows (`agglvl_code` 18/58), Private ownership (`own_code`
"5" -- the standard, most-complete-coverage choice), streaming each zip's
CSV member in chunks so the ~1.5-3GB uncompressed file per year is never
written to disk. All 24 years (2000-2023) fetched in well under an hour --
much faster than QWI's ~17hr per-state-per-industry API loop, since QCEW
ships one bulk file per year rather than requiring one API call per
state x industry.

`code/qcew/monthly_seasonal_index.py` builds the monthly analog of
`seasonal_index_emp`: `emp_share_{j,y,m} = emplvl_{j,y,m} / mean_m(emplvl_{j,y,*})`,
then the same shared cyclical-excess calculation used everywhere else in
this project, generalized from 4 periods to 12
(`excess_utils.cyclical_excess_by_month()`, added alongside the existing
`cyclical_excess_by_quarter()` -- both now thin wrappers over a new
`cyclical_excess_by_period(n_periods, label)`; refactor verified
byte-identical to the pre-refactor quarterly output on a full rerun, and
the December-January wraparound verified directly with a synthetic test).

**Two confounds found and corrected before trusting any headline number**
(published dataset carries all three layers so the corrections are
auditable -- see the file's own codebook):

1. **Common calendar effect.** 47.4% of industries have their *raw*
   monthly peak in December, 11.0% in January -- QCEW, like QWI, is not
   seasonally adjusted, and raw U.S. employment data has a well-documented
   broad, economy-wide December build-up/January pull-back that isn't
   industry-specific. This is the exact same class of confound
   `earnings_index.py` found for earnings (83% of industries peaking in
   Q4 there). `compute_common_calendar_effect()`/`add_idiosyncratic_excess()`
   net it out, mirroring that precedent exactly, into `idio_monthly_*`
   columns.
2. **Sample-size artifact.** A raw comparison of monthly vs. quarterly
   amplitude (even after step 1) still showed monthly seasonality higher
   for most industries -- but `max - min` over 12 monthly draws is
   *mechanically* larger than over 4 quarterly draws for the same
   underlying noise, since the expected range of a sample grows with its
   size even absent any true seasonal pattern. A placebo test
   (`compute_placebo_amplitude()`: permute month labels within each year,
   10 reps/industry, recompute the same idiosyncratic-excess calculation)
   confirms this is large: **mean placebo (null) amplitude is 89.7% of
   mean idiosyncratic amplitude**, and idiosyncratic amplitude exceeds its
   own industry's placebo in only 41.5% of industries.
   `add_placebo_correction()` nets this out too
   (`signal_amplitude = max(idio_amplitude - placebo_amplitude, 0)`,
   re-winsorized into `signal_index`) -- **the recommended column**, net
   of both confounds. 660 of 1,364 industries (48.4%) land at
   `signal_index = 0`: their apparent monthly seasonality was entirely
   explained by the two confounds combined, not genuine signal.

**Result, using `signal_index`**: correlation with the quarterly stock
index drops sharply once corrected (raw Pearson 0.788, Spearman 0.748 ->
corrected Pearson 0.444, Spearman 0.398) -- quarterly and monthly
seasonality are much more weakly related than the raw numbers implied.
Peak-quarter concordance (idiosyncratic peak month's containing quarter
vs. QWI's own peak quarter) is only 38.3%, similar in spirit to the
flow-vs-stock comparison's low timing concordance above. Two genuinely
different kinds of industries emerge:
- **Industries whose apparent monthly seasonality was entirely
  confound** (`signal_index` corrects to 0 despite a high raw monthly
  index): Skiing Facilities, RV Parks and Campgrounds, Drive-In Motion
  Picture Theaters, Recreational Goods Rental, Golf Courses,
  Scenic/Sightseeing Water Transportation, Marinas. These industries'
  seasonality genuinely operates at a quarterly-or-longer time scale (a
  whole ski season, a whole boating season), so quarterly QWI data
  already captures them well -- refining to monthly resolution adds
  noise, not information.
- **Industries with a robust, confound-surviving monthly-specific signal
  that quarterly data misses almost entirely**: **School and Employee Bus
  Transportation** is the standout example -- `signal_index = 1.00` (ties
  the winsorization ceiling) vs. a quarterly stock index of just **0.11**.
  The school year's abrupt September start gets averaged away with a much
  quieter July-August within the same Q3, so quarterly aggregation nearly
  erases a genuinely large, sharply-timed seasonal swing. Photography
  Studios/Portrait (December holiday portraits, 0.94 vs. 0.15 quarterly)
  and Corn Farming (July, 1.00 vs. 0.41 quarterly) show the same pattern.

**Implication**: this validates and quantifies the "quarterly is a coarser
measure of timing than CP's month-level data" caveat already noted in the
slide decks -- but the effect is concentrated in a specific, identifiable
kind of industry (a sharp calendar-driven event that falls mid-quarter or
spans a quarter boundary), not a uniform "everything looks more seasonal
monthly" story. An uncorrected monthly-vs-quarterly comparison would have
overstated this effect substantially and been actively misleading about
*which* industries it applies to.

A third, unrelated data-quality issue surfaced while building this:
`national_industry_title` is missing for ~30% of rows, because QCEW's
`industry_code` reflects whichever NAICS revision was current when each
year's data was published, while the title crosswalk is fixed at the 2022
vintage -- confirmed directly (e.g. code `211111`, retired by NAICS 2022,
appears through 2010 but not 2023; its replacement `211120` appears only
from 2023). Not corrected (would require reconciling pre- and post-2022
NAICS vintages into one continuous series); documented in the file's
codebook instead.

Published to the public repo: `data/qcew_clean/monthly_seasonal_index_naics6.csv`
with its own codebook (see "Public Index Files" above). Comparison outputs:
`output/tables/monthly_vs_quarterly_stock_comparison.csv`,
`output/figures/monthly_vs_quarterly_stock_scatter.pdf/png` (two-panel:
raw vs. bias-corrected).

### Hire (Accessions) Seasonal Index — code ready, awaiting refetch

`code/qwi/hire_index.py` builds a third gross-flow measure — **hiring**, from
QWI's `HirA` ("Hires All: Counts (Accessions)") — as the direct in-flow
counterpart to the separations-based flow index. `hire_rate = HirA / EmpEnd`,
then the same shared cyclical excess-by-quarter calc, then a 0-1
`hire_seasonal_index`. Unlike `sep_rate`, `hire_rate` is *not* truncated at
>=1 (a quarter's hires can legitimately exceed end-of-quarter headcount for a
ramping-up seasonal operation), so extreme values are left to the 99th-pct
winsorization instead.

The point is the **timing** vs. separations: `compare_hire_vs_separation_timing()`
computes `hire_to_sep_lag_quarters = (peak_quarter - peak_quarter_hire) % 4`
(0 = same quarter → likely high churn, same jobs different workers each cycle;
1-3 = separation follows the hiring peak by that many quarters — e.g. retail's
hire-Q4/separate-Q1) and a `likely_churn` flag (seasonal hiring AND
separations, but flat seasonal headcount).

`HirA` was **verified live** against the Census API (2026-09): populates with
sensible magnitudes and a "good" status flag on both a large and a thin cell,
unlike the broken Payroll field. It has been added to `QWI_VARS` in
`fetch_qwi.py`, but the existing `qwi_state_*.parquet` files predate it —
**`hire_index.py` needs a fresh full 51-state refetch (~17 hrs) before it can
run for real**, and raises a clear error otherwise. The full pipeline (rates →
excess → index → timing comparison → 3 figures) was dry-run-tested end to end
on real-scale data by standing in `Sep` for the not-yet-fetched `HirA`.

**Shared refactor (2026-09):** the cyclical excess-by-quarter calculation was
copy-pasted four times (separations, employment/stock, earnings, and
`geographic_analysis.py`'s state-level version). It now lives once in
`code/qwi/excess_utils.py::cyclical_excess_by_quarter()`; `seasonal_index.py`
(both functions), `earnings_index.py`, `geographic_analysis.py`, and
`hire_index.py` all call it. Verified byte-identical output on full reruns of
the first three against the pre-refactor committed CSVs. The per-caller
groupby-loop wrappers and "build a 0-1 index from the excess table" steps stay
separate (their output column names and, for earnings, extra trough columns,
differ enough that one generic function would cost more than it saves).

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
- **Naive/raw test: 52.3% of industries "dip"** — but this is almost entirely
  mechanical (8.3% dip when the separation peak falls in Q4, since Q4 is
  everyone's earnings peak; 93.6% "dip" when it falls in Q3, since Q3 is
  rarely anyone's earnings peak) — an artifact of the Q4-bonus confound, not
  a behavioral finding.
- **Idiosyncratic/credible test: 44.2% dip, mean excess +0.0055** (essentially
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

**Headline figures exclude agriculture (2026-09)** — see "Agriculture
Exclusion Check" below; `make_figures()` computes `geo_state_mean_seasonality`,
the tile map, `geo_sector_variation`, and the state×sector heatmap from the
data with NAICS sector 11 dropped, and re-labels them accordingly.
`geo_construction_agriculture_by_state.pdf` still uses the full sample,
since it's explicitly and only about agriculture's own within-sector
variation, shown for context.

Key finding: 3.2× gap between most seasonal (Alaska, 9.5 p.p.) and least seasonal
(Texas, 3.0 p.p.) states, excl. agriculture (up from 2.8× / 9.6 vs. 3.4 p.p.
when agriculture is included — see sensitivity check below). Earlier still,
this ratio was 2.4× (8.0 vs. 3.3 p.p.) before the degenerate-quarter fix
above; small/sparse states like Alaska had more single-quarter cells to
correct than large ones, so its true mean rose more.
Construction: MN 19.7 p.p. vs FL 2.6 p.p. (8× gap, unchanged) — Alaska newly
enters construction's state top-5 (18.7 p.p.) post-fix, previously absent.
Cross-state variation (SD), excl. agriculture, is now led by Arts/entertainment
(13.4 p.p.) and Accommodation/food (10.8 p.p.); Construction (9.6 p.p.) is
3rd, ahead of Educational services (9.3 p.p.), which enters the top 4 once
agriculture (formerly #1 at 14.5 p.p.) is dropped.
Implication: use state × NAICS index for firm classification when state is known.

**Data-coverage caveat (2026-09)**: per-state QWI year coverage is uneven.
40 of 51 states have the full 2000–2023 window, but 11 have a partial
range — most notably **Alaska (2000–2016 only, missing 2017–2023)** and
**Massachusetts (2010–2023 only, missing 2000–2009)**; the other 9 (AL, AZ,
AR, DC, KY, MI, MS, NH, WY) are missing 1–4 years each, mostly at the start
of the series. Verified live against the Census API (not a local fetch bug):
QWI returns `204 No Data` for Alaska/Massachusetts in these missing windows
for every industry tested, consistent with each state's own QWI/LEHD
data-sharing participation history rather than a gap in our pipeline. This
matters most for Alaska, since its 9.6 p.p. figure above — the headline
"most seasonal state" — reflects only 2000–2016 and includes nothing from
2017 onward.

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

**Excludes agriculture by default (2026-09)**, both from the within-state tail
selection and the national top/bottom-percentile flagging used for overlap —
same rationale as `agriculture_exclusion_check.py`/`geographic_analysis.py`
(`--include-agriculture` reproduces the full-sample version). Fixed a real bug
along the way: `seasonal_index_naics6_by_state.csv` already carries NAICS title
columns (merged in by `geographic_analysis.py`), so `add_naics_titles()`
re-merging the same crosswalk unconditionally collided and got silently
suffixed (`_x`/`_y`) by pandas, dropping the plain `national_industry_title`
column the frequency tables need — `add_naics_titles()` now only merges
columns not already present.

Key findings (ex-agriculture):
- **Most-seasonal tail is sector-concentrated but state-specific**:
  Arts/entertainment appears in the top 1% of 92% of states, Manufacturing-food
  in 53%, Accommodation/food in 49%, Other services in 45%, Construction in
  37% — but only **124 unique NAICS6 codes** ever make a state's top-1% tail
  (397 state-industry slots total), and 8 of those appear in 10+ states
  (Marinas 16, Ice Manufacturing 14, Specialty Trade Contractors 12,
  Racetracks 12, Golf Courses/Country Clubs 11).
- **Least-seasonal tail**: dominated by Mfg-metals (82% of states), Wholesale
  (69%), Mfg-chemicals (61%), Finance (51%), Mfg-food (39%) — essentially
  unchanged from the full-sample version, as expected since agriculture rarely
  appears in a "least seasonal" tail.
- **~16% overlap with the national ranking** (63 of 397 state top-tail slots
  are also in the national top 1%, up from ~11%/47 of 413 with agriculture
  included): most industries that are "most seasonal" in a given state are
  still not nationally seasonal, i.e. there's substantial state-specific
  heterogeneity a national-only index would miss.
- **Low cross-state overlap** (mean pairwise Jaccard ≈ 0.059 for top tails,
  0.013 for bottom): beyond the common sectors above, which specific
  industries make a state's top-1% list varies a lot state to state.
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

**Case study result (Yolo County, FIPS 06113)**: ranks **41st of 58** CA
counties (30th percentile) — *less* exposed to seasonal work than roughly
two-thirds of California counties, despite being an agricultural county.
Driver: employment-weighting means the county's large, stable
service/logistics employers (limited-service restaurants, general
warehousing, couriers — all with low seasonal_index) outweigh its genuinely
extreme agricultural niches (crop harvesting scores the maximum 1.0, but
employs only ~250 people locally vs. thousands in restaurants/warehousing).
Central Valley farm counties top the ranking (Madera 0.180, Colusa 0.169 —
roughly 2× Yolo's 0.088); the rest of the Sacramento metro area (Sacramento,
Solano, Placer) clusters near Yolo at the low-seasonality end.

Outputs: `output/tables/county_seasonal_exposure_06.csv` (one row per
county), `..._06_detail.csv` (one row per county × industry, with the
matched seasonal_index and which fallback tier supplied it — for auditing
any county's score), `output/figures/county_seasonal_exposure_06.pdf`.

Usage: `python code/county_seasonal_index.py --state 06 --highlight 06113`
(any state/county FIPS works once that state's CBP data is fetched).

### Establishment Size vs. Seasonality

`code/establishment_size_analysis.py` asks: do seasonal jobs concentrate in
small, independent businesses or large ones? It merges the **national** CBP
establishment-size distribution (NAICS6 × `EMPSZES` employment-size class,
fetched by `fetch_cbp.py --by-size` — ~1,000 calls, ~20 min, national not
county because county × NAICS6 × size is almost entirely EMP-suppressed while
national is fully populated) onto the QWI separation-based `seasonal_index`.

**Coverage caveat:** CBP does not cover NAICS 111/112 (crop & animal farm
production) — every farming code returns HTTP 204. So the single most seasonal
part of the QWI index drops out of this merge entirely; the analysis is
*conditional on the CBP universe* (construction, tourism/lodging, food
service, agricultural *support* services NAICS 115, seasonal retail, seasonal
manufacturing, etc.). 826 industries survive the CBP × QWI merge.

**Result: more seasonal industries do skew toward smaller establishments —
a real but concave relationship, and strongly sector-dependent.**
- Spearman correlation of `seasonal_index` with: employment share in
  <20-employee establishments +0.50; establishment-count share <20 emp +0.53;
  average establishment size **−0.52** (Pearson only −0.14 — the size
  relationship is monotone but very non-linear).
- By seasonality quartile, median average establishment size falls
  **51 → 33 → 15 → 10 employees**; small-establishment employment share rises
  **13% → 20% → 30% → 40%**; large-establishment (≥100 emp) share falls
  **59% → 50% → 41% → 31%**. The decile-mean curve rises steeply up to
  `seasonal_index ≈ 0.15` then plateaus near 40% — i.e. "very non-seasonal
  industries are large-establishment; anything with meaningful seasonality is
  ~40% small-establishment employment."
- **Sector split is the real story.** Seasonal *services / construction /
  ag-support* are small-establishment-heavy: mobile food services (80% of
  employment in <20-emp establishments), recreational goods rental (59%),
  drive-in theaters (64%), soil prep & planting (55%), RV parks (67%),
  specialty-trade contractors (~48%). Seasonal *manufacturing and large
  recreation venues* are the opposite: fruit & vegetable canning (6%,
  avg 56 employees/establishment), frozen fruit/juice/veg mfg (2%, avg 126),
  racetracks (4%, avg 60), golf courses (11%, avg 32). Manufacturing as a
  whole (329 CBP industries) averages just 10% small-establishment employment.

Outputs: `output/tables/establishment_size_naics6.csv` (per-industry size
metrics + merged seasonal_index), `output/tables/size_vs_seasonality_summary.csv`
(correlations + quartile + by-sector tables), `output/figures/size_vs_seasonality_scatter.pdf`
(with a decile-mean overlay), `output/figures/size_by_seasonality_quartile.pdf`.

Usage: `python code/fetch_cbp.py --by-size` then
`python code/establishment_size_analysis.py`.

### Industry Pay, Gender, and Seasonality Profile (ACS PUMS)

Motivating question: find a pair of industries with ~equal pay, opposite
seasonality, and a female-dominated workforce, to illustrate whether
seasonal work leaves otherwise-comparable (mostly female) workers more
rent-burdened. An initial ad hoc pass at this used QWI's `EarnBeg` (average
monthly pay *rate* among people employed that quarter) annualized by x12 as
"annual pay," and pulled gender composition from a coarser BLS table than
the NAICS6-level seasonality figure it was compared against. Both were
wrong in the same direction: annualizing a rate assumes year-round work,
which overstates realized income for exactly the seasonal population this
project studies, and mixing resolutions is an apples-to-oranges comparison.
This four-script pipeline fixes both:

- `code/build_indp_naics_crosswalk.py` — downloads and parses Census's
  official "2022 Census Industry Code List with Crosswalk" into a NAICS6 <->
  Census-industry-code (`INDP`, the ACS/CPS microdata classification) lookup,
  251 of 268 leaf codes resolved (17 dropped: military/government "Part of
  ..." references with no clean NAICS mapping). This classification is
  *coarser than 6-digit NAICS* for many service industries — e.g. NAICS
  611710 (Educational Support Services) is bundled with five sibling "other
  schools and instruction" codes into one Census industry code — a real
  resolution ceiling in the source data, not a tooling gap.
- `code/fetch_acs_pums.py` — fetches ACS 1-year PUMS person records (all 51
  states + DC, ~3.4M people, ~20 min): `WAGP` (wage/salary income actually
  received in the past 12 months — captures partial-year work, unlike a QWI
  rate), `SEX`, `WKWN` (weeks worked), `INDP`, `PWGTP` (person weight).
- `code/industry_pay_gender_profile.py` — per Census industry code: weighted
  median `WAGP`, weighted %female, weighted mean weeks worked; separately,
  aggregates QWI's NAICS6-level `excessQ1-4` up to the same industry-code
  level (weighted by each NAICS6's average QWI employment) and rebuilds
  amplitude/peak-quarter/seasonal_index from the aggregate, the same
  construction `seasonal_index.py` uses at the NAICS6 level. Merges both on
  the industry code. Output: `output/tables/industry_pay_gender_seasonality_profile.csv`
  (250 industry codes).
- `code/find_seasonal_pairs.py` — screens for pairs with similar realized pay
  (default: within 15%), a large seasonality gap (default: >=0.25), and a
  female-dominated workforce (default: >=55%), then reports rent burden at a
  configurable monthly rent (default: the Census ACS 2020-2024 national
  median gross rent, $1,413/mo). Output: `output/tables/seasonal_pair_candidates.csv`.

**Result: the fix changed the answer, not just its precision.** The initial
ad hoc estimate had Educational Support Services (and its bundled siblings)
at ~$52,000/year (QWI rate x 12); the realized PUMS figure is **$20,000** —
2.6x lower. At default thresholds, only 9 of 250 industry codes clear the
female-share and PUMS-sample-size bars, yielding 3 candidate pairs. The
standout: **"Other schools and instruction, and educational support
services"** (seasonal_index 0.51, peaks Q2/spring) vs. **"Other personal
services"** (seasonal_index 0.13, flat) — both at **exactly $20,000**
realized median annual pay, 65% and 68% female, and both **84.8%** rent-
burdened at the national median rent.

**But weeks-worked does not explain the pay level, in any of the 3 pairs**
(all three pairs show <1 week difference between their seasonal and flat
member, ~46-49 weeks/year either way) — undercutting the "seasonal workers
have fewer working weeks, hence lower realized pay" mechanism the whole
exercise set out to test. `WKWN` counts weeks worked *at any job in the past
year*, not specifically within the profiled industry, so a worker cycled out
of a seasonal role can still show a full work-year via other employment —
arguably a better fit to Coglianese & Price's own household-adaptation
framing than the mechanism this analysis was built to find: the income hit
from seasonality doesn't show up as fewer annual weeks worked, it shows up
in the underlying wage level of these low-wage service/education-support
categories, largely independent of whether that category happens to be
seasonal. The rent-burden finding stands; the mechanism guessed at going in
does not.

**Caveats:** `WAGP` is a *personal* wage — many workers in a $20k-median
category are secondary earners, students, or part-time workers, not
necessarily sole rent-payers, so comparing an individual wage to a full
apartment's rent can overstate personal burden. PUMS is a ~1% sample, so the
9-industry screening universe is real but small. The rent benchmark is
national, not localized to where these industries concentrate.

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
├── fetch_cbp.py                     # County Business Patterns fetcher: county x NAICS (CA done); --by-size = national NAICS x establishment-size
├── county_seasonal_index.py         # County-level seasonal exposure (CBP industry mix x QWI seasonal index)
├── establishment_size_analysis.py   # Do seasonal industries skew small? (national CBP-by-size x QWI seasonal index)
├── build_indp_naics_crosswalk.py    # Census industry code (ACS PUMS INDP) <-> NAICS6 crosswalk
├── fetch_acs_pums.py                # Download ACS 1-year PUMS person records (realized annual pay, sex, weeks worked)
├── industry_pay_gender_profile.py   # Per-industry-code: realized pay (PUMS) + %female + QWI seasonality, one resolution
├── find_seasonal_pairs.py           # Screen for close-pay/opposite-seasonality/female-dominated industry pairs + rent burden
├── replication/
│   ├── load_data.py                 # Extract zip; load .dta.gz files
│   ├── recurrence.py                # Core excess recurrence estimation (main replication)
│   └── gbt_classifier.py           # LightGBM classifier: train on CPS, apply to SIPP
├── qwi/
│   ├── fetch_qwi.py                 # Download QWI data via Census API (QWI_VARS now includes HirA)
│   ├── excess_utils.py              # Shared cyclical-excess-by-quarter calc (used by all four seasonal measures)
│   ├── seasonal_index.py            # Build 6-digit NAICS seasonal index (flow + stock, peak-flexible); flow-vs-stock comparison
│   ├── hire_index.py                # Hiring (accessions) seasonal index + hire-vs-separation timing lag [needs HirA refetch]
│   ├── geographic_analysis.py       # State-level variation (flow + stock): figures + tables
│   ├── state_index_percentiles.py   # Within-state top/bottom-1% seasonality + national/cross-state overlap
│   ├── covid_robustness_check.py    # Pre-COVID (2000-19) vs. full-sample index comparison
│   ├── seasonality_trend_check.py   # 2000-10 vs. 2011-23: has seasonality changed over time?
│   ├── agriculture_exclusion_check.py # Full sample vs. excluding agriculture: how much depends on ag's weaker UI coverage?
│   ├── earnings_index.py            # Income seasonality (EarnBeg): does income dip in the off-season?
│   ├── clean_naics_xwalk.py         # Census NAICS structure workbook -> naics6 title lookup
│   ├── national_state_scatterplot.py  # diagnostic: state vs. national seasonal index scatter
│   ├── plot_seasonality_figures.py    # diagnostic: employment time series + amplitude histogram
│   └── naics_codes.csv              # 1,012 six-digit NAICS codes (2022 vintage)
├── qcew/
│   ├── fetch_qcew.py                 # Download QCEW bulk annual singlefiles (no API key needed)
│   └── monthly_seasonal_index.py     # Monthly stock seasonal index + placebo-corrected comparison to QWI's quarterly stock index
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

### Public Index Files (`data/qwi_clean/`, `data/qcew_clean/`, committed to the repo)
Unlike the rest of `data/` (raw QWI/QCEW/CBP/replication inputs, all
gitignored due to size), these **derived** index CSVs are tracked and
shipped in the public repo, each with its own codebook markdown file
alongside it:
- `qwi_clean/seasonal_index_naics6.csv` — national flow + stock index ([codebook](data/qwi_clean/seasonal_index_naics6_codebook.md))
- `qwi_clean/seasonal_index_naics6_by_state.csv` — state x NAICS6 flow + stock index ([codebook](data/qwi_clean/seasonal_index_naics6_by_state_codebook.md))
- `qwi_clean/earnings_index_naics6.csv` — income/earnings seasonality index, raw + idiosyncratic ([codebook](data/qwi_clean/earnings_index_naics6_codebook.md))
- `qcew_clean/monthly_seasonal_index_naics6.csv` — monthly stock seasonal index (QCEW), raw + idiosyncratic + placebo-corrected ([codebook](data/qcew_clean/monthly_seasonal_index_naics6_codebook.md))

The `.gitignore` pattern is `data/*` + `!data/qwi_clean/` + `!data/qcew_clean/`
(a carve-out, same pattern as `output/*` + `!output/figures/` etc.) — raw
inputs under `data/qwi_raw/`, `data/qcew_raw/`, `data/replication/`,
`data/cbp_raw/`, and the NAICS crosswalk source workbook stay untracked.

### NAICS Title Crosswalk (optional)
- `data/naics_xwalk/2022_NAICS_Structure.xlsx` — Census 2022 NAICS structure workbook
  (not in repo; download from census.gov and place here to run `clean_naics_xwalk.py`)

### CBP Extensions (County-level in progress; national-by-size done)
- Same Census API key as the QWI extension
- `code/qwi/naics_codes.csv` (reused for the CBP NAICS code list)
- `fetch_cbp.py --by-size` (national NAICS x establishment-size, ~20 min) is
  done; the national county-level run (`fetch_cbp.py` with no `--by-size`,
  all 51 states, ~21-22 hrs) is still not launched

### QCEW Extension (Monthly Employment)
- No API key needed -- `fetch_qcew.py` downloads BLS's public bulk CSV
  files directly (`data.bls.gov/cew/data/files/{year}/csv/...`)
- Nothing else to configure; run `fetch_qcew.py` then `monthly_seasonal_index.py`

### Industry Pay/Gender/Seasonality Profile (ACS PUMS)
- Same Census API key as the QWI/CBP extensions
- `fetch_acs_pums.py` downloads ~3.4M ACS 1-year PUMS person records (~20 min);
  `build_indp_naics_crosswalk.py` downloads a small (~100KB) Census crosswalk
  workbook — both cache to `data/acs_raw/` (gitignored, like other raw inputs)

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
python code/qwi/agriculture_exclusion_check.py # full sample vs. excluding agriculture (weaker UI coverage concern)
python code/qwi/seasonal_index.py --compare-flow-stock both  # flow vs. stock index comparison
python code/qwi/earnings_index.py          # income/earnings seasonality + off-season income-dip test
python code/qwi/hire_index.py              # hiring seasonality + hire-vs-separation timing [needs fetch_qwi.py rerun for HirA]
python code/fetch_cbp.py --state 06 --naics-level 6  # county x NAICS establishment counts (CA done; other states in progress)
python code/fetch_cbp.py --state 06 --naics-level 4  # NAICS-4 fallback for suppressed county cells
python code/county_seasonal_index.py --state 06 --highlight 06113  # county-level seasonal exposure (example: Yolo, CA)
python code/build_indp_naics_crosswalk.py  # Census industry code <-> NAICS6 crosswalk (for the ACS PUMS profile below)
python code/fetch_acs_pums.py              # download ACS 1-year PUMS person records (~20 min)
python code/industry_pay_gender_profile.py # realized pay (PUMS) + %female + QWI seasonality, one consistent resolution
python code/find_seasonal_pairs.py         # find close-pay/opposite-seasonality/female-dominated pairs + rent burden
python code/fetch_cbp.py --by-size         # national CBP by NAICS x establishment-size class (~20 min)
python code/establishment_size_analysis.py # do seasonal industries skew toward small establishments?
python code/qcew/fetch_qcew.py             # download QCEW bulk annual files, all years (well under 1hr, no API key)
python code/qcew/monthly_seasonal_index.py # monthly stock index + placebo-corrected comparison to QWI's quarterly stock index
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
- [x] Built national seasonal index — 1,011 industries, ρ = 0.829 vs CP rankings
- [x] Geographic variation analysis — state × industry peak excess; 5 figures + 2 tables
- [x] Beamer presentation — output/slides/slides.tex (15 slides); sector table uses peak_excess; education note corrected (peaks Q2, positive)
- [x] NAICS title crosswalk — Census 2022 structure workbook cleaned to a NAICS6 lookup
- [x] State-level percentile/overlap analysis — within-state top/bottom 1%; only ~11% overlap with national tail; 8 output tables
- [x] COVID-era robustness check — rankings hold (Pearson 0.997, Spearman 0.986) after excluding 2020-2021; fixed a stale-threshold bug in seasonal_index.py along the way (3 industries had wrongly ranked #1 off 2 years of data, below the 5-year minimum) and a missing sys.path bootstrap that broke the script entirely
- [x] Figures/tables/final slide deck now synced to GitHub (previously all of output/ was gitignored)
- [x] Ran exploratory QWI diagnostic scripts (`national_state_scatterplot.py`, `plot_seasonality_figures.py`); fixed a pandas 3.0 `PeriodIndex` API break and a seaborn/pandas incompatibility found along the way
- [x] Deduplicated `STATE_NAMES`/`STATE_FIPS` — `fetch_qwi.py` and `geographic_analysis.py` now import from `config.py` instead of redefining locally
- [x] Validated CBP fetcher with a single-state test (fixed a NAICS-code-list path bug); full 51-state run (~21–22 hrs) deferred by choice, not run yet
- [x] Flow (separations) vs. stock (employment) seasonal index comparison — `compare_flow_vs_stock()`/`plot_flow_vs_stock()`/`--compare-flow-stock` in seasonal_index.py, stock-side state x NAICS computation added to geographic_analysis.py. Also fixed: fetch_qwi.py missing sys.path bootstrap (broke the script entirely), a partial/test fetch silently overwriting the shared all-states combined parquet (now only writes there when every state was actually fetched), and load_qwi()'s per-state fallback missing year/quarter parsing. **Numbers revised after the degenerate-quarter fix below**: national Pearson 0.784 (was 0.653), peak-quarter concordance still only ~15%; Construction still near-perfectly aligned (r=0.966), Agriculture revised from "near-zero" (r=0.207) to moderate (r=0.642) — the original "rapid worker replacement" hypothesis is retracted, it was largely a thin-cell artifact
- [x] **Degenerate single-quarter amplitude bug fixed at the source** — `build_seasonal_index()` and its stock/state/earnings analogs required only 1 non-missing quarter to compute `seasonal_amplitude`, which trivially collapses to 0 in that case (a data-thinness artifact, not genuine flatness). Affected only 6/1,011 industries nationally but **2,898/42,344 (6.8%) state x industry cells** — enough to move real headline numbers: Alaska's exposure 8.0->9.6 p.p., max/min ratio 2.4x->2.8x, and the flow-vs-stock correlations above. Surfaced while investigating thin cells for `plot_seasonality_figures.py`'s example industries, which now also carry NAICS titles and a stricter "all 4 quarters populated" check
- [x] Refetched QWI data with EarnBeg (earnings) in place of the broken Payroll field — complete 51-state run; rebuilt the national/state indices, flow-vs-stock comparison, and state percentiles on the final complete data (all numbers stable vs. the interim 50-state validation runs)
- [x] Income/earnings seasonality index (`earnings_index.py`) — found the raw index is ~83% dominated by a common Q4-bonus effect (not industry-specific), added a de-confounding step (`compute_common_calendar_effect`/`add_idiosyncratic_excess`). Credible test: only 44.2% of industries show an income dip at their own separation-peak quarter once de-confounded (mean excess +0.0055, not clearly negative); Construction shows the opposite (earnings *higher*, not lower, likely overtime pay before layoffs). No strong evidence of industry-average income dipping in the off-season — consistent with CP's effect being about the separating worker specifically, not the industry-wide average
- [x] County-level seasonal exposure index (`county_seasonal_index.py`) — fetched CBP NAICS-6 and NAICS-4 for California, built a 3-tier fallback (state NAICS6 -> national NAICS6 -> national NAICS4) reaching 100% employment coverage in all 58 CA counties. Case study: Yolo County ranks 41st of 58 (30th percentile) -- less seasonal than ~2/3 of CA counties despite being an ag county, because its employment is dominated by large stable service/logistics employers, not its small-but-extreme agricultural niches
- [ ] Fetch and merge County Business Patterns (CBP) data nationally for a 51-state county-level index (California pilot validated; full run not yet launched, ~21-22hrs)
- [x] Seasonality-over-time trend check (`seasonality_trend_check.py`) — 2000-2010 vs. 2011-2023. Economy-wide mean is essentially flat (0.131 -> 0.122), but masks real reshuffling: **as of a 2026-09 pipeline re-run, Educational services fell most (-0.053) and Construction kept falling (-0.047); Agriculture also fell (-0.037), reversing an earlier read of this same check that had reported Agriculture as the largest increaser off older QWI data (see correction note under "Has Seasonality Changed Over Time?" above) — the H-2A guest-worker hypothesis floated there is retracted**. Correlation (Pearson 0.931/Spearman 0.826) is meaningfully lower than the COVID check's, as expected for an 11-year period gap vs. excluding 2 years; the elevated peak-quarter-change rate (30.2%) concentrates in weakly-seasonal industries (noise), not strongly-seasonal ones
- [x] Agriculture exclusion check (`agriculture_exclusion_check.py`) — responds to feedback that agriculture's weaker/partial UI coverage could make it look more seasonal in the QWI than it truly is. Sector ranking reshuffles once agriculture is dropped and the index re-winsorized (Arts/entertainment becomes #1 at 0.551; the "most seasonal industries" list changes completely — Drive-In Theaters, Recreational Goods Rental, Mobile Food Services, etc.), but **state-level rankings barely move** (Pearson r=0.991, Spearman r=0.989 vs. full sample) — Alaska stays most seasonal (9.58 -> 9.45 p.p.) and Texas stays least seasonal (3.43 -> 2.99 p.p.) either way, implying Alaska's seasonality is driven mainly by non-agricultural industries, not farming. **Decision: excluding agriculture is now the primary/default presentation** — `geographic_analysis.py`'s headline figures and both slide decks were updated accordingly; the full sample is kept as a documented sensitivity check (new slide in each deck), not deleted
- [ ] Apply firm-level code on other machine
- [x] Published the derived index CSVs in `data/qwi_clean/` (national, state x NAICS6, earnings) to the public repo with a codebook per file; `.gitignore` carved out `data/*` + `!data/qwi_clean/` so raw inputs stay untracked
- [x] Deduplicated the cyclical excess-by-quarter calc into `code/qwi/excess_utils.py` — was copy-pasted 4x (separations, employment/stock, earnings, state-level); all callers verified byte-identical after the refactor
- [x] Hiring (accessions) seasonal index (`hire_index.py`) — code + dry-run test complete; `HirA` verified live and added to `QWI_VARS`. Blocked on a fresh ~17hr QWI refetch before it produces real numbers. The analysis it enables: hire-vs-separation *timing lag* per industry (0 quarters = churn; 1-3 = separation follows the hiring peak) and a `likely_churn` flag
- [x] Establishment size vs. seasonality (`establishment_size_analysis.py`) — fetched national CBP by NAICS x establishment-size class (`fetch_cbp.py --by-size`, 831 industries). More seasonal industries skew toward smaller establishments (Spearman ~0.5 for small-establishment employment share, -0.52 for average establishment size; median establishment size 51 emp in the least-seasonal quartile -> 10 emp in the most). But it's concave (plateaus at ~40% small-establishment employment above seasonal_index ~0.15) and sector-driven: seasonal services/construction/ag-support are small-establishment-heavy, seasonal manufacturing (canning, frozen foods) and big recreation venues (racetracks, golf courses) are the opposite. Conditional on the CBP universe — NAICS 111/112 farm production is not in CBP at all
- [ ] National CBP-by-*size* is done, but the national CBP-by-*county* run (for a 51-state county-level exposure index) is still not launched (~21-22hrs)
- [x] Industry pay/gender/seasonality profile (`industry_pay_gender_profile.py`, `find_seasonal_pairs.py`) — built to find a same-pay/opposite-seasonality/female-dominated industry pair for a rent-burden illustration. Fetched ACS 1-year PUMS (3.4M people, all states) and built a Census-industry-code <-> NAICS6 crosswalk (`build_indp_naics_crosswalk.py`) so pay (realized `WAGP`, not a QWI rate annualized x12), gender, and seasonality are all computed at one consistent resolution. Found 3 candidate pairs; best: "Other schools and instruction, and educational support services" (seasonal_index 0.51) vs. "Other personal services" (0.13) — both exactly $20,000 realized median annual pay, 65%/68% female, both 84.8% rent-burdened at the national median rent. Notably, weeks-worked is nearly identical in every pair, so the pay gap between seasonal and non-seasonal categories isn't explained by seasonal workers logging fewer annual weeks — it's a wage-level difference between these industry categories, independent of seasonality
- [x] QCEW monthly resolution check (`fetch_qcew.py`, `monthly_seasonal_index.py`) — QCEW has monthly employment but no separations, so it can only sharpen the existing *stock* index from quarterly to monthly, not replicate the flow-based measure. All 24 years fetched via bulk CSV in under an hour (no API key, much faster than QWI's per-state-per-industry loop). Two confounds found and corrected before trusting any number: (1) a common calendar effect — 47% of industries raw-peak in December, the same class of confound found in `earnings_index.py` for Q4 earnings — netted out via an idiosyncratic-excess correction; (2) a sample-size artifact — max-min range over 12 monthly draws is mechanically larger than over 4 quarterly draws even with zero true seasonality, confirmed via a permutation placebo test (**89.7% of the idiosyncratic amplitude is this artifact**, not real signal). After both corrections (`signal_index`): correlation with the quarterly stock index drops from Pearson 0.79 (raw) to 0.44 (corrected), peak-quarter concordance is only 38%, and 48% of industries show zero genuine monthly-specific signal. The genuine finding: quarterly aggregation specifically hides seasonality for industries with a sharp, brief calendar-driven event that falls mid-quarter (School and Employee Bus Transportation: corrected signal index 1.00 vs. quarterly stock index 0.11) — while several industries that looked "more seasonal monthly" in the raw comparison (Skiing Facilities, RV Parks, Golf Courses) correct to zero added signal, since their seasonality genuinely operates at a quarterly-or-longer scale already well captured by QWI. Published to the public repo as `data/qcew_clean/monthly_seasonal_index_naics6.csv` with a codebook

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

**Industry rankings** (both datasets): Agriculture, Educational services, and
Entertainment/recreation are the top three by excess recurrence in both CPS
and SIPP (order varies slightly by dataset). Healthcare is near the bottom in
both (14th/15 CPS, 13th/15 SIPP). Construction is solidly mid-pack in both
(8th/15 CPS, 7th/15 SIPP) — not top-tier, despite common intuition that it's
an archetypally seasonal industry. Mining's estimate is unstable (small
samples, n≈1,600–11,100 vs. tens of thousands for most other industries) and
flips sign between datasets, so it's excluded from ranking claims.

**Seam coding note**: `srefmon` in the pre-cleaned data cycles 1→4 within each 4-month wave. srefmon=1 marks the first month of each wave (right after the seam boundary). Seam variables: `first_seam = (srefmon==1) & (k<=4)`, `later_seam = (srefmon==1) & (k>=5)`.

---

## Key Numbers to Validate Against Paper

| Statistic | Paper value | Our value |
|---|---|---|
| CPS raw (Figure 2) | 1.4 p.p. (SE 0.14) | 1.44 ✓ |
| SIPP raw (Figure 2) | 2.0 p.p. (SE 0.11) | 2.04 ✓ |
| CPS preferred (App. F.1) | 1.5 p.p. (SE 0.14) | 1.42 ✓ |
| SIPP preferred (App. F.1) | 1.6 p.p. (SE 0.11) | 1.62 ✓ |
| Agriculture excess recurrence | highest industry | Confirmed ✓ |
| Education excess recurrence | high (top 3) | Confirmed ✓ |
| Construction excess recurrence | mid-pack, not highest (8th/15 CPS, 7th/15 SIPP) | Corrected 2026-09 — an earlier version of this table wrongly claimed Construction ranked highest |

---

## Notes

- Original code requires Stata 15+; our Python replication uses pandas + statsmodels + lightgbm.
- The pre-computed GBT predictions (`gbt_predictions_sipp.dta`) are in the replication zip
  and can be used directly, bypassing the need to retrain the model.
- QWI data is quarterly; CP data is monthly. The QWI seasonal index is therefore a
  coarser measure, but useful for industry-level classification.
- Firm-level code is designed to be self-contained: copy `code/firm_level/` and the
  QWI seasonal index CSV to the other machine.
