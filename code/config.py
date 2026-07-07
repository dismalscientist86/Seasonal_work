"""
config.py — Project-wide paths and parameters.

REPO_ROOT is auto-detected from this file's location, so the repo can be
cloned/copied to any machine without editing paths.  Set the Census API key
in a .env file at the repo root:

    CENSUS_API_KEY=your_key_here

Never put your API key directly in this file — .env is gitignored.
Get a free key at https://api.census.gov/data/key_signup.html
"""

import os
from pathlib import Path

# Load .env file if present (requires python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # dotenv optional; key can also be set as a shell env var

# ── Paths ────────────────────────────────────────────────────────────────────

# Root of the repository (auto-detected: this file lives at <REPO_ROOT>/code/config.py)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Replication package
REPKG_ZIP  = REPO_ROOT / "replication_package" / "238636-V1.zip"
REPKG_DIR  = REPO_ROOT / "data" / "replication"   # extracted files land here

# Pre-cleaned Stata datasets (inside the zip at replication/data/clean/)
DATA_CLEAN = REPKG_DIR / "replication" / "data" / "clean"

# QWI data
QWI_RAW    = REPO_ROOT / "data" / "qwi_raw"
QWI_CLEAN  = REPO_ROOT / "data" / "qwi_clean"
XWALK      = REPO_ROOT / "data" / "naics_xwalk"

# CBP (County Business Patterns) data
CBP_RAW    = REPO_ROOT / "data" / "cbp_raw"
CBP_CLEAN  = REPO_ROOT / "data" / "cbp_clean"

# Firm-level data (path on the other machine — override via env var or arg)
FIRM_DATA  = REPO_ROOT / "data" / "firm_level"

# Outputs
OUTPUT_DIR = REPO_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR  = OUTPUT_DIR / "tables"

# ── Census API ───────────────────────────────────────────────────────────────

# Get a free key at https://api.census.gov/data/key_signup.html
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")   # set in .env or shell

# ── Substantive parameters ───────────────────────────────────────────────────

# Sample years (CP paper: 1984–2013)
SAMPLE_YEARS = (1984, 2013)

# Prime-age range
AGE_MIN = 25
AGE_MAX = 54

# Earnings threshold for SIPP income analysis (monthly, $, matches CP)
EARNINGS_THRESHOLD = 450

# Excess recurrence horizon (months)
RECURRENCE_HORIZON = 12        # the spike we are measuring
RECURRENCE_ADJACENT = 1        # distance to counterfactual (±1 month)

# CPS: observable horizons (limited by 4-month observation window)
CPS_HORIZONS = list(range(10, 15))   # 10, 11, 12, 13, 14

# SIPP: observable horizons
SIPP_HORIZONS = list(range(1, 19))   # 1 .. 18

# Base month for normalization (November = 11, as in the paper)
BASE_MONTH_K = 11

# GBT hyperparameters (from CP paper's cross-validation)
GBT_N_TREES      = 1000
GBT_LEARNING_RATE = 0.01
GBT_N_FOLDS      = 5
GBT_RANDOM_SEED  = 257835478

# QWI: industry aggregation level (2, 3, 4, or 6)
QWI_NAICS_LEVEL = 6

# QWI: years to fetch
QWI_YEARS = list(range(2000, 2024))

# QWI: minimum number of year-observations required to report an excessQ estimate.
# Cells with fewer contributing years are set to NaN rather than treated as
# "zero seasonality" (a thin cell can mechanically produce a near-zero
# amplitude that looks like the absence of seasonality). Applied consistently
# across the national index, employment index, and state x industry index.
MIN_EXCESS_OBS = 5

# CBP: reference year (2023 is the most recent vintage as of mid-2026;
# CBP still uses NAICS2017 classification at this vintage, not NAICS2022)
CBP_YEAR = 2023

# CBP: NAICS digit levels to fetch at the county level. 6-digit matches the
# QWI seasonal index's granularity but is heavily suppressed at the county
# level; 4-digit (industry group) has much better county coverage and is a
# useful fallback when the 6-digit cell is missing.
CBP_NAICS_LEVELS = [6, 4]

# CBP: legal form of organization code for "all establishments" (the total
# row). LFO is "default displayed" in the CBP API, meaning if it is not
# passed as a filter (not just included in `get=`), the API returns every
# LFO breakdown rather than just the total.
CBP_LFO_TOTAL = "001"

# State FIPS codes (50 states + DC), shared across QWI and CBP fetchers.
STATE_FIPS = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
    "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
    "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
    "50","51","53","54","55","56",
]

STATE_NAMES = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas","06":"California",
    "08":"Colorado","09":"Connecticut","10":"Delaware","11":"DC","12":"Florida",
    "13":"Georgia","15":"Hawaii","16":"Idaho","17":"Illinois","18":"Indiana",
    "19":"Iowa","20":"Kansas","21":"Kentucky","22":"Louisiana","23":"Maine",
    "24":"Maryland","25":"Massachusetts","26":"Michigan","27":"Minnesota",
    "28":"Mississippi","29":"Missouri","30":"Montana","31":"Nebraska","32":"Nevada",
    "33":"New Hampshire","34":"New Jersey","35":"New Mexico","36":"New York",
    "37":"North Carolina","38":"North Dakota","39":"Ohio","40":"Oklahoma",
    "41":"Oregon","42":"Pennsylvania","44":"Rhode Island","45":"South Carolina",
    "46":"South Dakota","47":"Tennessee","48":"Texas","49":"Utah","50":"Vermont",
    "51":"Virginia","53":"Washington","54":"West Virginia","55":"Wisconsin","56":"Wyoming",
}

# ── Helper to create output directories ─────────────────────────────────────

def make_dirs():
    for d in [QWI_RAW, QWI_CLEAN, CBP_RAW, CBP_CLEAN, FIRM_DATA, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
