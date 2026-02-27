"""
config.py — Project-wide paths and parameters.

Edit REPO_ROOT below.  Set the Census API key in a .env file at the repo root:

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

# Root of the repository (edit this if running from a different machine)
REPO_ROOT = Path(r"M:\Users\Sandler\Documents\GitHub\Seasonal_work")

# Replication package
REPKG_ZIP  = REPO_ROOT / "replication_package" / "238636-V1.zip"
REPKG_DIR  = REPO_ROOT / "data" / "replication"   # extracted files land here

# Pre-cleaned Stata datasets (inside the zip at replication/data/clean/)
DATA_CLEAN = REPKG_DIR / "replication" / "data" / "clean"

# QWI data
QWI_RAW    = REPO_ROOT / "data" / "qwi_raw"
QWI_CLEAN  = REPO_ROOT / "data" / "qwi_clean"

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

# ── Helper to create output directories ─────────────────────────────────────

def make_dirs():
    for d in [QWI_RAW, QWI_CLEAN, FIRM_DATA, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
