"""
gbt_classifier.py — Python replication of the Coglianese & Price GBT seasonal classifier.

Replicates gbt_predictions.do using LightGBM.

The model:
  - Trained on CPS data: predict P(sep at t+lag) for lag ∈ {10,11,12,13,14}
  - Features: lagged 1-digit industry (categorical), lagged occupation (categorical),
              month encoded as sin/cos (continuous), Jan temperature (continuous)
  - 5-fold cross-validation to select n_trees and learning_rate
  - Applied to SIPP to predict pred_excess = P(sep_12) - (P(sep_11) + P(sep_13))/2

You can skip this script if you just want to use the pre-computed predictions
already included in the replication package (gbt_predictions_sipp.dta).

Usage:
    python code/replication/gbt_classifier.py   # trains and saves model + predictions
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from code.config import (
    GBT_N_TREES, GBT_LEARNING_RATE, GBT_N_FOLDS, GBT_RANDOM_SEED,
    OUTPUT_DIR, DATA_CLEAN,
)
from code.replication.load_data import load_cps_fullsample, load_sipp_fullsample

warnings.filterwarnings("ignore")

# Output locations
MODEL_DIR = OUTPUT_DIR / "models" / "gbt"
PRED_PATH = DATA_CLEAN / "gbt_predictions_python.parquet"


# ── Feature engineering ───────────────────────────────────────────────────────

def encode_month_cyclical(month_series: pd.Series) -> pd.DataFrame:
    """Encode month 1–12 as (sin, cos) pair — the same encoding used in the paper."""
    radians = month_series / 12 * 2 * np.pi
    return pd.DataFrame({
        "mth_dim1": np.sin(radians),
        "mth_dim2": np.cos(radians),
    }, index=month_series.index)


def prepare_cps_training_data(cps: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the CPS dataset for GBT training.

    For each separation event (sep==1) at time t, we create one row per
    forward lag l ∈ {10,11,12,13,14}, measuring fwd_sep = I(sep at t+l).

    Features:
        jan_max_temp  : state January temperature
        mth_dim1/2    : cyclical month encoding
        lag_ind       : 1-digit industry in month prior to separation (categorical)
        lag_occ       : occupation in month prior to separation (categorical)
        lag           : forward horizon l (used as a feature, as in the paper)
    """
    required = ["pid", "tm", "month", "wtfnl", "jan_max_temp", "sep", "ind15", "occ14"]
    missing = [c for c in required if c not in cps.columns]
    if missing:
        raise ValueError(f"CPS data missing columns: {missing}")

    # Sort panel
    cps = cps.sort_values(["pid", "tm"]).copy()

    # Lag industry and occupation (industry/occ in month BEFORE the separation)
    cps["lag_ind"] = cps.groupby("pid")["ind15"].shift(1).fillna(0).astype(int)
    cps["lag_occ"] = cps.groupby("pid")["occ14"].shift(1).fillna(0).astype(int)

    # Forward separations at lags 10–14
    lag_frames = []
    for l in range(10, 15):
        fwd = cps.groupby("pid")["sep"].shift(-l)
        sub = cps[cps["sep"] == 1].copy()
        sub["fwd_sep"] = fwd.reindex(sub.index)
        sub = sub.dropna(subset=["fwd_sep"])
        sub["lag"] = l
        lag_frames.append(sub)

    df = pd.concat(lag_frames, ignore_index=True)

    # Cyclical month features
    mth = encode_month_cyclical(df["month"])
    df = pd.concat([df, mth], axis=1)

    # Per-person cross-validation fold (random, assigned at person level)
    rng = np.random.default_rng(GBT_RANDOM_SEED)
    persons = cps["pid"].unique()
    fold_map = pd.Series(
        rng.integers(0, GBT_N_FOLDS, size=len(persons)),
        index=persons,
    )
    df["fold"] = df["pid"].map(fold_map)

    keep = ["pid", "fold", "fwd_sep", "jan_max_temp", "mth_dim1", "mth_dim2",
            "lag_ind", "lag_occ", "lag", "wtfnl"]
    return df[keep].copy()


def prepare_sipp_prediction_data(sipp: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare SIPP data for prediction: one row per separation, one per lag ∈ {11,12,13}.

    The paper harmonizes SIPP timing with CPS by adjusting month based on sep_week.
    We replicate that adjustment if sep_week is available.
    """
    required = ["pid", "tm", "month", "sep", "job1_ind15", "job1_occ14",
                "jan_max_temp"]
    missing = [c for c in required if c not in sipp.columns]
    if missing:
        raise ValueError(f"SIPP data missing columns: {missing}")

    sipp = sipp.sort_values(["pid", "tm"]).copy()

    # Adjust month for SIPP reference week (if sep_week available)
    if "sep_week" in sipp.columns:
        sipp["month_adj"] = sipp["month"]
        mask = sipp["sep_week"].isin([3, 4, 5])
        sipp.loc[mask, "month_adj"] = sipp.loc[mask, "month_adj"] + 1
        sipp.loc[sipp["month_adj"] == 13, "month_adj"] = 1
    else:
        sipp["month_adj"] = sipp["month"]

    # Lagged industry/occ
    sipp["lag_ind"] = sipp.groupby("pid")["job1_ind15"].shift(1).fillna(0).astype(int)
    sipp["lag_occ"] = sipp.groupby("pid")["job1_occ14"].shift(1).fillna(0).astype(int)

    # Keep only separations
    df = sipp[sipp["sep"] == 1].copy()

    # Cyclical month features
    mth = encode_month_cyclical(df["month_adj"])
    df = pd.concat([df, mth], axis=1)

    keep = ["pid", "tm", "jan_max_temp", "mth_dim1", "mth_dim2",
            "lag_ind", "lag_occ"]
    return df[keep].copy()


# ── Model training and cross-validation ──────────────────────────────────────

FEATURE_COLS = ["jan_max_temp", "mth_dim1", "mth_dim2", "lag"]
CAT_COLS     = ["lag_ind", "lag_occ"]
ALL_FEATURES = FEATURE_COLS + CAT_COLS


def _build_lgb_dataset(df: pd.DataFrame, label_col: str) -> lgb.Dataset:
    X = df[ALL_FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return lgb.Dataset(X, label=df[label_col], free_raw_data=False)


def cross_validate_gbt(
    train_df: pd.DataFrame,
    n_trees_grid: list[int] | None = None,
    lr_grid: list[float] | None = None,
) -> tuple[int, float]:
    """
    5-fold cross-validation over (n_trees, learning_rate) grid.

    Returns (best_n_trees, best_lr).
    """
    n_trees_grid = n_trees_grid or [50, 100, 200, 400, 600, 800, 1000]
    lr_grid      = lr_grid      or [0.05, 0.01, 0.005, 0.001]

    best_mse  = float("inf")
    best_params = (GBT_N_TREES, GBT_LEARNING_RATE)

    print("Cross-validating GBT hyperparameters …")
    for n in n_trees_grid:
        for lr in lr_grid:
            fold_mses = []
            for fold in range(GBT_N_FOLDS):
                tr = train_df[train_df["fold"] != fold]
                va = train_df[train_df["fold"] == fold]

                params = {
                    "objective":        "regression",
                    "n_estimators":     n,
                    "learning_rate":    lr,
                    "num_leaves":       31,
                    "verbose":          -1,
                    "seed":             GBT_RANDOM_SEED,
                    "n_jobs":           4,
                }
                model = lgb.LGBMRegressor(**params)

                X_tr = tr[ALL_FEATURES].copy()
                X_va = va[ALL_FEATURES].copy()
                for c in CAT_COLS:
                    X_tr[c] = X_tr[c].astype("category")
                    X_va[c] = X_va[c].astype("category")

                model.fit(X_tr, tr["fwd_sep"])
                preds = model.predict(X_va)
                fold_mses.append(mean_squared_error(va["fwd_sep"], preds))

            mse = np.mean(fold_mses)
            print(f"  N={n:4d}  lr={lr:.4f}  MSE={mse:.6f}")
            if mse < best_mse:
                best_mse    = mse
                best_params = (n, lr)

    print(f"Best: N={best_params[0]}, lr={best_params[1]}, MSE={best_mse:.6f}")
    return best_params


def train_gbt(
    train_df: pd.DataFrame,
    n_trees: int = GBT_N_TREES,
    learning_rate: float = GBT_LEARNING_RATE,
) -> lgb.LGBMRegressor:
    """Train the final GBT model on the full training dataset."""
    print(f"Training GBT (N={n_trees}, lr={learning_rate}) …")
    params = {
        "objective":      "regression",
        "n_estimators":   n_trees,
        "learning_rate":  learning_rate,
        "num_leaves":     31,
        "verbose":        -1,
        "seed":           GBT_RANDOM_SEED,
        "n_jobs":         4,
    }
    model = lgb.LGBMRegressor(**params)
    X = train_df[ALL_FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    model.fit(X, train_df["fwd_sep"])
    return model


def predict_for_sipp(
    sipp_pred_df: pd.DataFrame,
    model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    """
    Apply the trained model to SIPP at lags 11, 12, 13.

    Returns DataFrame with columns: pid, tm, sep_11, sep_12, sep_13, pred_excess.
    """
    results = []
    for l in [11, 12, 13]:
        df = sipp_pred_df.copy()
        df["lag"] = l
        X = df[ALL_FEATURES].copy()
        for c in CAT_COLS:
            X[c] = X[c].astype("category")
        df[f"sep_{l}"] = model.predict(X)
        results.append(df[["pid", "tm", f"sep_{l}"]])

    out = results[0].merge(results[1], on=["pid", "tm"]).merge(results[2], on=["pid", "tm"])
    out["pred_excess"] = out["sep_12"] - (out["sep_11"] + out["sep_13"]) / 2
    return out


# ── Decile assignment ─────────────────────────────────────────────────────────

def assign_deciles(pred_df: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    """
    Assign each SIPP separator to a decile of pred_excess (1=least, 10=most seasonal).

    Optionally weight by wtfnl to match the paper's weighted deciles.
    """
    pred_df = pred_df.copy()

    if weight_col and weight_col in pred_df.columns:
        # Weighted quantile cut
        pred_df = pred_df.sort_values("pred_excess").reset_index(drop=True)
        cum_wt  = pred_df[weight_col].cumsum()
        total   = pred_df[weight_col].sum()
        pred_df["pred_decile"] = (10 * cum_wt / total).clip(1, 10).astype(int)
    else:
        pred_df["pred_decile"] = pd.qcut(
            pred_df["pred_excess"], q=10, labels=False
        ).astype(int) + 1

    return pred_df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gbt_pipeline(
    cross_validate: bool = False,
    save_model: bool = True,
    save_predictions: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load data → (optional CV) → train → predict for SIPP.

    Returns predictions DataFrame.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CPS training data ───────────────────────────────────────────────
    print("\n=== Loading CPS full sample for GBT training ===")
    cps_cols = ["pid", "tm", "month", "wtfnl", "jan_max_temp",
                "sep", "ind15", "occ14"]
    cps = load_cps_fullsample(usecols=cps_cols)
    train_df = prepare_cps_training_data(cps)
    del cps
    print(f"  Training rows: {len(train_df):,}")

    # ── Cross-validate (optional) ────────────────────────────────────────────
    n_trees, lr = GBT_N_TREES, GBT_LEARNING_RATE
    if cross_validate:
        n_trees, lr = cross_validate_gbt(train_df)
        cv_result = {"n_trees": n_trees, "learning_rate": lr}
        (MODEL_DIR / "cv_parameters.json").write_text(json.dumps(cv_result, indent=2))

    # ── Train model ──────────────────────────────────────────────────────────
    model = train_gbt(train_df, n_trees=n_trees, learning_rate=lr)

    if save_model:
        model_path = MODEL_DIR / "gbt_cps.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved to {model_path}")

    # ── Load SIPP and predict ────────────────────────────────────────────────
    print("\n=== Loading SIPP full sample for predictions ===")
    sipp_cols = ["pid", "tm", "month", "wtfnl", "jan_max_temp",
                 "sep", "job1_ind15", "job1_occ14"]
    sipp_cols_try = sipp_cols + ["sep_week"]
    sipp = load_sipp_fullsample(usecols=sipp_cols_try)

    sipp_pred_df = prepare_sipp_prediction_data(sipp)
    del sipp

    predictions = predict_for_sipp(sipp_pred_df, model)
    print(f"  Predictions generated for {len(predictions):,} SIPP separations")

    if save_predictions:
        predictions.to_parquet(PRED_PATH, index=False)
        print(f"  Predictions saved to {PRED_PATH}")

    return predictions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GBT seasonal classifier")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation to select hyperparameters")
    args = parser.parse_args()

    preds = run_gbt_pipeline(cross_validate=args.cv)
    print("\nPredictions summary:")
    print(preds["pred_excess"].describe())
    print(f"\nTop decile threshold: {preds['pred_excess'].quantile(0.9):.4f}")
