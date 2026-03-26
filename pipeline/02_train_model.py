"""
pipeline/02_train_model.py
═══════════════════════════════════════════════════════════
Train UFC fight prediction model — walk-forward validation.

Architecture:
  - LightGBM as primary learner (handles missing values natively)
  - Walk-forward: train on [start, T), test on [T, T+6months)
  - Calibration via isotonic regression on the last 2 years
  - Two variants: with and without market_logit

Outputs:
  data/processed/preds_v2.parquet   ← OOS predictions (with odds data)
  data/processed/model_v2.pkl       ← Final model for live use
  data/processed/feat_importance_v2.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import lightgbm as lgb

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ── Feature sets ─────────────────────────────
W = 5

STATS_FEATURES = [
    # Elo
    "elo_diff",
    # Striking (offensive)
    f"diff_sig_lnd_L{W}",
    f"diff_sig_acc_L{W}",
    f"diff_net_sig_L{W}",
    f"diff_kd_L{W}",
    # Striking (defensive)
    f"diff_def_sig_absorbed_L{W}",
    # Grappling
    f"diff_td_lnd_L{W}",
    f"diff_td_acc_L{W}",
    f"diff_sub_att_L{W}",
    f"diff_ctrl_secs_L{W}",
    f"diff_def_td_absorbed_L{W}",
    # Form
    f"diff_result_win_L{W}",
    # Experience
    "diff_ufc_fights",
    "diff_days_off",
    # Physical
    "reach_diff",
    "age_diff",
    "southpaw_matchup",
]

FULL_FEATURES = STATS_FEATURES + ["market_logit"]

TARGET       = "y"
TRAIN_START  = "2013-01-01"
FIRST_TEST   = "2018-01-01"   # at least 5 years of training data
STEP_MONTHS  = 6              # retrain every 6 months

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.05,
    "num_leaves":        20,           # deliberately shallow — MMA data is noisy
    "max_depth":         4,
    "min_child_samples": 40,           # require meaningful samples per leaf
    "feature_fraction":  0.7,
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "reg_alpha":         0.2,
    "reg_lambda":        0.5,
    "n_estimators":      200,          # fewer trees to avoid overfitting
    "random_state":      42,
    "verbose":          -1,
}


# ── Walk-forward splits ───────────────────────
def walk_forward_splits(dates: pd.Series, first_test=FIRST_TEST, step_months=STEP_MONTHS):
    first_test_dt = pd.Timestamp(first_test)
    max_date      = dates.max()
    splits = []
    t = first_test_dt
    while t < max_date:
        t_end = t + pd.DateOffset(months=step_months)
        tr = dates < t
        te = (dates >= t) & (dates < t_end)
        if tr.sum() >= 200 and te.sum() >= 20:
            splits.append((tr, te))
        t = t_end
    return splits


# ── Train one LightGBM ────────────────────────
def train_lgbm(X_tr: pd.DataFrame, y_tr: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    return model


# ── Walk-forward evaluation ───────────────────
def run_walk_forward(df: pd.DataFrame, features: list, label: str) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET]).copy()
    dates = pd.to_datetime(df["event_date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    # For full-feature model, restrict to rows with market_logit
    if "market_logit" in features:
        df = df.dropna(subset=["market_logit"]).copy()
        dates = pd.to_datetime(df["event_date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)

    splits = walk_forward_splits(dates)
    print(f"  [{label}] Walk-forward splits: {len(splits)} | "
          f"fights: {len(df)} ({dates.min().date()} → {dates.max().date()})")

    records = []
    for i, (tr_mask, te_mask) in enumerate(splits):
        X_tr_full = df.loc[tr_mask, features].copy()
        y_tr_full = df.loc[tr_mask, TARGET]

        # Hold out last 20% of training set chronologically for calibration
        n_cal = max(50, int(len(X_tr_full) * 0.20))
        X_tr  = X_tr_full.iloc[:-n_cal]
        y_tr  = y_tr_full.iloc[:-n_cal]
        X_cal = X_tr_full.iloc[-n_cal:]
        y_cal = y_tr_full.iloc[-n_cal:]

        X_te  = df.loc[te_mask, features].copy()
        y_te  = df.loc[te_mask, TARGET]

        # Median imputation from training set
        medians = X_tr.median()
        X_tr  = X_tr.fillna(medians)
        X_cal = X_cal.fillna(medians)
        X_te  = X_te.fillna(medians)

        model = train_lgbm(X_tr, y_tr)

        # Calibrate using hold-out calibration set (no leakage)
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(model.predict_proba(X_cal)[:, 1], y_cal)

        raw   = model.predict_proba(X_te)[:, 1]
        proba = cal.predict(raw)   # calibrated

        brier = brier_score_loss(y_te, proba)
        auc   = roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan")
        print(f"    split {i+1:2d} | n={te_mask.sum():4d} | "
              f"Brier={brier:.4f} | AUC={auc:.4f}")

        batch = df.loc[te_mask, ["fight_id", "event_date",
                                  "fighter_1", "fighter_2", TARGET]].copy()
        batch["proba_model"] = proba
        records.append(batch)

    return pd.concat(records, ignore_index=True)


# ── Calibration ───────────────────────────────
def calibrate(preds_df: pd.DataFrame, target=TARGET) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(preds_df["proba_model"].values, preds_df[target].values)
    return cal


# ── Metrics summary ───────────────────────────
def print_metrics(preds_df: pd.DataFrame, label: str):
    y = preds_df[TARGET]
    p = preds_df["proba_model"]
    brier = brier_score_loss(y, p)
    auc   = roc_auc_score(y, p)
    ll    = log_loss(y, p)
    acc   = (y == (p > 0.5)).mean()

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Brier Score : {brier:.4f}  (baseline naïf ≈ 0.250)")
    print(f"  ROC AUC     : {auc:.4f}  (random = 0.500)")
    print(f"  Log-Loss    : {ll:.4f}")
    print(f"  Accuracy    : {acc:.3f}")
    print(f"  N fights    : {len(preds_df)}")


# ── Final model (all data) ────────────────────
def train_final_model(df: pd.DataFrame, features: list) -> dict:
    df = df.dropna(subset=[TARGET]).copy()
    if "market_logit" in features:
        df = df.dropna(subset=["market_logit"]).copy()

    X = df[features].copy()
    y = df[TARGET]
    medians = X.median()
    X = X.fillna(medians)

    print(f"  Training final model on {len(df)} fights …")
    model = train_lgbm(X, y)

    # Calibrate on most recent 2 years
    dates = pd.to_datetime(df["event_date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    cal_mask = dates > (dates.max() - pd.DateOffset(years=2))

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(model.predict_proba(X[cal_mask])[:, 1], y[cal_mask])

    return {
        "model":     model,
        "calibrator": cal,
        "features":  features,
        "medians":   medians.to_dict(),
    }


# ── Feature importance plot ───────────────────
def plot_importance(model: lgb.LGBMClassifier, features: list):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    colors = ["#e74c3c" if "market" in c else "#3498db" for c in imp.index]
    imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance — LightGBM v2", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gain importance")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = PROC / "feat_importance_v2.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── Main ─────────────────────────────────────
def main():
    print("Loading features …")
    df = pd.read_parquet(PROC / "features_v2.parquet")
    df = df[df["event_date"] >= TRAIN_START].copy()

    print(f"Dataset: {len(df)} fights\n")

    # ── Walk-forward: STATS model (independent of market) ──
    print("Walk-forward — STATS model (no market odds):")
    preds_stats = run_walk_forward(df, STATS_FEATURES, "STATS")
    preds_stats = preds_stats.merge(df[[TARGET, "fight_id"]], on="fight_id", how="left",
                                    suffixes=("", "_orig"))
    if TARGET + "_orig" in preds_stats.columns:
        preds_stats[TARGET] = preds_stats[TARGET + "_orig"]
        preds_stats = preds_stats.drop(columns=[TARGET + "_orig"])
    print_metrics(preds_stats, "STATS model (no market)")

    # ── Walk-forward: FULL model (with market logit) ──
    print("\nWalk-forward — FULL model (stats + market_logit):")
    preds_full = run_walk_forward(df, FULL_FEATURES, "FULL")
    preds_full = preds_full.merge(df[[TARGET, "fight_id"]], on="fight_id", how="left",
                                   suffixes=("", "_orig"))
    if TARGET + "_orig" in preds_full.columns:
        preds_full[TARGET] = preds_full[TARGET + "_orig"]
        preds_full = preds_full.drop(columns=[TARGET + "_orig"])
    print_metrics(preds_full, "FULL model (stats + market)")

    # ── Add odds + edge to predictions ──
    odds_cols = ["fight_id", "A_odds_1", "A_odds_2", "proba_market", "market_logit"]
    preds_full = preds_full.merge(df[odds_cols], on="fight_id", how="left")
    preds_full["edge"] = preds_full["proba_model"] - preds_full["proba_market"]

    # Save OOS predictions
    out_path = PROC / "preds_v2.parquet"
    preds_full.to_parquet(out_path, index=False)
    print(f"\nSaved OOS predictions → {out_path}")
    print(f"  {preds_full['proba_market'].notna().sum()} fights with odds / edge computed")

    # ── Train final model on all data ──
    print("\nTraining final model (all data) …")
    artifacts = train_final_model(df, FULL_FEATURES)
    model_path = PROC / "model_v2.pkl"
    joblib.dump(artifacts, model_path)
    print(f"  Saved → {model_path}")

    # ── Feature importance ──
    print("\nGenerating feature importance plot …")
    plot_importance(artifacts["model"], FULL_FEATURES)

    # ── Quick calibration check ──
    print("\nCalibration check (FULL model, OOS):")
    cal = calibrate(preds_full.dropna(subset=["proba_model", TARGET]), TARGET)
    preds_full["proba_calibrated"] = cal.predict(preds_full["proba_model"].fillna(0.5))
    brier_raw = brier_score_loss(
        preds_full.dropna(subset=["proba_model", TARGET])[TARGET],
        preds_full.dropna(subset=["proba_model", TARGET])["proba_model"]
    )
    brier_cal = brier_score_loss(
        preds_full.dropna(subset=["proba_calibrated", TARGET])[TARGET],
        preds_full.dropna(subset=["proba_calibrated", TARGET])["proba_calibrated"]
    )
    print(f"  Brier (raw)        : {brier_raw:.4f}")
    print(f"  Brier (calibrated) : {brier_cal:.4f}")


if __name__ == "__main__":
    main()
