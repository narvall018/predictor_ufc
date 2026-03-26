"""
pipeline/01_build_features.py
═══════════════════════════════════════════════════════════
Build comprehensive feature matrix for UFC fight prediction.

Features engineered:
  - Trailing L5 stats (no leakage): sig strike acc, TD acc,
    net striking, control time, knockdowns, submission attempts,
    defensive metrics, win rate
  - Physical: reach diff, height diff, age diff, stance matchup
  - Elo: global and divisional differential
  - Experience: UFC fight count diff, days since last fight diff
  - Market: market_logit, proba_market (from preds_cv when available)

Output: data/processed/features_v2.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW     = Path("data/raw")
INTERIM = Path("data/interim")
PROC    = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

WINDOW = 5   # trailing fights


# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────
def load_data():
    apps   = pd.read_parquet(RAW / "appearances.parquet")
    bio    = pd.read_parquet(RAW / "fighter_bio.parquet")
    fights = pd.read_parquet(INTERIM / "ratings_timeseries.parquet")
    preds  = pd.read_parquet(PROC / "preds_cv.parquet")
    return apps, bio, fights, preds


# ──────────────────────────────────────────────
# 2. Enrich appearances with defensive stats
# ──────────────────────────────────────────────
def add_defensive_stats(apps: pd.DataFrame) -> pd.DataFrame:
    """
    For each fighter-fight row, add the opponent's offensive stats
    as defensive stats (what you allowed the opponent to do).
    """
    df = apps.copy()

    # Per-fight accuracy rates
    df["sig_acc"] = df["sig_lnd"] / df["sig_att"].clip(lower=1)
    df["td_acc"]  = df["td_lnd"]  / df["td_att"].clip(lower=1)

    # Opponent's stats in the same fight
    opp = df[["fight_id", "fighter_id",
              "sig_lnd", "td_lnd", "ctrl_secs", "kd",
              "sig_acc", "td_acc"]].copy()
    opp.columns = ["fight_id", "opp_id",
                   "def_sig_absorbed", "def_td_absorbed",
                   "def_ctrl_secs",    "def_kd",
                   "def_sig_acc_opp",  "def_td_acc_opp"]

    df = df.merge(opp, on="fight_id", how="left")
    df = df[df["fighter_id"] != df["opp_id"]].reset_index(drop=True)

    # Net striking = landed minus absorbed
    df["net_sig"] = df["sig_lnd"] - df["def_sig_absorbed"]

    return df


# ──────────────────────────────────────────────
# 3. Trailing rolling stats per fighter (no leakage)
# ──────────────────────────────────────────────
STAT_COLS = [
    "sig_lnd", "sig_acc",
    "td_lnd",  "td_acc",
    "sub_att", "ctrl_secs", "kd",
    "def_sig_absorbed", "def_td_absorbed",
    "def_ctrl_secs",    "def_kd",
    "net_sig",
    "result_win",
]


def compute_trailing_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Rolling mean of last `window` fights, shifted by 1 to ensure
    stats used for fight N come only from fights 0..N-1.
    """
    df = df.sort_values(["fighter_id", "event_date"]).reset_index(drop=True)

    roll = {}
    for col in STAT_COLS:
        roll[f"{col}_L{window}"] = (
            df.groupby("fighter_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # UFC experience: fights before this one
    roll["ufc_fights"] = (
        df.groupby("fighter_id")["fight_id"]
        .transform(lambda x: x.expanding().count().shift(1).fillna(0))
    )

    # Days since last fight (layoff)
    roll["days_off"] = (
        df.groupby("fighter_id")["event_date"]
        .transform(lambda x: x.diff().dt.days)
    )

    trailing = pd.DataFrame(roll, index=df.index)
    return pd.concat([df, trailing], axis=1)


# ──────────────────────────────────────────────
# 4. Build fight-level feature matrix
# ──────────────────────────────────────────────
def build_fight_features(
    fights:       pd.DataFrame,
    apps_trail:   pd.DataFrame,
    bio:          pd.DataFrame,
    preds:        pd.DataFrame,
) -> pd.DataFrame:

    w = WINDOW
    trail_cols = [f"{c}_L{w}" for c in STAT_COLS] + ["ufc_fights", "days_off"]

    # Map fighter_id → fighter_url (for bio join)
    id_to_url = (
        apps_trail[["fighter_id", "fighter_url"]]
        .drop_duplicates("fighter_id")
        .set_index("fighter_id")["fighter_url"]
        .to_dict()
    )

    # Per-fighter trailing stats indexed by (fight_id, fighter_id)
    stats_idx = apps_trail.set_index(["fight_id", "fighter_id"])[trail_cols]

    feat = fights.copy()

    # Normalise target: 1 = fighter_1 wins, 0 = fighter_2 wins
    feat["y"] = (feat["winner"] == 1).astype(int)

    # ── Trailing stats for fighter 1 ──
    f1_stats = (
        feat.join(
            stats_idx.rename(columns={c: f"f1_{c}" for c in trail_cols}),
            on=["fight_id", "fighter_1_id"],
        )
    )
    feat = f1_stats.copy()

    # ── Trailing stats for fighter 2 ──
    f2_stats = (
        feat.join(
            stats_idx.rename(columns={c: f"f2_{c}" for c in trail_cols}),
            on=["fight_id", "fighter_2_id"],
        )
    )
    feat = f2_stats.copy()

    # ── Differential features ──
    for col in trail_cols:
        feat[f"diff_{col}"] = feat[f"f1_{col}"] - feat[f"f2_{col}"]

    # ── Elo differential ──
    feat["elo_diff"] = feat["elo_1_pre"] - feat["elo_2_pre"]

    # ── Bio features (reach, height, age, stance) ──
    bio_map = bio.set_index("fighter_url")[["reach_cm", "height_cm", "dob", "stance"]]

    feat["f1_url"] = feat["fighter_1_id"].map(id_to_url)
    feat["f2_url"] = feat["fighter_2_id"].map(id_to_url)

    for side, url_col in [("f1", "f1_url"), ("f2", "f2_url")]:
        joined = feat[url_col].map(bio_map["reach_cm"].to_dict())
        feat[f"{side}_reach"]  = feat[url_col].map(bio_map["reach_cm"].to_dict())
        feat[f"{side}_height"] = feat[url_col].map(bio_map["height_cm"].to_dict())
        feat[f"{side}_dob"]    = feat[url_col].map(bio_map["dob"].to_dict())
        feat[f"{side}_stance"] = feat[url_col].map(bio_map["stance"].to_dict())

    # Normalise event_date timezone
    ed = feat["event_date"]
    if hasattr(ed.dtype, "tz") and ed.dt.tz is not None:
        ed = ed.dt.tz_localize(None)

    dob1 = pd.to_datetime(feat["f1_dob"], errors="coerce")
    dob2 = pd.to_datetime(feat["f2_dob"], errors="coerce")

    feat["reach_diff"]  = feat["f1_reach"]  - feat["f2_reach"]
    feat["height_diff"] = feat["f1_height"] - feat["f2_height"]
    feat["age_diff"]    = (
        (ed - dob1).dt.days / 365.25 -
        (ed - dob2).dt.days / 365.25
    )

    # Southpaw vs Orthodox — well-documented striking advantage
    feat["southpaw_matchup"] = (
        ((feat["f1_stance"] == "Orthodox")  & (feat["f2_stance"] == "Southpaw")) |
        ((feat["f1_stance"] == "Southpaw")  & (feat["f2_stance"] == "Orthodox"))
    ).astype(float)

    # ── Market features (only available for ~5 225 fights) ──
    mkt_cols = ["fight_id", "A_odds_1", "A_odds_2", "proba_market", "market_logit"]
    preds_clean = preds[mkt_cols].copy()
    preds_clean["fight_id"] = preds_clean["fight_id"].astype(str)
    feat["fight_id"]         = feat["fight_id"].astype(str)

    feat = feat.merge(preds_clean, on="fight_id", how="left")

    # Drop helper columns
    feat = feat.drop(columns=["f1_url", "f2_url", "f1_dob", "f2_dob",
                               "f1_stance", "f2_stance", "f1_reach", "f2_reach",
                               "f1_height", "f2_height"], errors="ignore")

    return feat


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading raw data …")
    apps, bio, fights, preds = load_data()

    print("Adding defensive stats …")
    apps_rich = add_defensive_stats(apps)

    print(f"Computing trailing L{WINDOW} stats (strict no-leakage) …")
    apps_trail = compute_trailing_stats(apps_rich, WINDOW)

    print("Building fight-level feature matrix …")
    feat = build_fight_features(fights, apps_trail, bio, preds)

    n_total  = len(feat)
    n_odds   = feat["proba_market"].notna().sum()
    n_feats  = feat.select_dtypes(include=[np.number]).shape[1]
    ed       = feat["event_date"]
    if hasattr(ed.dtype, "tz") and ed.dt.tz is not None:
        ed = ed.dt.tz_localize(None)

    print(f"\n{'='*50}")
    print(f"Feature matrix : {n_total} fights × {n_feats} numeric features")
    print(f"Fights w/ odds : {n_odds} ({n_odds/n_total*100:.1f}%)")
    print(f"Date range     : {ed.min().date()} → {ed.max().date()}")
    print(f"{'='*50}")

    # Missing value report
    miss = feat.select_dtypes(include=[np.number]).isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print("\nMissing values (top 15):")
        print(miss.head(15).to_string())

    # ── Symmetry augmentation ──────────────────────────────────
    # Fighter_1 in ratings_timeseries is always the Red Corner
    # (higher-ranked fighter), who wins ~57% of the time.
    # If we train on this raw data, the model learns a F1 bias that
    # isn't a real edge (market already prices in red-corner advantage).
    # Fix: randomly flip F1/F2 labels for 50% of fights so the model
    # learns purely from stat differentials, not corner assignment.
    rng = np.random.default_rng(seed=42)
    flip_mask = rng.random(len(feat)) < 0.5

    diff_cols = [c for c in feat.columns if c.startswith("diff_")]
    sign_cols  = diff_cols + ["elo_diff", "reach_diff", "age_diff",
                               "market_logit"]
    for col in sign_cols:
        if col in feat.columns:
            feat.loc[flip_mask, col] = -feat.loc[flip_mask, col]

    feat.loc[flip_mask, "y"] = 1 - feat.loc[flip_mask, "y"]

    # Also swap fighter names and odds so backtest remains consistent
    feat_flip = feat.loc[flip_mask].copy()
    feat.loc[flip_mask, "fighter_1"]  = feat_flip["fighter_2"]
    feat.loc[flip_mask, "fighter_2"]  = feat_flip["fighter_1"]
    feat.loc[flip_mask, "A_odds_1"]   = feat_flip["A_odds_2"]
    feat.loc[flip_mask, "A_odds_2"]   = feat_flip["A_odds_1"]
    feat.loc[flip_mask, "proba_market"] = 1 - feat_flip["proba_market"]

    n_flipped = flip_mask.sum()
    print(f"  Symmetry augmentation: {n_flipped}/{len(feat)} fights flipped → "
          f"y=1 rate now {feat['y'].mean():.3f}")
    # ───────────────────────────────────────────────────────────

    feat.to_parquet(PROC / "features_v2.parquet", index=False)
    print(f"\nSaved → {PROC / 'features_v2.parquet'}")


if __name__ == "__main__":
    main()
