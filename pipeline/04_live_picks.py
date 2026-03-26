"""
pipeline/04_live_picks.py
═══════════════════════════════════════════════════════════
Generate live betting recommendations for upcoming UFC fights.

Pipeline:
  1. Fetch upcoming fights + odds from The Odds API (best odds multi-books)
  2. Match fighters to our database (normalised name matching)
  3. Compute features (trailing stats L5, Elo, physical)
  4. Predict with v2 model (LightGBM calibrated)
  5. Apply Kelly criterion  →  ranked recommendations

Usage:
  python pipeline/04_live_picks.py
  python pipeline/04_live_picks.py --bankroll 1200 --strategy BALANCED
  python pipeline/04_live_picks.py --strategy AGGRESSIVE --min_edge 0.03
"""

import json
import os
import re
import unicodedata
import urllib.request
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAW     = Path("data/raw")
INTERIM = Path("data/interim")
PROC    = Path("data/processed")

WINDOW = 5
PRIORITY_BOOKS = ["pinnacle", "betfair", "unibet", "1xbet", "draftkings"]

STRATEGIES = {
    "CONSERVATIVE": {"kelly_div": 4.0, "min_edge": 0.040, "max_bet_pct": 0.12, "odds_range": (1.20, 4.00)},
    "BALANCED":     {"kelly_div": 3.0, "min_edge": 0.035, "max_bet_pct": 0.18, "odds_range": (1.20, 5.00)},
    "AGGRESSIVE":   {"kelly_div": 2.0, "min_edge": 0.030, "max_bet_pct": 0.28, "odds_range": (1.20, 6.00)},
    "HIGH_VOLUME":  {"kelly_div": 4.5, "min_edge": 0.025, "max_bet_pct": 0.10, "odds_range": (1.15, 6.00)},
}

STAT_COLS = [
    "sig_lnd", "sig_acc",
    "td_lnd",  "td_acc",
    "sub_att", "ctrl_secs", "kd",
    "def_sig_absorbed", "def_td_absorbed",
    "def_ctrl_secs",    "def_kd",
    "net_sig",
    "result_win",
]


# ──────────────────────────────────────────────
# 1. The Odds API
# ──────────────────────────────────────────────
def fetch_odds_raw(api_key: str) -> list:
    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = f"?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
    req = urllib.request.Request(url + params, headers={"User-Agent": "UFC-Predictor-v2"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        data      = json.loads(resp.read().decode("utf-8"))
    print(f"  API quota: {used} utilisées / {remaining} restantes ce mois")
    return data


def parse_events(events: list) -> pd.DataFrame:
    """
    For each fight, extract:
      - best odds (highest price across all books)
      - consensus devigged probability (average across top books)
      - number of books
    """
    rows = []
    for ev in events:
        books = ev.get("bookmakers", [])
        if not books:
            continue

        home = ev["home_team"]
        away = ev["away_team"]
        start = pd.to_datetime(ev["commence_time"])

        home_odds_all, away_odds_all = [], []

        for book in books:
            for mkt in book.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for oc in mkt["outcomes"]:
                    if oc["name"] == home:
                        home_odds_all.append(oc["price"])
                    elif oc["name"] == away:
                        away_odds_all.append(oc["price"])

        if not home_odds_all or not away_odds_all:
            continue

        # Best odds across all books (for stake placement)
        best_home = max(home_odds_all)
        best_away = max(away_odds_all)

        # Consensus devig: average raw proba, then normalise
        p_home_avg = np.mean([1 / o for o in home_odds_all])
        p_away_avg = np.mean([1 / o for o in away_odds_all])
        total      = p_home_avg + p_away_avg
        p_market   = p_home_avg / total   # devigged consensus

        rows.append({
            "commence_time": start,
            "fighter_1":     home,
            "fighter_2":     away,
            "odds_1_best":   round(best_home, 3),
            "odds_2_best":   round(best_away, 3),
            "p_market":      round(p_market, 4),
            "market_logit":  round(np.log(p_market / (1 - p_market)), 4),
            "n_books":       len(books),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 2. Fighter name matching
# ──────────────────────────────────────────────
def norm(name: str) -> str:
    """Normalise: remove accents, lowercase, strip punctuation."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[^a-z\s]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def match_name(api_name: str, db_names: list) -> str | None:
    """
    Match API fighter name to database name.
    Priority: exact → lastname+initial → lastname only
    """
    norm_api  = norm(api_name)
    parts_api = norm_api.split()

    # 1. Exact match
    for db in db_names:
        if norm(db) == norm_api:
            return db

    # 2. Last name + first initial
    for db in db_names:
        parts_db = norm(db).split()
        if (parts_api and parts_db
                and parts_api[-1] == parts_db[-1]
                and len(parts_api) > 1 and len(parts_db) > 1
                and parts_api[0][0] == parts_db[0][0]):
            return db

    # 3. Last name only
    for db in db_names:
        parts_db = norm(db).split()
        if parts_api and parts_db and parts_api[-1] == parts_db[-1]:
            return db

    return None


# ──────────────────────────────────────────────
# 3. Fighter database (trailing stats + Elo)
# ──────────────────────────────────────────────
def build_fighter_db() -> pd.DataFrame:
    """
    For each fighter in our database, compute their latest trailing
    stats (L5, INCLUDING their most recent fight — no shift, since
    we are predicting a future fight).
    """
    apps = pd.read_parquet(RAW / "appearances.parquet")
    bio  = pd.read_parquet(RAW / "fighter_bio.parquet")
    rat  = pd.read_parquet(INTERIM / "ratings_timeseries.parquet")

    # Per-fight rates
    apps["sig_acc"] = apps["sig_lnd"] / apps["sig_att"].clip(lower=1)
    apps["td_acc"]  = apps["td_lnd"]  / apps["td_att"].clip(lower=1)

    # Defensive stats
    opp = apps[["fight_id", "fighter_id", "sig_lnd", "td_lnd",
                 "ctrl_secs", "kd", "sig_acc", "td_acc"]].copy()
    opp.columns = ["fight_id", "opp_id",
                   "def_sig_absorbed", "def_td_absorbed",
                   "def_ctrl_secs",    "def_kd",
                   "def_sig_acc_opp",  "def_td_acc_opp"]
    apps = apps.merge(opp, on="fight_id", how="left")
    apps = apps[apps["fighter_id"] != apps["opp_id"]].copy()
    apps["net_sig"] = apps["sig_lnd"] - apps["def_sig_absorbed"]

    apps = apps.sort_values(["fighter_id", "event_date"])

    # Trailing L5 (NO shift — include all fights up to today)
    for col in STAT_COLS:
        apps[f"{col}_L{WINDOW}"] = (
            apps.groupby("fighter_id")[col]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        )
    apps["ufc_fights"]  = apps.groupby("fighter_id").cumcount() + 1
    apps["days_off"]    = apps.groupby("fighter_id")["event_date"].transform(
        lambda x: x.diff().dt.days
    )

    # Keep only latest entry per fighter
    latest = apps.sort_values("event_date").groupby("fighter_id").last().reset_index()

    # Latest Elo
    elo1 = rat[["fighter_1_id", "fighter_1", "elo_1_post", "event_date"]].rename(
        columns={"fighter_1_id": "fighter_id", "fighter_1": "name_from_rat",
                 "elo_1_post": "elo"})
    elo2 = rat[["fighter_2_id", "fighter_2", "elo_2_post", "event_date"]].rename(
        columns={"fighter_2_id": "fighter_id", "fighter_2": "name_from_rat",
                 "elo_2_post": "elo"})
    all_elo   = pd.concat([elo1, elo2]).sort_values("event_date")
    latest_elo = all_elo.groupby("fighter_id").last()[["elo", "name_from_rat"]].reset_index()

    db = latest.merge(latest_elo[["fighter_id", "elo", "name_from_rat"]],
                      on="fighter_id", how="left")
    db["elo"] = db["elo"].fillna(1500.0)

    # Add bio
    db = db.merge(bio[["fighter_url", "reach_cm", "height_cm", "dob", "stance"]],
                  on="fighter_url", how="left")

    # Canonical display name: prefer name_from_rat (cleaner) else fighter_name
    db["display_name"] = db["name_from_rat"].fillna(db["fighter_name"])

    return db


# ──────────────────────────────────────────────
# 4. Feature computation for one fight
# ──────────────────────────────────────────────
def fight_features(f1: pd.Series, f2: pd.Series,
                   market_logit: float, event_date: pd.Timestamp) -> dict:
    W = WINDOW

    def d(col):
        v1 = f1.get(col, np.nan)
        v2 = f2.get(col, np.nan)
        if pd.isna(v1) or pd.isna(v2):
            return np.nan
        return float(v1) - float(v2)

    # Age
    age_diff = np.nan
    dob1 = f1.get("dob")
    dob2 = f2.get("dob")
    if pd.notna(dob1) and pd.notna(dob2):
        age_diff = (
            (event_date - pd.Timestamp(dob1)).days / 365.25 -
            (event_date - pd.Timestamp(dob2)).days / 365.25
        )

    southpaw = float(
        (f1.get("stance") == "Orthodox"  and f2.get("stance") == "Southpaw") or
        (f1.get("stance") == "Southpaw"  and f2.get("stance") == "Orthodox")
    )

    return {
        "elo_diff":                  d("elo"),
        f"diff_sig_lnd_L{W}":        d(f"sig_lnd_L{W}"),
        f"diff_sig_acc_L{W}":        d(f"sig_acc_L{W}"),
        f"diff_net_sig_L{W}":        d(f"net_sig_L{W}"),
        f"diff_kd_L{W}":             d(f"kd_L{W}"),
        f"diff_def_sig_absorbed_L{W}": d(f"def_sig_absorbed_L{W}"),
        f"diff_td_lnd_L{W}":         d(f"td_lnd_L{W}"),
        f"diff_td_acc_L{W}":         d(f"td_acc_L{W}"),
        f"diff_sub_att_L{W}":        d(f"sub_att_L{W}"),
        f"diff_ctrl_secs_L{W}":      d(f"ctrl_secs_L{W}"),
        f"diff_def_td_absorbed_L{W}": d(f"def_td_absorbed_L{W}"),
        f"diff_result_win_L{W}":     d(f"result_win_L{W}"),
        "diff_ufc_fights":           d("ufc_fights"),
        "diff_days_off":             d("days_off"),
        "reach_diff":                d("reach_cm"),
        "age_diff":                  age_diff,
        "southpaw_matchup":          southpaw,
        "market_logit":              market_logit,
    }


# ──────────────────────────────────────────────
# 5. Kelly stake
# ──────────────────────────────────────────────
def kelly_stake(p: float, odds: float, bankroll: float,
                kelly_div: float, max_bet_pct: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (p * b - (1 - p)) / b
    frac  = max(0.0, kelly / kelly_div)
    pct   = min(frac, max_bet_pct)
    return round(bankroll * pct, 2)


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UFC live picks v2")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--strategy", default="BALANCED",
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--min_edge", type=float, default=None,
                        help="Overrides strategy min_edge")
    args = parser.parse_args()

    cfg = STRATEGIES[args.strategy].copy()
    if args.min_edge is not None:
        cfg["min_edge"] = args.min_edge

    print("=" * 62)
    print("  UFC PREDICTOR v2 — Live Picks")
    print(f"  Stratégie: {args.strategy} | Bankroll: €{args.bankroll:.0f}")
    print(f"  Min edge: {cfg['min_edge']*100:.1f}% | Kelly: 1/{cfg['kelly_div']:.1f}")
    print("=" * 62)

    # ── Load model ──
    print("\nChargement du modèle …")
    arts     = joblib.load(PROC / "model_v2.pkl")
    model    = arts["model"]
    cal      = arts.get("calibrator")
    features = arts["features"]
    medians  = arts["medians"]

    # ── Build fighter DB ──
    print("Construction base fighters …")
    db = build_fighter_db()
    db_names = db["display_name"].dropna().tolist()
    db_idx   = db.set_index("display_name")

    # ── Fetch odds ──
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        print("\nERREUR: ODDS_API_KEY non trouvé.")
        print("  → export ODDS_API_KEY='votre_cle'")
        return

    print("Récupération des cotes (The Odds API) …")
    try:
        raw_events = fetch_odds_raw(api_key)
    except Exception as e:
        print(f"Erreur API: {e}")
        return

    fights_df = parse_events(raw_events)
    if fights_df.empty:
        print("Aucun combat trouvé.")
        return

    print(f"  {len(fights_df)} combats récupérés\n")

    # ── Process each fight ──
    results = []

    for _, fight in fights_df.iterrows():
        api_name1 = fight["fighter_1"]
        api_name2 = fight["fighter_2"]

        matched1 = match_name(api_name1, db_names)
        matched2 = match_name(api_name2, db_names)

        if matched1 is None or matched2 is None:
            miss = api_name1 if matched1 is None else api_name2
            print(f"  ⚠  Inconnu en DB : {miss}  →  {api_name1} vs {api_name2}")
            continue

        f1 = db_idx.loc[matched1]
        f2 = db_idx.loc[matched2]

        # Compute features
        fdict = fight_features(f1, f2, fight["market_logit"], fight["commence_time"])

        X = pd.DataFrame([fdict])[features]
        for col, med in medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)

        # Predict
        raw_p = model.predict_proba(X)[0][1]
        p1    = float(cal.predict([raw_p])[0]) if cal is not None else raw_p
        p2    = 1 - p1

        pm1   = fight["p_market"]
        pm2   = 1 - pm1
        edge1 = p1 - pm1
        edge2 = p2 - pm2

        # Determine bet side
        min_e  = cfg["min_edge"]
        lo, hi = cfg["odds_range"]
        bet_side, edge_bet, p_bet, pm_bet, odds_bet = None, 0, None, None, None

        if edge1 >= min_e and edge1 >= edge2 and lo <= fight["odds_1_best"] <= hi:
            bet_side, edge_bet = matched1, edge1
            p_bet, pm_bet, odds_bet = p1, pm1, fight["odds_1_best"]
        elif edge2 >= min_e and lo <= fight["odds_2_best"] <= hi:
            bet_side, edge_bet = matched2, edge2
            p_bet, pm_bet, odds_bet = p2, pm2, fight["odds_2_best"]

        stake = kelly_stake(p_bet, odds_bet, args.bankroll,
                            cfg["kelly_div"], cfg["max_bet_pct"]) if bet_side else 0.0
        ev    = round(p_bet * (odds_bet - 1) - (1 - p_bet), 4) if bet_side else None

        results.append({
            "date":      fight["commence_time"].date(),
            "fighter_1": matched1,
            "fighter_2": matched2,
            "odds_1":    fight["odds_1_best"],
            "odds_2":    fight["odds_2_best"],
            "p_model_1": round(p1, 3),
            "p_market_1": round(pm1, 3),
            "edge_1":    round(edge1 * 100, 2),
            "edge_2":    round(edge2 * 100, 2),
            "n_books":   fight["n_books"],
            "bet_on":    bet_side,
            "odds_bet":  odds_bet,
            "stake":     stake,
            "ev_pct":    round(ev * 100, 2) if ev else None,
        })

    if not results:
        print("Aucun résultat.")
        return

    df = pd.DataFrame(results)

    # ── Recommended bets ──
    reco = df[df["bet_on"].notna()].sort_values("edge_1", ascending=False)

    print(f"{'='*62}")
    print(f"  PARIS RECOMMANDÉS ({len(reco)} / {len(df)} combats analysés)")
    print(f"{'='*62}")

    total_stake = 0.0
    for _, r in reco.iterrows():
        is_f1 = r["bet_on"] == r["fighter_1"]
        edge  = r["edge_1"] if is_f1 else r["edge_2"]
        print(f"\n  {r['fighter_1']}  vs  {r['fighter_2']}")
        print(f"  📅 {r['date']}  |  {r['n_books']} bookmakers")
        print(f"  ✅ PARI : {r['bet_on']}  @  {r['odds_bet']:.2f}")
        print(f"  📊 Modèle: {r['p_model_1']*100:.1f}%  |  Marché: {r['p_market_1']*100:.1f}%  |  Edge: {edge:.2f}%")
        print(f"  💰 Mise: €{r['stake']:.2f}  |  EV: {r['ev_pct']:+.2f}%")
        total_stake += r["stake"]

    pct_bankroll = total_stake / args.bankroll * 100
    print(f"\n  ──────────────────────────────────────")
    print(f"  Total exposé : €{total_stake:.2f}  ({pct_bankroll:.1f}% bankroll)")

    # ── All fights summary ──
    print(f"\n{'='*62}")
    print("  TOUS LES COMBATS ANALYSÉS")
    print(f"{'='*62}")
    display = df[["date", "fighter_1", "fighter_2", "odds_1", "odds_2",
                  "p_model_1", "p_market_1", "edge_1", "edge_2",
                  "n_books", "bet_on"]].copy()
    display.columns = ["Date", "F1", "F2", "Cote1", "Cote2",
                       "Modèle%", "Marché%", "Edge1%", "Edge2%",
                       "Books", "Pari"]
    display["Modèle%"] = (display["Modèle%"] * 100).round(1)
    display["Marché%"] = (display["Marché%"] * 100).round(1)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
