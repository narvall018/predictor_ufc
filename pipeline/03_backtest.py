"""
pipeline/03_backtest.py
═══════════════════════════════════════════════════════════
Full backtest of the v2 model with CLV tracking and
detailed performance analytics.

Key metrics:
  CLV  (Closing Line Value) — the only reliable long-run edge test
  ROI, Sharpe ratio, max drawdown
  Annual & monthly P&L breakdown
  Edge distribution analysis

Usage:
  python pipeline/03_backtest.py
  python pipeline/03_backtest.py --min_date 2020-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

PROC = Path("data/processed")

INITIAL_BANKROLL = 1000.0

# ── Stratégies originales de l'app ───────────────────────────────
# Réplication exacte des paramètres de app.py / BETTING_STRATEGIES
# kelly_fraction : diviseur Kelly (identique à app.py)
# min_edge       : edge min = proba_model - 1/odds  (avec vig, comme app.py)
# max_bet_pct    : max_bet_fraction (cap dur sur la mise)
# min_bet_pct    : mise minimale en % bankroll
# odds_range     : (min_odds, max_odds)
# max_ev         : EV max autorisé (filtre anti-suspects)
STRATEGIES = {
    "SAFE": {
        "kelly_fraction": 2.75,
        "min_edge":       0.035,
        "max_bet_pct":    0.25,
        "min_bet_pct":    0.01,
        "odds_range":     (1.0, 5.0),
        "max_ev":         1.0,
    },
    "EQUILIBREE": {
        "kelly_fraction": 2.5,
        "min_edge":       0.042,
        "max_bet_pct":    0.30,
        "min_bet_pct":    0.01,
        "odds_range":     (1.0, 5.0),
        "max_ev":         1.0,
    },
    "AGRESSIVE": {
        "kelly_fraction": 2.0,
        "min_edge":       0.042,
        "max_bet_pct":    0.36,
        "min_bet_pct":    0.01,
        "odds_range":     (1.0, 5.0),
        "max_ev":         1.0,
    },
    "VOLUME+": {
        "kelly_fraction": 3.0,
        "min_edge":       0.030,
        "max_bet_pct":    0.20,
        "min_bet_pct":    0.01,
        "odds_range":     (1.0, 5.0),
        "max_ev":         1.0,
    },
    "SELECTIF": {
        "kelly_fraction": 2.2,
        "min_edge":       0.063,
        "max_bet_pct":    0.37,
        "min_bet_pct":    0.01,
        "odds_range":     (1.0, 5.0),
        "max_ev":         1.0,
    },
}

STOP_LOSS_PCT = 0.70   # pause si bankroll chute de 70% depuis le pic


# ── Kelly stake — réplication exacte app.py ──
def kelly_stake_orig(p: float, odds: float, bankroll: float, cfg: dict) -> float:
    """Même formule que calculate_kelly_stake() dans app.py."""
    kelly_fraction = cfg["kelly_fraction"]
    max_bet_pct    = cfg["max_bet_pct"]
    min_bet_pct    = cfg["min_bet_pct"]

    b = odds - 1.0
    q = 1 - p
    kelly_raw     = (p * b - q) / b
    kelly_adjusted = kelly_raw / kelly_fraction
    pct = max(min_bet_pct, min(kelly_adjusted, max_bet_pct))
    return bankroll * pct


def should_bet(p_model: float, odds: float, cfg: dict) -> tuple:
    """
    Réplication exacte des conditions de l'app.py :
      edge  = proba_model - 1/odds  (prob marché brute, avec vig)
      ev    = proba_model * odds - 1
      Conditions: edge >= min_edge  AND  ev > 0  AND  ev <= max_ev
                  AND  min_odds <= odds <= max_odds
    Retourne (bet: bool, edge, ev)
    """
    p_market_raw = 1.0 / odds if odds > 0 else 0.0
    edge = p_model - p_market_raw
    ev   = p_model * odds - 1.0

    odds_lo, odds_hi = cfg["odds_range"]
    bet = (
        edge >= cfg["min_edge"]
        and ev   >  0
        and ev   <= cfg["max_ev"]
        and odds >= odds_lo
        and odds <= odds_hi
    )
    return bet, edge, ev


# ── Simulate one strategy ─────────────────────
def run_backtest(preds: pd.DataFrame, cfg: dict, name: str) -> pd.DataFrame:
    df = preds.dropna(subset=["proba_model", "A_odds_1", "A_odds_2"]).copy()
    df = df.sort_values("event_date").reset_index(drop=True)

    bankroll  = INITIAL_BANKROLL
    peak_bk   = INITIAL_BANKROLL
    bets      = []

    for _, row in df.iterrows():
        p1    = row["proba_model"]
        p2    = 1 - p1
        odds1 = row["A_odds_1"]
        odds2 = row["A_odds_2"]

        bet1, edge1, ev1 = should_bet(p1, odds1, cfg)
        bet2, edge2, ev2 = should_bet(p2, odds2, cfg)

        # Choisir le meilleur côté (plus grand EV) — même logique que app.py
        candidates = []
        if bet1:
            candidates.append((ev1, 1, p1, odds1, edge1))
        if bet2:
            candidates.append((ev2, 2, p2, odds2, edge2))

        if not candidates:
            continue

        _, side, p_bet, odds_bet, edge = max(candidates, key=lambda x: x[0])
        pm_bet = row.get("proba_market", 1.0 / odds_bet)

        # Stop-loss guard
        if bankroll < peak_bk * (1 - STOP_LOSS_PCT):
            continue

        stake = kelly_stake_orig(p_bet, odds_bet, bankroll, cfg)
        if stake < 0.50:
            continue

        y     = int(row["y"])
        won   = (side == 1 and y == 1) or (side == 2 and y == 0)
        profit = stake * (odds_bet - 1) if won else -stake
        bankroll += profit
        peak_bk   = max(peak_bk, bankroll)

        # Model EV: expected profit per unit staked according to model
        # Positive = model predicts value over the quoted odds
        clv = p_bet * (odds_bet - 1) - (1 - p_bet)   # = EV of this bet

        bets.append({
            "event_date":  row["event_date"],
            "fighter_1":   row["fighter_1"],
            "fighter_2":   row["fighter_2"],
            "bet_side":    side,
            "bet_fighter": row["fighter_1"] if side == 1 else row["fighter_2"],
            "odds":        round(odds_bet, 3),
            "stake":       round(stake, 2),
            "p_model":     round(p_bet, 4),
            "p_market":    round(pm_bet, 4),
            "edge":        round(edge, 4),
            "ev":          round(p_bet * (odds_bet - 1) - (1 - p_bet), 4),
            "won":         int(won),
            "profit":      round(profit, 2),
            "bankroll":    round(bankroll, 2),
            "clv":         round(clv, 4),
        })

    return pd.DataFrame(bets)


# ── Compute & print metrics ───────────────────
def metrics(bets: pd.DataFrame, name: str) -> dict:
    if len(bets) == 0:
        print(f"  {name}: no bets placed")
        return {}

    staked  = bets["stake"].sum()
    profit  = bets["profit"].sum()
    roi     = profit / staked * 100
    wr      = bets["won"].mean() * 100
    n       = len(bets)

    # Drawdown
    bk = pd.concat([pd.Series([INITIAL_BANKROLL]), bets["bankroll"].reset_index(drop=True)])
    peak = bk.cummax()
    dd   = ((bk - peak) / peak).min() * 100

    # Monthly Sharpe
    bets["month"] = pd.to_datetime(bets["event_date"]).dt.to_period("M")
    monthly = bets.groupby("month")["profit"].sum()
    sharpe  = (monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0

    # CLV
    avg_clv = bets["clv"].mean() * 100    # avg model EV per bet

    # Final bankroll
    final_bk    = bets["bankroll"].iloc[-1]
    total_ret   = (final_bk - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100

    # Annual breakdown
    bets["year"] = pd.to_datetime(bets["event_date"]).dt.year
    annual = bets.groupby("year").agg(
        n_bets  = ("profit", "count"),
        profit  = ("profit", "sum"),
        staked  = ("stake",  "sum"),
    )
    annual["roi_pct"] = (annual["profit"] / annual["staked"] * 100).round(1)
    profitable_yrs    = (annual["roi_pct"] > 0).sum()

    print(f"\n{'═'*58}")
    print(f"  STRATEGY : {name}")
    print(f"{'═'*58}")
    print(f"  Période      : {pd.to_datetime(bets['event_date']).min().date()} "
          f"→ {pd.to_datetime(bets['event_date']).max().date()}")
    print(f"  Nb paris     : {n}")
    print(f"  Win rate     : {wr:.1f}%")
    print(f"  ROI          : {roi:+.2f}%")
    print(f"  Bankroll     : €{INITIAL_BANKROLL:.0f} → €{final_bk:.0f} ({total_ret:+.1f}%)")
    print(f"  Max Drawdown : {dd:.1f}%")
    print(f"  Sharpe (ann) : {sharpe:.2f}")
    print(f"  EV moyen/bet : {avg_clv:+.2f}%  {'✅' if avg_clv > 0 else '❌'}")
    print(f"  Années profit: {profitable_yrs}/{len(annual)}")
    print(f"\n  Détail annuel:")
    print(annual[["n_bets", "roi_pct"]].to_string())

    return {
        "name": name, "n": n, "wr": wr, "roi": roi,
        "total_ret": total_ret, "dd": dd, "sharpe": sharpe,
        "clv": avg_clv, "final_bk": final_bk,
        "profitable_yrs": f"{profitable_yrs}/{len(annual)}",
        "bets": bets,
    }


# ── Plots ─────────────────────────────────────
def make_plots(results: dict, save_path: Path):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_bk   = fig.add_subplot(gs[0, :])      # bankroll full-width
    ax_roi  = fig.add_subplot(gs[1, 0])      # ROI by year
    ax_clv  = fig.add_subplot(gs[1, 1])      # CLV distribution
    ax_edge = fig.add_subplot(gs[1, 2])      # edge vs outcome

    palette = {
        "CONSERVATIVE": "#2ecc71",
        "BALANCED":     "#3498db",
        "AGGRESSIVE":   "#e74c3c",
        "HIGH_VOLUME":  "#f39c12",
    }

    # ── Bankroll evolution ──
    for name, res in results.items():
        if "bets" not in res or res["bets"].empty:
            continue
        bets  = res["bets"]
        dates = pd.to_datetime(bets["event_date"])
        bk    = pd.concat([pd.Series([INITIAL_BANKROLL]),
                           bets["bankroll"].reset_index(drop=True)])
        ax_bk.plot(list(dates) + [dates.iloc[-1]], bk,
                   label=f"{name}  ROI={res['roi']:+.1f}%  CLV={res['clv']:+.2f}%",
                   color=palette.get(name, "gray"), linewidth=1.8)

    ax_bk.axhline(INITIAL_BANKROLL, color="black", linestyle="--",
                  alpha=0.4, linewidth=1, label="Mise initiale")
    ax_bk.set_title("Évolution bankroll — backtest walk-forward", fontweight="bold")
    ax_bk.set_ylabel("Bankroll (€)")
    ax_bk.legend(fontsize=8)
    ax_bk.grid(True, alpha=0.25)

    # ── ROI par an (BALANCED) ──
    best_name = "BALANCED" if "BALANCED" in results else list(results.keys())[0]
    if "bets" in results.get(best_name, {}):
        b   = results[best_name]["bets"].copy()
        b["year"] = pd.to_datetime(b["event_date"]).dt.year
        ann = b.groupby("year").agg(p=("profit","sum"), s=("stake","sum"))
        ann["roi"] = ann["p"] / ann["s"] * 100
        colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in ann["roi"]]
        ax_roi.bar(ann.index.astype(str), ann["roi"], color=colors_bar, alpha=0.85)
        ax_roi.axhline(0, color="black", linewidth=0.8)
        ax_roi.set_title(f"ROI annuel ({best_name})", fontweight="bold")
        ax_roi.set_ylabel("ROI (%)")
        ax_roi.tick_params(axis="x", rotation=45)
        ax_roi.grid(True, alpha=0.25)

    # ── CLV distribution ──
    if "bets" in results.get(best_name, {}):
        b   = results[best_name]["bets"]
        won = b.loc[b["won"] == 1, "clv"] * 100
        los = b.loc[b["won"] == 0, "clv"] * 100
        ax_clv.hist(won, bins=25, alpha=0.6, color="#2ecc71", label="Gagné")
        ax_clv.hist(los, bins=25, alpha=0.6, color="#e74c3c", label="Perdu")
        ax_clv.axvline(b["clv"].mean() * 100, color="navy",
                       linestyle="--", linewidth=1.5,
                       label=f"CLV moy={b['clv'].mean()*100:+.2f}%")
        ax_clv.set_title(f"Distribution CLV ({best_name})", fontweight="bold")
        ax_clv.set_xlabel("CLV (%)")
        ax_clv.legend(fontsize=8)
        ax_clv.grid(True, alpha=0.25)

    # ── Edge vs outcome ──
    if "bets" in results.get(best_name, {}):
        b = results[best_name]["bets"].copy()
        b["edge_bin"] = pd.cut(b["edge"] * 100, bins=8)
        grp = b.groupby("edge_bin").agg(wr=("won", "mean"), n=("won", "count"))
        grp["wr"] *= 100
        ax_edge.bar(range(len(grp)), grp["wr"], color="#3498db", alpha=0.8)
        ax_edge.set_xticks(range(len(grp)))
        ax_edge.set_xticklabels([str(x) for x in grp.index], rotation=45, fontsize=7)
        ax_edge.axhline(50, color="black", linestyle="--", linewidth=0.8)
        ax_edge.set_title(f"Win rate par tranche d'edge ({best_name})", fontweight="bold")
        ax_edge.set_ylabel("Win rate (%)")
        ax_edge.grid(True, alpha=0.25)

    plt.suptitle("UFC Predictor v2 — Backtest complet", fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGraphique sauvegardé → {save_path}")


# ── Summary table ─────────────────────────────
def summary_table(results: dict):
    print(f"\n{'═'*85}")
    print("  TABLEAU RÉCAPITULATIF")
    print(f"{'═'*85}")
    hdr = f"{'Stratégie':<14}{'Paris':>7}{'WR%':>7}{'ROI%':>8}{'Retour%':>9}{'DrawD%':>9}{'Sharpe':>8}{'CLV%':>7}{'Années':>8}"
    print(hdr)
    print("─" * 85)
    for name, r in results.items():
        if "roi" not in r:
            continue
        print(f"{name:<14}{r['n']:>7}{r['wr']:>6.1f}%{r['roi']:>7.2f}%"
              f"{r['total_ret']:>8.1f}%{r['dd']:>8.1f}%"
              f"{r['sharpe']:>8.2f}{r['clv']:>6.1f}%{r['profitable_yrs']:>8}")


# ── Main ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_date", default="2015-01-01",
                        help="Début backtest (YYYY-MM-DD)")
    parser.add_argument("--preds_file", default=None,
                        help="Fichier parquet de prédictions (défaut: preds_v2.parquet)")
    args = parser.parse_args()

    preds_path = Path(args.preds_file) if args.preds_file else PROC / "preds_v2.parquet"
    print(f"Chargement des prédictions : {preds_path.name} …")
    preds = pd.read_parquet(preds_path)
    preds["event_date"] = pd.to_datetime(preds["event_date"])
    if preds["event_date"].dt.tz is not None:
        preds["event_date"] = preds["event_date"].dt.tz_localize(None)

    preds = preds[preds["event_date"] >= args.min_date].copy()
    preds = preds.dropna(subset=["proba_model", "proba_market"]).copy()

    print(f"Prédictions : {len(preds)} combats | "
          f"{preds['event_date'].min().date()} → {preds['event_date'].max().date()}")

    results = {}
    for name, cfg in STRATEGIES.items():
        bets = run_backtest(preds, cfg, name)
        results[name] = metrics(bets, name)

    summary_table(results)

    make_plots(results, PROC / "backtest_v2.png")

    # Save best bets for inspection
    if "BALANCED" in results and "bets" in results["BALANCED"]:
        out = PROC / "backtest_bets_v2.csv"
        results["BALANCED"]["bets"].to_csv(out, index=False)
        print(f"Paris BALANCED sauvegardés → {out}")

    # CLV interpretation
    print("\n── EV modèle (valeur espérée par pari selon le modèle) ──────────")
    for name, r in results.items():
        if "clv" not in r:
            continue
        sign = "✅ EV positif" if r["clv"] > 0 else "❌ EV négatif"
        print(f"  {name:<14} EV={r['clv']:+.1f}%  {sign}")
    print()
    print("  EV moyen > 0 = le modèle prédit de la valeur sur les paris placés.")
    print("  Pour valider l'edge réel, comparer aux cotes de clôture Pinnacle.")


if __name__ == "__main__":
    main()
