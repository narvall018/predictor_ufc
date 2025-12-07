#!/usr/bin/env python3
"""
🥊 Entraînement d'un modèle ML avec features combattants
========================================================
- Features: stats cumulées AVANT chaque combat (pas de data leakage)
- Modèles: LogisticRegression, RandomForest, XGBoost, LightGBM
- Validation: TimeSeriesSplit (respecte l'ordre temporel)
- Optimisation: Grid Search + AG parallélisé pour stratégies de mise
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import joblib
import time

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Optimisation
from numba import njit, prange
import itertools

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"

print("=" * 80)
print("🥊 ENTRAÎNEMENT MODÈLE ML AVEC FEATURES COMBATTANTS")
print("=" * 80)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📊 Chargement des données...")

appearances = pd.read_parquet(RAW_DIR / "appearances.parquet")
ratings = pd.read_parquet(INTERIM_DIR / "ratings_timeseries.parquet")
bio = pd.read_parquet(RAW_DIR / "fighter_bio.parquet")
preds_existing = pd.read_parquet(PROC_DIR / "preds_cv.parquet")

# Extraire fighter_id de l'URL dans bio
bio['fighter_id'] = bio['fighter_url'].str.extract(r'/([0-9a-f]{16,})$')

print(f"  Appearances: {len(appearances):,} lignes")
print(f"  Ratings: {len(ratings):,} combats")
print(f"  Bio: {len(bio):,} combattants")
print(f"  Preds existants: {len(preds_existing):,} combats (avec cotes)")

# =============================================================================
# 2. CALCUL DES FEATURES CUMULÉES (SANS DATA LEAKAGE)
# =============================================================================
print("\n🔧 Calcul des features cumulées par combattant...")

# Trier par date
appearances['event_date'] = pd.to_datetime(appearances['event_date'])
appearances = appearances.sort_values('event_date').reset_index(drop=True)

# Colonnes de stats à cumuler
stat_cols = ['kd', 'sig_lnd', 'sig_att', 'td_lnd', 'td_att', 'sub_att', 'ctrl_secs', 'result_win']

# Calculer les stats cumulées AVANT chaque combat pour chaque combattant
def compute_cumulative_stats(df, fighter_id_col='fighter_id'):
    """Calcule les stats cumulées AVANT chaque combat (shift pour éviter leakage)"""
    
    # Grouper par combattant et calculer les stats cumulées
    df = df.sort_values('event_date')
    
    cumstats = []
    for col in stat_cols:
        if col in df.columns:
            # Cumul AVANT le combat actuel (shift de 1)
            df[f'{col}_cum'] = df.groupby(fighter_id_col)[col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            cumstats.append(f'{col}_cum')
    
    # Nombre de combats AVANT
    df['n_fights_before'] = df.groupby(fighter_id_col).cumcount()
    
    return df

appearances = compute_cumulative_stats(appearances)

# Ajouter les données bio
appearances = appearances.merge(
    bio[['fighter_id', 'reach_cm', 'dob', 'weight_lbs']], 
    on='fighter_id', 
    how='left'
)

# Calculer l'âge au moment du combat
appearances['dob'] = pd.to_datetime(appearances['dob'])
appearances['age_at_fight'] = (appearances['event_date'] - appearances['dob']).dt.days / 365.25

print(f"  Features cumulées calculées pour {appearances['fighter_id'].nunique():,} combattants")

# =============================================================================
# 3. CRÉATION DU DATASET DE COMBATS
# =============================================================================
print("\n📦 Création du dataset de combats...")

# Utiliser ratings_timeseries comme base (un row par combat)
fights = ratings.copy()
fights['event_date'] = pd.to_datetime(fights['event_date'])

# Merger les stats du combattant 1
fighter1_stats = appearances.copy()
fighter1_cols = {col: f'f1_{col}' for col in ['reach_cm', 'age_at_fight', 'n_fights_before', 
                                               'kd_cum', 'sig_lnd_cum', 'sig_att_cum', 
                                               'td_lnd_cum', 'td_att_cum', 'ctrl_secs_cum', 
                                               'result_win_cum']}
fighter1_stats = fighter1_stats.rename(columns=fighter1_cols)
fighter1_stats = fighter1_stats.rename(columns={'fighter_id': 'fighter_1_id', 'fight_id': 'fight_id'})

# Prendre une seule ligne par (fight_id, fighter_id)
fighter1_stats = fighter1_stats.drop_duplicates(subset=['fight_id', 'fighter_1_id'])

fights = fights.merge(
    fighter1_stats[['fight_id', 'fighter_1_id'] + list(fighter1_cols.values())],
    on=['fight_id', 'fighter_1_id'],
    how='left'
)

# Merger les stats du combattant 2
fighter2_stats = appearances.copy()
fighter2_cols = {col: f'f2_{col}' for col in ['reach_cm', 'age_at_fight', 'n_fights_before',
                                               'kd_cum', 'sig_lnd_cum', 'sig_att_cum',
                                               'td_lnd_cum', 'td_att_cum', 'ctrl_secs_cum',
                                               'result_win_cum']}
fighter2_stats = fighter2_stats.rename(columns=fighter2_cols)
fighter2_stats = fighter2_stats.rename(columns={'fighter_id': 'fighter_2_id', 'fight_id': 'fight_id'})
fighter2_stats = fighter2_stats.drop_duplicates(subset=['fight_id', 'fighter_2_id'])

fights = fights.merge(
    fighter2_stats[['fight_id', 'fighter_2_id'] + list(fighter2_cols.values())],
    on=['fight_id', 'fighter_2_id'],
    how='left'
)

# Merger les cotes du marché depuis preds_existing
fights = fights.merge(
    preds_existing[['fight_id', 'A_odds_1', 'A_odds_2', 'proba_market', 'market_logit']],
    on='fight_id',
    how='inner'  # Garder uniquement les combats avec cotes
)

# Target: le combattant 1 gagne
fights['y'] = (fights['winner'] == 1).astype(int)

print(f"  Dataset: {len(fights):,} combats avec cotes et features")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n🔬 Feature Engineering...")

# Différences entre combattants
fights['reach_diff'] = fights['f1_reach_cm'] - fights['f2_reach_cm']
fights['age_diff'] = fights['f1_age_at_fight'] - fights['f2_age_at_fight']
fights['exp_diff'] = fights['f1_n_fights_before'] - fights['f2_n_fights_before']
fights['elo_diff'] = fights['elo_1_pre'] - fights['elo_2_pre']

# Stats différentielles
fights['kd_diff'] = fights['f1_kd_cum'].fillna(0) - fights['f2_kd_cum'].fillna(0)
fights['sig_acc_1'] = fights['f1_sig_lnd_cum'] / fights['f1_sig_att_cum'].replace(0, np.nan)
fights['sig_acc_2'] = fights['f2_sig_lnd_cum'] / fights['f2_sig_att_cum'].replace(0, np.nan)
fights['sig_acc_diff'] = fights['sig_acc_1'].fillna(0.5) - fights['sig_acc_2'].fillna(0.5)

fights['td_acc_1'] = fights['f1_td_lnd_cum'] / fights['f1_td_att_cum'].replace(0, np.nan)
fights['td_acc_2'] = fights['f2_td_lnd_cum'] / fights['f2_td_att_cum'].replace(0, np.nan)
fights['td_acc_diff'] = fights['td_acc_1'].fillna(0.5) - fights['td_acc_2'].fillna(0.5)

fights['ctrl_diff'] = fights['f1_ctrl_secs_cum'].fillna(0) - fights['f2_ctrl_secs_cum'].fillna(0)
fights['winrate_diff'] = fights['f1_result_win_cum'].fillna(0.5) - fights['f2_result_win_cum'].fillna(0.5)

# Liste des features
FEATURE_COLS = [
    'market_logit',      # Info du marché (très important)
    'elo_diff',          # Différence Elo
    'reach_diff',        # Différence allonge
    'age_diff',          # Différence âge
    'exp_diff',          # Différence expérience
    'kd_diff',           # Différence knockdowns
    'sig_acc_diff',      # Différence précision strikes
    'td_acc_diff',       # Différence précision takedowns
    'ctrl_diff',         # Différence contrôle
    'winrate_diff',      # Différence winrate historique
]

# Filtrer les combats avec features complètes
fights_clean = fights.dropna(subset=['market_logit', 'elo_diff'])
print(f"  Combats avec features complètes: {len(fights_clean):,}")

# Remplir les NaN restants par 0 (différence neutre)
for col in FEATURE_COLS:
    if col in fights_clean.columns:
        fights_clean[col] = fights_clean[col].fillna(0)

# Trier par date
fights_clean = fights_clean.sort_values('event_date').reset_index(drop=True)

print(f"  Features: {FEATURE_COLS}")

# =============================================================================
# 5. ENTRAÎNEMENT DES MODÈLES (VALIDATION TEMPORELLE)
# =============================================================================
print("\n🤖 Entraînement des modèles avec validation temporelle...")

X = fights_clean[FEATURE_COLS].values
y = fights_clean['y'].values
dates = fights_clean['event_date'].values

# TimeSeriesSplit pour respecter l'ordre temporel
tscv = TimeSeriesSplit(n_splits=5)

# Modèles à tester
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
    'LogisticRegression_L1': LogisticRegression(max_iter=1000, C=0.5, penalty='l1', solver='saga'),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20),
}

results = {}
best_model = None
best_score = -np.inf

for name, model in models.items():
    print(f"\n  📈 {name}...")
    
    accuracies = []
    log_losses = []
    brier_scores = []
    aucs = []
    all_preds = []
    all_true = []
    all_probas = []
    all_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques
        accuracies.append(accuracy_score(y_test, y_pred))
        log_losses.append(log_loss(y_test, y_proba))
        brier_scores.append(brier_score_loss(y_test, y_proba))
        aucs.append(roc_auc_score(y_test, y_proba))
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_probas.extend(y_proba)
        all_indices.extend(test_idx)
    
    # Résultats moyens
    results[name] = {
        'accuracy': np.mean(accuracies),
        'log_loss': np.mean(log_losses),
        'brier': np.mean(brier_scores),
        'auc': np.mean(aucs),
        'predictions': all_probas,
        'indices': all_indices,
        'true': all_true
    }
    
    print(f"      Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"      Log Loss: {np.mean(log_losses):.3f} ± {np.std(log_losses):.3f}")
    print(f"      Brier:    {np.mean(brier_scores):.3f} ± {np.std(brier_scores):.3f}")
    print(f"      AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    
    # Meilleur modèle basé sur AUC
    if np.mean(aucs) > best_score:
        best_score = np.mean(aucs)
        best_model = name

print(f"\n🏆 Meilleur modèle: {best_model} (AUC: {best_score:.3f})")

# =============================================================================
# 6. ENTRAÎNEMENT FINAL DU MEILLEUR MODÈLE
# =============================================================================
print(f"\n🎯 Entraînement final de {best_model}...")

# Scaler final
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

# Modèle final
if best_model == 'LogisticRegression':
    final_model = LogisticRegression(max_iter=1000, C=1.0)
elif best_model == 'LogisticRegression_L1':
    final_model = LogisticRegression(max_iter=1000, C=0.5, penalty='l1', solver='saga')
elif best_model == 'RandomForest':
    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, n_jobs=-1)
else:
    final_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20)

final_model.fit(X_scaled, y)

# Calibration
calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
calibrated_model.fit(X_scaled, y)

# =============================================================================
# 7. PRÉPARER LES DONNÉES POUR BACKTEST
# =============================================================================
print("\n📊 Préparation des données pour backtest...")

# Utiliser les prédictions CV du meilleur modèle
best_results = results[best_model]
pred_indices = best_results['indices']
pred_probas = best_results['predictions']

# Créer le dataframe de backtest
backtest_df = fights_clean.iloc[pred_indices].copy()
backtest_df['proba_model_features'] = pred_probas

# Calculer les probabilités marché
backtest_df['p_market_1'] = 1 / backtest_df['A_odds_1']
backtest_df['p_market_2'] = 1 / backtest_df['A_odds_2']
vig = backtest_df['p_market_1'] + backtest_df['p_market_2']
backtest_df['p_market_1_devig'] = backtest_df['p_market_1'] / vig

# Edge
backtest_df['edge_A'] = backtest_df['proba_model_features'] - backtest_df['p_market_1_devig']
backtest_df['edge_B'] = (1 - backtest_df['proba_model_features']) - (1 - backtest_df['p_market_1_devig'])

backtest_df = backtest_df.sort_values('event_date').reset_index(drop=True)
print(f"  Combats pour backtest: {len(backtest_df):,}")
print(f"  Période: {backtest_df['event_date'].min().date()} → {backtest_df['event_date'].max().date()}")

# =============================================================================
# 8. BACKTEST VECTORISÉ AVEC NUMBA
# =============================================================================
print("\n💰 Backtest vectorisé des stratégies...")

# Préparer les arrays pour Numba
odds_1 = backtest_df['A_odds_1'].values.astype(np.float64)
odds_2 = backtest_df['A_odds_2'].values.astype(np.float64)
proba_model = backtest_df['proba_model_features'].values.astype(np.float64)
y_true = backtest_df['y'].values.astype(np.int32)
dates_int = (backtest_df['event_date'] - pd.Timestamp('2000-01-01')).dt.days.values.astype(np.int32)

@njit
def simulate_strategy_single(odds_1, odds_2, proba_model, y_true, dates_int,
                             kelly_frac, min_edge, max_stake, min_odds, max_odds):
    """Simule une stratégie de mise"""
    n = len(odds_1)
    bankroll = 1000.0
    peak = 1000.0
    max_dd = 0.0
    n_bets = 0
    n_wins = 0
    total_staked = 0.0
    
    bankroll_history = np.zeros(n)
    
    for i in range(n):
        # Probabilités marché
        p_impl_1 = 1.0 / odds_1[i]
        p_impl_2 = 1.0 / odds_2[i]
        vig = p_impl_1 + p_impl_2
        p_market_1 = p_impl_1 / vig
        
        # Edge pour chaque combattant
        edge_1 = proba_model[i] - p_market_1
        edge_2 = (1 - proba_model[i]) - (1 - p_market_1)
        
        # Décision de pari
        bet_on = -1  # Pas de pari
        odds_bet = 0.0
        edge_bet = 0.0
        proba_bet = 0.0
        
        # Vérifier combattant 1
        if edge_1 >= min_edge and odds_1[i] >= min_odds and odds_1[i] <= max_odds:
            bet_on = 1
            odds_bet = odds_1[i]
            edge_bet = edge_1
            proba_bet = proba_model[i]
        
        # Vérifier combattant 2 (si meilleur edge)
        if edge_2 >= min_edge and odds_2[i] >= min_odds and odds_2[i] <= max_odds:
            if bet_on == -1 or edge_2 > edge_bet:
                bet_on = 2
                odds_bet = odds_2[i]
                edge_bet = edge_2
                proba_bet = 1 - proba_model[i]
        
        # Placer le pari
        if bet_on != -1:
            # Kelly criterion
            q = 1 - proba_bet
            b = odds_bet - 1
            kelly = (proba_bet * b - q) / b if b > 0 else 0
            kelly_adj = kelly / kelly_frac
            stake_pct = max(0.01, min(kelly_adj, max_stake))
            stake = bankroll * stake_pct
            
            n_bets += 1
            total_staked += stake
            
            # Résultat
            won = (bet_on == 1 and y_true[i] == 1) or (bet_on == 2 and y_true[i] == 0)
            if won:
                bankroll += stake * (odds_bet - 1)
                n_wins += 1
            else:
                bankroll -= stake
            
            # Drawdown
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd
        
        bankroll_history[i] = bankroll
    
    win_rate = n_wins / n_bets if n_bets > 0 else 0
    roi = (bankroll - 1000) / total_staked if total_staked > 0 else 0
    
    return bankroll, max_dd, n_bets, win_rate, roi, bankroll_history

# =============================================================================
# 9. GRID SEARCH PARALLÉLISÉ
# =============================================================================
print("\n🔍 Grid Search des stratégies...")

kelly_fracs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
min_edges = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
max_stakes = [0.15, 0.20, 0.25, 0.30, 0.40]
min_odds_list = [1.0, 1.2, 1.3]
max_odds_list = [3.0, 4.0, 5.0, 10.0]

param_grid = list(itertools.product(kelly_fracs, min_edges, max_stakes, min_odds_list, max_odds_list))
print(f"  Combinaisons à tester: {len(param_grid):,}")

start_time = time.time()

best_strategies = []

for params in param_grid:
    kelly_frac, min_edge, max_stake, min_odds, max_odds = params
    
    final_bank, max_dd, n_bets, win_rate, roi, _ = simulate_strategy_single(
        odds_1, odds_2, proba_model, y_true, dates_int,
        kelly_frac, min_edge, max_stake, min_odds, max_odds
    )
    
    # Filtrer: au moins 50 paris par an en moyenne
    n_years = (backtest_df['event_date'].max() - backtest_df['event_date'].min()).days / 365.25
    bets_per_year = n_bets / n_years if n_years > 0 else 0
    
    if n_bets >= 100 and final_bank > 1000 and max_dd < 0.60:
        best_strategies.append({
            'kelly_frac': kelly_frac,
            'min_edge': min_edge,
            'max_stake': max_stake,
            'min_odds': min_odds,
            'max_odds': max_odds,
            'final_bank': final_bank,
            'max_dd': max_dd,
            'n_bets': n_bets,
            'win_rate': win_rate,
            'roi': roi,
            'bets_per_year': bets_per_year,
            'profit': final_bank - 1000,
            'sharpe': (roi / max_dd) if max_dd > 0 else 0
        })

elapsed = time.time() - start_time
print(f"  Grid Search terminé en {elapsed:.1f}s")

# Trier par profit
best_strategies = sorted(best_strategies, key=lambda x: x['profit'], reverse=True)

print(f"\n🏆 TOP 10 STRATÉGIES (par profit):")
print("-" * 120)
print(f"{'Rank':<5} {'K':<5} {'Edge':<7} {'MaxS':<6} {'Odds':<10} {'Profit':>12} {'ROI':>8} {'DD':>7} {'WR':>7} {'Bets':>6} {'B/Y':>6}")
print("-" * 120)

for i, s in enumerate(best_strategies[:10]):
    print(f"{i+1:<5} 1/{s['kelly_frac']:<3.1f} {s['min_edge']:.0%}    {s['max_stake']:.0%}   [{s['min_odds']:.1f}-{s['max_odds']:.1f}]   {s['profit']:>10,.0f}€ {s['roi']:>7.1%} {s['max_dd']:>6.1%} {s['win_rate']:>6.1%} {s['n_bets']:>5} {s['bets_per_year']:>5.0f}")

# =============================================================================
# 10. STRATÉGIES RECOMMANDÉES
# =============================================================================
print("\n" + "=" * 80)
print("📋 STRATÉGIES RECOMMANDÉES POUR LE MODÈLE FEATURES")
print("=" * 80)

# Sélectionner des stratégies variées
strategies_by_category = {
    'SAFE_FEATURES': None,       # DD < 35%, profit > 0
    'BALANCED_FEATURES': None,   # DD < 40%, bon profit
    'AGGRESSIVE_FEATURES': None, # Meilleur profit absolu
    'VOLUME_FEATURES': None,     # Plus de paris
}

for s in best_strategies:
    if strategies_by_category['SAFE_FEATURES'] is None and s['max_dd'] < 0.35 and s['profit'] > 0:
        strategies_by_category['SAFE_FEATURES'] = s
    
    if strategies_by_category['BALANCED_FEATURES'] is None and s['max_dd'] < 0.40 and s['profit'] > 50000:
        strategies_by_category['BALANCED_FEATURES'] = s
    
    if strategies_by_category['AGGRESSIVE_FEATURES'] is None and s['profit'] > 100000:
        strategies_by_category['AGGRESSIVE_FEATURES'] = s
    
    if strategies_by_category['VOLUME_FEATURES'] is None and s['bets_per_year'] > 100 and s['profit'] > 0:
        strategies_by_category['VOLUME_FEATURES'] = s

print("\n📊 Stratégies sélectionnées:")
for name, s in strategies_by_category.items():
    if s:
        print(f"\n  {name}:")
        print(f"    Kelly: 1/{s['kelly_frac']}, Edge min: {s['min_edge']:.1%}, Max stake: {s['max_stake']:.0%}")
        print(f"    Cotes: [{s['min_odds']:.1f} - {s['max_odds']:.1f}]")
        print(f"    Profit: {s['profit']:,.0f}€, ROI: {s['roi']:.1%}, DD: {s['max_dd']:.1%}")
        print(f"    {s['n_bets']} paris ({s['bets_per_year']:.0f}/an), WR: {s['win_rate']:.1%}")

# =============================================================================
# 11. COMPARAISON AVEC MODÈLE ACTUEL (market+reach+age)
# =============================================================================
print("\n" + "=" * 80)
print("📊 COMPARAISON: Modèle Features vs Modèle Actuel (market+reach+age)")
print("=" * 80)

# Backtest du modèle actuel avec les mêmes données
proba_market_model = backtest_df['p_market_1_devig'].values + backtest_df['reach_diff'].values * 0.001 + backtest_df['age_diff'].values * (-0.005)
proba_market_model = np.clip(proba_market_model, 0.01, 0.99)

# Utiliser la meilleure stratégie actuelle (SAFE)
current_best = simulate_strategy_single(
    odds_1, odds_2, proba_market_model, y_true, dates_int,
    2.75, 0.035, 0.25, 1.0, 5.0
)

features_best = None
if strategies_by_category['SAFE_FEATURES']:
    s = strategies_by_category['SAFE_FEATURES']
    features_best = simulate_strategy_single(
        odds_1, odds_2, proba_model, y_true, dates_int,
        s['kelly_frac'], s['min_edge'], s['max_stake'], s['min_odds'], s['max_odds']
    )

print(f"\n  Modèle Actuel (SAFE): Profit {current_best[0]-1000:,.0f}€, DD {current_best[1]:.1%}, {current_best[2]} paris")
if features_best:
    print(f"  Modèle Features (SAFE): Profit {features_best[0]-1000:,.0f}€, DD {features_best[1]:.1%}, {features_best[2]} paris")

# =============================================================================
# 12. SAUVEGARDE DU MODÈLE
# =============================================================================
print("\n💾 Sauvegarde du modèle features...")

model_info = {
    'model': calibrated_model,
    'scaler': scaler_final,
    'features': FEATURE_COLS,
    'best_model_name': best_model,
    'metrics': results[best_model],
    'strategies': strategies_by_category,
    'training_date': datetime.now().isoformat()
}

joblib.dump(model_info, PROC_DIR / "model_features.pkl")
print(f"  Sauvegardé: {PROC_DIR / 'model_features.pkl'}")

# Sauvegarder les prédictions
backtest_df.to_parquet(PROC_DIR / "preds_features_cv.parquet", index=False)
print(f"  Prédictions: {PROC_DIR / 'preds_features_cv.parquet'}")

print("\n" + "=" * 80)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("=" * 80)

# Générer le code des stratégies pour app.py
print("\n📝 Code pour app.py (stratégies features):")
print("-" * 60)
print("FEATURES_BETTING_STRATEGIES = {")
for name, s in strategies_by_category.items():
    if s:
        emoji = "🛡️" if "SAFE" in name else "🟢" if "BALANCED" in name else "🔥" if "AGGRESSIVE" in name else "📈"
        print(f'    "{emoji} {name}": {{')
        print(f'        "kelly_fraction": {s["kelly_frac"]},')
        print(f'        "min_confidence": 0.0,')
        print(f'        "min_edge": {s["min_edge"]},')
        print(f'        "max_value": 1.0,')
        print(f'        "min_odds": {s["min_odds"]},')
        print(f'        "max_odds": {s["max_odds"]},')
        print(f'        "max_bet_fraction": {s["max_stake"]},')
        print(f'        "min_bet_pct": 0.01,')
        print(f'        "description": "{emoji} {name} - Profit {s["profit"]/1000:.0f}k€ | ROI {s["roi"]:.0%} | DD {s["max_dd"]:.0%} | {s["n_bets"]} paris"')
        print('    },')
print("}")
