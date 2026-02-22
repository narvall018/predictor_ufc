#!/usr/bin/env python3
"""
Backtest complet du modèle UFC sur la période 2015-2025
Analyse les performances de toutes les stratégies de paris
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Vérifier et installer les dépendances si nécessaire
try:
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Installation des dépendances nécessaires...")
    os.system("pip install -q pandas numpy joblib matplotlib seaborn pyarrow scikit-learn")
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'processed', 'model_pipeline.pkl')
CALIBRATOR_PATH = os.path.join(DATA_DIR, 'processed', 'calibrator.pkl')
APPEARANCES_PATH = os.path.join(DATA_DIR, 'raw', 'appearances.parquet')
FIGHTER_BIO_PATH = os.path.join(DATA_DIR, 'raw', 'fighter_bio.parquet')
RATINGS_PATH = os.path.join(DATA_DIR, 'interim', 'ratings_timeseries.parquet')

# Constantes
BASE_ELO = 1500
INITIAL_BANKROLL = 1000.0

# Définition des 5 stratégies optimisées
STRATEGIES = {
    'SAFE': {
        'name': '🛡️ SAFE',
        'kelly_fraction': 2.75,
        'min_confidence': 0.0,
        'min_edge': 0.035,
        'max_bet_fraction': 0.25,
        'min_bet_pct': 0.01,
        'min_odds': 1.0,
        'max_odds': 5.0,
        'max_value': 1.0
    },
    'EQUILIBREE': {
        'name': '🟢 ÉQUILIBRÉE',
        'kelly_fraction': 2.5,
        'min_confidence': 0.0,
        'min_edge': 0.042,
        'max_bet_fraction': 0.30,
        'min_bet_pct': 0.01,
        'min_odds': 1.0,
        'max_odds': 5.0,
        'max_value': 1.0
    },
    'AGRESSIVE': {
        'name': '🔥 AGRESSIVE',
        'kelly_fraction': 2.0,
        'min_confidence': 0.0,
        'min_edge': 0.042,
        'max_bet_fraction': 0.36,
        'min_bet_pct': 0.01,
        'min_odds': 1.0,
        'max_odds': 5.0,
        'max_value': 1.0
    },
    'VOLUME_PLUS': {
        'name': '📈 VOLUME+',
        'kelly_fraction': 3.0,
        'min_confidence': 0.0,
        'min_edge': 0.030,
        'max_bet_fraction': 0.20,
        'min_bet_pct': 0.01,
        'min_odds': 1.0,
        'max_odds': 5.0,
        'max_value': 1.0
    },
    'SELECTIF': {
        'name': '💎 SÉLECTIF',
        'kelly_fraction': 2.2,
        'min_confidence': 0.0,
        'min_edge': 0.063,
        'max_bet_fraction': 0.37,
        'min_bet_pct': 0.01,
        'min_odds': 1.0,
        'max_odds': 5.0,
        'max_value': 1.0
    }
}


def calculate_kelly_stake(proba_model, odds, bankroll, strategy_params):
    """Calcule la mise selon le critère de Kelly"""
    kelly_fraction = strategy_params['kelly_fraction']
    min_confidence = strategy_params['min_confidence']
    min_edge = strategy_params['min_edge']
    max_ev = strategy_params.get('max_value', 1.0)
    max_bet_fraction = strategy_params['max_bet_fraction']
    min_bet_pct = strategy_params['min_bet_pct']
    min_odds = strategy_params.get('min_odds', 1.0)
    max_odds = strategy_params.get('max_odds', 999.0)

    p_market = 1.0 / odds if odds > 0 else 0
    edge = proba_model - p_market
    ev = (proba_model * odds) - 1

    should_bet = (
        proba_model >= min_confidence and
        edge >= min_edge and
        ev <= max_ev and
        odds >= min_odds and
        odds <= max_odds and
        ev > 0
    )

    if not should_bet:
        return 0, edge, ev

    q = 1 - proba_model
    b = odds - 1
    kelly_fraction_value = (proba_model * b - q) / b
    kelly_adjusted = kelly_fraction_value / kelly_fraction
    kelly_pct = max(min_bet_pct, min(kelly_adjusted, max_bet_fraction))
    stake = bankroll * kelly_pct

    return stake, edge, ev


def simulate_odds_from_elo(elo_a, elo_b, noise_std=0.15):
    """
    Simule des cotes réalistes basées sur les ratings Elo
    Ajoute du bruit pour simuler l'inefficience du marché
    """
    # Proba Elo "vraie"
    proba_elo_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    # Ajouter du bruit gaussien pour simuler l'inefficience du marché
    noise = np.random.normal(0, noise_std)
    proba_market_a = np.clip(proba_elo_a + noise, 0.05, 0.95)
    proba_market_b = 1 - proba_market_a

    # Ajouter le vigorish (marge du bookmaker) ~5%
    vig_factor = 1.05
    proba_with_vig_a = proba_market_a * vig_factor
    proba_with_vig_b = proba_market_b * vig_factor

    # Convertir en cotes décimales
    odds_a = 1 / proba_with_vig_a
    odds_b = 1 / proba_with_vig_b

    return odds_a, odds_b


def load_data():
    """Charge toutes les données nécessaires"""
    print("Chargement des données...")

    # Charger le modèle
    try:
        model_data = joblib.load(MODEL_PATH)
        # Le modèle est un dict avec la clé 'model'
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_medians = model_data.get('feature_medians', {'reach_diff': 0.0, 'age_diff': 0.0})
        else:
            model = model_data
            feature_medians = {'reach_diff': 0.0, 'age_diff': 0.0}
        print(f"✓ Modèle chargé: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Erreur chargement modèle: {e}")
        return None

    # Charger les données de combats
    try:
        appearances = pd.read_parquet(APPEARANCES_PATH)
        print(f"✓ Combats chargés: {len(appearances)} apparitions")
    except Exception as e:
        print(f"✗ Erreur chargement appearances: {e}")
        return None

    # Charger les bios des combattants
    try:
        fighter_bio = pd.read_parquet(FIGHTER_BIO_PATH)
        print(f"✓ Bios chargées: {len(fighter_bio)} combattants")
    except Exception as e:
        print(f"✗ Erreur chargement bios: {e}")
        fighter_bio = pd.DataFrame()

    # Charger les ratings Elo
    try:
        ratings = pd.read_parquet(RATINGS_PATH)
        print(f"✓ Ratings Elo chargés: {len(ratings)} entrées")
    except Exception as e:
        print(f"✗ Erreur chargement ratings: {e}")
        ratings = pd.DataFrame()

    return {
        'model': model,
        'feature_medians': feature_medians,
        'appearances': appearances,
        'fighter_bio': fighter_bio,
        'ratings': ratings
    }


def prepare_backtest_data(data):
    """Prépare les données pour le backtest"""
    print("\nPréparation des données de backtest...")

    appearances = data['appearances'].copy()
    fighter_bio = data['fighter_bio']
    ratings = data['ratings']

    # Filtrer par date (2015-2025)
    appearances['event_date'] = pd.to_datetime(appearances['event_date'])
    appearances = appearances[
        (appearances['event_date'] >= '2015-01-01') &
        (appearances['event_date'] <= '2025-12-31')
    ].copy()

    print(f"Période: {appearances['event_date'].min()} à {appearances['event_date'].max()}")

    # Regrouper par combat (2 apparitions par combat)
    fights = []
    grouped = appearances.groupby('fight_id')

    for fight_id, group in grouped:
        if len(group) != 2:
            continue

        row1, row2 = group.iloc[0], group.iloc[1]

        # Déterminer le gagnant (result_win: 1.0 = victoire, 0.0 = défaite)
        if pd.notna(row1['result_win']) and row1['result_win'] == 1.0:
            winner_idx = 0
        elif pd.notna(row2['result_win']) and row2['result_win'] == 1.0:
            winner_idx = 1
        else:
            continue  # Skip draws/NC/no result

        # Récupérer les infos des combattants
        fighter_1_url = row1['fighter_url']
        fighter_2_url = row2['fighter_url']

        # Récupérer reach et age depuis les bios
        bio1 = fighter_bio[fighter_bio['fighter_url'] == fighter_1_url]
        bio2 = fighter_bio[fighter_bio['fighter_url'] == fighter_2_url]

        reach_1 = bio1['reach_cm'].values[0] if len(bio1) > 0 and not pd.isna(bio1['reach_cm'].values[0]) else None
        reach_2 = bio2['reach_cm'].values[0] if len(bio2) > 0 and not pd.isna(bio2['reach_cm'].values[0]) else None

        # Calculer l'âge au moment du combat
        if len(bio1) > 0 and 'dob' in bio1.columns and not pd.isna(bio1['dob'].values[0]):
            dob1 = pd.to_datetime(bio1['dob'].values[0])
            age_1 = (row1['event_date'] - dob1).days / 365.25
        else:
            age_1 = None

        if len(bio2) > 0 and 'dob' in bio2.columns and not pd.isna(bio2['dob'].values[0]):
            dob2 = pd.to_datetime(bio2['dob'].values[0])
            age_2 = (row2['event_date'] - dob2).days / 365.25
        else:
            age_2 = None

        # Récupérer les Elo pré-combat
        fight_ratings = ratings[ratings['fight_id'] == fight_id]
        if len(fight_ratings) > 0:
            elo_1 = fight_ratings['elo_1_pre'].values[0]
            elo_2 = fight_ratings['elo_2_pre'].values[0]
        else:
            elo_1 = BASE_ELO
            elo_2 = BASE_ELO

        fights.append({
            'fight_id': fight_id,
            'event_date': row1['event_date'],
            'fighter_1': row1['fighter_name'],
            'fighter_2': row2['fighter_name'],
            'winner_idx': winner_idx,
            'reach_1': reach_1,
            'reach_2': reach_2,
            'age_1': age_1,
            'age_2': age_2,
            'elo_1': elo_1,
            'elo_2': elo_2
        })

    df_fights = pd.DataFrame(fights)
    df_fights = df_fights.sort_values('event_date').reset_index(drop=True)

    print(f"✓ {len(df_fights)} combats préparés ({df_fights['event_date'].dt.year.min()}-{df_fights['event_date'].dt.year.max()})")
    print(f"Distribution par année:")
    print(df_fights['event_date'].dt.year.value_counts().sort_index())

    return df_fights


def run_backtest(df_fights, model, feature_medians, strategy_params, strategy_name):
    """Exécute le backtest pour une stratégie donnée"""

    # Initialisation
    bankroll = INITIAL_BANKROLL
    bankroll_history = [bankroll]
    dates_history = [df_fights.iloc[0]['event_date']]

    bets_placed = []
    wins = 0
    losses = 0
    total_staked = 0
    total_profit = 0

    for idx, row in df_fights.iterrows():
        # Simuler les cotes basées sur Elo
        odds_1, odds_2 = simulate_odds_from_elo(row['elo_1'], row['elo_2'])

        # Préparer les features pour le modèle
        # Market logit
        p_impl_1 = 1 / odds_1
        p_impl_2 = 1 / odds_2
        vig = p_impl_1 + p_impl_2
        proba_market = p_impl_1 / vig

        proba_market_clipped = np.clip(proba_market, 0.01, 0.99)
        market_logit = np.log(proba_market_clipped / (1 - proba_market_clipped))

        # Reach diff
        if row['reach_1'] is not None and row['reach_2'] is not None and pd.notna(row['reach_1']) and pd.notna(row['reach_2']):
            reach_diff = float(row['reach_1']) - float(row['reach_2'])
        else:
            reach_diff = feature_medians['reach_diff']

        # Age diff
        if row['age_1'] is not None and row['age_2'] is not None and pd.notna(row['age_1']) and pd.notna(row['age_2']):
            age_diff = float(row['age_1']) - float(row['age_2'])
        else:
            age_diff = feature_medians['age_diff']

        # S'assurer qu'il n'y a pas de NaN
        if pd.isna(market_logit) or pd.isna(reach_diff) or pd.isna(age_diff):
            continue  # Skip ce combat si les features sont invalides

        # Prédiction du modèle
        X = np.array([[market_logit, reach_diff, age_diff]])
        proba_1 = model.predict_proba(X)[0][1]
        proba_2 = 1 - proba_1

        # Calculer les stakes pour les deux combattants
        stake_1, edge_1, ev_1 = calculate_kelly_stake(proba_1, odds_1, bankroll, strategy_params)
        stake_2, edge_2, ev_2 = calculate_kelly_stake(proba_2, odds_2, bankroll, strategy_params)

        # Placer le pari (on ne parie que sur un seul combattant par combat)
        bet_placed = False

        if stake_1 > 0 and stake_1 >= stake_2:
            # Parier sur combattant 1
            result = 'win' if row['winner_idx'] == 0 else 'loss'
            profit = stake_1 * (odds_1 - 1) if result == 'win' else -stake_1

            bankroll += profit
            total_staked += stake_1
            total_profit += profit

            if result == 'win':
                wins += 1
            else:
                losses += 1

            bets_placed.append({
                'date': row['event_date'],
                'fighter': row['fighter_1'],
                'odds': odds_1,
                'stake': stake_1,
                'proba_model': proba_1,
                'edge': edge_1,
                'ev': ev_1,
                'result': result,
                'profit': profit
            })
            bet_placed = True

        elif stake_2 > 0:
            # Parier sur combattant 2
            result = 'win' if row['winner_idx'] == 1 else 'loss'
            profit = stake_2 * (odds_2 - 1) if result == 'win' else -stake_2

            bankroll += profit
            total_staked += stake_2
            total_profit += profit

            if result == 'win':
                wins += 1
            else:
                losses += 1

            bets_placed.append({
                'date': row['event_date'],
                'fighter': row['fighter_2'],
                'odds': odds_2,
                'stake': stake_2,
                'proba_model': proba_2,
                'edge': edge_2,
                'ev': ev_2,
                'result': result,
                'profit': profit
            })
            bet_placed = True

        # Enregistrer l'historique de bankroll
        if bet_placed:
            bankroll_history.append(bankroll)
            dates_history.append(row['event_date'])

        # Fail-safe: si la bankroll tombe à 0 ou moins
        if bankroll <= 0:
            print(f"⚠️  Bankroll épuisée à la date {row['event_date']}")
            break

    # Calculer les métriques
    df_bets = pd.DataFrame(bets_placed)

    if len(df_bets) == 0:
        return None

    final_bankroll = bankroll
    total_return = final_bankroll - INITIAL_BANKROLL
    roi = (total_return / INITIAL_BANKROLL) * 100

    # Calculer le drawdown
    bankroll_series = pd.Series(bankroll_history)
    running_max = bankroll_series.expanding().max()
    drawdown = ((bankroll_series - running_max) / running_max) * 100
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    # ROI par stake
    roi_per_stake = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # Années profitables
    df_bets['year'] = pd.to_datetime(df_bets['date']).dt.year
    yearly_profit = df_bets.groupby('year')['profit'].sum()
    profitable_years = (yearly_profit > 0).sum()
    total_years = len(yearly_profit)

    # Statistiques par année
    yearly_stats = df_bets.groupby('year').agg({
        'profit': 'sum',
        'stake': 'sum',
        'result': lambda x: (x == 'win').sum() / len(x) * 100
    }).round(2)
    yearly_stats.columns = ['Profit (€)', 'Stake Total (€)', 'Win Rate (%)']
    yearly_stats['ROI (%)'] = (yearly_stats['Profit (€)'] / yearly_stats['Stake Total (€)'] * 100).round(2)

    results = {
        'strategy_name': strategy_name,
        'num_bets': len(df_bets),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'initial_bankroll': INITIAL_BANKROLL,
        'final_bankroll': final_bankroll,
        'total_return': total_return,
        'roi': roi,
        'roi_per_stake': roi_per_stake,
        'total_staked': total_staked,
        'max_drawdown': max_drawdown,
        'profitable_years': profitable_years,
        'total_years': total_years,
        'bankroll_history': bankroll_history,
        'dates_history': dates_history,
        'bets_df': df_bets,
        'yearly_stats': yearly_stats,
        'avg_odds': df_bets['odds'].mean(),
        'avg_stake_pct': (df_bets['stake'] / INITIAL_BANKROLL * 100).mean(),
        'avg_edge': df_bets['edge'].mean() * 100
    }

    return results


def print_results(results):
    """Affiche les résultats de manière formatée"""

    print("\n" + "="*80)
    print(f"RÉSULTATS BACKTEST - {results['strategy_name']}")
    print("="*80)

    print(f"\n📊 STATISTIQUES GÉNÉRALES")
    print(f"Période: 2015-2025")
    print(f"Nombre de paris: {results['num_bets']}")
    print(f"Paris gagnés: {results['wins']} ({results['win_rate']:.2f}%)")
    print(f"Paris perdus: {results['losses']}")

    print(f"\n💰 PERFORMANCE FINANCIÈRE")
    print(f"Bankroll initiale: {results['initial_bankroll']:.2f}€")
    print(f"Bankroll finale: {results['final_bankroll']:.2f}€")
    print(f"Profit total: {results['total_return']:.2f}€")
    print(f"ROI global: {results['roi']:.2f}%")
    print(f"ROI par € misé: {results['roi_per_stake']:.2f}%")
    print(f"Total misé: {results['total_staked']:.2f}€")

    print(f"\n📉 RISQUE")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Années profitables: {results['profitable_years']}/{results['total_years']}")

    print(f"\n📈 MOYENNES")
    print(f"Cote moyenne: {results['avg_odds']:.2f}")
    print(f"Mise moyenne: {results['avg_stake_pct']:.2f}% de la bankroll")
    print(f"Edge moyen: {results['avg_edge']:.2f}%")

    print(f"\n📅 PERFORMANCE ANNUELLE")
    print(results['yearly_stats'].to_string())

    print("\n" + "="*80)


def plot_results(all_results):
    """Génère des graphiques de performance"""

    # Créer une figure avec 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Backtest UFC 2015-2025 - Analyse Comparative des Stratégies', fontsize=16, fontweight='bold')

    # 1. Évolution de la bankroll
    ax1 = axes[0, 0]
    for strategy_name, results in all_results.items():
        ax1.plot(results['dates_history'], results['bankroll_history'],
                label=results['strategy_name'], linewidth=2)
    ax1.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5, label='Bankroll initiale')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bankroll (€)')
    ax1.set_title('Évolution de la Bankroll')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Comparaison ROI
    ax2 = axes[0, 1]
    strategies_names = [r['strategy_name'] for r in all_results.values()]
    rois = [r['roi'] for r in all_results.values()]
    colors = ['#2ecc71' if roi > 0 else '#e74c3c' for roi in rois]
    ax2.barh(strategies_names, rois, color=colors)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('ROI (%)')
    ax2.set_title('ROI par Stratégie')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Win Rate vs Nombre de Paris
    ax3 = axes[1, 0]
    win_rates = [r['win_rate'] for r in all_results.values()]
    num_bets = [r['num_bets'] for r in all_results.values()]
    ax3.scatter(num_bets, win_rates, s=200, alpha=0.6)
    for i, name in enumerate(strategies_names):
        ax3.annotate(name, (num_bets[i], win_rates[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Nombre de Paris')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate vs Volume de Paris')
    ax3.grid(True, alpha=0.3)

    # 4. Tableau de comparaison
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Créer un tableau de données
    table_data = []
    for results in all_results.values():
        table_data.append([
            results['strategy_name'],
            f"{results['roi']:.1f}%",
            f"{results['max_drawdown']:.1f}%",
            f"{results['num_bets']}",
            f"{results['win_rate']:.1f}%",
            f"{results['profitable_years']}/{results['total_years']}"
        ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Stratégie', 'ROI', 'Max DD', 'Paris', 'Win%', 'Années+'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Colorer l'en-tête
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Colorer les lignes alternées
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.tight_layout()

    # Sauvegarder
    output_path = os.path.join(BASE_DIR, 'backtest_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvegardés: {output_path}")

    return output_path


def main():
    """Fonction principale"""

    print("="*80)
    print("BACKTEST UFC PREDICTOR - Période 2015-2025")
    print("="*80)

    # Charger les données
    data = load_data()
    if data is None:
        print("Erreur lors du chargement des données")
        return

    # Préparer les données de backtest
    df_fights = prepare_backtest_data(data)
    if df_fights is None or len(df_fights) == 0:
        print("Pas de combats disponibles pour le backtest")
        return

    # Exécuter le backtest pour chaque stratégie
    print("\n" + "="*80)
    print("EXÉCUTION DES BACKTESTS")
    print("="*80)

    all_results = {}

    for strategy_key, strategy_params in STRATEGIES.items():
        print(f"\n🔄 Backtest en cours: {strategy_params['name']}...")

        results = run_backtest(
            df_fights,
            data['model'],
            data['feature_medians'],
            strategy_params,
            strategy_params['name']
        )

        if results:
            all_results[strategy_key] = results
            print_results(results)
        else:
            print(f"⚠️  Aucun pari placé pour {strategy_params['name']}")

    # Générer les graphiques comparatifs
    if all_results:
        print("\n" + "="*80)
        print("GÉNÉRATION DES GRAPHIQUES")
        print("="*80)
        plot_results(all_results)

    # Résumé comparatif
    print("\n" + "="*80)
    print("RÉSUMÉ COMPARATIF")
    print("="*80)

    summary_df = pd.DataFrame({
        'Stratégie': [r['strategy_name'] for r in all_results.values()],
        'ROI (%)': [f"{r['roi']:.2f}" for r in all_results.values()],
        'Profit (€)': [f"{r['total_return']:.2f}" for r in all_results.values()],
        'Bankroll Finale (€)': [f"{r['final_bankroll']:.2f}" for r in all_results.values()],
        'Max DD (%)': [f"{r['max_drawdown']:.2f}" for r in all_results.values()],
        'Win Rate (%)': [f"{r['win_rate']:.2f}" for r in all_results.values()],
        'Nb Paris': [r['num_bets'] for r in all_results.values()],
        'Années +': [f"{r['profitable_years']}/{r['total_years']}" for r in all_results.values()]
    })

    print("\n" + summary_df.to_string(index=False))

    # Recommandation
    best_roi = max(all_results.values(), key=lambda x: x['roi'])
    best_sharpe = min(all_results.values(), key=lambda x: abs(x['max_drawdown']) / (x['roi'] + 0.001))

    print("\n" + "="*80)
    print("RECOMMANDATIONS")
    print("="*80)
    print(f"\n🏆 Meilleur ROI: {best_roi['strategy_name']} ({best_roi['roi']:.2f}%)")
    print(f"🛡️  Meilleur ratio risque/rendement: {best_sharpe['strategy_name']}")

    print("\n✅ Backtest terminé avec succès!")
    print("="*80)


if __name__ == "__main__":
    # Fixer la seed pour la reproductibilité
    np.random.seed(42)
    main()
