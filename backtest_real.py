#!/usr/bin/env python3
"""
Backtest RÉALISTE du modèle UFC sur la période 2015-2025
Utilise les vraies cotes historiques et prédictions du modèle
"""

import os
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Installation des dépendances...")
    os.system("pip install -q pandas numpy matplotlib seaborn pyarrow")
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_CV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'preds_cv.parquet')

# Constantes
INITIAL_BANKROLL = 1000.0

# Les 5 stratégies optimisées
STRATEGIES = {
    'SAFE': {
        'name': '🛡️ SAFE',
        'kelly_fraction': 2.75,
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

    if kelly_fraction_value <= 0:
        return 0, edge, ev

    kelly_adjusted = kelly_fraction_value / kelly_fraction
    kelly_pct = max(min_bet_pct, min(kelly_adjusted, max_bet_fraction))
    stake = bankroll * kelly_pct

    return stake, edge, ev


def load_data():
    """Charge les prédictions CV avec les vraies cotes"""
    print("Chargement des données...")

    df = pd.read_parquet(PREDS_CV_PATH)
    print(f"✓ {len(df)} combats chargés")

    # Convertir les dates
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Filtrer 2015-2025
    df = df[(df['event_date'] >= '2015-01-01') & (df['event_date'] <= '2025-12-31')].copy()
    df = df.sort_values('event_date').reset_index(drop=True)

    print(f"✓ Période: {df['event_date'].min().date()} à {df['event_date'].max().date()}")
    print(f"✓ {len(df)} combats sur la période 2015-2025")

    return df


def run_backtest(df, strategy_params, strategy_name):
    """Exécute le backtest pour une stratégie donnée"""

    bankroll = INITIAL_BANKROLL
    bankroll_history = [bankroll]
    dates_history = [df.iloc[0]['event_date']]

    bets_placed = []
    wins = 0
    losses = 0
    total_staked = 0
    total_profit = 0

    for idx, row in df.iterrows():
        # On peut parier sur A ou B selon les edges
        edge_A = row['edge_A']
        edge_B = row['edge_B']

        # Calculer les probas
        proba_A = row['proba_model']
        proba_B = 1 - proba_A

        # Cotes
        odds_A = row['A_odds_1']
        odds_B = row['A_odds_2']

        # Résultat réel (y=1 si A gagne, y=0 si B gagne)
        actual_winner = row['y']

        # Calculer les stakes
        stake_A, edge_A_calc, ev_A = calculate_kelly_stake(proba_A, odds_A, bankroll, strategy_params)
        stake_B, edge_B_calc, ev_B = calculate_kelly_stake(proba_B, odds_B, bankroll, strategy_params)

        # Parier sur celui qui a le meilleur edge
        bet_placed = False

        if stake_A > 0 and stake_A >= stake_B:
            # Parier sur A
            if actual_winner == 1:
                result = 'win'
                profit = stake_A * (odds_A - 1)
            else:
                result = 'loss'
                profit = -stake_A

            bankroll += profit
            total_staked += stake_A
            total_profit += profit

            if result == 'win':
                wins += 1
            else:
                losses += 1

            bets_placed.append({
                'date': row['event_date'],
                'fighter': row['fighter_1'],
                'odds': odds_A,
                'stake': stake_A,
                'proba_model': proba_A,
                'edge': edge_A,
                'ev': ev_A,
                'result': result,
                'profit': profit
            })
            bet_placed = True

        elif stake_B > 0:
            # Parier sur B
            if actual_winner == 0:
                result = 'win'
                profit = stake_B * (odds_B - 1)
            else:
                result = 'loss'
                profit = -stake_B

            bankroll += profit
            total_staked += stake_B
            total_profit += profit

            if result == 'win':
                wins += 1
            else:
                losses += 1

            bets_placed.append({
                'date': row['event_date'],
                'fighter': row['fighter_2'],
                'odds': odds_B,
                'stake': stake_B,
                'proba_model': proba_B,
                'edge': edge_B,
                'ev': ev_B,
                'result': result,
                'profit': profit
            })
            bet_placed = True

        # Enregistrer l'historique
        if bet_placed:
            bankroll_history.append(bankroll)
            dates_history.append(row['event_date'])

        # Fail-safe
        if bankroll <= 0:
            print(f"⚠️  Bankroll épuisée à {row['event_date'].date()}")
            break

    # Calculer les métriques
    df_bets = pd.DataFrame(bets_placed)

    if len(df_bets) == 0:
        return None

    final_bankroll = bankroll
    total_return = final_bankroll - INITIAL_BANKROLL
    roi = (total_return / INITIAL_BANKROLL) * 100

    # Drawdown
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

    # Stats annuelles
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
    """Affiche les résultats"""

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
    print(f"Mise moyenne: {results['avg_stake_pct']:.2f}% de la bankroll initiale")
    print(f"Edge moyen: {results['avg_edge']:.2f}%")

    print(f"\n📅 PERFORMANCE ANNUELLE")
    print(results['yearly_stats'].to_string())

    print("\n" + "="*80)


def plot_results(all_results):
    """Génère les graphiques"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Backtest UFC 2015-2025 - Analyse Comparative (COTES RÉELLES)',
                 fontsize=16, fontweight='bold')

    # 1. Évolution bankroll
    ax1 = axes[0, 0]
    for strategy_name, results in all_results.items():
        ax1.plot(results['dates_history'], results['bankroll_history'],
                label=results['strategy_name'], linewidth=2)
    ax1.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bankroll (€)')
    ax1.set_title('Évolution de la Bankroll')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. ROI
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

    # 4. Tableau comparatif
    ax4 = axes[1, 1]
    ax4.axis('off')

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

    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.tight_layout()

    output_path = os.path.join(BASE_DIR, 'backtest_results_REAL.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvegardés: {output_path}")

    return output_path


def main():
    """Fonction principale"""

    print("="*80)
    print("BACKTEST RÉALISTE UFC - Période 2015-2025")
    print("Utilise les vraies cotes historiques et prédictions du modèle")
    print("="*80)

    # Charger les données
    df = load_data()

    # Exécuter les backtests
    print("\n" + "="*80)
    print("EXÉCUTION DES BACKTESTS")
    print("="*80)

    all_results = {}

    for strategy_key, strategy_params in STRATEGIES.items():
        print(f"\n🔄 Backtest: {strategy_params['name']}...")

        results = run_backtest(df, strategy_params, strategy_params['name'])

        if results:
            all_results[strategy_key] = results
            print_results(results)
        else:
            print(f"⚠️  Aucun pari pour {strategy_params['name']}")

    # Graphiques
    if all_results:
        print("\n" + "="*80)
        print("GÉNÉRATION DES GRAPHIQUES")
        print("="*80)
        plot_results(all_results)

    # Résumé
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

    # Recommandations
    if all_results:
        best_roi = max(all_results.values(), key=lambda x: x['roi'])
        best_sharpe = min(all_results.values(),
                         key=lambda x: abs(x['max_drawdown']) / (x['roi'] + 0.001) if x['roi'] > 0 else 999)

        print("\n" + "="*80)
        print("RECOMMANDATIONS")
        print("="*80)
        print(f"\n🏆 Meilleur ROI: {best_roi['strategy_name']} ({best_roi['roi']:.2f}%)")
        print(f"🛡️  Meilleur ratio risque/rendement: {best_sharpe['strategy_name']}")
        print(f"\n💡 Note: Ces résultats sont basés sur les vraies cotes historiques")
        print(f"    et les prédictions de votre modèle en cross-validation.")

    print("\n✅ Backtest terminé!")
    print("="*80)


if __name__ == "__main__":
    main()
