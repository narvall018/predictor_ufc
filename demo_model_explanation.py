#!/usr/bin/env python3
"""
Démonstration Interactive : Comment le Modèle Utilise les Cotes
Montre étape par étape l'analyse d'un combat réel
"""

import os
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    import joblib
except ImportError:
    print("Installation des dépendances...")
    os.system("pip install -q pandas numpy joblib pyarrow scikit-learn")
    import pandas as pd
    import numpy as np
    import joblib

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'model_pipeline.pkl')
PREDS_CV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'preds_cv.parquet')


def print_separator(char='=', length=80):
    """Affiche une ligne de séparation"""
    print(char * length)


def print_section(title):
    """Affiche un titre de section"""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_step(step_number, title):
    """Affiche un numéro d'étape"""
    print(f"\n{'='*5} ÉTAPE {step_number}: {title} {'='*5}\n")


def analyze_fight(fight_data, model):
    """Analyse complète d'un combat avec explications détaillées"""

    print("\n" * 2)
    print_section("🥊 ANALYSE DÉTAILLÉE D'UN COMBAT UFC")

    # Extraire les données
    fighter_1 = fight_data['fighter_1']
    fighter_2 = fight_data['fighter_2']
    odds_1 = fight_data['A_odds_1']
    odds_2 = fight_data['A_odds_2']
    reach_diff = fight_data['reach_diff']
    age_diff = fight_data['age_diff']
    actual_winner = fight_data['y']
    event_date = pd.to_datetime(fight_data['event_date']).strftime('%d/%m/%Y')

    print(f"\n📅 Date: {event_date}")
    print(f"🔴 Fighter A: {fighter_1}")
    print(f"🔵 Fighter B: {fighter_2}")

    # =========================================================================
    # ÉTAPE 1: LES COTES DU BOOKMAKER
    # =========================================================================
    print_step(1, "LES COTES DU BOOKMAKER")

    print(f"🎲 Cotes décimales:")
    print(f"   - {fighter_1}: {odds_1:.2f}")
    print(f"   - {fighter_2}: {odds_2:.2f}")

    print(f"\n💡 Explication:")
    print(f"   Si je mise 10€ sur {fighter_1} et qu'il gagne,")
    print(f"   je récupère {odds_1:.2f} × 10€ = {odds_1 * 10:.2f}€ (profit de {(odds_1-1)*10:.2f}€)")

    # =========================================================================
    # ÉTAPE 2: CONVERSION EN PROBABILITÉS
    # =========================================================================
    print_step(2, "CONVERSION EN PROBABILITÉS")

    p_impl_1 = 1 / odds_1
    p_impl_2 = 1 / odds_2

    print(f"📊 Probabilités implicites (brutes):")
    print(f"   - {fighter_1}: 1 / {odds_1:.2f} = {p_impl_1:.1%}")
    print(f"   - {fighter_2}: 1 / {odds_2:.2f} = {p_impl_2:.1%}")
    print(f"   - TOTAL: {p_impl_1 + p_impl_2:.1%}")

    overround = (p_impl_1 + p_impl_2 - 1) * 100
    print(f"\n⚠️  Le total dépasse 100% de {overround:.1f}%")
    print(f"    C'est la MARGE du bookmaker (vigorish)")
    print(f"    Le bookmaker gagne cette marge quel que soit le résultat !")

    # =========================================================================
    # ÉTAPE 3: RETIRER LA MARGE (DEVIG)
    # =========================================================================
    print_step(3, "RETIRER LA MARGE DU BOOKMAKER")

    vig = p_impl_1 + p_impl_2
    proba_market_1 = p_impl_1 / vig
    proba_market_2 = p_impl_2 / vig

    print(f"🧮 Probabilités \"vraies\" (après devig):")
    print(f"   - {fighter_1}: {p_impl_1:.1%} / {vig:.3f} = {proba_market_1:.1%}")
    print(f"   - {fighter_2}: {p_impl_2:.1%} / {vig:.3f} = {proba_market_2:.1%}")
    print(f"   - TOTAL: {proba_market_1 + proba_market_2:.1%} ✓")

    print(f"\n💭 Interprétation:")
    if proba_market_1 > 0.6:
        print(f"   Le marché considère {fighter_1} comme GRAND FAVORI")
    elif proba_market_1 > 0.52:
        print(f"   Le marché considère {fighter_1} comme FAVORI")
    elif proba_market_1 > 0.48:
        print(f"   Le marché considère ce combat comme ÉQUILIBRÉ")
    elif proba_market_1 > 0.4:
        print(f"   Le marché considère {fighter_1} comme UNDERDOG")
    else:
        print(f"   Le marché considère {fighter_1} comme GRAND UNDERDOG")

    # =========================================================================
    # ÉTAPE 4: TRANSFORMATION EN LOGIT
    # =========================================================================
    print_step(4, "TRANSFORMATION EN LOGIT (FEATURE 1)")

    proba_market_clipped = np.clip(proba_market_1, 0.01, 0.99)
    market_logit = np.log(proba_market_clipped / (1 - proba_market_clipped))

    print(f"🔢 Formule du logit:")
    print(f"   market_logit = log({proba_market_1:.1%} / {proba_market_2:.1%})")
    print(f"   market_logit = log({proba_market_1:.3f} / {proba_market_2:.3f})")
    print(f"   market_logit = {market_logit:.3f}")

    print(f"\n💡 Interprétation du logit:")
    print(f"   - Si > 0  → {fighter_1} est FAVORI")
    print(f"   - Si < 0  → {fighter_1} est UNDERDOG")
    print(f"   - Si ≈ 0  → Combat ÉQUILIBRÉ")
    print(f"   → Ici: {market_logit:.3f} → ", end="")

    if market_logit > 0.5:
        print(f"{fighter_1} est FORT FAVORI")
    elif market_logit > 0:
        print(f"{fighter_1} est FAVORI")
    elif market_logit > -0.5:
        print(f"{fighter_1} est UNDERDOG")
    else:
        print(f"{fighter_1} est FORT UNDERDOG")

    # =========================================================================
    # ÉTAPE 5: CARACTÉRISTIQUES PHYSIQUES
    # =========================================================================
    print_step(5, "CARACTÉRISTIQUES PHYSIQUES")

    print(f"📏 REACH DIFF (FEATURE 2):")
    print(f"   Différence d'allonge = {reach_diff:+.1f} cm")

    if reach_diff > 5:
        print(f"   → {fighter_1} a un AVANTAGE d'allonge significatif")
        print(f"      (peut frapper de plus loin, contrôler la distance)")
    elif reach_diff > 0:
        print(f"   → {fighter_1} a un léger avantage d'allonge")
    elif reach_diff > -5:
        print(f"   → {fighter_2} a un léger avantage d'allonge")
    else:
        print(f"   → {fighter_2} a un AVANTAGE d'allonge significatif")

    print(f"\n👴 AGE DIFF (FEATURE 3):")
    print(f"   Différence d'âge = {age_diff:+.1f} ans")

    if age_diff < -3:
        print(f"   → {fighter_1} est NETTEMENT plus jeune")
        print(f"      (meilleure récupération, reflexes plus vifs)")
    elif age_diff < 0:
        print(f"   → {fighter_1} est plus jeune")
    elif age_diff < 3:
        print(f"   → {fighter_2} est plus jeune")
    else:
        print(f"   → {fighter_2} est NETTEMENT plus jeune")

    # =========================================================================
    # ÉTAPE 6: PRÉDICTION DU MODÈLE
    # =========================================================================
    print_step(6, "PRÉDICTION DU MODÈLE ML")

    X = np.array([[market_logit, reach_diff, age_diff]])

    print(f"🔮 Vecteur de features passé au modèle:")
    print(f"   X = [market_logit, reach_diff, age_diff]")
    print(f"   X = [{market_logit:.3f}, {reach_diff:.1f}, {age_diff:.1f}]")

    proba_model_1 = model.predict_proba(X)[0][1]
    proba_model_2 = 1 - proba_model_1

    print(f"\n🤖 Le modèle prédit:")
    print(f"   - Probabilité {fighter_1}: {proba_model_1:.1%}")
    print(f"   - Probabilité {fighter_2}: {proba_model_2:.1%}")

    # =========================================================================
    # ÉTAPE 7: COMPARAISON MARCHÉ VS MODÈLE
    # =========================================================================
    print_step(7, "COMPARAISON: MARCHÉ vs MODÈLE")

    edge_1 = proba_model_1 - proba_market_1
    edge_2 = proba_model_2 - proba_market_2

    print(f"\n📊 Tableau comparatif:\n")
    print(f"{'':20} {'Marché':>12} {'Modèle':>12} {'Edge':>12}")
    print(f"{'-'*60}")
    print(f"{fighter_1:20} {proba_market_1:>11.1%} {proba_model_1:>11.1%} {edge_1:>+11.1%}")
    print(f"{fighter_2:20} {proba_market_2:>11.1%} {proba_model_2:>11.1%} {edge_2:>+11.1%}")

    print(f"\n💭 Analyse de l'écart:")

    if abs(edge_1) < 0.02:
        print(f"   Le modèle est GLOBALEMENT D'ACCORD avec le marché")
        print(f"   Écart faible ({edge_1:+.1%}) → Pas d'opportunité")
    else:
        if edge_1 > 0:
            print(f"   ✅ Le modèle pense que {fighter_1} a {edge_1:.1%} de chances")
            print(f"      EN PLUS que ce que le marché estime !")
            print(f"\n   🎯 RAISON: Les caractéristiques physiques suggèrent que")
            if reach_diff > 0 and age_diff < 0:
                print(f"      l'avantage d'allonge (+{reach_diff:.0f}cm) et de jeunesse")
                print(f"      ({abs(age_diff):.0f} ans de moins) ne sont pas assez valorisés")
            elif reach_diff > 0:
                print(f"      l'avantage d'allonge (+{reach_diff:.0f}cm) n'est pas assez valorisé")
            elif age_diff < 0:
                print(f"      l'avantage d'âge ({abs(age_diff):.0f} ans de moins) n'est pas assez valorisé")
            else:
                print(f"      le marché sous-estime {fighter_1}")
        else:
            print(f"   ✅ Le modèle pense que {fighter_2} a {abs(edge_1):.1%} de chances")
            print(f"      EN PLUS que ce que le marché estime !")

    # =========================================================================
    # ÉTAPE 8: CALCUL DE L'EV (EXPECTED VALUE)
    # =========================================================================
    print_step(8, "CALCUL DE LA VALEUR ATTENDUE (EV)")

    ev_1 = (proba_model_1 * odds_1) - 1
    ev_2 = (proba_model_2 * odds_2) - 1

    print(f"💰 Formule de l'EV:")
    print(f"   EV = (Probabilité × Cote) - 1")

    print(f"\n   Pour {fighter_1}:")
    print(f"   EV = ({proba_model_1:.1%} × {odds_1:.2f}) - 1")
    print(f"   EV = {proba_model_1 * odds_1:.3f} - 1")
    print(f"   EV = {ev_1:+.1%}")

    print(f"\n   Pour {fighter_2}:")
    print(f"   EV = ({proba_model_2:.1%} × {odds_2:.2f}) - 1")
    print(f"   EV = {proba_model_2 * odds_2:.3f} - 1")
    print(f"   EV = {ev_2:+.1%}")

    print(f"\n💡 Interprétation de l'EV:")
    print(f"   Si je mise 100€ sur {fighter_1} à long terme:")
    print(f"   - Gain attendu moyen = {ev_1 * 100:+.2f}€ par pari")

    if ev_1 > 0.1:
        print(f"   → EXCELLENT pari (EV > 10%)")
    elif ev_1 > 0.05:
        print(f"   → BON pari (EV > 5%)")
    elif ev_1 > 0:
        print(f"   → Pari légèrement positif")
    else:
        print(f"   → Pari non rentable à long terme")

    # =========================================================================
    # ÉTAPE 9: DÉCISION DE PARI
    # =========================================================================
    print_step(9, "DÉCISION DE PARI")

    MIN_EDGE = 0.035  # 3.5% pour stratégie SAFE

    print(f"📋 Critères de la stratégie SAFE:")
    print(f"   1. Edge minimum: {MIN_EDGE:.1%}")
    print(f"   2. EV positif")
    print(f"   3. Cotes entre 1.0 et 5.0")

    # Déterminer le meilleur pari
    best_bet = None
    if edge_1 >= MIN_EDGE and ev_1 > 0 and 1.0 <= odds_1 <= 5.0:
        best_bet = 1
    elif edge_2 >= MIN_EDGE and ev_2 > 0 and 1.0 <= odds_2 <= 5.0:
        best_bet = 2

    print(f"\n🎯 DÉCISION:")

    if best_bet == 1:
        print(f"   ✅ PARIER sur {fighter_1}")
        print(f"      - Edge: {edge_1:+.1%} (> {MIN_EDGE:.1%} ✓)")
        print(f"      - EV: {ev_1:+.1%} (> 0% ✓)")
        print(f"      - Cote: {odds_1:.2f} (dans [1.0, 5.0] ✓)")

        # Calcul Kelly
        kelly_fraction = 2.75
        q = 1 - proba_model_1
        b = odds_1 - 1
        kelly_full = (proba_model_1 * b - q) / b
        kelly_adjusted = kelly_full / kelly_fraction
        kelly_pct = min(kelly_adjusted, 0.25) * 100  # Max 25%

        print(f"\n   💵 Mise recommandée (Kelly fractionné):")
        print(f"      Kelly complet = {kelly_full:.1%}")
        print(f"      Kelly fractionné (/{kelly_fraction}) = {kelly_adjusted:.1%}")
        print(f"      Mise finale = {kelly_pct:.1%} de la bankroll")
        print(f"\n      Si bankroll = 1,000€ → Miser {kelly_pct * 10:.2f}€")

    elif best_bet == 2:
        print(f"   ✅ PARIER sur {fighter_2}")
        print(f"      - Edge: {edge_2:+.1%} (> {MIN_EDGE:.1%} ✓)")
        print(f"      - EV: {ev_2:+.1%} (> 0% ✓)")
        print(f"      - Cote: {odds_2:.2f} (dans [1.0, 5.0] ✓)")
    else:
        print(f"   ❌ NE PAS PARIER")

        if max(edge_1, edge_2) < MIN_EDGE:
            print(f"      Raison: Edge trop faible")
            print(f"      (Edge max = {max(edge_1, edge_2):.1%} < {MIN_EDGE:.1%})")
        elif max(ev_1, ev_2) <= 0:
            print(f"      Raison: EV négatif")
        else:
            print(f"      Raison: Critères non remplis")

    # =========================================================================
    # ÉTAPE 10: RÉSULTAT RÉEL
    # =========================================================================
    print_step(10, "RÉSULTAT RÉEL DU COMBAT")

    winner_name = fighter_1 if actual_winner == 1 else fighter_2
    print(f"🏆 Vainqueur réel: {winner_name}")

    if best_bet:
        if (best_bet == 1 and actual_winner == 1) or (best_bet == 2 and actual_winner == 0):
            odds_used = odds_1 if best_bet == 1 else odds_2
            kelly_pct_used = kelly_pct if best_bet == 1 else 15  # Approximation
            profit = kelly_pct_used * 10 * (odds_used - 1)

            print(f"✅ PARI GAGNANT !")
            print(f"   Mise: {kelly_pct_used * 10:.2f}€")
            print(f"   Retour: {kelly_pct_used * 10 * odds_used:.2f}€")
            print(f"   Profit: +{profit:.2f}€")
        else:
            kelly_pct_used = kelly_pct if best_bet == 1 else 15
            print(f"❌ PARI PERDANT")
            print(f"   Mise perdue: -{kelly_pct_used * 10:.2f}€")
    else:
        print(f"ℹ️  Pas de pari placé (critères non remplis)")

    # =========================================================================
    # RÉSUMÉ
    # =========================================================================
    print_section("📝 RÉSUMÉ DE L'ANALYSE")

    print(f"\n1. Le marché estimait: {fighter_1} {proba_market_1:.1%} vs {fighter_2} {proba_market_2:.1%}")
    print(f"2. Le modèle prédit: {fighter_1} {proba_model_1:.1%} vs {fighter_2} {proba_model_2:.1%}")
    print(f"3. Edge trouvé: {max(abs(edge_1), abs(edge_2)):.1%}")

    if best_bet:
        print(f"4. Décision: PARIER sur {fighter_1 if best_bet == 1 else fighter_2}")
    else:
        print(f"4. Décision: NE PAS PARIER")

    print(f"5. Résultat: {winner_name} a gagné")

    if best_bet:
        if (best_bet == 1 and actual_winner == 1) or (best_bet == 2 and actual_winner == 0):
            print(f"6. Performance: ✅ GAGNANT (+{profit:.2f}€)")
        else:
            print(f"6. Performance: ❌ PERDANT (-{kelly_pct_used * 10:.2f}€)")

    print_separator()


def main():
    """Fonction principale"""

    print("\n" * 2)
    print_section("🎓 DÉMONSTRATION: COMMENT LE MODÈLE UTILISE LES COTES")

    print("\nCe script va analyser UN combat réel étape par étape")
    print("pour montrer EXACTEMENT comment votre modèle fonctionne.\n")

    # Charger le modèle
    print("Chargement du modèle...")
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model'] if isinstance(model_data, dict) else model_data
    print("✓ Modèle chargé\n")

    # Charger les données
    print("Chargement des combats...")
    df = pd.read_parquet(PREDS_CV_PATH)
    df = df[(df['event_date'] >= '2023-01-01') & (df['event_date'] <= '2025-12-31')]
    print(f"✓ {len(df)} combats 2023-2025 chargés\n")

    # Filtrer pour avoir des exemples intéressants
    df['edge_max'] = df[['edge_A', 'edge_B']].abs().max(axis=1)
    df_interesting = df[df['edge_max'] > 0.04].head(10)

    if len(df_interesting) == 0:
        print("Aucun combat intéressant trouvé")
        return

    # Analyser le premier combat intéressant
    fight = df_interesting.iloc[0]
    analyze_fight(fight, model)

    # Proposer d'en voir d'autres
    print("\n" * 2)
    print("="*80)
    print("Voulez-vous voir d'autres exemples ?")
    print("="*80)
    print(f"\nIl y a {len(df_interesting)} combats intéressants disponibles.")
    print("\nPour voir un autre exemple, relancez le script !")
    print("Vous pouvez aussi modifier la ligne 'df_interesting.iloc[0]' en")
    print("'df_interesting.iloc[1]', 'df_interesting.iloc[2]', etc.")

    # Afficher la liste des combats disponibles
    print("\n" + "="*80)
    print("COMBATS DISPONIBLES POUR ANALYSE:")
    print("="*80)

    for idx, row in df_interesting.iterrows():
        edge_max = row['edge_max']
        date = pd.to_datetime(row['event_date']).strftime('%d/%m/%Y')
        print(f"\n{idx}. {date}: {row['fighter_1']} vs {row['fighter_2']}")
        print(f"   Edge: {edge_max:.1%} | Cotes: {row['A_odds_1']:.2f} / {row['A_odds_2']:.2f}")


if __name__ == "__main__":
    main()
