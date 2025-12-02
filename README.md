# 🥊 UFC Betting Predictor

Application de prédiction de paris UFC basée sur un modèle ML sans data leakage.

## 📊 Performance

- **Accuracy**: ~56%
- **ROI TRAIN**: +20.8%
- **ROI TEST**: +50% (25 paris)
- **Combattants**: 2075+

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/votre-username/predictor_ufc.git
cd predictor_ufc

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

## 🎯 Fonctionnalités

- **Événements à venir**: Récupère les prochains combats UFC
- **Recommandations de paris**: Calcul automatique avec critère de Kelly
- **Gestion Bankroll**: Suivi des paris et performances
- **Classement Elo**: Ranking des combattants
- **Mise à jour des données**: Scraping automatique

## 📈 Stratégie REALISTIC (Recommandée)

| Paramètre | Valeur |
|-----------|--------|
| Confiance min | 60% |
| Edge min | 10% |
| EV max | 50% |
| Cotes | 1.20 - 3.0 |
| Kelly | 1/10 |

## ⚠️ Avertissement

Les paris sportifs comportent des risques. Cette application fournit des recommandations basées sur des modèles statistiques mais ne garantit pas les résultats. Pariez de manière responsable.

## 📁 Structure

```
predictor_ufc/
├── app.py                    # Application Streamlit
├── requirements.txt          # Dépendances
├── data/
│   ├── raw/
│   │   └── appearances.parquet
│   ├── interim/
│   │   └── ratings_timeseries.parquet
│   └── processed/
│       └── model_pipeline.pkl
└── bets/                     # Données personnelles (gitignore)
```

## 🔧 Technologies

- Python 3.10+
- Streamlit
- Scikit-learn (LogisticRegression)
- Pandas / NumPy
- Plotly
- BeautifulSoup4
