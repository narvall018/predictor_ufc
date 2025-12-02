# 🥊 UFC Betting Predictor

Application de prédiction de paris UFC basée sur un modèle ML sans data leakage.

## 📊 Performance

- **Accuracy**: ~56%
- **ROI TRAIN**: +20.8%
- **ROI TEST**: +50% (25 paris)
- **Combattants**: 2075+

## 🚀 Déploiement sur Streamlit Cloud

### 1. Fork/Clone ce repo sur GitHub

### 2. Configurer Streamlit Cloud
1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Connecter votre repo GitHub
3. Dans **Settings > Secrets**, ajouter :

```toml
GITHUB_TOKEN = "github_pat_VOTRE_TOKEN"
GITHUB_REPO = "votre-username/predictor_ufc"
```

### 3. Créer un GitHub Personal Access Token
1. GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens
2. Créer un token avec les permissions:
   - **Contents**: Read and write (pour sauvegarder les paris)
3. Copier le token dans les secrets Streamlit

## 💻 Installation locale

```bash
# Cloner le repo
git clone https://github.com/votre-username/predictor_ufc.git
cd predictor_ufc

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit run app.py
```

## �� Fonctionnalités

- **Événements à venir**: Récupère les prochains combats UFC
- **Recommandations de paris**: Calcul automatique avec critère de Kelly
- **Gestion Bankroll**: Suivi des paris (synchronisé avec GitHub)
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
└── bets/                     # Paris (sync GitHub)
```

## ⚠️ Avertissement

Les paris sportifs comportent des risques. Pariez de manière responsable.
