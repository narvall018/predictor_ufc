import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re
import json
import joblib
import unicodedata
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import subprocess
import time
import base64
import io
import urllib.request
import urllib.error
import hashlib

# ============================================================================
# 🔐 SYSTÈME DE PROFILS / SESSIONS
# ============================================================================
# Chaque profil a son propre mot de passe hashé, sa bankroll et son historique
# Le profil "visiteur" a un accès limité (pas de bankroll, pas de paris)

USER_PROFILES = {
    "narvall018": {
        "password_hash": "30085bd9342911e82fa94982d4cc7320921c8fdb5732ad7e8f335e7bf61919fc",  # Jumanji_75
        "display_name": "🏆 narvall018",
        "is_admin": True,
        "bets_folder": "bets",  # Dossier des paris pour ce profil
        "can_bet": True,
        "can_view_bankroll": True,
    },
    # 🔮 Futurs profils à ajouter ici:
    # "user2": {
    #     "password_hash": "hash_sha256_du_mot_de_passe",
    #     "display_name": "👤 User 2",
    #     "is_admin": False,
    #     "bets_folder": "bets_user2",
    #     "can_bet": True,
    #     "can_view_bankroll": True,
    # },
}

def _hash_password(password):
    """Hash un mot de passe en SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(password):
    """
    Authentifie un utilisateur par son mot de passe.
    Retourne le nom du profil si authentifié, None sinon.
    """
    password_hash = _hash_password(password)
    
    for username, profile in USER_PROFILES.items():
        if profile["password_hash"] == password_hash:
            return username
    
    return None

def get_current_user():
    """Retourne le profil de l'utilisateur connecté ou None"""
    if 'logged_in_user' in st.session_state and st.session_state.logged_in_user:
        username = st.session_state.logged_in_user
        if username in USER_PROFILES:
            return {
                "username": username,
                **USER_PROFILES[username]
            }
    return None

def is_logged_in():
    """Vérifie si un utilisateur est connecté"""
    return get_current_user() is not None

def can_access_betting():
    """Vérifie si l'utilisateur peut accéder aux fonctions de paris"""
    user = get_current_user()
    return user is not None and user.get("can_bet", False)

def can_view_bankroll():
    """Vérifie si l'utilisateur peut voir la bankroll"""
    user = get_current_user()
    return user is not None and user.get("can_view_bankroll", False)

def get_user_bets_folder():
    """Retourne le dossier des paris pour l'utilisateur connecté"""
    user = get_current_user()
    if user:
        return Path(user.get("bets_folder", "bets"))
    return Path("bets")  # Par défaut

def logout_user():
    """Déconnecte l'utilisateur"""
    if 'logged_in_user' in st.session_state:
        del st.session_state.logged_in_user
    if 'unlocked_api_key' in st.session_state:
        del st.session_state.unlocked_api_key

# ============================================================================
# CONFIGURATION GITHUB (pour Streamlit Cloud)
# ============================================================================

def get_github_config():
    """Récupère la config GitHub depuis les secrets Streamlit"""
    try:
        return {
            "token": st.secrets.get("GITHUB_TOKEN", ""),
            "repo": st.secrets.get("GITHUB_REPO", ""),
            "enabled": bool(st.secrets.get("GITHUB_TOKEN", ""))
        }
    except:
        return {"token": "", "repo": "", "enabled": False}

def github_api_request(method, endpoint, data=None, github_config=None):
    """Effectue une requête à l'API GitHub"""
    if not github_config or not github_config.get("enabled"):
        return None
    
    import urllib.request
    import urllib.error
    
    url = f"https://api.github.com/repos/{github_config['repo']}/{endpoint}"
    headers = {
        "Authorization": f"token {github_config['token']}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "UFC-Predictor-App"
    }
    
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            headers["Content-Type"] = "application/json"
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        st.warning(f"GitHub API error: {e.code}")
        return None
    except Exception as e:
        st.warning(f"GitHub connection error: {e}")
        return None

def load_file_from_github(file_path, github_config):
    """Charge un fichier depuis GitHub"""
    if not github_config.get("enabled"):
        return None, None
    
    result = github_api_request("GET", f"contents/{file_path}", github_config=github_config)
    if result and "content" in result:
        content = base64.b64decode(result["content"])
        sha = result.get("sha")
        return content, sha
    return None, None

def save_file_to_github(file_path, content, message, github_config, sha=None):
    """Sauvegarde un fichier sur GitHub"""
    if not github_config.get("enabled"):
        return False
    
    if isinstance(content, (pd.DataFrame,)):
        buffer = io.BytesIO()
        content.to_parquet(buffer, index=False)
        content = buffer.getvalue()
    elif isinstance(content, str):
        content = content.encode('utf-8')
    
    data = {
        "message": message,
        "content": base64.b64encode(content).decode('utf-8'),
        "branch": "main"
    }
    
    if sha:
        data["sha"] = sha
    
    result = github_api_request("PUT", f"contents/{file_path}", data=data, github_config=github_config)
    return result is not None

def load_parquet_from_github(file_path, github_config):
    """Charge un fichier parquet depuis GitHub"""
    content, sha = load_file_from_github(file_path, github_config)
    if content:
        try:
            return pd.read_parquet(io.BytesIO(content)), sha
        except:
            pass
    return None, None

def load_csv_from_github(file_path, github_config):
    """Charge un fichier CSV depuis GitHub"""
    content, sha = load_file_from_github(file_path, github_config)
    if content:
        try:
            return pd.read_csv(io.BytesIO(content)), sha
        except:
            pass
    return None, None

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="UFC Betting Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Config GitHub
GITHUB_CONFIG = get_github_config()

# Chemins
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"
BETS_DIR = Path("bets")

for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROC_DIR, BETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Paramètres Elo
K_FACTOR = 24
BASE_ELO = 1500.0

# ============================================================================
# ✅ STRATÉGIES DE PARIS OPTIMISÉES - Grid Search + AG Multi-Îles Parallélisé
# Backtest 2014-2025 sur 5,099 combats UFC | Bankroll initiale: 1000€
# Optimisation: Kelly fraction, Edge threshold, plage de cotes, max stake
# Grid Search: 20,790 combinaisons | AG: 6 îles × 500 individus × 300 générations
# Validation Out-of-Sample 2023-2025: Toutes stratégies ✅ cohérentes
# ============================================================================
BETTING_STRATEGIES = {
    "🛡️ SAFE (RECOMMANDÉE)": {
        "kelly_fraction": 2.75,
        "min_confidence": 0.0,
        "min_edge": 0.035,  # Edge minimum 3.5%
        "max_value": 1.0,
        "min_odds": 1.0,
        "max_odds": 5.0,
        "max_bet_fraction": 0.25,
        "min_bet_pct": 0.01,
        "description": "🛡️ SAFE - Profit 119k€ | ROI 17% | DD 34% | 11/12 ans | ~140 paris/an | Pour débutants"
    },
    "🟢 ÉQUILIBRÉE (DD<35%)": {
        "kelly_fraction": 2.5,
        "min_confidence": 0.0,
        "min_edge": 0.042,  # Edge minimum 4.2%
        "max_value": 1.0,
        "min_odds": 1.0,
        "max_odds": 5.0,
        "max_bet_fraction": 0.30,
        "min_bet_pct": 0.01,
        "description": "🟢 ÉQUILIBRÉE - Profit 202k€ | ROI 19% | DD 35% | 11/12 ans | ~122 paris/an | Recommandée"
    },
    "🔥 AGRESSIVE (DD<40%)": {
        "kelly_fraction": 2.0,
        "min_confidence": 0.0,
        "min_edge": 0.042,  # Edge minimum 4.2%
        "max_value": 1.0,
        "min_odds": 1.0,
        "max_odds": 5.0,
        "max_bet_fraction": 0.36,
        "min_bet_pct": 0.01,
        "description": "🔥 AGRESSIVE - Profit 418k€ | ROI 20% | DD 40% | 11/12 ans | ~122 paris/an | Traders expérimentés"
    },
    "📈 VOLUME+ (Plus de paris)": {
        "kelly_fraction": 3.0,
        "min_confidence": 0.0,
        "min_edge": 0.03,  # Edge minimum 3%
        "max_value": 1.0,
        "min_odds": 1.0,
        "max_odds": 5.0,
        "max_bet_fraction": 0.20,
        "min_bet_pct": 0.01,
        "description": "📈 VOLUME+ - Profit 82k€ | ROI 15% | DD 34% | 10/12 ans | ~157 paris/an | Plus d'opportunités"
    },
    "💎 SÉLECTIF (Meilleur Sharpe)": {
        "kelly_fraction": 2.2,
        "min_confidence": 0.0,
        "min_edge": 0.063,  # Edge minimum 6.3%
        "max_value": 1.0,
        "min_odds": 1.0,
        "max_odds": 5.0,
        "max_bet_fraction": 0.37,
        "min_bet_pct": 0.01,
        "description": "💎 SÉLECTIF - Profit 367k€ | ROI 32% | DD 40% | Sharpe 1.44 | 12/12 ans | ~77 paris/an | Meilleur ratio"
    },
}

# ============================================================================
# THE ODDS API - RÉCUPÉRATION AUTOMATIQUE DES COTES
# ============================================================================
# API gratuite: 500 requêtes/mois - https://the-odds-api.com
# Sport key: mma_mixed_martial_arts

# 🔐 Clé API encodée (disponible pour les utilisateurs connectés)
_ENCODED_API_KEY = "MTI4NTcwMTFmZjI3MDcwYWYxZTI4NTc2MTZkYWM1YjQ="  # Base64

def _decode_api_key():
    """Décode la clé API si l'utilisateur est connecté"""
    # Seuls les utilisateurs connectés peuvent utiliser la clé intégrée
    if not is_logged_in():
        return None
    
    try:
        return base64.b64decode(_ENCODED_API_KEY).decode('utf-8')
    except:
        return None

def get_odds_api_key():
    """Récupère la clé API depuis les secrets Streamlit, session ou variable d'env"""
    key = None
    
    # 1. Clé intégrée si utilisateur connecté
    if is_logged_in():
        key = _decode_api_key()
    
    # 2. Clé temporaire en session (saisie manuelle)
    if not key and 'temp_odds_api_key' in st.session_state and st.session_state.temp_odds_api_key:
        key = st.session_state.temp_odds_api_key
    
    # 3. Secrets Streamlit
    if not key:
        try:
            key = st.secrets.get("ODDS_API_KEY", "")
        except:
            pass
    
    # 4. Variable d'environnement
    if not key:
        key = os.environ.get("ODDS_API_KEY", "")
    
    # ✅ Nettoyer la clé (retirer espaces, retours à la ligne)
    if key:
        key = key.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    
    return key

def fetch_mma_odds(api_key=None, bookmaker="pinnacle"):
    """
    Récupère les cotes MMA depuis The Odds API
    
    Args:
        api_key: Clé API (optionnel, utilise secrets sinon)
        bookmaker: Bookmaker préféré (pinnacle par défaut - meilleures cotes)
    
    Returns:
        dict: {event_id: {fighter1: odds1, fighter2: odds2, ...}}
    """
    if not api_key:
        api_key = get_odds_api_key()
    
    if not api_key:
        return None, "❌ Clé API manquante. Ajoutez ODDS_API_KEY dans les secrets Streamlit."
    
    url = f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",  # Europe pour avoir Pinnacle
        "markets": "h2h",  # Head to head (moneyline)
        "oddsFormat": "decimal"
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{url}?{query_string}"
    
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "UFC-Predictor"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            # Extraire les headers pour le quota
            remaining = response.headers.get('x-requests-remaining', '?')
            used = response.headers.get('x-requests-used', '?')
            
            # Parser les données
            odds_data = {}
            for event in data:
                event_key = f"{event.get('home_team', '')} vs {event.get('away_team', '')}"
                event_time = event.get('commence_time', '')
                
                # Trouver le bookmaker préféré ou le premier disponible
                bookmakers = event.get('bookmakers', [])
                selected_book = None
                
                # Priorité: pinnacle > betfair > unibet > premier dispo
                priority_books = ['pinnacle', 'betfair', 'unibet', '1xbet']
                for prio in priority_books:
                    for book in bookmakers:
                        if book.get('key', '').lower() == prio:
                            selected_book = book
                            break
                    if selected_book:
                        break
                
                if not selected_book and bookmakers:
                    selected_book = bookmakers[0]
                
                if selected_book:
                    markets = selected_book.get('markets', [])
                    for market in markets:
                        if market.get('key') == 'h2h':
                            outcomes = market.get('outcomes', [])
                            fight_odds = {}
                            for outcome in outcomes:
                                fighter_name = outcome.get('name', '')
                                price = outcome.get('price', 0)
                                fight_odds[fighter_name] = price
                            
                            odds_data[event_key] = {
                                'odds': fight_odds,
                                'bookmaker': selected_book.get('title', 'Unknown'),
                                'last_update': selected_book.get('last_update', ''),
                                'commence_time': event_time,
                                'home_team': event.get('home_team', ''),
                                'away_team': event.get('away_team', '')
                            }
            
            return odds_data, f"✅ {len(odds_data)} combats récupérés (Quota: {used}/{int(used)+int(remaining) if remaining != '?' else '?'})"
            
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "❌ Clé API invalide"
        elif e.code == 429:
            return None, "❌ Quota API dépassé (500 req/mois gratuit)"
        else:
            return None, f"❌ Erreur API: {e.code}"
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

def normalize_fighter_name_for_matching(name):
    """Normalise un nom pour le matching entre UFC et The Odds API"""
    if not name:
        return ""
    # Retirer accents
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    # Minuscules, retirer ponctuation
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def match_fighter_to_odds(fighter_name, odds_dict):
    """
    Trouve les cotes correspondant à un combattant
    
    Args:
        fighter_name: Nom du combattant (ex: "Jon Jones")
        odds_dict: Dict des cotes de l'API
        
    Returns:
        (odds_value, matched_name, bookmaker) ou (None, None, None)
    """
    if not fighter_name or not odds_dict:
        return None, None, None
    
    norm_name = normalize_fighter_name_for_matching(fighter_name)
    name_parts = norm_name.split()
    
    # Chercher dans toutes les données de cotes
    for event_key, event_data in odds_dict.items():
        odds = event_data.get('odds', {})
        bookmaker = event_data.get('bookmaker', '')
        
        for api_fighter, api_odds in odds.items():
            norm_api = normalize_fighter_name_for_matching(api_fighter)
            api_parts = norm_api.split()
            
            # Match exact
            if norm_name == norm_api:
                return api_odds, api_fighter, bookmaker
            
            # Match par nom de famille (dernier mot)
            if name_parts and api_parts:
                if name_parts[-1] == api_parts[-1]:
                    # Vérifier au moins une partie du prénom
                    if len(name_parts) > 1 and len(api_parts) > 1:
                        if name_parts[0][0] == api_parts[0][0]:  # Même initiale
                            return api_odds, api_fighter, bookmaker
                    elif len(name_parts) == 1 or len(api_parts) == 1:
                        return api_odds, api_fighter, bookmaker
    
    return None, None, None

def find_fight_odds(fighter_a, fighter_b, odds_dict):
    """
    Trouve les cotes pour un combat spécifique
    
    Returns:
        (odds_a, odds_b, bookmaker, matched_a, matched_b) ou None si non trouvé
    """
    if not odds_dict:
        return None
    
    odds_a, matched_a, book_a = match_fighter_to_odds(fighter_a, odds_dict)
    odds_b, matched_b, book_b = match_fighter_to_odds(fighter_b, odds_dict)
    
    if odds_a and odds_b:
        return {
            'odds_a': odds_a,
            'odds_b': odds_b,
            'bookmaker': book_a or book_b,
            'matched_a': matched_a,
            'matched_b': matched_b
        }
    
    return None

# STYLES CSS
# ============================================================================

st.markdown("""
<style>
    :root {
        --primary-red: #E53935;
        --primary-blue: #1E88E5;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #E53935 0%, #1E88E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 30px;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .fighter-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .fighter-card-red {
        background: linear-gradient(135deg, rgba(229, 57, 53, 0.1) 0%, rgba(229, 57, 53, 0.05) 100%);
        border-left: 3px solid var(--primary-red);
    }
    
    .fighter-card-blue {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(30, 136, 229, 0.05) 100%);
        border-left: 3px solid var(--primary-blue);
    }
    
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-blue);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .bet-recommendation {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid var(--success-color);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%);
    }
    
    .section-fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def normalize_name(s):
    """Normalise un nom de combattant"""
    if not isinstance(s, str):
        return None
    s_norm = unicodedata.normalize('NFKD', s)
    s_norm = ''.join(c for c in s_norm if not unicodedata.combining(c))
    s_norm = s_norm.lower()
    s_norm = re.sub(r"[^a-z0-9\s']", " ", s_norm)
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    return s_norm

def id_from_url(u: str):
    """Extrait l'ID d'une URL"""
    if not isinstance(u, str) or not u:
        return None
    m = re.search(r"/([0-9a-f]{16,})$", u.strip())
    return m.group(1) if m else u

def dec_to_prob(dec):
    """Convertit cote décimale en probabilité"""
    try:
        d = float(dec)
        return 1.0/d if d > 0 else np.nan
    except:
        return np.nan

def devig_two_way(odds1_dec, odds2_dec):
    """Retire le vig (dé-vigorish) de deux cotes"""
    p1 = dec_to_prob(odds1_dec)
    p2 = dec_to_prob(odds2_dec)
    if pd.isna(p1) or pd.isna(p2):
        return np.nan, np.nan
    s = p1 + p2
    if s <= 0:
        return np.nan, np.nan
    return p1/s, p2/s

def get_elo_for_fighter(fighter_id, elo_dict):
    """Récupère l'Elo d'un combattant avec valeur par défaut"""
    return elo_dict.get(fighter_id, BASE_ELO)

def get_fighter_data_with_fallback(fighter_url, fighter_name, fighters_data, model_data):
    """
    Récupère les données d'un combattant avec plusieurs méthodes de fallback:
    1. Par URL complète
    2. Par fighter_id (extrait de l'URL)
    3. Par nom normalisé
    4. Valeurs par défaut si non trouvé
    """
    # Méthode 1: Par URL complète
    if fighter_url and fighter_url in fighters_data:
        return fighters_data[fighter_url]
    
    # Méthode 2: Par fighter_id
    fighter_id = id_from_url(fighter_url) if fighter_url else None
    if fighter_id and fighter_id in fighters_data:
        return fighters_data[fighter_id]
    
    # Méthode 3: Par nom normalisé
    if fighter_name:
        normalized_name = fighter_name.lower().strip()
        if normalized_name in fighters_data:
            return fighters_data[normalized_name]
    
    # Méthode 4: Valeurs par défaut
    elo = get_elo_for_fighter(fighter_id, model_data['elo_dict']) if fighter_id else BASE_ELO
    
    # Essayer de récupérer les données bio depuis model_data
    fighter_bio = model_data.get('fighter_bio', {})
    bio = fighter_bio.get(fighter_id, {}) if fighter_id else {}
    
    return {
        'name': fighter_name or 'Unknown',
        'fighter_id': fighter_id,
        'elo_global': elo,
        'elo_div': BASE_ELO,
        'reach_cm': bio.get('reach_cm'),  # ✅ Données bio pour le modèle
        'age': bio.get('age'),  # ✅ Données bio pour le modèle
        'sig_lnd': 0,
        'sig_att': 0,
        'kd': 0,  # ✅ Knockdowns
        'td_lnd': 0,
        'td_att': 0,
        'adv_elo_mean_3': BASE_ELO
    }

def clean_text(s: str) -> str:
    """Nettoie un texte"""
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s))
    return s.strip()

def parse_mmss_to_seconds(s):
    """Parse MM:SS en secondes"""
    if s is None:
        return np.nan
    m = re.match(r"^(\d+):(\d{2})$", str(s).strip())
    if not m:
        return np.nan
    return int(m.group(1))*60 + int(m.group(2))

def to_float_safe(x):
    """Conversion sûre en float"""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        return float(m.group(0)) if m else np.nan
    except:
        return np.nan

# ============================================================================
# FONCTIONS DE SCRAPING
# ============================================================================

import subprocess

_last_request_time = 0

def make_request(url, max_retries=3):
    """Effectue une requête HTTP avec curl (plus fiable que requests pour ce site)"""
    global _last_request_time
    
    # Rate limiting: minimum 1.5 seconde entre les requêtes
    elapsed = time.time() - _last_request_time
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    
    for i in range(max_retries):
        try:
            _last_request_time = time.time()
            result = subprocess.run(
                ['curl', '-s', '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0', 
                 '--max-time', '30', url],
                capture_output=True,
                text=True,
                timeout=35
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                # Créer un objet response-like
                class CurlResponse:
                    def __init__(self, text):
                        self.text = text
                        self.status_code = 200
                return CurlResponse(result.stdout)
            time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return None

def get_completed_events_urls(max_pages=1):
    """Récupère les URLs des événements complétés"""
    # Note: page=0 cause une erreur 500, on commence à page=1 ou sans paramètre
    urls = []
    
    for page in range(max_pages):
        if page == 0:
            url = "http://ufcstats.com/statistics/events/completed"
        else:
            url = f"http://ufcstats.com/statistics/events/completed?page={page}"
        
        response = make_request(url)
        if not response:
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='b-statistics__table-events')
        
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 1:
                    link = cells[0].find('a')
                    if link:
                        urls.append(link.get('href'))
    
    return urls

def extract_fights_from_event_detailed(event_url):
    """Extrait les combats détaillés d'un événement"""
    response = make_request(event_url)
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    fights = []
    
    # ✅ Extraire la date de l'événement
    event_date = None
    date_span = soup.find('span', class_='b-statistics__date')
    if date_span:
        try:
            event_date = pd.to_datetime(date_span.text.strip(), format='%B %d, %Y')
        except:
            pass
    
    if not event_date:
        for item in soup.select('.b-list__box-list-item'):
            text = item.get_text().strip()
            if 'Date' in text:
                import re
                date_match = re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})', text)
                if date_match:
                    try:
                        event_date = pd.to_datetime(date_match.group(1), format='%B %d, %Y')
                    except:
                        pass
                break
    
    table = soup.find("table", class_="b-fight-details__table")
    if table:
        rows = table.select("tbody > tr")
        
        for row in rows:
            # ✅ L'URL du combat est dans data-link de la ligne TR
            fight_url = row.get('data-link')
            if not fight_url:
                continue
            
            # Les combattants sont dans la 2ème cellule
            fighter_links = row.select("td:nth-child(2) a.b-link")
            if len(fighter_links) >= 2:
                fights.append({
                    'fight_url': fight_url,
                    'event_url': event_url,
                    'event_date': event_date,
                    'red_fighter': fighter_links[0].text.strip(),
                    'blue_fighter': fighter_links[1].text.strip(),
                    'red_url': fighter_links[0].get('href'),
                    'blue_url': fighter_links[1].get('href')
                })
    
    return fights

def extract_fight_details(fight_url):
    """Extrait les détails complets d'un combat"""
    response = make_request(fight_url)
    if not response:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    sections = soup.select('.b-fight-details__person')
    if len(sections) < 2:
        return None
    
    fighters = []
    for section in sections[:2]:
        name_elem = section.select_one('.b-fight-details__person-name a')
        if not name_elem:
            continue
        
        stats_rows = section.select('.b-fight-details__person-stat')
        
        fighter_data = {
            'fighter_url': name_elem.get('href'),
            'fighter_name': clean_text(name_elem.text)
        }
        
        for stat_row in stats_rows:
            label_elem = stat_row.select_one('.b-fight-details__person-title')
            value_elem = stat_row.select_one('.b-fight-details__person-text')
            
            if label_elem and value_elem:
                label = clean_text(label_elem.text).lower()
                value = clean_text(value_elem.text)
                
                if 'kd' in label:
                    fighter_data['kd'] = to_float_safe(value)
                elif 'sig. str' in label:
                    parts = value.split(' of ')
                    if len(parts) == 2:
                        fighter_data['sig_lnd'] = to_float_safe(parts[0])
                        fighter_data['sig_att'] = to_float_safe(parts[1])
                elif 'total str' in label:
                    parts = value.split(' of ')
                    if len(parts) == 2:
                        fighter_data['tot_lnd'] = to_float_safe(parts[0])
                        fighter_data['tot_att'] = to_float_safe(parts[1])
                elif 'td' in label and 'sub' not in label:
                    parts = value.split(' of ')
                    if len(parts) == 2:
                        fighter_data['td_lnd'] = to_float_safe(parts[0])
                        fighter_data['td_att'] = to_float_safe(parts[1])
                elif 'sub' in label:
                    fighter_data['sub_att'] = to_float_safe(value)
                elif 'ctrl' in label:
                    fighter_data['ctrl_secs'] = parse_mmss_to_seconds(value)
        
        fighters.append(fighter_data)
    
    # ✅ Trouver le gagnant avec le bon sélecteur (style_green = winner)
    for section in sections[:2]:
        status = section.select_one('.b-fight-details__person-status')
        name_elem = section.select_one('.b-fight-details__person-name a')
        
        if status and name_elem:
            status_classes = status.get('class', [])
            fighter_name = clean_text(name_elem.text)
            
            # Trouver le fighter correspondant
            for fighter in fighters:
                if fighter.get('fighter_name') == fighter_name:
                    if 'b-fight-details__person-status_style_green' in status_classes:
                        fighter['result_win'] = 1
                    else:
                        fighter['result_win'] = 0
    
    return fighters

def compute_elo_ratings(appearances_df, K=24):
    """Calcule les ratings Elo et retourne aussi le format ratings_timeseries"""
    df = appearances_df.sort_values(["event_date", "fight_id"]).copy()
    
    base = BASE_ELO
    elo_global = {}
    elo_div = {}
    rows_out = []
    ratings_timeseries = []  # ✅ Format pour ratings_timeseries.parquet
    
    for event_date, event_group in df.groupby("event_date", sort=False):
        elo_snapshot = {
            "global": dict(elo_global),
            "div": dict(elo_div)
        }
        
        for fight_id, fight_group in event_group.groupby("fight_id", sort=False):
            if fight_group.shape[0] != 2:
                continue
            
            a, b = fight_group.iloc[0], fight_group.iloc[1]
            
            fa, fb = a["fighter_id"], b["fighter_id"]
            div = a.get("weight_class") or "Unknown"
            
            Ra_g = elo_snapshot["global"].get(fa, base)
            Rb_g = elo_snapshot["global"].get(fb, base)
            Ra_d = elo_snapshot["div"].get((fa, div), base)
            Rb_d = elo_snapshot["div"].get((fb, div), base)
            
            for idx, r in fight_group.iterrows():
                fighter_id = r["fighter_id"]
                rows_out.append({
                    **r.to_dict(),
                    "elo_global_pre": elo_snapshot["global"].get(fighter_id, base),
                    "elo_div_pre": elo_snapshot["div"].get((fighter_id, div), base)
                })
            
            if not pd.isna(a.get("result_win")) and not pd.isna(b.get("result_win")):
                Sa, Sb = float(a["result_win"]), float(b["result_win"])
                
                Ea_g = 1.0 / (1.0 + 10 ** ((Rb_g - Ra_g) / 400))
                Eb_g = 1.0 - Ea_g
                
                new_Ra_g = Ra_g + K * (Sa - Ea_g)
                new_Rb_g = Rb_g + K * (Sb - Eb_g)
                
                elo_global[fa] = new_Ra_g
                elo_global[fb] = new_Rb_g
                
                Ea_d = 1.0 / (1.0 + 10 ** ((Rb_d - Ra_d) / 400))
                Eb_d = 1.0 - Ea_d
                
                new_Ra_d = Ra_d + K * (Sa - Ea_d)
                new_Rb_d = Rb_d + K * (Sb - Eb_d)
                
                elo_div[(fa, div)] = new_Ra_d
                elo_div[(fb, div)] = new_Rb_d
                
                # ✅ Ajouter au format ratings_timeseries (format cohérent)
                ratings_timeseries.append({
                    'fight_url': a.get('fight_url', ''),
                    'fight_id': fight_id,
                    'event_date': event_date,
                    'fighter_1': a.get('fighter_name', ''),
                    'fighter_2': b.get('fighter_name', ''),
                    'fighter_1_id': fa,
                    'fighter_2_id': fb,
                    'elo_1_pre': Ra_g,
                    'elo_2_pre': Rb_g,
                    'elo_1_post': new_Ra_g,
                    'elo_2_post': new_Rb_g,
                    'winner': 1 if Sa == 1 else 2
                })
    
    return pd.DataFrame(rows_out), elo_global, elo_div, pd.DataFrame(ratings_timeseries)

# ============================================================================
# VÉRIFICATION ET MISE À JOUR DES DONNÉES
# ============================================================================

def check_data_freshness():
    """
    Vérifie l'état des données LOCALEMENT (sans scraping web).
    Rapide car ne fait que lire les fichiers locaux.
    Utilise appearances.parquet pour les dates (source de vérité).
    """
    appearances_path = RAW_DIR / "appearances.parquet"
    ratings_path = INTERIM_DIR / "ratings_timeseries.parquet"
    
    if not appearances_path.exists() and not ratings_path.exists():
        return {
            'has_data': False,
            'last_event_date': None,
            'days_old': None,
            'fight_count': 0,
            'fighter_count': 0,
            'message': '📭 Aucune donnée existante. Lancez une mise à jour pour scraper les données.'
        }
    
    try:
        # Compter les combats et combattants depuis appearances
        fight_count = 0
        fighter_count = 0
        last_date = None
        
        # ✅ Utiliser appearances pour les dates (source de vérité)
        if appearances_path.exists():
            appearances_df = pd.read_parquet(appearances_path)
            fight_count = appearances_df['fight_id'].nunique() if 'fight_id' in appearances_df.columns else len(appearances_df) // 2
            fighter_count = appearances_df['fighter_id'].nunique() if 'fighter_id' in appearances_df.columns else 0
            if 'event_date' in appearances_df.columns:
                last_date = pd.to_datetime(appearances_df['event_date']).max()
                if hasattr(last_date, 'tz') and last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
        
        # Vérifier si la date est valide
        if last_date is None or pd.isna(last_date):
            return {
                'has_data': True,
                'last_event_date': None,
                'days_old': None,
                'fight_count': fight_count,
                'fighter_count': fighter_count,
                'message': '⚠️ Aucune date trouvée dans les données'
            }
        
        days_old = (pd.Timestamp.now() - last_date).days
        
        # Message basé sur l'âge des données
        if days_old <= 7:
            status = "✅"
            freshness = "à jour"
        elif days_old <= 14:
            status = "🟡"
            freshness = "récentes"
        else:
            status = "🟠"
            freshness = "à mettre à jour"
        
        return {
            'has_data': True,
            'last_event_date': last_date,
            'days_old': days_old,
            'fight_count': fight_count,
            'fighter_count': fighter_count,
            'message': f'{status} Données {freshness} (dernier événement: {last_date.date()}, il y a {days_old} jours)'
        }
    
    except Exception as e:
        return {
            'has_data': False,
            'last_event_date': None,
            'days_old': None,
            'fight_count': 0,
            'fighter_count': 0,
            'message': f'❌ Erreur lecture données: {str(e)}'
        }

def scrape_new_events(progress_callback=None):
    """Scrappe les nouveaux événements non présents dans les données"""
    appearances_path = RAW_DIR / "appearances.parquet"
    
    existing_fight_ids = set()
    if appearances_path.exists():
        try:
            appearances_df = pd.read_parquet(appearances_path)
            
            fight_id_col = None
            for col in ['fight_id', 'fight_url', 'bout_url']:
                if col in appearances_df.columns:
                    fight_id_col = col
                    break
            
            if fight_id_col:
                if 'url' in fight_id_col.lower():
                    existing_fight_ids = set(appearances_df[fight_id_col].apply(id_from_url))
                else:
                    existing_fight_ids = set(appearances_df[fight_id_col].unique())
            else:
                st.warning("⚠️ Aucune colonne d'identification de combat trouvée.")
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du chargement: {e}")
    
    if progress_callback:
        progress_callback("🔍 Récupération des derniers événements...")
    
    # ✅ Ne scraper qu'une seule page d'abord (les ~12 derniers événements)
    event_urls = get_completed_events_urls(max_pages=1)
    
    new_fights = []
    new_appearances = []
    found_existing = False  # Flag pour arrêter dès qu'on trouve un combat existant
    today = pd.Timestamp.now().normalize()  # Date d'aujourd'hui à minuit
    
    total_events = len(event_urls)
    
    for i, event_url in enumerate(event_urls):
        if progress_callback:
            progress_callback(f"📊 Analyse événement {i+1}/{total_events}...")
        
        fights = extract_fights_from_event_detailed(event_url)
        
        # ✅ Ignorer les événements futurs (après aujourd'hui)
        if fights and fights[0].get('event_date'):
            event_date = fights[0]['event_date']
            if pd.notna(event_date) and event_date > today:
                if progress_callback:
                    progress_callback(f"⏭️ Événement futur ignoré ({event_date.strftime('%Y-%m-%d')})")
                continue
        
        event_has_new_fights = False
        for fight in fights:
            fight_id = id_from_url(fight['fight_url'])
            
            if fight_id in existing_fight_ids:
                # Combat déjà existant, on peut s'arrêter après cet événement
                found_existing = True
                continue
            
            # Nouveau combat trouvé
            event_has_new_fights = True
            new_fights.append(fight)
            
            if progress_callback:
                progress_callback(f"⚔️ Nouveau: {fight['red_fighter']} vs {fight['blue_fighter']}")
            
            fight_details = extract_fight_details(fight['fight_url'])
            
            if fight_details:
                for fighter_data in fight_details:
                    fighter_data['fight_id'] = fight_id
                    fighter_data['fight_url'] = fight['fight_url']
                    fighter_data['event_url'] = event_url
                    fighter_data['event_date'] = fight['event_date']
                    fighter_data['fighter_id'] = id_from_url(fighter_data['fighter_url'])
                    new_appearances.append(fighter_data)
        
        # ✅ Si on a trouvé des combats existants et pas de nouveaux dans cet événement, on s'arrête
        if found_existing and not event_has_new_fights:
            if progress_callback:
                progress_callback("✅ Tous les événements récents ont été vérifiés")
            break
        
        time.sleep(0.3)  # Réduire le délai (0.5 -> 0.3)
    
    return {
        'new_fights': new_fights,
        'new_appearances': new_appearances,
        'count': len(new_fights)
    }

def update_data_files(new_appearances):
    """Met à jour les fichiers de données"""
    appearances_path = RAW_DIR / "appearances.parquet"
    
    if appearances_path.exists():
        existing_df = pd.read_parquet(appearances_path)
    else:
        existing_df = pd.DataFrame()
    
    new_df = pd.DataFrame(new_appearances)
    
    if not new_df.empty:
        if 'event_date' in new_df.columns:
            new_df['event_date'] = pd.to_datetime(new_df['event_date'])
        
        if 'fight_id' not in new_df.columns and 'fight_url' in new_df.columns:
            new_df['fight_id'] = new_df['fight_url'].apply(id_from_url)
        
        if not existing_df.empty:
            if 'fight_id' not in existing_df.columns:
                if 'fight_url' in existing_df.columns:
                    existing_df['fight_id'] = existing_df['fight_url'].apply(id_from_url)
                elif 'bout_url' in existing_df.columns:
                    existing_df['fight_id'] = existing_df['bout_url'].apply(id_from_url)
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            if 'fight_id' in combined_df.columns and 'fighter_id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['fight_id', 'fighter_id'], keep='last')
        else:
            combined_df = new_df
        
        combined_df.to_parquet(appearances_path, index=False)
        
        return combined_df
    
    return existing_df

def recalculate_features_and_elo(progress_callback=None):
    """Recalcule toutes les features et les Elo"""
    appearances_path = RAW_DIR / "appearances.parquet"
    
    if not appearances_path.exists():
        raise FileNotFoundError("Fichier appearances.parquet non trouvé")
    
    if progress_callback:
        progress_callback("📊 Chargement des données...")
    
    appearances_df = pd.read_parquet(appearances_path)
    
    if 'fight_id' not in appearances_df.columns:
        if 'fight_url' in appearances_df.columns:
            appearances_df['fight_id'] = appearances_df['fight_url'].apply(id_from_url)
        elif 'bout_url' in appearances_df.columns:
            appearances_df['fight_id'] = appearances_df['bout_url'].apply(id_from_url)
        else:
            raise ValueError("Aucune colonne d'identification de combat trouvée")
    
    if 'fighter_id' not in appearances_df.columns:
        if 'fighter_url' in appearances_df.columns:
            appearances_df['fighter_id'] = appearances_df['fighter_url'].apply(id_from_url)
        else:
            raise ValueError("Aucune colonne d'identification de combattant trouvée")
    
    if progress_callback:
        progress_callback("🎯 Calcul des ratings Elo...")
    
    appearances_with_elo, elo_global_dict, elo_div_dict, ratings_ts = compute_elo_ratings(appearances_df, K=K_FACTOR)
    
    # Sauvegarder asof_full.parquet
    asof_path = INTERIM_DIR / "asof_full.parquet"
    appearances_with_elo.to_parquet(asof_path, index=False)
    
    # ✅ Sauvegarder ratings_timeseries.parquet (pour les dates et Elo POST)
    ratings_path = INTERIM_DIR / "ratings_timeseries.parquet"
    if not ratings_ts.empty:
        ratings_ts.to_parquet(ratings_path, index=False)
        if progress_callback:
            progress_callback(f"💾 Sauvegardé {len(ratings_ts)} combats dans ratings_timeseries")
    
    if progress_callback:
        progress_callback("✅ Features et Elo recalculés avec succès!")
    
    return {
        'appearances_count': len(appearances_with_elo),
        'fighters_count': len(elo_global_dict),
        'elo_global': elo_global_dict,
        'elo_div': elo_div_dict
    }

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data(ttl=3600)
def load_model_and_data():
    """Charge le modèle ML et les données nécessaires"""
    data = {
        "model": None,
        "calibrator": None,
        "feat_cols": None,
        "features": None,  # Liste des features du modèle
        "feature_medians": {},  # Valeurs médianes pour imputation
        "strategy": {},  # Stratégie de mise
        "ratings": None,
        "elo_dict": {},
        "fighter_bio": {}  # Données biographiques (reach, age)
    }
    
    # Charger le modèle (nouveau format avec features market+reach+age)
    model_path = PROC_DIR / "model_pipeline.pkl"
    if model_path.exists():
        try:
            model_info = joblib.load(model_path)
            data["model"] = model_info.get("model")
            data["features"] = model_info.get("features", ["market_logit", "reach_diff", "age_diff"])
            data["feature_medians"] = model_info.get("feature_medians", {"reach_diff": 0, "age_diff": 0})
            data["strategy"] = model_info.get("strategy", {"edge_threshold": 0.03, "stake_pct": 0.05})
            # Pour compatibilité avec l'ancien code
            data["feat_cols"] = data["features"]
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement modèle: {e}")
    
    # Charger les données biographiques (reach, dob)
    bio_path = RAW_DIR / "fighter_bio.parquet"
    if bio_path.exists():
        try:
            bio_df = pd.read_parquet(bio_path)
            fighter_bio = {}
            for _, row in bio_df.iterrows():
                fighter_id = id_from_url(row.get("fighter_url", ""))
                if fighter_id:
                    # Calculer l'âge à partir de dob
                    age = None
                    if pd.notna(row.get("dob")):
                        try:
                            dob = pd.to_datetime(row["dob"])
                            age = (pd.Timestamp.now() - dob).days / 365.25
                        except:
                            pass
                    fighter_bio[fighter_id] = {
                        "reach_cm": row.get("reach_cm"),
                        "age": age,
                        "name": row.get("fighter_name", "")
                    }
            data["fighter_bio"] = fighter_bio
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement fighter_bio: {e}")
    
    # Charger le calibrateur
    calib_path = PROC_DIR / "calibrator.pkl"
    if calib_path.exists():
        try:
            data["calibrator"] = joblib.load(calib_path)
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement calibrateur: {e}")
    
    # ✅ LOGIQUE CORRECTE: Charger ratings_timeseries et prendre les derniers Elo POST
    ratings_path = INTERIM_DIR / "ratings_timeseries.parquet"
    if ratings_path.exists():
        try:
            ratings_df = pd.read_parquet(ratings_path)
            data["ratings"] = ratings_df
            
            # ✅ Pour chaque combattant, prendre le DERNIER Elo POST
            # (qui sera son Elo PRE pour son prochain combat)
            elo_dict = {}
            
            # Détecter le format du fichier
            if 'fighter_1_id' in ratings_df.columns and 'fighter_2_id' in ratings_df.columns:
                # ✅ Format actuel: fighter_1_id, fighter_2_id, elo_1_post, elo_2_post
                ratings_sorted = ratings_df.sort_values('event_date')
                for _, row in ratings_sorted.iterrows():
                    f1_id = row.get('fighter_1_id')
                    f1_elo = row.get('elo_1_post', row.get('elo_1_pre', BASE_ELO))
                    f2_id = row.get('fighter_2_id')
                    f2_elo = row.get('elo_2_post', row.get('elo_2_pre', BASE_ELO))
                    
                    if f1_id and pd.notna(f1_id):
                        elo_dict[f1_id] = f1_elo
                    if f2_id and pd.notna(f2_id):
                        elo_dict[f2_id] = f2_elo
            
            elif 'fa' in ratings_df.columns and 'fb' in ratings_df.columns:
                # Ancien format (fa, fb, elo_global_fa_post, elo_global_fb_post)
                for fighter_id in ratings_df['fa'].unique():
                    last_fight = ratings_df[ratings_df['fa'] == fighter_id].iloc[-1]
                    elo_dict[fighter_id] = last_fight['elo_global_fa_post']
                
                for fighter_id in ratings_df['fb'].unique():
                    if fighter_id not in elo_dict:
                        last_fight = ratings_df[ratings_df['fb'] == fighter_id].iloc[-1]
                        elo_dict[fighter_id] = last_fight['elo_global_fb_post']
                    else:
                        last_fight_b = ratings_df[ratings_df['fb'] == fighter_id].iloc[-1]
                        last_fight_a = ratings_df[ratings_df['fa'] == fighter_id].iloc[-1]
                        if 'event_date' in ratings_df.columns:
                            date_a = last_fight_a.get('event_date')
                            date_b = last_fight_b.get('event_date')
                            if pd.notna(date_b) and pd.notna(date_a) and date_b > date_a:
                                elo_dict[fighter_id] = last_fight_b['elo_global_fb_post']
                            elif pd.notna(date_b) and pd.isna(date_a):
                                elo_dict[fighter_id] = last_fight_b['elo_global_fb_post']
            
            data["elo_dict"] = elo_dict
            
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement ratings: {e}")
    
    # Fallback sur asof_full si ratings_timeseries n'existe pas
    elif (INTERIM_DIR / "asof_full.parquet").exists():
        try:
            asof_df = pd.read_parquet(INTERIM_DIR / "asof_full.parquet")
            data["ratings"] = asof_df
            
            elo_dict = {}
            for _, row in asof_df.iterrows():
                fighter_id = row.get('fighter_id')
                if fighter_id:
                    if 'elo_global_post' in row:
                        elo_dict[fighter_id] = row['elo_global_post']
                    elif 'elo_global_pre' in row:
                        elo_dict[fighter_id] = row['elo_global_pre']
            
            data["elo_dict"] = elo_dict
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement depuis asof_full: {e}")
    
    return data

@st.cache_data(ttl=3600)
def load_fighters_data():
    """Charge les données des combattants avec Elo POST et données bio (reach, age)"""
    fighters = {}
    
    # ✅ Charger les données biographiques (reach, dob) d'abord
    fighter_bio = {}
    bio_path = RAW_DIR / "fighter_bio.parquet"
    if bio_path.exists():
        try:
            bio_df = pd.read_parquet(bio_path)
            for _, row in bio_df.iterrows():
                fighter_id = id_from_url(row.get("fighter_url", ""))
                if fighter_id:
                    # Calculer l'âge à partir de dob
                    age = None
                    if pd.notna(row.get("dob")):
                        try:
                            dob = pd.to_datetime(row["dob"])
                            age = (pd.Timestamp.now() - dob).days / 365.25
                        except:
                            pass
                    fighter_bio[fighter_id] = {
                        "reach_cm": row.get("reach_cm"),
                        "age": age,
                        "bio_name": row.get("fighter_name", "")
                    }
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement fighter_bio: {e}")
    
    # ✅ Charger les Elo POST depuis ratings_timeseries
    elo_post_dict = {}
    ratings_path = INTERIM_DIR / "ratings_timeseries.parquet"
    if ratings_path.exists():
        try:
            ratings_df = pd.read_parquet(ratings_path)
            
            # Détecter le format du fichier
            if 'fighter_1_id' in ratings_df.columns and 'fighter_2_id' in ratings_df.columns:
                # Format actuel: fighter_1_id, fighter_2_id, elo_1_post, elo_2_post
                ratings_sorted = ratings_df.sort_values('event_date')
                for _, row in ratings_sorted.iterrows():
                    f1_id = row.get('fighter_1_id')
                    f1_elo = row.get('elo_1_post', row.get('elo_1_pre', BASE_ELO))
                    f2_id = row.get('fighter_2_id')
                    f2_elo = row.get('elo_2_post', row.get('elo_2_pre', BASE_ELO))
                    
                    if f1_id and pd.notna(f1_id):
                        elo_post_dict[f1_id] = f1_elo
                    if f2_id and pd.notna(f2_id):
                        elo_post_dict[f2_id] = f2_elo
            
            elif 'fa' in ratings_df.columns and 'fb' in ratings_df.columns:
                # Ancien format
                for fighter_id in ratings_df['fa'].unique():
                    last_fight = ratings_df[ratings_df['fa'] == fighter_id].iloc[-1]
                    elo_post_dict[fighter_id] = last_fight['elo_global_fa_post']
                
                for fighter_id in ratings_df['fb'].unique():
                    if fighter_id not in elo_post_dict:
                        last_fight = ratings_df[ratings_df['fb'] == fighter_id].iloc[-1]
                        elo_post_dict[fighter_id] = last_fight['elo_global_fb_post']
                    else:
                        last_fight_b = ratings_df[ratings_df['fb'] == fighter_id].iloc[-1]
                        last_fight_a = ratings_df[ratings_df['fa'] == fighter_id].iloc[-1]
                        if 'event_date' in ratings_df.columns:
                            date_a = last_fight_a.get('event_date')
                            date_b = last_fight_b.get('event_date')
                            if pd.notna(date_b) and pd.notna(date_a) and date_b > date_a:
                                elo_post_dict[fighter_id] = last_fight_b['elo_global_fb_post']
                        
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement Elo POST: {e}")
    
    # ✅ Charger les stats depuis asof_full ET appearances (fusion)
    asof_path = INTERIM_DIR / "asof_full.parquet"
    appearances_path = RAW_DIR / "appearances.parquet"
    
    # Charger les deux sources et les fusionner
    all_fighters_urls = set()
    source_dfs = []
    
    if asof_path.exists():
        try:
            asof_df = pd.read_parquet(asof_path)
            if not asof_df.empty and 'fighter_url' in asof_df.columns:
                source_dfs.append(asof_df)
                all_fighters_urls.update(asof_df['fighter_url'].unique())
        except:
            pass
    
    # Ajouter appearances pour les combattants manquants
    if appearances_path.exists():
        try:
            appearances_df = pd.read_parquet(appearances_path)
            if not appearances_df.empty and 'fighter_url' in appearances_df.columns:
                # Filtrer seulement les combattants pas encore chargés
                missing_mask = ~appearances_df['fighter_url'].isin(all_fighters_urls)
                if missing_mask.any():
                    source_dfs.append(appearances_df[missing_mask])
        except:
            pass
    
    # Combiner toutes les sources
    if source_dfs:
        source_df = pd.concat(source_dfs, ignore_index=True)
    else:
        source_df = None
    
    if source_df is not None and not source_df.empty and 'fighter_url' in source_df.columns:
        try:
            for fighter_url in source_df['fighter_url'].unique():
                fighter_data = source_df[source_df['fighter_url'] == fighter_url].iloc[-1]
                fighter_id = id_from_url(fighter_url)
                fighter_name = fighter_data.get('fighter_name', 'Unknown')
                
                # ✅ Récupérer les données bio (reach, age)
                bio = fighter_bio.get(fighter_id, {})
                
                data_entry = {
                    'fighter_url': fighter_url,
                    'fighter_id': fighter_id,
                    'name': fighter_name,
                    # ✅ Utiliser Elo POST depuis ratings_timeseries
                    'elo_global': elo_post_dict.get(fighter_id, BASE_ELO),
                    'elo_div': fighter_data.get('elo_div_pre', BASE_ELO) if 'elo_div_pre' in fighter_data else BASE_ELO,
                    # ✅ Données bio pour le nouveau modèle
                    'reach_cm': bio.get('reach_cm'),
                    'age': bio.get('age'),
                    # Stats de combat
                    'sig_lnd': fighter_data.get('his_mean_sig_lnd', fighter_data.get('sig_lnd', 0)),
                    'sig_att': fighter_data.get('his_mean_sig_att', fighter_data.get('sig_att', 0)),
                    'kd': fighter_data.get('his_mean_kd', fighter_data.get('kd', 0)),
                    'td_lnd': fighter_data.get('his_mean_td_lnd', fighter_data.get('td_lnd', 0)),
                    'td_att': fighter_data.get('his_mean_td_att', fighter_data.get('td_att', 0)),
                    'adv_elo_mean_3': fighter_data.get('adv_elo_mean_3', BASE_ELO)
                }
                
                # ✅ Index par URL
                fighters[fighter_url] = data_entry
                
                # ✅ Index par fighter_id (pour fallback par ID dans l'URL)
                if fighter_id:
                    fighters[fighter_id] = data_entry
                
                # ✅ Index par nom normalisé (pour fallback par nom)
                if fighter_name and fighter_name != 'Unknown':
                    normalized_name = fighter_name.lower().strip()
                    fighters[normalized_name] = data_entry
                    
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement combattants: {e}")
    
    return fighters

@st.cache_data(ttl=3600)
def get_fighter_recent_fights(fighter_id, n_fights=3):
    """
    Récupère les n derniers combats d'un combattant.
    
    Returns:
        Liste de dicts avec: opponent_name, result (W/L), event_date, method
    """
    ratings_path = INTERIM_DIR / "ratings_timeseries.parquet"
    
    if not ratings_path.exists():
        return []
    
    try:
        ratings_df = pd.read_parquet(ratings_path)
        
        # Filtrer les combats du combattant (position 1 ou 2)
        fights_as_1 = ratings_df[ratings_df['fighter_1_id'] == fighter_id].copy()
        fights_as_1['position'] = 1
        fights_as_1['opponent_name'] = fights_as_1['fighter_2']
        fights_as_1['result'] = fights_as_1['winner'].apply(lambda w: 'W' if w == 1 else 'L')
        
        fights_as_2 = ratings_df[ratings_df['fighter_2_id'] == fighter_id].copy()
        fights_as_2['position'] = 2
        fights_as_2['opponent_name'] = fights_as_2['fighter_1']
        fights_as_2['result'] = fights_as_2['winner'].apply(lambda w: 'W' if w == 2 else 'L')
        
        # Combiner et trier par date
        all_fights = pd.concat([fights_as_1, fights_as_2])
        if all_fights.empty:
            return []
        
        all_fights['event_date'] = pd.to_datetime(all_fights['event_date'])
        all_fights = all_fights.sort_values('event_date', ascending=False)
        
        # Prendre les n derniers combats
        recent = all_fights.head(n_fights)
        
        result = []
        for _, row in recent.iterrows():
            result.append({
                'opponent': row['opponent_name'],
                'result': row['result'],
                'date': row['event_date'].strftime('%d/%m/%Y') if pd.notna(row['event_date']) else 'N/A'
            })
        
        return result
        
    except Exception as e:
        return []

# ============================================================================
# CALCUL DES MISES (STRATÉGIE KELLY)
# ============================================================================

def calculate_kelly_stake(proba_model, odds, bankroll, strategy_params):
    """Calcule la mise selon le critère de Kelly"""
    kelly_fraction = strategy_params['kelly_fraction']
    min_confidence = strategy_params['min_confidence']
    min_edge = strategy_params['min_edge']
    max_ev = strategy_params.get('max_value', 1.0)  # EV maximum (0.50 = 50%)
    max_bet_fraction = strategy_params['max_bet_fraction']
    min_bet_pct = strategy_params['min_bet_pct']
    min_odds = strategy_params.get('min_odds', 1.0)  # Cote minimum
    max_odds = strategy_params.get('max_odds', 999.0)  # Cote maximum
    
    p_market = 1.0 / odds if odds > 0 else 0
    edge = proba_model - p_market
    ev = (proba_model * odds) - 1
    
    should_bet = (
        proba_model >= min_confidence and
        edge >= min_edge and
        ev <= max_ev and              # ✅ EV max (éviter les EV trop élevés = suspects)
        odds >= min_odds and          # ✅ Cote minimum
        odds <= max_odds and          # ✅ Cote maximum
        ev > 0
    )
    
    if not should_bet:
        reason = []
        if proba_model < min_confidence:
            reason.append(f'Confiance {proba_model:.1%} < {min_confidence:.1%}')
        if edge < min_edge:
            reason.append(f'Edge {edge:.1%} < {min_edge:.1%}')
        if ev > max_ev:
            reason.append(f'EV {ev:.1%} > {max_ev:.1%} (suspect)')
        if ev <= 0:
            reason.append(f'EV {ev:.1%} <= 0')
        if odds < min_odds:
            reason.append(f'Cote {odds:.2f} < {min_odds:.2f}')
        if odds > max_odds:
            reason.append(f'Cote {odds:.2f} > {max_odds:.2f}')
        
        return {
            'stake': 0,
            'edge': edge,
            'ev': ev,
            'should_bet': False,
            'kelly_pct': 0,
            'reason': ', '.join(reason) if reason else 'Contraintes non respectées'
        }
    
    q = 1 - proba_model
    b = odds - 1
    kelly_fraction_value = (proba_model * b - q) / b
    kelly_adjusted = kelly_fraction_value / kelly_fraction
    kelly_pct = max(min_bet_pct, min(kelly_adjusted, max_bet_fraction))
    stake = bankroll * kelly_pct
    
    return {
        'stake': stake,
        'edge': edge,
        'ev': ev,
        'should_bet': True,
        'kelly_pct': kelly_pct,
        'kelly_raw': kelly_fraction_value,
        'reason': 'OK'
    }

# ============================================================================
# PRÉDICTION DE COMBAT (Nouveau modèle Market + Reach + Age)
# ============================================================================

def predict_fight_with_odds(fighter_a_data, fighter_b_data, model_data, odds_a, odds_b):
    """
    Prédit l'issue d'un combat avec le nouveau modèle basé sur:
    - market_logit: log-odds du marché
    - reach_diff: différence d'allonge (cm)
    - age_diff: différence d'âge (années)
    
    Returns:
        dict avec proba_a, proba_b, edge_a, edge_b, recommendation
    """
    if not model_data.get("model"):
        return None
    
    try:
        # Calculer la probabilité marché (dévigée)
        p_impl_a = 1 / odds_a
        p_impl_b = 1 / odds_b
        vig = p_impl_a + p_impl_b
        proba_market = p_impl_a / vig  # Proba marché pour A
        
        # Market logit
        proba_market_clipped = np.clip(proba_market, 0.01, 0.99)
        market_logit = np.log(proba_market_clipped / (1 - proba_market_clipped))
        
        # Reach diff (A - B) - utilise les valeurs réelles ou 0 si manquant
        reach_a = fighter_a_data.get('reach_cm')
        reach_b = fighter_b_data.get('reach_cm')
        
        # Si les deux reach sont disponibles, calculer la diff
        # Sinon, utiliser la médiane de reach_diff (qui est 0)
        if reach_a is not None and reach_b is not None and not pd.isna(reach_a) and not pd.isna(reach_b):
            reach_diff = float(reach_a) - float(reach_b)
        else:
            reach_diff = model_data.get('feature_medians', {}).get('reach_diff', 0.0)
        
        # Age diff (A - B) - utilise les valeurs réelles ou médiane si manquant
        age_a = fighter_a_data.get('age')
        age_b = fighter_b_data.get('age')
        
        if age_a is not None and age_b is not None and not pd.isna(age_a) and not pd.isna(age_b):
            age_diff = float(age_a) - float(age_b)
        else:
            age_diff = model_data.get('feature_medians', {}).get('age_diff', 0.0)
        
        # Créer le vecteur de features
        X = np.array([[market_logit, reach_diff, age_diff]])
        
        # Prédire
        proba_a = model_data["model"].predict_proba(X)[0][1]
        proba_b = 1 - proba_a
        
        # Calculer les edges
        edge_a = proba_a - p_impl_a
        edge_b = proba_b - p_impl_b
        
        # Déterminer la recommandation
        threshold = model_data.get('strategy', {}).get('edge_threshold', 0.03)
        
        if edge_a >= threshold:
            recommendation = {
                'bet_on': 'A',
                'fighter': fighter_a_data.get('name', 'Fighter A'),
                'odds': odds_a,
                'edge': edge_a,
                'proba_model': proba_a
            }
        elif edge_b >= threshold:
            recommendation = {
                'bet_on': 'B',
                'fighter': fighter_b_data.get('name', 'Fighter B'),
                'odds': odds_b,
                'edge': edge_b,
                'proba_model': proba_b
            }
        else:
            recommendation = None
        
        return {
            'proba_a': proba_a,
            'proba_b': proba_b,
            'proba_market': proba_market,
            'edge_a': edge_a,
            'edge_b': edge_b,
            'reach_diff': reach_diff,
            'age_diff': age_diff,
            'recommendation': recommendation,
            'winner': 'A' if proba_a > 0.5 else 'B',
            'confidence': 'Élevée' if abs(proba_a - 0.5) > 0.15 else 'Modérée'
        }
        
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return None

def predict_fight(fighter_a_data, fighter_b_data, model_data, odds_a=None, odds_b=None):
    """
    Prédit l'issue d'un combat.
    - Si odds_a et odds_b fournis: utilise le nouveau modèle market+physique
    - Sinon: fallback sur proba Elo uniquement
    """
    # Si cotes fournies, utiliser le nouveau modèle
    if odds_a is not None and odds_b is not None:
        return predict_fight_with_odds(fighter_a_data, fighter_b_data, model_data, odds_a, odds_b)
    
    # Sinon, fallback sur Elo
    elo_a = fighter_a_data.get('elo_global', BASE_ELO)
    elo_b = fighter_b_data.get('elo_global', BASE_ELO)
    
    # Proba Elo classique
    proba_elo_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    return {
        'proba_a': proba_elo_a,
        'proba_b': 1 - proba_elo_a,
        'proba_raw': proba_elo_a,
        'winner': 'A' if proba_elo_a > 0.5 else 'B',
        'confidence': 'Élevée' if abs(proba_elo_a - 0.5) > 0.2 else 'Modérée',
        'note': 'Basé sur Elo uniquement (entrez les cotes pour prédiction complète)'
    }

# ============================================================================
# SCRAPING ÉVÉNEMENTS UFC À VENIR
# ============================================================================

@st.cache_data(ttl=86400)
def get_upcoming_events(max_events=5):
    """Récupère les événements UFC à venir"""
    url = "http://ufcstats.com/statistics/events/upcoming"
    response = make_request(url)
    
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    events = []
    
    table = soup.find('table', class_='b-statistics__table-events')
    if table:
        rows = table.find_all('tr')[1:]
        
        for row in rows[:max_events]:
            cells = row.find_all('td')
            if len(cells) >= 1:
                link = cells[0].find('a')
                if link:
                    events.append({
                        'name': link.text.strip(),
                        'url': link.get('href')
                    })
    
    return events

@st.cache_data(ttl=86400)
def extract_fights_from_event(event_url):
    """Extrait les combats d'un événement"""
    response = make_request(event_url)
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    fights = []
    
    table = soup.find("table", class_="b-fight-details__table")
    if table:
        rows = table.select("tbody > tr")
        
        for row in rows:
            links = row.select("td:nth-child(2) a")
            if len(links) >= 2:
                fights.append({
                    'red_fighter': links[0].text.strip(),
                    'blue_fighter': links[1].text.strip(),
                    'red_url': links[0].get('href'),
                    'blue_url': links[1].get('href')
                })
    
    return fights

# ============================================================================
# GESTION BANKROLL
# ============================================================================

def init_bankroll():
    """Initialise la bankroll (avec support GitHub pour Streamlit Cloud)"""
    bankroll_file = BETS_DIR / "bankroll.csv"
    
    if GITHUB_CONFIG.get("enabled"):
        df, sha = load_csv_from_github("bets/bankroll.csv", GITHUB_CONFIG)
        if df is not None and not df.empty:
            df.to_csv(bankroll_file, index=False)
            return float(df.iloc[-1]["amount"])
    
    if bankroll_file.exists():
        df = pd.read_csv(bankroll_file)
        if not df.empty:
            return float(df.iloc[-1]["amount"])
    
    df = pd.DataFrame({
        "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "amount": [1000.0],
        "action": ["initial"],
        "note": ["Bankroll initiale"]
    })
    df.to_csv(bankroll_file, index=False)
    
    if GITHUB_CONFIG.get("enabled"):
        save_file_to_github("bets/bankroll.csv", df.to_csv(index=False), 
                           "Init bankroll", GITHUB_CONFIG)
    
    return 1000.0

def update_bankroll(new_amount, action="update", note=""):
    """Met a jour la bankroll (avec sync GitHub)"""
    bankroll_file = BETS_DIR / "bankroll.csv"
    sha = None
    
    if GITHUB_CONFIG.get("enabled"):
        df, sha = load_csv_from_github("bets/bankroll.csv", GITHUB_CONFIG)
        if df is None:
            df = pd.DataFrame(columns=["date", "amount", "action", "note"])
    elif bankroll_file.exists():
        df = pd.read_csv(bankroll_file)
    else:
        df = pd.DataFrame(columns=["date", "amount", "action", "note"])
    
    new_entry = pd.DataFrame({
        "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "amount": [new_amount],
        "action": [action],
        "note": [note]
    })
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(bankroll_file, index=False)
    
    if GITHUB_CONFIG.get("enabled"):
        save_file_to_github("bets/bankroll.csv", df.to_csv(index=False),
                           f"Update bankroll: {action}", GITHUB_CONFIG, sha)
    
    return new_amount

def add_bet(event_name, fighter_red, fighter_blue, pick, odds, stake, 
            model_probability, kelly_fraction, edge, ev):
    """Ajoute un pari (avec sync GitHub)"""
    bets_file = BETS_DIR / "bets.csv"
    sha = None
    
    if GITHUB_CONFIG.get("enabled"):
        df, sha = load_csv_from_github("bets/bets.csv", GITHUB_CONFIG)
        if df is not None and not df.empty:
            next_id = int(df["bet_id"].max()) + 1
        else:
            df = pd.DataFrame(columns=[
                "bet_id", "date", "event", "fighter_red", "fighter_blue",
                "pick", "odds", "stake", "model_probability", "kelly_fraction",
                "edge", "ev", "status", "result", "profit", "roi"
            ])
            next_id = 1
    elif bets_file.exists():
        df = pd.read_csv(bets_file)
        next_id = int(df["bet_id"].max()) + 1 if not df.empty else 1
    else:
        df = pd.DataFrame(columns=[
            "bet_id", "date", "event", "fighter_red", "fighter_blue",
            "pick", "odds", "stake", "model_probability", "kelly_fraction",
            "edge", "ev", "status", "result", "profit", "roi"
        ])
        next_id = 1
    
    new_bet = pd.DataFrame({
        "bet_id": [next_id],
        "date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        "event": [event_name],
        "fighter_red": [fighter_red],
        "fighter_blue": [fighter_blue],
        "pick": [pick],
        "odds": [odds],
        "stake": [stake],
        "model_probability": [model_probability],
        "kelly_fraction": [kelly_fraction],
        "edge": [edge],
        "ev": [ev],
        "status": ["open"],
        "result": [np.nan],
        "profit": [0.0],
        "roi": [0.0]
    })
    
    df = pd.concat([df, new_bet], ignore_index=True)
    df.to_csv(bets_file, index=False)
    
    if GITHUB_CONFIG.get("enabled"):
        save_file_to_github("bets/bets.csv", df.to_csv(index=False),
                           f"Add bet: {pick} @ {odds}", GITHUB_CONFIG, sha)
    
    return True

def get_open_bets():
    """Récupère les paris ouverts (avec sync GitHub)"""
    bets_file = BETS_DIR / "bets.csv"
    
    # ✅ Priorité GitHub sur Streamlit Cloud
    if GITHUB_CONFIG.get("enabled"):
        df, sha = load_csv_from_github("bets/bets.csv", GITHUB_CONFIG)
        if df is not None:
            return df[df["status"] == "open"]
    
    if not bets_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(bets_file)
    return df[df["status"] == "open"]

def close_bet(bet_id, result):
    """Clôture un pari (avec sync GitHub)"""
    bets_file = BETS_DIR / "bets.csv"
    sha = None
    
    # ✅ Charger depuis GitHub si activé
    if GITHUB_CONFIG.get("enabled"):
        df, sha = load_csv_from_github("bets/bets.csv", GITHUB_CONFIG)
        if df is None:
            return False
    elif bets_file.exists():
        df = pd.read_csv(bets_file)
    else:
        return False
    
    if bet_id not in df["bet_id"].values:
        return False
    
    bet = df[df["bet_id"] == bet_id].iloc[0]
    stake = float(bet["stake"])
    odds = float(bet["odds"])
    
    if result == "win":
        profit = stake * (odds - 1)
    elif result == "loss":
        profit = -stake
    else:  # cancelled / push
        profit = 0
    
    roi = (profit / stake) * 100 if stake > 0 else 0
    
    df.loc[df["bet_id"] == bet_id, "status"] = "closed"
    df.loc[df["bet_id"] == bet_id, "result"] = result
    df.loc[df["bet_id"] == bet_id, "profit"] = profit
    df.loc[df["bet_id"] == bet_id, "roi"] = roi
    
    # Sauvegarder localement
    df.to_csv(bets_file, index=False)
    
    # ✅ Sync GitHub
    if GITHUB_CONFIG.get("enabled"):
        save_file_to_github("bets/bets.csv", df.to_csv(index=False),
                           f"Close bet #{bet_id}: {result}", GITHUB_CONFIG, sha)
    
    return True

# ============================================================================
# INTERFACE - PAGE ACCUEIL
# ============================================================================

def show_home_page(model_data=None):
    """Affiche la page d'accueil"""
    
    # Calculer les stats dynamiquement
    n_fighters = len(model_data.get('elo_dict', {})) if model_data else 0
    
    st.markdown("""
    <div class="section-fade-in" style="text-align: center; padding: 50px 0;">
        <h1>🥊 Application de Paris Sportifs 🥊</h1>
        <p style="font-size: 1.3rem; color: #888;">
            Modèle ML sans data leakage - Stratégie réaliste validée
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance du Modèle (Market + Reach + Age)")
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">17-20%</div>
            <div class="metric-label">ROI Backtest</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">11-12/12</div>
            <div class="metric-label">Années profit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">34-40%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{n_fighters}</div>
            <div class="metric-label">Combattants</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Fonctionnalités")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div class="card">
            <h3 style="color: var(--primary-blue);">📅 Événements à venir</h3>
            <p>Consultez les prochains combats UFC avec recommandations de paris automatiques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="card">
            <h3 style="color: var(--success-color);">💰 Gestion de Bankroll</h3>
            <p>Suivez vos paris et gérez votre bankroll avec la stratégie Kelly optimisée</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="card">
            <h3 style="color: var(--warning-color);">🏆 Classement Elo</h3>
            <p>Consultez le classement des combattants par rating Elo</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📖 Comment utiliser")
    
    st.markdown("""
    <div class="card">
        <ol style="line-height: 2;">
            <li><b>Événements à venir</b> : Récupérez les prochains combats et obtenez des recommandations de paris</li>
            <li><b>Saisissez les cotes</b> : Entrez les cotes proposées par votre bookmaker</li>
            <li><b>Suivez les recommandations</b> : L'application calcule automatiquement les mises optimales selon Kelly</li>
            <li><b>Enregistrez vos paris</b> : Ajoutez les paris à votre historique pour suivre vos performances</li>
            <li><b>Mettez à jour les résultats</b> : Après les combats, enregistrez les résultats pour suivre votre ROI</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
                padding: 20px; border-radius: 12px; margin-top: 30px; border-left: 3px solid var(--warning-color);">
        <h3 style="color: var(--warning-color); margin-top: 0;">⚠️ Avertissement</h3>
        <p>Les paris sportifs comportent des risques. Cette application fournit des recommandations basées sur 
        des modèles statistiques mais ne garantit pas les résultats. Pariez de manière responsable.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INTERFACE - ÉVÉNEMENTS À VENIR
# ============================================================================

def show_events_page(model_data, fighters_data, current_bankroll):
    """Affiche la page des événements à venir"""
    
    st.title("📅 Événements UFC à venir")
    
    # Boutons principaux
    btn_cols = st.columns([2, 2, 1])
    
    with btn_cols[0]:
        if st.button("🔄 Récupérer les événements", type="primary"):
            with st.spinner("Récupération des événements..."):
                events = get_upcoming_events()
                st.session_state.events = events
                
                if events:
                    st.success(f"✅ {len(events)} événements récupérés")
                else:
                    st.error("❌ Aucun événement trouvé")
    
    with btn_cols[1]:
        if st.button("💰 Récupérer cotes (API)", help="Récupère automatiquement les cotes MMA depuis The Odds API"):
            with st.spinner("Récupération des cotes..."):
                odds_data, message = fetch_mma_odds()
                if odds_data:
                    st.session_state.api_odds = odds_data
                    st.success(message)
                else:
                    st.warning(message)
    
    # Afficher les cotes disponibles si récupérées
    if 'api_odds' in st.session_state and st.session_state.api_odds:
        with st.expander(f"📊 Cotes API disponibles ({len(st.session_state.api_odds)} combats)", expanded=False):
            for event_key, event_data in st.session_state.api_odds.items():
                odds = event_data.get('odds', {})
                bookmaker = event_data.get('bookmaker', '')
                fighters_str = " | ".join([f"{f}: {o:.2f}" for f, o in odds.items()])
                st.write(f"**{event_key}** ({bookmaker}): {fighters_str}")
    
    if 'events' in st.session_state and st.session_state.events:
        
        st.markdown("### ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_name = st.selectbox(
                "Stratégie de paris",
                options=list(BETTING_STRATEGIES.keys()),
                index=0
            )
            strategy = BETTING_STRATEGIES[strategy_name]
            
            st.info(f"📝 {strategy['description']}")
        
        with col2:
            st.metric("💰 Bankroll actuelle", f"{current_bankroll:.2f} €")
        
        with st.expander("📊 Détails de la stratégie"):
            param_cols = st.columns(3)
            with param_cols[0]:
                st.metric("Confiance min", f"{strategy['min_confidence']:.0%}")
                st.metric("Edge min", f"{strategy['min_edge']:.1%}")
            with param_cols[1]:
                st.metric("Kelly fraction", f"1/{strategy['kelly_fraction']}")
                st.metric("Mise max", f"{strategy['max_bet_fraction']:.0%}")
            with param_cols[2]:
                st.metric("Mise min", f"{strategy['min_bet_pct']:.1%}")
        
        tabs = st.tabs([event['name'] for event in st.session_state.events])
        
        for i, (event, tab) in enumerate(zip(st.session_state.events, tabs)):
            with tab:
                st.subheader(f"🥊 {event['name']}")
                
                if st.button(f"Charger les combats", key=f"load_fights_{i}"):
                    with st.spinner("Récupération des combats..."):
                        fights = extract_fights_from_event(event['url'])
                        st.session_state[f"fights_{i}"] = fights
                        
                        if fights:
                            st.success(f"✅ {len(fights)} combats chargés")
                        else:
                            st.warning("⚠️ Aucun combat trouvé")
                
                if f"fights_{i}" in st.session_state:
                    fights = st.session_state[f"fights_{i}"]
                    
                    if fights:
                        st.markdown("---")
                        st.markdown("### 🎯 Recommandations de paris")
                        
                        for j, fight in enumerate(fights):
                            st.markdown(f"#### Combat {j+1}")
                            
                            # ✅ Utiliser la fonction avec fallback par nom
                            fighter_a_data = get_fighter_data_with_fallback(
                                fight['red_url'], 
                                fight['red_fighter'], 
                                fighters_data, 
                                model_data
                            )
                            
                            fighter_b_data = get_fighter_data_with_fallback(
                                fight['blue_url'], 
                                fight['blue_fighter'], 
                                fighters_data, 
                                model_data
                            )
                            
                            # ✅ Détecter les nouveaux combattants (Elo = 1500)
                            elo_a = fighter_a_data.get('elo_global', BASE_ELO)
                            elo_b = fighter_b_data.get('elo_global', BASE_ELO)
                            is_new_fighter_a = abs(elo_a - BASE_ELO) < 1  # Elo ~= 1500
                            is_new_fighter_b = abs(elo_b - BASE_ELO) < 1
                            has_new_fighter = is_new_fighter_a or is_new_fighter_b
                            
                            fight_cols = st.columns(2)
                            
                            with fight_cols[0]:
                                new_badge_a = " 🆕" if is_new_fighter_a else ""
                                elo_display_a = f"Elo: {elo_a:.0f}" if not is_new_fighter_a else "Elo: 1500 (nouveau)"
                                st.markdown(f"""
                                <div class="fighter-card fighter-card-red">
                                    <h4>🔴 {fight['red_fighter']}{new_badge_a}</h4>
                                    <p>{elo_display_a}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 📜 Derniers combats du combattant rouge
                                fighter_a_id = id_from_url(fight['red_url']) if fight.get('red_url') else None
                                if fighter_a_id and not is_new_fighter_a:
                                    recent_a = get_fighter_recent_fights(fighter_a_id, 3)
                                    if recent_a:
                                        history_html = "<div style='font-size: 0.85em; margin-top: 5px;'><b>📜 Derniers combats:</b><br>"
                                        for f in recent_a:
                                            emoji = "✅" if f['result'] == 'W' else "❌"
                                            history_html += f"{emoji} vs {f['opponent']} ({f['date']})<br>"
                                        history_html += "</div>"
                                        st.markdown(history_html, unsafe_allow_html=True)
                            
                            with fight_cols[1]:
                                new_badge_b = " 🆕" if is_new_fighter_b else ""
                                elo_display_b = f"Elo: {elo_b:.0f}" if not is_new_fighter_b else "Elo: 1500 (nouveau)"
                                st.markdown(f"""
                                <div class="fighter-card fighter-card-blue">
                                    <h4>🔵 {fight['blue_fighter']}{new_badge_b}</h4>
                                    <p>{elo_display_b}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 📜 Derniers combats du combattant bleu
                                fighter_b_id = id_from_url(fight['blue_url']) if fight.get('blue_url') else None
                                if fighter_b_id and not is_new_fighter_b:
                                    recent_b = get_fighter_recent_fights(fighter_b_id, 3)
                                    if recent_b:
                                        history_html = "<div style='font-size: 0.85em; margin-top: 5px;'><b>📜 Derniers combats:</b><br>"
                                        for f in recent_b:
                                            emoji = "✅" if f['result'] == 'W' else "❌"
                                            history_html += f"{emoji} vs {f['opponent']} ({f['date']})<br>"
                                        history_html += "</div>"
                                        st.markdown(history_html, unsafe_allow_html=True)
                            
                            # ⚠️ Avertissement si nouveau combattant
                            if has_new_fighter:
                                new_fighters = []
                                if is_new_fighter_a:
                                    new_fighters.append(fight['red_fighter'])
                                if is_new_fighter_b:
                                    new_fighters.append(fight['blue_fighter'])
                                st.warning(f"⚠️ **Nouveau(x) combattant(s) détecté(s)** : {', '.join(new_fighters)}. "
                                          f"Elo par défaut (1500) = manque de données historiques. **Pari non recommandé.**")
                            
                            # 💵 D'abord entrer les cotes (nécessaires pour le nouveau modèle)
                            st.markdown("##### 💵 Cotes du bookmaker")
                            
                            # ✅ Chercher les cotes automatiques depuis l'API
                            api_odds_info = None
                            default_odds_a = 2.0
                            default_odds_b = 2.0
                            
                            if 'api_odds' in st.session_state and st.session_state.api_odds:
                                api_odds_info = find_fight_odds(
                                    fight['red_fighter'], 
                                    fight['blue_fighter'], 
                                    st.session_state.api_odds
                                )
                                if api_odds_info:
                                    default_odds_a = api_odds_info['odds_a']
                                    default_odds_b = api_odds_info['odds_b']
                            
                            # Afficher info si cotes trouvées automatiquement
                            if api_odds_info:
                                st.success(f"🔄 Cotes auto ({api_odds_info['bookmaker']}): "
                                          f"{api_odds_info['matched_a']} @ {api_odds_info['odds_a']:.2f} | "
                                          f"{api_odds_info['matched_b']} @ {api_odds_info['odds_b']:.2f}")
                            
                            odds_cols = st.columns(2)
                            
                            with odds_cols[0]:
                                odds_a = st.number_input(
                                    f"Cote {fight['red_fighter']}",
                                    min_value=1.01,
                                    max_value=50.0,
                                    value=default_odds_a,
                                    step=0.01,
                                    key=f"odds_a_{i}_{j}"
                                )
                            
                            with odds_cols[1]:
                                odds_b = st.number_input(
                                    f"Cote {fight['blue_fighter']}",
                                    min_value=1.01,
                                    max_value=50.0,
                                    value=default_odds_b,
                                    step=0.01,
                                    key=f"odds_b_{i}_{j}"
                                )
                            
                            # Prédiction avec les cotes (nouveau modèle market + physique)
                            prediction = predict_fight(fighter_a_data, fighter_b_data, model_data, odds_a, odds_b)
                            
                            if prediction:
                                # Calcul des probabilités implicites du marché
                                proba_market_a = 1 / odds_a
                                proba_market_b = 1 / odds_b
                                
                                st.markdown(f"""
                                <div class="card">
                                    <h5>📊 Prédiction du modèle (mkt+phys)</h5>
                                    <table style="width:100%; text-align:center;">
                                        <tr>
                                            <th></th>
                                            <th>🔴 {fight['red_fighter']}</th>
                                            <th>🔵 {fight['blue_fighter']}</th>
                                        </tr>
                                        <tr>
                                            <td><b>Modèle</b></td>
                                            <td style="color: {'green' if prediction['proba_a'] > proba_market_a else 'red'};">{prediction['proba_a']:.1%}</td>
                                            <td style="color: {'green' if prediction['proba_b'] > proba_market_b else 'red'};">{prediction['proba_b']:.1%}</td>
                                        </tr>
                                        <tr>
                                            <td><b>Marché</b></td>
                                            <td>{proba_market_a:.1%}</td>
                                            <td>{proba_market_b:.1%}</td>
                                        </tr>
                                        <tr>
                                            <td><b>Edge</b></td>
                                            <td style="color: {'green' if prediction['proba_a'] - proba_market_a > 0 else 'red'};">{(prediction['proba_a'] - proba_market_a)*100:+.1f}%</td>
                                            <td style="color: {'green' if prediction['proba_b'] - proba_market_b > 0 else 'red'};">{(prediction['proba_b'] - proba_market_b)*100:+.1f}%</td>
                                        </tr>
                                    </table>
                                    <p style="margin-top: 10px;"><small>Reach diff: {prediction.get('reach_diff', 'N/A')} cm | Age diff: {prediction.get('age_diff', 'N/A')} ans</small></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                stake_a = calculate_kelly_stake(
                                    prediction['proba_a'],
                                    odds_a,
                                    current_bankroll,
                                    strategy
                                )
                                
                                stake_b = calculate_kelly_stake(
                                    prediction['proba_b'],
                                    odds_b,
                                    current_bankroll,
                                    strategy
                                )
                                
                                # ⚠️ Warning si données bio manquantes (mais pas bloquant)
                                has_bio_warning = False
                                if prediction.get('reach_diff') == 0 and prediction.get('age_diff') in [0, 0.5, None]:
                                    has_bio_warning = True
                                    st.warning("⚠️ **Données physiques incomplètes** : reach/age utilisent les médianes. L'edge est basé principalement sur les cotes du marché.")
                                
                                # ✅ NOUVELLE LOGIQUE: Parier sur le combattant avec edge ≥ seuil (pas juste le favori)
                                best_bet = None
                                
                                # Vérifier si A a un edge suffisant
                                if stake_a['should_bet']:
                                    best_bet = {
                                        'fighter': fight['red_fighter'],
                                        'stake_info': stake_a,
                                        'odds': odds_a,
                                        'proba': prediction['proba_a'],
                                        'color': '🔴'
                                    }
                                
                                # Vérifier si B a un edge suffisant (et meilleur que A)
                                if stake_b['should_bet']:
                                    if best_bet is None or stake_b['edge'] > best_bet['stake_info']['edge']:
                                        best_bet = {
                                            'fighter': fight['blue_fighter'],
                                            'stake_info': stake_b,
                                            'odds': odds_b,
                                            'proba': prediction['proba_b'],
                                            'color': '🔵'
                                        }
                                
                                if best_bet:
                                    st.markdown(f"""
                                    <div class="bet-recommendation">
                                        <h5>✅ RECOMMANDATION DE PARI</h5>
                                        <p><b>Parier sur:</b> {best_bet['color']} {best_bet['fighter']}</p>
                                        <p><b>Cote:</b> {best_bet['odds']:.2f}</p>
                                        <p><b>Mise recommandée:</b> {best_bet['stake_info']['stake']:.2f} €</p>
                                        <p><b>Edge:</b> {best_bet['stake_info']['edge']:.1%}</p>
                                        <p><b>EV:</b> {best_bet['stake_info']['ev']:.1%}</p>
                                        <p><b>% Bankroll:</b> {best_bet['stake_info']['kelly_pct']:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # 🔒 Bouton d'enregistrement uniquement pour utilisateurs connectés
                                    if can_access_betting():
                                        if st.button(f"💾 Enregistrer ce pari", key=f"save_bet_{i}_{j}"):
                                            success = add_bet(
                                                event_name=event['name'],
                                                fighter_red=fight['red_fighter'],
                                                fighter_blue=fight['blue_fighter'],
                                                pick=best_bet['fighter'],
                                                odds=best_bet['odds'],
                                                stake=best_bet['stake_info']['stake'],
                                                model_probability=best_bet['proba'],
                                                kelly_fraction=strategy['kelly_fraction'],
                                                edge=best_bet['stake_info']['edge'],
                                                ev=best_bet['stake_info']['ev']
                                            )
                                            
                                            if success:
                                                st.success(f"✅ Pari enregistré : {best_bet['stake_info']['stake']:.2f}€ sur {best_bet['fighter']}")
                                            else:
                                                st.error("❌ Erreur lors de l'enregistrement")
                                    else:
                                        st.info("🔒 Connectez-vous pour enregistrer ce pari")
                                else:
                                    st.info(f"ℹ️ Aucun pari recommandé (edge < {strategy['min_edge']:.1%} pour les deux combattants)")
                                    
                                    with st.expander("Voir les détails"):
                                        st.write(f"**{fight['red_fighter']}**: Edge {stake_a['edge']:.1%}")
                                        if stake_a.get('reason'):
                                            st.write(f"  → {stake_a['reason']}")
                                        st.write(f"**{fight['blue_fighter']}**: Edge {stake_b['edge']:.1%}")
                                        if stake_b.get('reason'):
                                            st.write(f"  → {stake_b['reason']}")
                            
                            st.markdown("---")
                    else:
                        st.info("Cliquez sur 'Charger les combats' pour voir les affrontements")

# ============================================================================
# INTERFACE - GESTION BANKROLL
# ============================================================================

def show_bankroll_page(current_bankroll):
    """Affiche la page de gestion de bankroll"""
    
    st.title("💰 Gestion de la Bankroll")
    
    # 🔒 Vérification des permissions
    if not can_view_bankroll():
        st.error("🔒 **Accès refusé** - Connectez-vous pour accéder à cette page")
        st.info("👈 Utilisez le formulaire de connexion dans la barre latérale")
        return
    
    # Afficher le profil connecté
    user = get_current_user()
    if user:
        st.success(f"📊 Bankroll de **{user['display_name']}**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💵 Bankroll actuelle", f"{current_bankroll:.2f} €")
    
    st.markdown("### ⚙️ Ajuster la bankroll")
    
    adj_cols = st.columns([2, 1, 1])
    
    with adj_cols[0]:
        adjustment = st.number_input(
            "Montant de l'ajustement (€)",
            min_value=-current_bankroll,
            max_value=10000.0,
            value=0.0,
            step=10.0
        )
    
    with adj_cols[1]:
        action = st.selectbox("Action", ["Dépôt", "Retrait"])
    
    with adj_cols[2]:
        if st.button("✅ Valider", type="primary"):
            if adjustment != 0:
                if action == "Retrait":
                    adjustment = -abs(adjustment)
                else:
                    adjustment = abs(adjustment)
                
                new_bankroll = current_bankroll + adjustment
                
                if new_bankroll < 0:
                    st.error("❌ La bankroll ne peut pas être négative")
                else:
                    update_bankroll(
                        new_bankroll,
                        action.lower(),
                        f"{action} de {abs(adjustment):.2f}€"
                    )
                    st.success(f"✅ Bankroll mise à jour : {new_bankroll:.2f}€")
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Paris en cours")
    
    open_bets = get_open_bets()
    
    if not open_bets.empty:
        
        total_stake = open_bets['stake'].sum()
        potential_profit = ((open_bets['odds'] - 1) * open_bets['stake']).sum()
        
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("📊 Nombre de paris", len(open_bets))
        with metric_cols[1]:
            st.metric("💵 Mise totale", f"{total_stake:.2f} €")
        with metric_cols[2]:
            st.metric("🎯 Profit potentiel", f"{potential_profit:.2f} €")
        
        for idx, bet in open_bets.iterrows():
            with st.expander(f"Pari #{int(bet['bet_id'])} - {bet['pick']} @ {bet['odds']:.2f}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Événement:** {bet['event']}")
                    st.write(f"**Combat:** {bet['fighter_red']} vs {bet['fighter_blue']}")
                    st.write(f"**Sélection:** {bet['pick']}")
                    st.write(f"**Cote:** {bet['odds']:.2f}")
                    st.write(f"**Mise:** {bet['stake']:.2f} €")
                
                with col2:
                    st.write(f"**Date:** {bet['date']}")
                    st.write(f"**Probabilité:** {bet['model_probability']:.1%}")
                    st.write(f"**Edge:** {bet['edge']:.1%}")
                    st.write(f"**EV:** {bet['ev']:.1%}")
                    st.write(f"**Kelly:** 1/{int(bet['kelly_fraction'])}")
                
                st.markdown("**Résultat du combat:**")
                result_cols = st.columns(3)
                
                with result_cols[0]:
                    if st.button("✅ Victoire", key=f"win_{int(bet['bet_id'])}"):
                        if close_bet(int(bet['bet_id']), "win"):
                            profit = bet['stake'] * (bet['odds'] - 1)
                            new_bankroll = current_bankroll + profit
                            update_bankroll(new_bankroll, "win", f"Pari #{int(bet['bet_id'])} gagné")
                            st.success(f"✅ Pari gagné ! +{profit:.2f}€")
                            time.sleep(0.5)  # Attendre sync GitHub
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de la clôture du pari")
                
                with result_cols[1]:
                    if st.button("❌ Défaite", key=f"loss_{int(bet['bet_id'])}"):
                        if close_bet(int(bet['bet_id']), "loss"):
                            new_bankroll = current_bankroll - bet['stake']
                            update_bankroll(new_bankroll, "loss", f"Pari #{int(bet['bet_id'])} perdu")
                            st.warning(f"❌ Pari perdu ! -{bet['stake']:.2f}€")
                            time.sleep(0.5)  # Attendre sync GitHub
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de la clôture du pari")
                
                with result_cols[2]:
                    if st.button("⚪ Annulé", key=f"void_{int(bet['bet_id'])}"):
                        if close_bet(int(bet['bet_id']), "void"):
                            st.info("⚪ Pari annulé")
                            time.sleep(0.5)  # Attendre sync GitHub
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de l'annulation du pari")
    
    else:
        st.info("📭 Aucun pari en cours")
    
    st.markdown("---")
    st.markdown("### 📊 Historique des paris")
    
    # ✅ Charger depuis GitHub si activé
    all_bets = None
    if GITHUB_CONFIG.get("enabled"):
        all_bets, _ = load_csv_from_github("bets/bets.csv", GITHUB_CONFIG)
    
    if all_bets is None:
        bets_file = BETS_DIR / "bets.csv"
        if bets_file.exists():
            all_bets = pd.read_csv(bets_file)
    
    if all_bets is not None and not all_bets.empty:
        closed_bets = all_bets[all_bets['status'] == 'closed']
        
        if not closed_bets.empty:
            
            total_bets = len(closed_bets)
            wins = len(closed_bets[closed_bets['result'] == 'win'])
            losses = len(closed_bets[closed_bets['result'] == 'loss'])
            win_rate = wins / total_bets if total_bets > 0 else 0
            
            total_profit = closed_bets['profit'].sum()
            total_staked = closed_bets['stake'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            st.markdown("#### 📈 Statistiques globales")
            
            stats_cols = st.columns(5)
            
            with stats_cols[0]:
                st.metric("Paris total", total_bets)
            with stats_cols[1]:
                st.metric("Victoires", wins)
            with stats_cols[2]:
                st.metric("Défaites", losses)
            with stats_cols[3]:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with stats_cols[4]:
                st.metric("ROI", f"{roi:.1f}%", delta=f"{total_profit:.2f}€")
            
            st.markdown("#### 📉 Évolution du profit")
            
            closed_bets = closed_bets.sort_values('date')
            closed_bets['cumulative_profit'] = closed_bets['profit'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(closed_bets) + 1)),
                y=closed_bets['cumulative_profit'],
                mode='lines+markers',
                name='Profit cumulé',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Évolution du profit cumulé",
                xaxis_title="Nombre de paris",
                yaxis_title="Profit (€)",
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📋 Détails des paris fermés")
            
            display_df = closed_bets[[
                'bet_id', 'date', 'event', 'pick', 'odds',
                'stake', 'result', 'profit', 'roi'
            ]].copy()
            
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['stake'] = display_df['stake'].apply(lambda x: f"{x:.2f}€")
            display_df['profit'] = display_df['profit'].apply(lambda x: f"{x:.2f}€")
            display_df['roi'] = display_df['roi'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("📭 Aucun pari fermé pour le moment")
    else:
        st.info("📭 Aucun historique disponible")

# ============================================================================
# INTERFACE - CLASSEMENT ELO
# ============================================================================

def show_rankings_page(model_data):
    """Affiche le classement des combattants par Elo"""
    
    st.title("🏆 Classement des combattants (Elo)")
    
    if model_data["ratings"] is not None and not model_data["ratings"].empty:
        
        ratings_df = model_data["ratings"].copy()
        
        # Détecter le format du fichier ratings
        if 'fighter_1_id' in ratings_df.columns and 'fighter_2_id' in ratings_df.columns:
            # ✅ Nouveau format: fighter_1_id, fighter_2_id, elo_1_post, elo_2_post
            latest_ratings = {}
            id_to_name = {}
            
            ratings_sorted = ratings_df.sort_values('event_date')
            
            for _, row in ratings_sorted.iterrows():
                f1_id = row.get('fighter_1_id')
                f1_name = row.get('fighter_1', f1_id)
                f1_elo = row.get('elo_1_post', row.get('elo_1_pre', BASE_ELO))
                
                f2_id = row.get('fighter_2_id')
                f2_name = row.get('fighter_2', f2_id)
                f2_elo = row.get('elo_2_post', row.get('elo_2_pre', BASE_ELO))
                
                if f1_id and pd.notna(f1_id):
                    latest_ratings[f1_id] = f1_elo
                    id_to_name[f1_id] = f1_name
                
                if f2_id and pd.notna(f2_id):
                    latest_ratings[f2_id] = f2_elo
                    id_to_name[f2_id] = f2_name
            
            ranking_data = []
            for fighter_id, elo in latest_ratings.items():
                ranking_data.append({
                    'fighter_id': fighter_id,
                    'fighter_name': id_to_name.get(fighter_id, fighter_id),
                    'elo': elo
                })
            
            ranking_df = pd.DataFrame(ranking_data)
        
        elif 'fa' in ratings_df.columns and 'fb' in ratings_df.columns:
            # Ancien format ratings_timeseries (fa, fb, elo_global_fa_post, etc.)
            id_to_name = {}
            
            # D'abord, utiliser les noms directement depuis ratings_timeseries si disponibles
            if 'fa_name' in ratings_df.columns and 'fb_name' in ratings_df.columns:
                for _, row in ratings_df.iterrows():
                    if row.get('fa') and row.get('fa_name'):
                        id_to_name[row['fa']] = row['fa_name']
                    if row.get('fb') and row.get('fb_name'):
                        id_to_name[row['fb']] = row['fb_name']
            
            # Fallback: charger les noms depuis asof_full ou appearances
            if not id_to_name:
                asof_path = INTERIM_DIR / "asof_full.parquet"
                appearances_path = RAW_DIR / "appearances.parquet"
                
                if asof_path.exists():
                    try:
                        asof_df = pd.read_parquet(asof_path)
                        if not asof_df.empty and 'fighter_id' in asof_df.columns:
                            for _, row in asof_df.iterrows():
                                fighter_id = row.get('fighter_id')
                                fighter_name = row.get('fighter_name', fighter_id)
                                if fighter_id:
                                    id_to_name[fighter_id] = fighter_name
                    except:
                        pass
                
                if not id_to_name and appearances_path.exists():
                    try:
                        app_df = pd.read_parquet(appearances_path)
                        for _, row in app_df.iterrows():
                            fighter_id = row.get('fighter_id')
                            fighter_name = row.get('fighter_name', fighter_id)
                            if fighter_id:
                                id_to_name[fighter_id] = fighter_name
                    except:
                        pass
            
            # Obtenir le dernier Elo POST de chaque combattant
            latest_ratings = []
            
            for fighter_id in ratings_df['fa'].unique():
                last_fight = ratings_df[ratings_df['fa'] == fighter_id].iloc[-1]
                fighter_name = id_to_name.get(fighter_id, fighter_id)
                latest_ratings.append({
                    'fighter_id': fighter_id,
                    'fighter_name': fighter_name,
                    'elo': last_fight['elo_global_fa_post']
                })
            
            for fighter_id in ratings_df['fb'].unique():
                if fighter_id not in [r['fighter_id'] for r in latest_ratings]:
                    last_fight = ratings_df[ratings_df['fb'] == fighter_id].iloc[-1]
                    fighter_name = id_to_name.get(fighter_id, fighter_id)
                    latest_ratings.append({
                        'fighter_id': fighter_id,
                        'fighter_name': fighter_name,
                        'elo': last_fight['elo_global_fb_post']
                    })
            
            ranking_df = pd.DataFrame(latest_ratings)
        
        else:
            # asof_full format (fighter_id, elo_global_pre)
            latest_elo = {}
            id_to_name = {}
            
            if 'fighter_id' not in ratings_df.columns:
                st.warning("Format de données non reconnu")
                return
            
            for fighter_id in ratings_df['fighter_id'].unique():
                if pd.isna(fighter_id):
                    continue
                fighter_data = ratings_df[ratings_df['fighter_id'] == fighter_id].iloc[-1]
                elo = fighter_data.get('elo_global_pre', BASE_ELO)
                name = fighter_data.get('fighter_name', fighter_id)
                latest_elo[fighter_id] = elo
                id_to_name[fighter_id] = name
            
            ranking_data = []
            for fighter_id, elo in latest_elo.items():
                ranking_data.append({
                    'fighter_id': fighter_id,
                    'fighter_name': id_to_name.get(fighter_id, fighter_id),
                    'elo': elo
                })
            
            ranking_df = pd.DataFrame(ranking_data)
        
        ranking_df = ranking_df.sort_values('elo', ascending=False).reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            search = st.text_input("🔍 Rechercher un combattant", "")
        
        with col2:
            top_n = st.slider("Afficher le top", 10, 100, 50, 10)
        
        if search:
            mask = ranking_df['fighter_name'].str.contains(search, case=False, na=False)
            display_df = ranking_df[mask].head(top_n)
        else:
            display_df = ranking_df.head(top_n)
        
        st.markdown(f"### Top {len(display_df)} combattants")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=display_df['fighter_name'],
            y=display_df['elo'],
            marker=dict(
                color=display_df['elo'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Elo")
            ),
            text=display_df['elo'].apply(lambda x: f"{x:.0f}"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Top {len(display_df)} - Classement Elo",
            xaxis_title="Combattant",
            yaxis_title="Rating Elo",
            height=600,
            template='plotly_dark',
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            display_df[['fighter_name', 'elo']].rename(columns={
                'fighter_name': 'Combattant',
                'elo': 'Rating Elo'
            }),
            use_container_width=True
        )
    
    else:
        st.warning("⚠️ Aucune donnée de rating disponible.")

# ============================================================================
# INTERFACE - MISE À JOUR DES STATS
# ============================================================================

def show_stats_update_page():
    """Affiche la page de mise à jour des statistiques"""
    
    st.title("🔄 Mise à jour des données")
    
    # ✅ Bouton pour vider le cache
    col_cache1, col_cache2 = st.columns([3, 1])
    with col_cache2:
        if st.button("🗑️ Vider le cache", help="Force le rechargement des données"):
            st.cache_data.clear()
            st.success("✅ Cache vidé ! Rechargez la page.")
            st.rerun()
    
    # ✅ Vérification LOCALE rapide (pas de scraping)
    st.markdown("### 📊 État des données locales")
    
    freshness = check_data_freshness()
    
    # Afficher le message principal
    st.info(freshness['message'])
    
    # Afficher les métriques si on a des données
    if freshness['has_data']:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if freshness['last_event_date'] is not None and pd.notna(freshness['last_event_date']):
                try:
                    date_str = freshness['last_event_date'].strftime('%Y-%m-%d')
                except:
                    date_str = str(freshness['last_event_date'])[:10]
                st.metric("📅 Dernier événement", date_str)
            else:
                st.metric("📅 Dernier événement", "N/A")
        
        with col2:
            if freshness['days_old'] is not None and pd.notna(freshness['days_old']):
                st.metric("🕐 Âge", f"{int(freshness['days_old'])} jours")
            else:
                st.metric("🕐 Âge", "N/A")
        
        with col3:
            st.metric("🥊 Combats", freshness['fight_count'])
        
        with col4:
            st.metric("👤 Combattants", freshness['fighter_count'])
    
    st.markdown("---")
    st.markdown("### 🔄 Mettre à jour les données")
    
    st.markdown("""
    > 💡 **Cliquez sur le bouton ci-dessous** pour vérifier s'il y a de nouveaux événements UFC 
    > et mettre à jour automatiquement vos données.
    """)
    
    if st.button("🚀 Lancer la mise à jour", type="primary", use_container_width=True):
        
        progress_placeholder = st.empty()
        
        def update_progress(message):
            progress_placeholder.info(message)
        
        try:
            with st.spinner("🔍 Connexion à UFC Stats et recherche de nouveaux événements..."):
                new_data = scrape_new_events(progress_callback=update_progress)
            
            if new_data['count'] == 0:
                # Vérifier si ratings_timeseries est en retard par rapport à appearances
                appearances_df = pd.read_parquet(RAW_DIR / "appearances.parquet")
                ratings_df = pd.read_parquet(INTERIM_DIR / "ratings_timeseries.parquet")
                app_date = pd.to_datetime(appearances_df['event_date']).max()
                rat_date = pd.to_datetime(ratings_df['event_date']).max()
                
                if app_date > rat_date:
                    st.info(f"📊 Les ratings Elo sont en retard ({rat_date.strftime('%Y-%m-%d')} vs {app_date.strftime('%Y-%m-%d')}). Recalcul...")
                    update_progress("🎯 Recalcul des features et des ratings Elo...")
                    result = recalculate_features_and_elo(progress_callback=update_progress)
                    st.cache_data.clear()
                    st.success(f"✅ Ratings recalculés ! ({result['appearances_count']} combats, {result['fighters_count']} combattants)")
                else:
                    st.success("✅ Aucun nouveau combat à ajouter. Vos données sont à jour !")
            else:
                st.success(f"✅ {new_data['count']} nouveaux combats trouvés !")
                
                with st.expander(f"Voir les {new_data['count']} nouveaux combats"):
                    for fight in new_data['new_fights'][:10]:
                        st.write(f"🥊 {fight['red_fighter']} vs {fight['blue_fighter']} - {fight.get('event_date', 'Date inconnue')}")
                    
                    if len(new_data['new_fights']) > 10:
                        st.write(f"... et {len(new_data['new_fights']) - 10} autres combats")
                
                update_progress("💾 Intégration des nouvelles données...")
                update_data_files(new_data['new_appearances'])
                
                update_progress("🎯 Recalcul des features et des ratings Elo...")
                result = recalculate_features_and_elo(progress_callback=update_progress)
                
                # ✅ Vider le cache pour recharger les nouvelles données
                st.cache_data.clear()
                
                st.success("✅ Mise à jour terminée avec succès !")
                
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.metric("📊 Combats total", result['appearances_count'])
                with stats_cols[1]:
                    st.metric("🥊 Combattants", result['fighters_count'])
                with stats_cols[2]:
                    st.metric("🆕 Nouveaux ajoutés", new_data['count'])
                
                st.info("💡 Rechargez la page (F5) pour voir les nouvelles données")
                
                if st.button("🔄 Recharger l'application", type="primary"):
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la mise à jour : {str(e)}")
            st.exception(e)
        
        finally:
            progress_placeholder.empty()
    
    st.markdown("---")
    st.markdown("### ⚙️ Recalcul manuel complet")
    
    st.warning("""
    ⚠️ Utilisez cette option uniquement si vous avez modifié manuellement les fichiers de données.
    Cela va recalculer toutes les features et tous les Elo depuis le début.
    """)
    
    if st.button("🔄 Recalculer toutes les features et Elo", use_container_width=True):
        progress_placeholder = st.empty()
        
        def update_progress(message):
            progress_placeholder.info(message)
        
        try:
            with st.spinner("Recalcul en cours..."):
                result = recalculate_features_and_elo(progress_callback=update_progress)
            
            st.success("✅ Recalcul terminé !")
            
            stats_cols = st.columns(2)
            with stats_cols[0]:
                st.metric("📊 Combats total", result['appearances_count'])
            with stats_cols[1]:
                st.metric("🥊 Combattants", result['fighters_count'])
            
            st.info("💡 Rechargez la page (F5) pour voir les nouvelles données")
            
            if st.button("🔄 Recharger l'application maintenant", type="primary"):
                st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors du recalcul : {str(e)}")
            st.exception(e)
        
        finally:
            progress_placeholder.empty()
    
    st.markdown("---")
    st.markdown("""
    <div class="card">
        <h4>📖 Informations</h4>
        <ul>
            <li>Les données sont récupérées depuis <code>ufcstats.com</code></li>
            <li>Seuls les nouveaux événements sont scrapés pour économiser du temps</li>
            <li>Les ratings Elo sont recalculés automatiquement après chaque mise à jour</li>
            <li>Il est recommandé de mettre à jour les données après chaque événement UFC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    
    model_data = load_model_and_data()
    fighters_data = load_fighters_data()
    
    st.markdown('<div class="main-title">🥊 Combat Sports Betting App 🥊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modèle ML sans data leakage - Stratégies optimisées Grid Search + AG</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # SIDEBAR - CONNEXION UTILISATEUR
    # ============================================================================
    with st.sidebar:
        st.markdown("### 👤 Profil")
        
        current_user = get_current_user()
        
        if current_user:
            # Utilisateur connecté
            st.success(f"Connecté: {current_user['display_name']}")
            
            if st.button("🚪 Déconnexion", use_container_width=True):
                logout_user()
                st.rerun()
            
            # Afficher la bankroll si autorisé
            if can_view_bankroll():
                current_bankroll = init_bankroll()
                st.metric("💰 Bankroll", f"{current_bankroll:.2f} €")
        else:
            # Formulaire de connexion
            st.info("🔒 Connectez-vous pour accéder aux paris et à la bankroll")
            
            with st.form("login_form"):
                password = st.text_input("Mot de passe", type="password")
                submitted = st.form_submit_button("🔐 Connexion", use_container_width=True)
                
                if submitted and password:
                    username = authenticate_user(password)
                    if username:
                        st.session_state.logged_in_user = username
                        st.rerun()
                    else:
                        st.error("❌ Mot de passe incorrect")
            
            current_bankroll = 0  # Pas de bankroll pour les visiteurs
        
        st.markdown("---")
        
        # ============================================================================
        # SIDEBAR - CONFIGURATION API COTES
        # ============================================================================
        with st.expander("🔑 API Cotes (The Odds API)"):
            st.markdown("""
            **The Odds API** permet de récupérer automatiquement les cotes MMA.
            
            - 🆓 **Gratuit**: 500 requêtes/mois
            - 📊 Cotes de Pinnacle, Betfair, Unibet...
            - 🔗 [Obtenir une clé API](https://the-odds-api.com/#get-access)
            """)
            
            # Vérifier si une clé est déjà configurée
            current_key = get_odds_api_key()
            key_status = "✅ Configurée" if current_key else "❌ Non configurée"
            st.markdown(f"**Status:** {key_status}")
            
            if not current_key:
                if is_logged_in():
                    st.success("🔓 Clé API disponible (connecté)")
                else:
                    st.markdown("---")
                    st.markdown("**Saisir une clé manuellement:**")
                    
                    # Option pour tester une clé temporairement
                    temp_key = st.text_input("Clé API (temporaire)", type="password", key="temp_api_key")
                    if temp_key:
                        st.session_state.temp_odds_api_key = temp_key
                        st.success("Clé temporaire enregistrée pour cette session")
    
    # ============================================================================
    # ONGLETS PRINCIPAUX
    # ============================================================================
    
    # Définir les onglets selon le statut de connexion
    if is_logged_in() and can_view_bankroll():
        tabs = st.tabs([
            "🏠 Accueil",
            "📅 Événements à venir",
            "💰 Gestion Bankroll",
            "🏆 Classement Elo",
            "🔄 Mise à jour"
        ])
        
        with tabs[0]:
            show_home_page(model_data)
        
        with tabs[1]:
            show_events_page(model_data, fighters_data, current_bankroll)
        
        with tabs[2]:
            show_bankroll_page(current_bankroll)
        
        with tabs[3]:
            show_rankings_page(model_data)
        
        with tabs[4]:
            show_stats_update_page()
    else:
        # Mode visiteur - accès limité
        tabs = st.tabs([
            "🏠 Accueil",
            "📅 Événements à venir",
            "🏆 Classement Elo",
            "🔄 Mise à jour"
        ])
        
        with tabs[0]:
            show_home_page(model_data)
        
        with tabs[1]:
            # Mode lecture seule pour les visiteurs
            st.warning("🔒 **Mode visiteur** - Connectez-vous pour enregistrer des paris et gérer votre bankroll")
            show_events_page(model_data, fighters_data, 0)  # Bankroll = 0 pour visiteurs
        
        with tabs[2]:
            show_rankings_page(model_data)
        
        with tabs[3]:
            show_stats_update_page()

if __name__ == "__main__":
    main()