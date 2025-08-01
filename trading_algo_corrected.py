import pandas as pd
import requests
import time
from datetime import datetime

def long_trend_part():
    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
            
    def recency_weight(date):
        if not date:
            return 0.5
        try:
            year = pd.to_datetime(date).year 
            now = datetime.now().year
            delta = now - year
            return max(0.2, 1 - 0.2 * delta)
        except:
            return 0.5
            
    def get_diff(url, impact_if_rising, crit, parser=None):
        headers = {"User-Agent": "Mozilla/5.0"}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10).json()
                # Parsing pour la Fed
                if "observations" in response and isinstance(response["observations"], list):
                    values = [(safe_float(obs["value"]), obs["date"]) for obs in response["observations"] if obs.get("value") not in [None, "."]]
                    sorted_values = sorted(values, key=lambda x: pd.to_datetime(x[1]), reverse=True)
                    if len(sorted_values) >= 2:
                        diff = sorted_values[0][0] - sorted_values[1][0]
                        date = sorted_values[0][1]
                        mul = recency_weight(date)
                        score = safe_float(diff * mul)
                        return round(score, 2) if impact_if_rising else -score
                    elif len(sorted_values) == 1:
                        return 0.0
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt+1}/{max_retries} for {crit}")
                    time.sleep(2)
                    continue
                print(f"Failed after {max_retries} attempts for {crit}: {e}")
                return 0.0
        print(f'unexpected response for {crit}: {response}')
        return 0.0
        
    indicators = ["interest_rate", "cpi", "retail_sales", "gdp", "unemployment"]
    data = {
        'US': ["FEDFUNDS", "CPIAUCSL", "RSAFS", "GDP", "UNRATE"],
        'CAD': ["IR3TIB01CAM156N", "CPALCY01CAM661N", "CANSLRTTO02IXOBSAM", "NGDPRSAXDCCAQ", "LRUNTTTTCAA156S"],
        'GBP': ["IR3TIB01GBM156N", "GBRCPIALLMINMEI", "GBRSLRTTO02IXOBSAM", "CLVMNACSCAB1GQUK", "LRHUTTTTGBQ156S"],
        'EUR': ["ECBMRRFR", "CP0000EZ19M086NEST", "EA19SLRTTO01IXOBSAM", "CLVMNACSCAB1GQEA19", "LRHUTTTTEZM156S"],
        'JPY': ["IR3TIB01JPM156N", "JPNCPIALLMINMEI", "JPNSLRTTO02IXOBSAM", "JPNRGDPR", "LRUNTTTTJPM156S"]
    }
    data = pd.DataFrame(data, index=indicators).transpose()
    poids = pd.Series({
        'interest_rate': 1.7,
        'cpi': 1.5,
        'retail_sales': 1.3,
        'gdp': 1.5,
        'unemployment': 1.3
    })

    url_fred = "https://api.stlouisfed.org/fred/series/observations?series_id={crit}&api_key=029fc913002866780d080d671f0da98b&file_type=json"
    data_matrix = {}
    for country, row in data.iterrows():
        results = []
        for series_id in row:
            if series_id == 'interest_rate' or series_id == 'unemployment':
                impact_if_rising = False
            else:
                impact_if_rising = True
            url = url_fred.format(crit=series_id)
            results.append(get_diff(url, impact_if_rising, series_id))
        data_matrix[country] = results
        
    df_results = pd.DataFrame(data_matrix, index=indicators).multiply(poids, axis=0)
    scores_par_pays = df_results.sum(axis=0).round(2)
    df_final = pd.DataFrame({
        'pays': scores_par_pays.index,
        'score': scores_par_pays.values
    })
    df_final = df_final.sort_values(by='score', ascending=False)
    devises_fortes = df_final[:2]
    devises_faibles = df_final[-2:]
 
    # tendance long terme graphique
    class RateLimiter:
        def __init__(self, max_calls=8, period=60):
            self.max_calls = max_calls
            self.period = period
            self.calls = []
            
        def wait_if_needed(self):
            """Attend si nécessaire pour respecter la limite d'appels"""
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                wait_time = self.period - (now - oldest_call)
                if wait_time > 0:
                    print(f"Limite d'API atteinte. Attente de {wait_time:.1f} secondes...")
                    time.sleep(wait_time + 1)
                    self.calls = []
            self.calls.append(time.time())
            
    api_limiter = RateLimiter(max_calls=8, period=60)
    
    def calculer_intensite(paire):
        api_limiter.wait_if_needed()
        url = f"https://api.twelvedata.com/ema?symbol={paire}&interval=1day&time_period=50&apikey=e46dfe11b3544e8cb324143354c3ec42&outputsize=400"
        try:
            response = requests.get(url).json()
            if "values" not in response:
                print(f"Erreur pour {paire}: {response}")
                return 0
            df = pd.DataFrame(response["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["ema"] = df["ema"].astype(float)
            df = df.sort_values("datetime")
            date_actuelle = df["datetime"].max()
            date_ancienne = date_actuelle - pd.Timedelta(days=365)
            mm_actuelle = df.iloc[-1]["ema"]
            idx_ancien = (df["datetime"] - date_ancienne).abs().idxmin()
            mm_ancienne = df.loc[idx_ancien, "ema"]
            variation = mm_actuelle - mm_ancienne
            if mm_ancienne == 0:
                return 0
            variation_relative = variation / abs(mm_ancienne)
            intensite = max(min(variation_relative, 1), -1)
            return intensite
        except Exception as e:
            print(f"Erreur lors du calcul de l'intensité pour {paire}: {e}")
            return 0
    
    def calculer_intensite_moyenne_par_devise(devise):
        """Calcule l'intensité moyenne de variation pour une devise donnée à travers les paires spécifiées."""
        if devise == 'EUR':
            paires = ['EUR/AUD', 'EUR/CHF', 'EUR/NOK', 'EUR/TRY']
        elif devise == 'USD':
            paires = ['USD/CHF', 'USD/SEK', 'USD/MXN', 'USD/SGD']
        elif devise == 'GBP':
            paires = ['GBP/AUD', 'GBP/CHF', 'GBP/NZD', 'GBP/SGD']
        elif devise == 'JPY':
            paires = ['JPY/SGD', 'JPY/NOK', 'JPY/SEK', 'JPY/MXN']
        elif devise == 'CAD':
            paires = ['CAD/CHF', 'CAD/SEK', 'CAD/SGD', 'CAD/NOK']
        else:
            print(f"Devise {devise} non supportée")
            return 0
        intensites = []
        for paire in paires:
            intensite = calculer_intensite(paire)
            intensites.append(intensite)
            print(f"Paire {paire}: intensité = {intensite:.4f}")
        if intensites:
            moyenne = sum(intensites) / len(intensites)
            return moyenne
        else:
            return 0

    # creation dataframe devise fortes 
    resultats_devises = {}
    for devise in devises_fortes['pays']:
        print(f"\nCalcul de l'intensité moyenne pour {devise}:")
        moyenne = calculer_intensite_moyenne_par_devise(devise)
        resultats_devises[devise] = moyenne
        print(f"Intensité moyenne pour {devise}: {moyenne:.4f}")

    # dataframe devises fortes
    df_devises_fortes = pd.DataFrame({
        'devise': list(resultats_devises.keys()),
        'intensite': list(resultats_devises.values()),
        'tendance': ['haussière' if x > 0 else 'baissière' for x in resultats_devises.values()],
        'force': [abs(x) for x in resultats_devises.values()]
    })

    # creation dataframe devise faibles
    resultats_devises_faibles = {}
    for devise in devises_faibles['pays']:
        print(f"\nCalcul de l'intensité moyenne pour {devise} (devise faible):")
        moyenne = calculer_intensite_moyenne_par_devise(devise)
        resultats_devises_faibles[devise] = moyenne
        print(f"Intensité moyenne pour {devise}: {moyenne:.4f}")

    # dataframe devises faibles 
    df_devises_faibles = pd.DataFrame({
        'devise': list(resultats_devises_faibles.keys()),
        'intensite': list(resultats_devises_faibles.values()),
        'tendance': ['haussière' if x > 0 else 'baissière' for x in resultats_devises_faibles.values()],
        'force': [abs(x) for x in resultats_devises_faibles.values()]
    })

    # Afficher les DataFrames
    print("\nDataFrame des devises fortes:")
    print(df_devises_fortes)

    print("\nDataFrame des devises faibles:")
    print(df_devises_faibles)

    # dataframe complet pour trend graphique
    toutes_devises = {}
    for devise in devises_fortes['pays']:
        score_macro = float(devises_fortes[devises_fortes['pays'] == devise]['score'].values[0])
        intensite = resultats_devises.get(devise, 0)
        score_tendance = 0.5 * intensite + 0.5 * (score_macro / 10)
        
        toutes_devises[devise] = {
            'score_macro': score_macro,
            'intensite': intensite,
            'score_tendance': score_tendance,
            'categorie': 'forte'
        }
    for devise in devises_faibles['pays']:
        score_macro = float(devises_faibles[devises_faibles['pays'] == devise]['score'].values[0])
        intensite = resultats_devises_faibles.get(devise, 0)
        score_tendance = 0.5 * intensite + 0.5 * (score_macro / 10)
        
        toutes_devises[devise] = {
            'score_macro': score_macro,
            'intensite': intensite,
            'score_tendance': score_tendance,
            'categorie': 'faible'
        }

    # dataframe complet final 
    df_complet = pd.DataFrame([
        {
            'devise': devise,
            'score_macro': info['score_macro'],
            'intensite': info['intensite'],
            'tendance_mm': 'haussière' if info['intensite'] > 0 else 'baissière',
            'force_mm': abs(info['intensite']),
            'score_tendance': info['score_tendance'],
            'tendance_globale': 'haussière' if info['score_tendance'] > 0 else 'baissière',
            'force_globale': abs(info['score_tendance']),
            'categorie': info['categorie']
        }
        for devise, info in toutes_devises.items()
    ])
    df_complet = df_complet.sort_values(by='score_tendance', ascending=False)
    print("\nTableau complet avec score de tendance à long terme:")
    print(df_complet)
    meilleure_devise = df_complet.iloc[0]['devise']
    pire_devise = df_complet.iloc[-1]['devise']
    print(f"\nAcheter {meilleure_devise}/{pire_devise}")
    
    return df_complet, meilleure_devise, pire_devise

# Exemple d'utilisation
if __name__ == "__main__":
    df_complet, meilleure_devise, pire_devise = long_trend_part()
