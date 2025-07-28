import pandas as pd
import requests 
from datetime import datetime, timedelta
import time
import random
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from bs4 import BeautifulSoup as bs
import webdriver_manager
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.service import Service
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)
# long term trend part 
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
print(devises_fortes)
print(devises_faibles)
print(f'acheter {df_final.iloc[0, 0]}/{df_final.iloc[-1, 0]}')

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

# news scrapping part
print("\n--- Récupération du calendrier économique de Trading Economics ---\n")
main_countries_tradingecon = ['JP', 'US', 'GB', 'CA', 'EU']
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")
data_calendar = []

driver = webdriver.Chrome(
    service=webdriver.ChromeService(ChromeDriverManager().install()),
    options=options
)

try:
    driver.get("https://tradingeconomics.com/calendar")
    print("Page ouverte, attente du chargement...")
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "calendar"))
    )

    # cookies
    try:
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".fc-button-label, .fc-button.fc-cta-consent"))
        )
        cookie_button.click()
        time.sleep(2)
        print("Cookies acceptés")
    except Exception as e:
        print("Pas de popup de cookies ou erreur:", e)

    # filtre d'importance
    try:
        print("Application du filtre d'importance (simplifié)...")

        # bouton d'importance
        importance_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-calendar[aria-label='Select calendar importance']"))
        )
        driver.execute_script("arguments[0].click();", importance_button)
        time.sleep(1)
        
        #bouton 3 étoiles
        three_stars_option = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[onclick="setCalendarImportance(\'3\');"]'))
        )
        driver.execute_script("arguments[0].click();", three_stars_option)
        time.sleep(2)
        
        print("Filtre d'importance appliqué (3 étoiles)")
        driver.save_screenshot("importance_filter_applied.png")
        
    except Exception as e:
        print(f"Erreur lors de l'application du filtre d'importance: {e}")

    # filtre par pays
    try:
        print("Application du filtre de pays (méthode directe)...")

        # bouton de filtre pays
        country_filter = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-calendar[aria-label='Select countries']"))
        )
        driver.execute_script("arguments[0].click();", country_filter)
        time.sleep(2)
        
        # bouton Clear
        clear_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.te-c-option[onclick='clearSelection();']"))
        )
        driver.execute_script("arguments[0].click();", clear_button)
        time.sleep(2)
        print("Filtres pays réinitialisés")

        # bouton pour les pays dont j'ai besoin
        country_selectors = {
            "Canada": "li.te-c-option-can",
            "Euro Area": "li.te-c-option-emu", 
            "Japan": "li.te-c-option-jpn",
            "United Kingdom": "li.te-c-option-gbr",
            "United States": "li.te-c-option-usa"
        }
        
        for country_name, selector in country_selectors.items():
            try:
                country_option = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", country_option)
                print(f"Pays sélectionné: {country_name}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Erreur lors de la sélection du pays {country_name}: {e}")
        
        # bouton Save
        save_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn-success[onclick='saveSelectionAndGO();']"))
        )
        driver.execute_script("arguments[0].click();", save_button)
        time.sleep(4)
        print("Filtres de pays appliqués")
        
        # Capture d'écran pour vérifier
        driver.save_screenshot("countries_filtered.png")
        
    except Exception as e:
        print(f"Erreur lors du filtrage par pays: {e}")
        import traceback
        traceback.print_exc()

    driver.save_screenshot("trading_economics_filtered.png")
    print("Filtres appliqués, extraction des données...")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#calendar tbody tr"))
    )
    html = driver.page_source
    soup = bs(html, "html.parser")

    # débogage avanncé
    html = driver.page_source
    soup = bs(html, "html.parser")

    with open("calendar_debug.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML enregistré dans calendar_debug.html pour inspection")

    headers_selectors = [
        "thead.table-header", 
        "thead", 
        "tr.calendar-date", 
        ".calendar__row--day-breaker"
    ]

    headers = []
    for selector in headers_selectors:
        found = soup.select(selector)
        if found:
            headers.extend(found)
            print(f"Trouvé {len(found)} en-têtes avec le sélecteur '{selector}'")

    print(f"Total: {len(headers)} en-têtes potentiels trouvés")

    date_map = {}
    all_dates = []
    # Extraction des dates
    for i, header in enumerate(headers):
        date_text = None

        date_th = header.select_one("th[colspan='3']")
        if date_th and date_th.text.strip():
            date_text = date_th.text.strip()
        
        if not date_text:
            first_cell = header.select_one("th") or header.select_one("td")
            if first_cell and first_cell.text.strip():
                date_text = first_cell.text.strip()
        
        if not date_text:
            date_text = header.text.strip()
        
        if date_text:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            if any(month in date_text for month in months) and any(c.isdigit() for c in date_text):
                header_id = header.get('id', f'auto_id_{i}')
                date_map[header_id] = date_text
                all_dates.append(date_text)
                print(f"Date trouvée: '{date_text}' (ID: {header_id})")

    print(f"Total: {len(date_map)} dates extraites")
    if all_dates:
        print(f"Dates trouvées: {all_dates[:5]}...")

    table = soup.select_one("#calendar")
    if not table:
        print("Table #calendar non trouvée")
    else:
        current_date = None
        for tbody in table.select("tbody"):
            prev_thead = tbody.find_previous("thead")
            found_date = False
            
            if prev_thead:
                thead_id = prev_thead.get('id', '')
                if thead_id in date_map:
                    current_date = date_map[thead_id]
                    found_date = True
                    print(f"Date trouvée via ID: {current_date}")
                if not found_date:
                    date_cell = prev_thead.select_one("th[colspan='3']")
                    if date_cell and date_cell.text.strip():
                        current_date = date_cell.text.strip()
                        found_date = True
                        print(f"Date trouvée via texte direct: {current_date}")
            if not found_date and all_dates:
                all_tbodies = table.select("tbody")
                tbody_index = all_tbodies.index(tbody) if tbody in all_tbodies else -1
                if 0 <= tbody_index < len(all_tbodies):
                    date_index = min(int(tbody_index * len(all_dates) / len(all_tbodies)), len(all_dates) - 1)
                    current_date = all_dates[date_index]
                    print(f"Date estimée par position: {current_date}")
                else:
                    current_date = None
            for row in tbody.select("tr"):
                cells = row.select("td")
                if len(cells) < 6:
                    continue
                try:
                    time_cell = cells[0].text.strip() if len(cells) > 0 else ""
                    country_cell = cells[1] if len(cells) > 1 else None
                    country_code = country_cell.text.strip() if country_cell else ""
                    country = country_cell.get('title', country_code) if country_cell else ""
                    if not country:
                        country = country_code
                    print(f"Cellule pays: Code='{country_code}', Nom='{country}'")
                    importance = "High"
                    event = cells[3].text.strip() if len(cells) > 3 else ""
                    actual = cells[4].text.strip() if len(cells) > 4 else ""
                    forecast = cells[5].text.strip() if len(cells) > 5 else ""
                    previous = cells[6].text.strip() if len(cells) > 6 else ""
                    # creation du dataframe
                    if country_code in main_countries_tradingecon and event and current_date:
                        for prefix in [f"{country} ", f"{country}: "]:
                            event = event.replace(prefix, "")
                        
                        data_calendar.append({
                            'date': current_date,
                            'time': time_cell,
                            'country': country,
                            'country_code': country_code,
                            'importance': importance,
                            'event': event,
                            'actual': actual,
                            'forecast': forecast,
                            'previous': previous
                        })
                        print(f"  - Événement AJOUTÉ: {event} ({country_code})")
                    else:
                        print(f"  - Événement IGNORÉ: {event} ({country_code})")
                except Exception as e:
                    print(f"Erreur lors de l'extraction d'une ligne: {e}")

except Exception as e:
    print(f"Erreur principale: {e}")
    import traceback
    traceback.print_exc()

finally:
    df_events = pd.DataFrame(data_calendar)
    print("\n--- Événements économiques importants ---\n")
    if df_events.empty:
        print("Aucun événement récupéré! Vérifiez trading_economics_filtered.png")
    else:
        print(f"{len(df_events)} événements importants trouvés après filtrage:")
        print(df_events)
    
    driver.quit()
