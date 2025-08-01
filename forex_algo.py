import pandas as pd
print("test")
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
print(devises_fortes)
print(devises_faibles)
print(f'acheter {df_final.iloc[0, 0]}/{df_final.iloc[-1, 0]}')
print(df_complet)

print("\n--- Calcul du score de journée par devise ---\n")
def scrape_trading_economics_stream():
    print("\n--- Récupération du flux d'actualités Trading Economics ---\n")
    country_to_currency = {
        'United States': 'USD',
        'US': 'USD',
        'USA': 'USD',
        'U.S.': 'USD',
        'U.S': 'USD',
        'America': 'USD',
        'Fed': 'USD',
        'France': 'EUR',
        'Germany': 'EUR',
        'Italy': 'EUR',
        'Spain': 'EUR',
        'Netherlands': 'EUR',
        'Belgium': 'EUR',
        'Euro Area': 'EUR',
        'Eurozone': 'EUR',
        'Europe': 'EUR',
        'EU': 'EUR',
        'ECB': 'EUR',
        'Japan': 'JPY',
        'Tokyo': 'JPY',
        'BOJ': 'JPY',
        'United Kingdom': 'GBP',
        'UK': 'GBP',
        'Britain': 'GBP',
        'England': 'GBP',
        'BOE': 'GBP',
        'Canada': 'CAD',
        'Ottawa': 'CAD',
        'BOC': 'CAD'
    }
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    
    driver = webdriver.Chrome(
        service=webdriver.ChromeService(ChromeDriverManager().install()),
        options=options
    )
    news_data = []
    try:
        driver.get("https://tradingeconomics.com/stream?i=economy")
        print("Page ouverte, attente du chargement...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "te-stream-item"))
        )
        max_scrolls = 50
        scrolls_without_new_content = 0
        last_item_count = 0
        hr = 0

        for scroll in range(max_scrolls):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1.2)
            
            elements = driver.find_elements(By.CSS_SELECTOR, "li.list-group-item.te-stream-item.indc_news_stream")
            if len(elements) == last_item_count:
                scrolls_without_new_content += 1
                if scrolls_without_new_content >= 5:
                    print("Fin du chargement - aucun nouvel élément")
                    break
            else:
                scrolls_without_new_content = 0
                last_item_count = len(elements)
            if elements and scroll % 3 == 0:
                try:
                    time_element = elements[-1].find_element(By.CSS_SELECTOR, "small[style*='color:#808080']")
                    time_ago = time_element.text.strip()
                    if "hours" in time_ago:
                        current_hr = int(time_ago.split()[0])
                        hr = max(hr, current_hr)
                        print(f"Article le plus ancien: {hr} heures")
                        if hr > 24:
                            print("Atteinte d'articles de plus de 24h")
                            break
                except Exception as e:
                    print(f"Erreur lors de la vérification du temps: {e}")
        try:
            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".fc-button-label, .fc-button.fc-cta-consent"))
            )
            cookie_button.click()
            time.sleep(1)
            print("Cookies acceptés")
        except Exception as e:
            print("Pas de popup de cookies ou erreur:", e)
        news_items = driver.find_elements(By.CSS_SELECTOR, "li.list-group-item.te-stream-item.indc_news_stream")
        news_items2 = driver.find_elements(By.CSS_SELECTOR, "li.list-group-item.te-stream-item.indc_news_stream.te-stream-item-2")
        print(f"Nombre d'événements trouvés: {len(news_items)}")
        print(f"Nombre d'événements trouvés (type 2): {len(news_items2)}")

        for item in news_items + news_items2:
            try:
                country_element = item.find_element(By.CSS_SELECTOR, ".te-stream-country")
                country = country_element.text.strip()
                category_element = item.find_element(By.CSS_SELECTOR, ".te-stream-category")
                category = category_element.text.strip()
                description_element = item.find_element(By.CSS_SELECTOR, ".te-stream-item-description")
                description = description_element.text.strip()
                time_element = item.find_element(By.CSS_SELECTOR, "small[style*='color:#808080']")
                time_ago = time_element.text.strip()
                title_element = item.find_element(By.CSS_SELECTOR, ".te-stream-title")
                title = title_element.text.strip()
                currency = None
                country_text = country.lower()
                print(f"Tentative de correspondance pour: '{country_text}'")
                for country_name, currency_code in country_to_currency.items():
                    if country_name.lower() in country_text or country_text in country_name.lower():
                        currency = currency_code
                        break
                if country == 'France' or country == 'Germany' or country == 'Italy' or country == 'Spain' or country == 'Netherlands' or country == 'Belgium':
                    impact_multiplier = 0.5
                else:
                    impact_multiplier = 1.0
                if not currency:
                    print(f"⚠️ Pays non reconnu: '{country}' - Titre: {title[:30]}...")
                
                if "hours" in time_ago:
                    hours_ago = int(time_ago.split()[0])
                    if hours_ago > 24:
                        print(f"Événement ignoré (trop ancien): {title} ({country})")
                        continue

                if currency:
                    news_data.append({
                        'country': country,
                        'currency': currency,
                        'category': category,
                        'title': title,
                        'description': description,
                        'time_ago': time_ago,
                        'impact_multiplier': impact_multiplier,
                        'full_text': f"{title}. {description}"
                    })
                    print(f"Événement ajouté: {title} ({country} / {currency})")
                else:
                    print(f"Événement ignoré (pays non suivi): {title} ({country})")
                
            except Exception as e:
                print(f"Erreur lors de l'extraction d'un événement: {e}")
                continue
        
    except Exception as e:
        print(f"Erreur principale: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    df_te_news = pd.DataFrame(news_data)
    
    if df_te_news.empty:
        print("Aucun événement récupéré!")
    else:
        print(f"\n{len(df_te_news)} événements importants trouvés:")
        print(df_te_news[['currency', 'title', 'time_ago']])
    
    return df_te_news

def analyze_te_sentiment(df_news):
    print("\n--- Analyse de sentiment des données Trading Economics ---\n")
    
    if df_news.empty:
        print("Aucun événement à analyser!")
        return pd.DataFrame()
    print("Chargement du modèle FinBERT...")
    tokenizer_te = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model_te = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    news_by_currency = {dev: [] for dev in ["EUR", "USD", "CAD", "JPY", "GBP"]}
    for _, row in df_news.iterrows():
        if row["currency"] in news_by_currency:
            news_by_currency[row["currency"]].append({
                'text': row["full_text"],
                'impact': row["impact_multiplier"]
            })

    sentiment_results = {}
    for currency, texts in news_by_currency.items():
        print(f"\nAnalyse des événements pour {currency}: {len(texts)} événements")
        
        if not texts:
            print(f"  Aucun événement trouvé pour {currency}")
            continue
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        confidence_sum = 0
        
        for item in texts:
            if isinstance(item, dict):
                text = item['text']
                impact_multiplier = float(item['impact'])
            else:
                text = item
                impact_multiplier = 1.0
                
            inputs = tokenizer_te(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model_te(**inputs)
            
            probs = F.softmax(outputs.logits, dim=1)
            sentiment_labels = ["negative", "neutral", "positive"]
            sentiment_scores = {label: float(prob) for label, prob in zip(sentiment_labels, probs[0])}
            sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = float(sentiment_scores[sentiment])
            if sentiment == "positive":
                positive_count += 1
                confidence_sum += confidence * impact_multiplier
            elif sentiment == "negative":
                negative_count += 1
                confidence_sum -= confidence * impact_multiplier
            else:
                neutral_count += 1
            print(f"  - Événement: {text[:50]}... | Sentiment: {sentiment} ({confidence:.2f}) [Impact: {impact_multiplier}]")
        total_count = positive_count + negative_count + neutral_count
        avg_confidence = abs(confidence_sum / total_count) if total_count > 0 else 0
        sentiment_score = confidence_sum / total_count if total_count > 0 else 0
        
        sentiment_results[currency] = {
            'total_events': total_count,
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'avg_confidence': avg_confidence,
            'score_journee': sentiment_score 
        }
    df_scores_journee = pd.DataFrame([
        {
            'devise': currency,
            'nb_evenements': data['total_events'],
            'positifs': data['positive'],
            'negatifs': data['negative'],
            'neutres': data['neutral'],
            'confiance': data['avg_confidence'],
            'score_journee': data['score_journee']
        }
        for currency, data in sentiment_results.items()
    ])
    print("\n=== Scores de sentiment par devise (Trading Economics) ===")
    print(df_scores_journee)
    
    return df_scores_journee
te_news = scrape_trading_economics_stream()
df_scores_journee = analyze_te_sentiment(te_news)
