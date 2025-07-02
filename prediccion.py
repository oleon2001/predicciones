# Importar las bibliotecas necesarias
# Asegúrate de tenerlas instaladas: 
# pip install python-binance pandas ta numpy newsapi-python matplotlib
from binance.client import Client
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
import time # Para manejar límites de la API
from newsapi import NewsApiClient # Para obtener noticias
import matplotlib.pyplot as plt # Para graficar
import matplotlib.dates as mdates # Para formatear fechas en el gráfico

# --- Configuración de las Claves API ---
# Estas claves han sido proporcionadas por el usuario.
# ¡¡¡ADVERTENCIA DE SEGURIDAD MUY IMPORTANTE!!!
# NUNCA tengas tus claves API directamente en el código en un entorno de producción o si compartes el script.
# Considera usar variables de entorno o un gestor de secretos para mayor seguridad.
API_KEY = "26Sh06jOHDKVRqZY5puDa7q16hQ1CU0aitWM0OWy1iMF0jU8h8jqKqUFsxzLC5Ze"  # Clave API de Binance
API_SECRET = "yOWjPf3wHDuQbBJVKll3kDYGG6c57GiUgbdu6Xp79VnAG3dzC5AUMU4IDd3LhsnT" # Clave Secreta API de Binance

# Clave API para NewsAPI.org proporcionada por el usuario.
NEWS_API_KEY = "95a1ebc226f34eb38842c95fd4ce1932" 

# --- Parámetros de Análisis ---
# Se incluyen ejemplos de meme coins y otras altcoins. ¡Invertir en ellas es de alto riesgo!
PARES_A_ANALIZAR_MANUALMENTE = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT",  # Principales
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", # Meme Coins Populares (MUY ALTO RIESGO)
    "SOLUSDT", "ADAUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "LINKUSDT" # Otras Altcoins
    # Para monedas como las temáticas "TRUMP", verifica el ticker exacto en Binance (ej. MAGAUSDT si estuviera listada) y añádelo aquí.
] 
NUMERO_DE_PARES_POR_VOLUMEN = 5 # Se usará si PARES_A_ANALIZAR_MANUALMENTE está vacío.
INTERVALO_VELAS = '4h' # Intervalo de tiempo para las velas (ej: '15m', '1h', '4h', '1d')
PERIODO_HISTORICO = "90 day ago UTC" # Periodo para descargar datos históricos (ej: "30 day ago UTC", "90 day ago UTC")

# Parámetros para indicadores técnicos
SMA_CORTO_PERIODO = 10 
SMA_LARGO_PERIODO = 50 
RSI_PERIODO = 14     
MACD_FAST_PERIODO = 12 
MACD_SLOW_PERIODO = 26 
MACD_SIGN_PERIODO = 9  
BOLLINGER_WINDOW = 20  
BOLLINGER_STD_DEV = 2  

# Ratios estándar de Fibonacci
FIBO_RATIOS_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIBO_RATIOS_EXTENSION = [0.236, 0.382, 0.618, 1.0, 1.618] # Usados para proyectar más allá del rango

# Mapeo simple de símbolos a nombres para mejorar búsqueda de noticias
CRYPTO_NAMES_MAP = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin", "SOL": "Solana",
    "XRP": "Ripple", "ADA": "Cardano", "DOGE": "Dogecoin", "SHIB": "Shiba Inu",
    "PEPE": "Pepe", "WIF": "dogwifhat", "DOT": "Polkadot", "MATIC": "Polygon", 
    "AVAX": "Avalanche", "LINK": "Chainlink",
}

# --- Inicialización de Clientes ---
client = None
try:
    # Inicializa el cliente de Binance con las claves proporcionadas.
    client = Client(API_KEY, API_SECRET)
    # Realiza una llamada simple para verificar la conexión y los permisos básicos de las claves.
    client.get_account() 
    print(">>> Conexión exitosa a Binance (cuenta verificada).\n")
except Exception as e:
    # Si falla la conexión con claves, intenta inicializar sin claves para datos públicos.
    print(f"Error al conectar con Binance o al verificar la cuenta con las claves proporcionadas: {e}")
    print(">>> ADVERTENCIA: Intentando inicializar cliente de Binance sin autenticación (solo para datos públicos).")
    print(">>> Esto puede ocurrir si las claves API no son válidas, han expirado o no tienen los permisos necesarios para 'get_account'.")
    try:
        client = Client() # Cliente sin autenticación
        # Probar una llamada pública para ver si el cliente (sin auth) funciona
        client.ping()
        print(">>> Cliente de Binance inicializado en modo solo datos públicos (sin autenticación) debido a un error previo con las claves.\n")
    except Exception as e_public:
        print(f"Error al inicializar cliente de Binance incluso en modo público: {e_public}")
        print(">>> El script podría no funcionar correctamente sin acceso a la API de Binance.")
        client = None # Asegurar que client es None si todo falla

newsapi = None
# Verifica que la NEWS_API_KEY no sea el placeholder original y que no esté vacía.
if NEWS_API_KEY and NEWS_API_KEY != "TU_NEWS_API_KEY_AQUI": 
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        print(">>> NewsAPI client inicializado correctamente.\n")
    except Exception as e:
        print(f"Error inicializando NewsAPI client: {e}. Las noticias no estarán disponibles.\n")
else:
    # Advierte si la clave de NewsAPI es el placeholder o está vacía.
    print(">>> ADVERTENCIA: NEWS_API_KEY no configurada o es un placeholder ('TU_NEWS_API_KEY_AQUI').")
    print(">>> Las noticias no estarán disponibles. Consigue una clave gratuita en https://newsapi.org\n")

# --- Funciones de Obtención y Cálculo ---

def obtener_pares_populares(num_pares=5):
    """
    Obtiene los 'num_pares' de trading más populares contra USDT por volumen de 'quoteVolume'.
    'quoteVolume' es el volumen total comerciado en la moneda de cotización (USDT en este caso) durante las últimas 24 horas.
    """
    if not client: 
        print("Error: Cliente de Binance no inicializado en obtener_pares_populares.")
        return []
    try:
        print(f"Obteniendo los {num_pares} pares más populares por volumen (contra USDT)...")
        tickers = client.get_ticker() # Obtiene datos de las últimas 24h para todos los pares
        usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
        
        # Asegurar que 'quoteVolume' existe, es numérico y positivo para un ordenamiento significativo
        usdt_pairs_valid_volume = []
        for p in usdt_pairs:
            try:
                volume = float(p.get('quoteVolume', 0)) # Convertir a float, default a 0 si no existe
                if volume > 0: # Solo considerar pares con volumen positivo
                    usdt_pairs_valid_volume.append(p)
            except (ValueError, TypeError):
                # Ignorar pares donde quoteVolume no es convertible a float
                continue 
        
        # Ordenar por 'quoteVolume' de mayor a menor
        sorted_usdt_pairs = sorted(usdt_pairs_valid_volume, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        pares_seleccionados = [pair['symbol'] for pair in sorted_usdt_pairs[:num_pares]]
        if pares_seleccionados:
            print(f"Pares seleccionados por volumen: {pares_seleccionados}\n")
        else:
            print("No se encontraron pares con volumen suficiente o datos válidos para seleccionar por popularidad.\n")
        return pares_seleccionados
    except Exception as e:
        print(f"Error al obtener pares populares: {e}")
        return []


def obtener_datos_historicos(par, intervalo, periodo_historico_str):
    """
    Obtiene datos históricos de velas (k-lines) para un par específico desde Binance.
    Convierte los datos a un DataFrame de Pandas y realiza una limpieza básica.
    """
    if not client: 
        print(f"Error: Cliente de Binance no inicializado en obtener_datos_historicos para {par}.")
        return pd.DataFrame()
    try:
        print(f"Obteniendo datos históricos para {par} (Intervalo: {intervalo}, Periodo: {periodo_historico_str})...")
        # Binance puede limitar el número de velas por solicitud (usualmente 1000)
        klines = client.get_historical_klines(par, intervalo, periodo_historico_str, limit=1000)
        
        if not klines: # Si la API devuelve una lista vacía
            print(f"No se recibieron datos históricos de la API para {par}. Puede que el par no exista o no tenga historial en este periodo.\n")
            return pd.DataFrame()

        columnas = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 
                    'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 
                    'Taker_buy_quote_asset_volume', 'Ignore']
        df = pd.DataFrame(klines, columns=columnas)

        # Convertir Timestamp a datetime y asegurar tipos numéricos para columnas relevantes
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote_asset_volume', 
                           'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' convierte errores de parsing a NaN
        
        # Eliminar filas donde las columnas numéricas clave son NaN después de la conversión (indica datos corruptos o faltantes)
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True) 

        if df.empty:
            print(f"Después de la limpieza, no quedaron datos válidos para {par}.\n")
            return pd.DataFrame()

        print(f"Datos obtenidos y procesados para {par}: {len(df)} velas válidas.\n")
        return df
    except Exception as e:
        print(f"Error al obtener o procesar datos históricos para {par}: {e}\n")
        return pd.DataFrame()

def calcular_indicadores_tecnicos(df):
    """
    Calcula un conjunto de indicadores técnicos (SMA, RSI, MACD, Bandas de Bollinger)
    para el DataFrame de datos de precios.
    """
    if df.empty: 
        print("DataFrame vacío, no se pueden calcular indicadores.")
        return df
    
    # Determinar el mínimo de datos necesarios basado en la ventana más larga de los indicadores
    min_datos_requeridos = max(SMA_LARGO_PERIODO, RSI_PERIODO, MACD_SLOW_PERIODO, BOLLINGER_WINDOW, 1) # Mínimo 1 para evitar error con max([])
    if len(df) < min_datos_requeridos:
        print(f"No hay suficientes datos ({len(df)} velas) para calcular todos los indicadores (se requieren al menos {min_datos_requeridos} velas).")
        # Añadir columnas vacías para los indicadores para evitar errores posteriores si no se pueden calcular
        for col_name in ['SMA_corto', 'SMA_largo', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Bollinger_High', 'Bollinger_Mid', 'Bollinger_Low']:
            if col_name not in df.columns:
                 df[col_name] = np.nan # Asignar NaN si la columna no existe
        return df

    print("Calculando indicadores técnicos...")
    
    # Medias Móviles Simples (SMA)
    df['SMA_corto'] = SMAIndicator(close=df['Close'], window=SMA_CORTO_PERIODO, fillna=True).sma_indicator()
    df['SMA_largo'] = SMAIndicator(close=df['Close'], window=SMA_LARGO_PERIODO, fillna=True).sma_indicator()

    # Índice de Fuerza Relativa (RSI)
    df['RSI'] = RSIIndicator(close=df['Close'], window=RSI_PERIODO, fillna=True).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd_indicator = MACD(close=df['Close'], 
                          window_slow=MACD_SLOW_PERIODO, 
                          window_fast=MACD_FAST_PERIODO, 
                          window_sign=MACD_SIGN_PERIODO, 
                          fillna=True)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_hist'] = macd_indicator.macd_diff() # Histograma MACD (diferencia entre MACD y señal)

    # Bandas de Bollinger
    bollinger_indicator = BollingerBands(close=df['Close'], window=BOLLINGER_WINDOW, window_dev=BOLLINGER_STD_DEV, fillna=True)
    df['Bollinger_High'] = bollinger_indicator.bollinger_hband() # Banda superior
    df['Bollinger_Mid'] = bollinger_indicator.bollinger_mavg()   # Línea media (es una SMA de 'BOLLINGER_WINDOW' periodos)
    df['Bollinger_Low'] = bollinger_indicator.bollinger_lband()  # Banda inferior
    
    print("Indicadores técnicos calculados.\n")
    return df


def calcular_niveles_fibonacci(df):
    """
    Calcula niveles de retroceso y extensión de Fibonacci basados en el máximo y mínimo del período de datos proporcionado.
    Los niveles de extensión aquí son proyecciones simples del rango principal.
    """
    if df.empty or len(df) < 2: # Se necesitan al menos dos puntos para definir un rango
        return None, "Datos insuficientes para calcular niveles de Fibonacci."

    # Identificar el precio más alto ('High') y más bajo ('Low') en el período, y sus fechas
    highest_high_price = df['High'].max()
    idx_high = df['High'].idxmax() # Índice de la primera ocurrencia del máximo
    date_high = df.loc[idx_high, 'Timestamp']

    lowest_low_price = df['Low'].min()
    idx_low = df['Low'].idxmin() # Índice de la primera ocurrencia del mínimo
    date_low = df.loc[idx_low, 'Timestamp']
    
    # Verificar que se encontraron valores válidos y que hay una diferencia para calcular
    if pd.isna(highest_high_price) or pd.isna(lowest_low_price) or highest_high_price == lowest_low_price:
        return None, "No se pudo determinar un rango válido (máximo y mínimo iguales o no numéricos) para Fibonacci."

    # Determinar la dirección del movimiento principal DENTRO DEL PERÍODO SELECCIONADO
    # Si el mínimo ocurrió antes que el máximo, consideramos el movimiento principal del período como alcista.
    trend_is_up_in_period = date_low < date_high 
    
    diff = highest_high_price - lowest_low_price # El rango total del movimiento en el período
    
    levels = {
        "p_alto": (highest_high_price, date_high), 
        "p_bajo": (lowest_low_price, date_low), 
        "direccion_periodo": "alcista" if trend_is_up_in_period else "bajista", # Almacenar la dirección para usarla después
        "retrocesos": {},
        "extensiones": {} # Extensiones simples proyectadas desde el rango
    }
    
    # Calcular niveles de Retroceso
    if trend_is_up_in_period: # Movimiento de abajo hacia arriba (lowest_low_price a highest_high_price)
        for ratio in FIBO_RATIOS_RETRACEMENT:
            level = highest_high_price - (diff * ratio) # Restar del máximo para retrocesos
            levels["retrocesos"][f"{ratio*100:.1f}%"] = level
        # Calcular niveles de Extensión (proyectados hacia arriba desde el highest_high_price)
        for ratio in FIBO_RATIOS_EXTENSION:
            level = highest_high_price + (diff * ratio)
            levels["extensiones"][f"Ext {100 + ratio*100:.1f}%"] = level
    else: # Movimiento de arriba hacia abajo (highest_high_price a lowest_low_price)
        for ratio in FIBO_RATIOS_RETRACEMENT:
            level = lowest_low_price + (diff * ratio) # Sumar al mínimo para retrocesos
            levels["retrocesos"][f"{ratio*100:.1f}%"] = level
        # Calcular niveles de Extensión (proyectados hacia abajo desde el lowest_low_price)
        for ratio in FIBO_RATIOS_EXTENSION:
            level = lowest_low_price - (diff * ratio)
            levels["extensiones"][f"Ext {100 + ratio*100:.1f}%"] = level
            
    # Añadir los niveles 0% y 100% que definen el rango del movimiento
    levels["retrocesos"]["0.0% (Extremo Superior del Rango)" if trend_is_up_in_period else "100.0% (Extremo Superior del Rango)"] = highest_high_price
    levels["retrocesos"]["100.0% (Extremo Inferior del Rango)" if trend_is_up_in_period else "0.0% (Extremo Inferior del Rango)"] = lowest_low_price

    # Crear un reporte textual de los niveles de Fibonacci
    reporte_fibo = (f"Niveles Fibonacci basados en el rango del periodo ({PERIODO_HISTORICO}):\n"
                    f"  - Mínimo del periodo: {lowest_low_price:.4f} (el {date_low.strftime('%Y-%m-%d %H:%M')})\n"
                    f"  - Máximo del periodo: {highest_high_price:.4f} (el {date_high.strftime('%Y-%m-%d %H:%M')})\n"
                    f"  - Dirección principal del movimiento DENTRO de este periodo: {levels['direccion_periodo']}.\n\n"
                    f"  Posibles Soportes (si el precio baja) / Resistencias (si el precio sube) - Retrocesos del rango total:\n")
    # Ordenar retrocesos para la presentación
    sorted_retrocesos = sorted(levels["retrocesos"].items(), key=lambda item: item[1], reverse=trend_is_up_in_period)
    for name, val in sorted_retrocesos:
        reporte_fibo += f"    - Nivel {name}: {val:.4f}\n"
    
    reporte_fibo += f"\n  Posibles Objetivos/Extensiones más allá del rango del periodo (proyecciones simples):\n"
    sorted_extensiones = sorted(levels["extensiones"].items(), key=lambda item: item[1], reverse=trend_is_up_in_period)
    for name, val in sorted_extensiones:
        reporte_fibo += f"    - Nivel {name}: {val:.4f}\n"
    
    reporte_fibo += ("\n  IMPORTANTE: Los niveles de Fibonacci son teóricos y se basan en el rango de precios del periodo analizado. "
                     "El precio puede o no reaccionar a estos niveles. Deben usarse como una herramienta más junto con otros indicadores y análisis, "
                     "y no como predicciones certeras.")
    
    return levels, reporte_fibo


def obtener_noticias_relevantes(simbolo_base):
    """
    Obtiene hasta 3 noticias relevantes para un símbolo de criptomoneda base usando NewsAPI.
    La relevancia es determinada por los algoritmos de NewsAPI.
    """
    if not newsapi: 
        return ["Servicio de noticias no disponible (NewsAPI client no inicializado)."]
    
    nombre_cripto_busqueda = CRYPTO_NAMES_MAP.get(simbolo_base, simbolo_base) # Usar nombre mapeado si existe
    # Query para buscar noticias relacionadas con la criptomoneda
    query_noticias = f'"{nombre_cripto_busqueda}" AND (criptomoneda OR crypto OR blockchain OR token OR altcoin OR memecoin)'
    
    try:
        print(f"Buscando noticias relevantes para {simbolo_base} (Término de búsqueda: '{nombre_cripto_busqueda}')...")
        all_articles = newsapi.get_everything(
            q=query_noticias,
            language='es', # Buscar noticias en español
            sort_by='relevancy', # Ordenar por relevancia (otras opciones: 'popularity', 'publishedAt')
            page_size=3, # Número máximo de artículos a obtener
        )

        if all_articles['status'] == 'ok' and all_articles['articles']:
            noticias = [f"- \"{article['title']}\" (Fuente: {article['source']['name']})" for article in all_articles['articles']]
            return noticias
        elif not all_articles['articles']:
            return [f"No se encontraron noticias recientes para '{nombre_cripto_busqueda}' con los criterios de búsqueda actuales. Esto puede ser normal para monedas menos cubiertas o si no hay noticias relevantes recientes."]
        else:
            # Si NewsAPI devuelve un error específico
            error_msg = all_articles.get('message', 'Respuesta no esperada de NewsAPI')
            return [f"No se pudieron obtener noticias para '{nombre_cripto_busqueda}': {error_msg}"]
    except Exception as e:
        # Capturar otras excepciones (ej. problemas de red con NewsAPI)
        return [f"Excepción al obtener noticias para '{nombre_cripto_busqueda}': {str(e)}"]

def analizar_tendencia_basica(df_par):
    """
    Proporciona una interpretación básica y educativa de los indicadores técnicos estándar.
    Recordatorio: Esto NO es asesoramiento financiero. Es altamente especulativo.
    Los indicadores ofrecen perspectivas, no certezas. Considera múltiples factores.
    """
    if df_par.empty or len(df_par) < 2: # Se necesitan al menos dos velas para comparar
        return "No hay suficientes datos (velas) para un análisis técnico básico de indicadores."

    ultimo = df_par.iloc[-1] # Última vela completa
    anterior = df_par.iloc[-2] # Penúltima vela completa

    interpretaciones = [f"**Precio Actual de Cierre: {ultimo['Close']:.4f}** (correspondiente a la vela que cerró el {ultimo['Timestamp'].strftime('%Y-%m-%d %H:%M')})"]
    explicaciones_generales = ["\n**Explicación de los Indicadores Técnicos (Interpretación Simplificada y Educativa):**"]

    # SMA (Medias Móviles Simples)
    # DEFINICIÓN: Suavizan los datos de precios para ayudar a identificar la dirección de la tendencia. Una SMA corta reacciona más rápido a los cambios de precio que una SMA larga.
    if pd.notna(ultimo['SMA_corto']) and pd.notna(ultimo['SMA_largo']) and \
       pd.notna(anterior['SMA_corto']) and pd.notna(anterior['SMA_largo']):
        sma_info = f"SMA Corta ({SMA_CORTO_PERIODO}p): {ultimo['SMA_corto']:.4f}, SMA Larga ({SMA_LARGO_PERIODO}p): {ultimo['SMA_largo']:.4f}"
        if anterior['SMA_corto'] <= anterior['SMA_largo'] and ultimo['SMA_corto'] > ultimo['SMA_largo']:
            interpretaciones.append(f"SMA: **POTENCIAL CRUCE ALCISTA (Cruce Dorado)** - La media corta ({SMA_CORTO_PERIODO}p) ha cruzado POR ENCIMA de la larga ({SMA_LARGO_PERIODO}p). Históricamente, esto A VECES es visto como una señal de posible inicio de tendencia alcista, pero requiere confirmación y no es infalible. ({sma_info})")
        elif anterior['SMA_corto'] >= anterior['SMA_largo'] and ultimo['SMA_corto'] < ultimo['SMA_largo']:
            interpretaciones.append(f"SMA: **POTENCIAL CRUCE BAJISTA (Cruce de la Muerte)** - La media corta ({SMA_CORTO_PERIODO}p) ha cruzado POR DEBAJO de la larga ({SMA_LARGO_PERIODO}p). Históricamente, esto A VECES es visto como una señal de posible inicio de tendencia bajista, pero requiere confirmación. ({sma_info})")
        elif ultimo['SMA_corto'] > ultimo['SMA_largo']:
            interpretaciones.append(f"SMA: Tendencia General Alcista - La media corta está POR ENCIMA de la larga, lo que sugiere que el impulso de precios a corto plazo es más fuerte que a largo plazo. ({sma_info})")
        else:
            interpretaciones.append(f"SMA: Tendencia General Bajista - La media corta está POR DEBAJO de la larga, sugiriendo que el impulso de precios a corto plazo es más débil que a largo plazo. ({sma_info})")
        explicaciones_generales.append(f"  - Medias Móviles (SMA {SMA_CORTO_PERIODO}p y {SMA_LARGO_PERIODO}p): Compara el precio promedio de corto plazo con el de largo plazo. Un 'Cruce Dorado' (corta sobre larga) puede indicar optimismo; un 'Cruce de la Muerte' (corta bajo larga) puede indicar pesimismo. La posición relativa también sugiere la tendencia predominante.")
    else:
        interpretaciones.append("SMA: Datos insuficientes o no calculados para una interpretación completa en el último periodo.")
    
    # RSI (Índice de Fuerza Relativa)
    # DEFINICIÓN: Mide la velocidad y el cambio de los movimientos de precios para evaluar si un activo está sobrecomprado (potencialmente caro y propenso a bajar) o sobrevendido (potencialmente barato y propenso a subir). Oscila entre 0 y 100.
    if pd.notna(ultimo['RSI']):
        rsi_val = ultimo['RSI']
        rsi_status_text = ""
        if rsi_val > 70:
            rsi_status_text = f"RSI: **Nivel de Sobrecompra ({rsi_val:.2f})** - El RSI está por encima de 70. Tradicionalmente, esto sugiere que el activo ha subido mucho y rápido, y podría estar 'sobrecomprado', lo que A VECES precede a una corrección de precios a la baja o una consolidación. No es una señal de venta automática."
        elif rsi_val < 30:
            rsi_status_text = f"RSI: **Nivel de Sobreventa ({rsi_val:.2f})** - El RSI está por debajo de 30. Tradicionalmente, esto sugiere que el activo ha bajado mucho y rápido, y podría estar 'sobrevendido', lo que A VECES precede a un rebote de precios al alza. No es una señal de compra automática."
        else:
            rsi_status_text = f"RSI: Zona Neutral ({rsi_val:.2f}) - El RSI está entre 30 y 70, lo que generalmente no indica condiciones extremas de sobrecompra o sobreventa. La dirección del RSI dentro de esta zona puede dar pistas sobre el impulso."
        interpretaciones.append(rsi_status_text)
        explicaciones_generales.append(f"  - RSI ({RSI_PERIODO}p): Un valor >70 puede indicar 'sobrecompra' (el precio podría estar listo para bajar). Un valor <30 puede indicar 'sobreventa' (el precio podría estar listo para subir). La zona media (30-70) es considerada neutral, aunque movimientos hacia 50 pueden indicar cambios de momento.")
    else:
        interpretaciones.append("RSI: Datos insuficientes o no calculado.")

    # MACD (Convergencia/Divergencia de Medias Móviles)
    # DEFINICIÓN: Es un indicador de seguimiento de tendencia que muestra la relación entre dos medias móviles exponenciales del precio. Consta de la línea MACD, la línea de Señal (una EMA de la línea MACD) y el Histograma (diferencia entre MACD y Señal).
    if pd.notna(ultimo['MACD']) and pd.notna(ultimo['MACD_signal']) and \
       pd.notna(anterior['MACD']) and pd.notna(anterior['MACD_signal']): 
        macd_vals = f"MACD: {ultimo['MACD']:.4f}, Señal: {ultimo['MACD_signal']:.4f}, Histograma: {ultimo['MACD_hist']:.4f}"
        # Cruce Alcista: MACD cruza POR ENCIMA de la línea de Señal
        if anterior['MACD'] <= anterior['MACD_signal'] and ultimo['MACD'] > ultimo['MACD_signal']:
             interpretaciones.append(f"MACD: **POTENCIAL CRUCE ALCISTA** - La línea MACD ha cruzado POR ENCIMA de su línea de Señal. Esto A VECES es una señal de compra o de fortalecimiento del impulso alcista. ({macd_vals})")
        # Cruce Bajista: MACD cruza POR DEBAJO de la línea de Señal
        elif anterior['MACD'] >= anterior['MACD_signal'] and ultimo['MACD'] < ultimo['MACD_signal']: 
             interpretaciones.append(f"MACD: **POTENCIAL CRUCE BAJISTA** - La línea MACD ha cruzado POR DEBAJO de su línea de Señal. Esto A VECES es una señal de venta o de fortalecimiento del impulso bajista. ({macd_vals})")
        elif ultimo['MACD'] > ultimo['MACD_signal']:
            interpretaciones.append(f"MACD: Momento Alcista General - La línea MACD está POR ENCIMA de su línea de Señal, lo que generalmente sugiere un impulso positivo predominante. ({macd_vals})")
        else: # ultimo['MACD'] < ultimo['MACD_signal']
            interpretaciones.append(f"MACD: Momento Bajista General - La línea MACD está POR DEBAJO de su línea de Señal, sugiriendo un impulso negativo predominante. ({macd_vals})")
        
        # Interpretación del Histograma (como confirmación o señal temprana)
        if ultimo['MACD_hist'] > 0 and (len(df_par) > 2 and df_par['MACD_hist'].iloc[-2] <= 0):
            interpretaciones.append("MACD Histograma: Cruzó a positivo, lo que puede reforzar el momento alcista o anticipar un cruce alcista de líneas.")
        elif ultimo['MACD_hist'] < 0 and (len(df_par) > 2 and df_par['MACD_hist'].iloc[-2] >= 0):
            interpretaciones.append("MACD Histograma: Cruzó a negativo, lo que puede reforzar el momento bajista o anticipar un cruce bajista de líneas.")

        explicaciones_generales.append(f"  - MACD ({MACD_FAST_PERIODO}p,{MACD_SLOW_PERIODO}p,{MACD_SIGN_PERIODO}p): Un cruce de la línea MACD sobre su línea de señal puede ser una señal alcista. Un cruce por debajo, bajista. El histograma muestra la fuerza de la convergencia/divergencia.")
    else:
        interpretaciones.append("MACD: Datos insuficientes o no calculado.")

    # Bandas de Bollinger
    # DEFINICIÓN: Consisten en una media móvil central (generalmente una SMA de 20 periodos) más dos bandas de desviación estándar por encima y por debajo. Miden la volatilidad del mercado y los niveles de precios relativos (caro/barato).
    if pd.notna(ultimo['Close']) and pd.notna(ultimo['Bollinger_High']) and pd.notna(ultimo['Bollinger_Low']):
        bb_info = f"Precio: {ultimo['Close']:.4f}, BB Superior: {ultimo['Bollinger_High']:.4f}, BB Medio: {ultimo['Bollinger_Mid']:.4f}, BB Inferior: {ultimo['Bollinger_Low']:.4f}"
        if ultimo['Close'] > ultimo['Bollinger_High']:
            interpretaciones.append(f"Bandas Bollinger: **Precio POR ENCIMA de Banda Superior** - Indica que el precio está en un nivel alto relativo a su volatilidad reciente. A VECES precede una consolidación o un retroceso hacia la media (banda media). ({bb_info})")
        elif ultimo['Close'] < ultimo['Bollinger_Low']:
            interpretaciones.append(f"Bandas Bollinger: **Precio POR DEBAJO de Banda Inferior** - Indica que el precio está en un nivel bajo relativo a su volatilidad reciente. A VECES precede un rebote hacia la media (banda media). ({bb_info})")
        else:
            interpretaciones.append(f"Bandas Bollinger: Precio DENTRO de las bandas. El precio está operando dentro de su rango de volatilidad esperado. La proximidad a la banda media puede ser relevante. ({bb_info})")
        explicaciones_generales.append(f"  - Bandas de Bollinger ({BOLLINGER_WINDOW}p, {BOLLINGER_STD_DEV}dev): Muestran la volatilidad. El precio tocando la banda superior puede indicar sobreextensión alcista; tocando la inferior, bajista. A menudo el precio tiende a volver hacia la banda media (SMA {BOLLINGER_WINDOW}p).")

    if len(explicaciones_generales) > 1: # Solo añadir si hay explicaciones
        interpretaciones.extend(explicaciones_generales)
    
    interpretaciones.append("\nNOTA MUY IMPORTANTE: Esta interpretación es una simplificación con fines educativos. NO ES ASESORAMIENTO FINANCIERO. Las señales de los indicadores deben ser confirmadas, usadas en conjunto y dentro de una estrategia de gestión de riesgos. Los mercados son complejos y pueden no seguir estos patrones.")
    return "\n".join(interpretaciones)

# --- NUEVA FUNCIÓN PARA RESUMEN EDUCATIVO ---
def generar_resumen_educativo_inversion(df_par, fib_levels, par_nombre):
    """
    Genera un resumen educativo (NO CONSEJO DE INVERSIÓN) basado en indicadores técnicos.
    Esta función es una simplificación extrema y tiene fines puramente ilustrativos y educativos.
    NO DEBE USARSE PARA TOMAR DECISIONES DE INVERSIÓN REALES.
    """
    # Descargo de responsabilidad inicial y fundamental, aún más enfático
    resumen_inicial = "\n========================================================================================================\n"
    resumen_inicial += "!!! SECCIÓN ESTRICTAMENTE EDUCATIVA - NO ES ASESORAMIENTO FINANCIERO !!!\n"
    resumen_inicial += "========================================================================================================\n"
    resumen_inicial += "ADVERTENCIA: La siguiente sección es una INTERPRETACIÓN AUTOMATIZADA y ALTAMENTE SIMPLIFICADA de\n"
    resumen_inicial += "indicadores técnicos, creada con el ÚNICO PROPÓSITO DE ILUSTRAR cómo un trader podría\n"
    resumen_inicial += "intentar combinar diferentes señales. ESTO NO ES UNA RECOMENDACIÓN DE COMPRA O VENTA.\n"
    resumen_inicial += "El 'modelo de puntuación' es un ejemplo de juguete y NO una estrategia de trading probada o válida.\n"
    resumen_inicial += "El trading de criptomonedas es EXTREMADAMENTE RIESGOSO. Puede perder TODO su capital.\n"
    resumen_inicial += "NO TOME DECISIONES FINANCIERAS BASADO EN ESTA INFORMACIÓN.\n"
    resumen_inicial += "--------------------------------------------------------------------------------------------------------\n"
    print(resumen_inicial)


    if df_par.empty or len(df_par) < 2:
        return "No hay suficientes datos para generar un resumen educativo detallado."

    ultimo = df_par.iloc[-1]
    anterior = df_par.iloc[-2]
    
    # Modelo de "puntuación" simplificado para ilustrar la combinación de señales
    # Ponderaciones y umbrales son arbitrarios y solo para demostración.
    puntos_alcistas = 0.0
    puntos_bajistas = 0.0
    observaciones_detalladas = [] 
    confianza_modelo = 0 # Suma de pesos de señales activas

    # Evaluación SMA
    if pd.notna(ultimo['SMA_corto']) and pd.notna(ultimo['SMA_largo']) and pd.notna(anterior['SMA_corto']) and pd.notna(anterior['SMA_largo']):
        peso_sma_cruce = 2.0
        peso_sma_tendencia = 1.0
        if anterior['SMA_corto'] <= anterior['SMA_largo'] and ultimo['SMA_corto'] > ultimo['SMA_largo']:
            puntos_alcistas += peso_sma_cruce
            confianza_modelo += peso_sma_cruce
            observaciones_detalladas.append(f"SMA: CRUCE DORADO (alcista fuerte) detectado (SMA {SMA_CORTO_PERIODO}p sobre {SMA_LARGO_PERIODO}p). Teóricamente, una señal alcista significativa, pero puede ser tardía o falsa. Requiere confirmación (ej. volumen, otros indicadores).")
        elif anterior['SMA_corto'] >= anterior['SMA_largo'] and ultimo['SMA_corto'] < ultimo['SMA_largo']:
            puntos_bajistas += peso_sma_cruce
            confianza_modelo += peso_sma_cruce
            observaciones_detalladas.append(f"SMA: CRUCE DE LA MUERTE (bajista fuerte) detectado (SMA {SMA_CORTO_PERIODO}p bajo {SMA_LARGO_PERIODO}p). Teóricamente, una señal bajista significativa, pero también puede ser tardía o falsa.")
        elif ultimo['SMA_corto'] > ultimo['SMA_largo']:
            puntos_alcistas += peso_sma_tendencia
            confianza_modelo += peso_sma_tendencia
            observaciones_detalladas.append(f"SMA: Tendencia alcista general indicada (SMA {SMA_CORTO_PERIODO}p por encima de {SMA_LARGO_PERIODO}p). Sugiere que el momentum reciente es positivo.")
        else:
            puntos_bajistas += peso_sma_tendencia
            confianza_modelo += peso_sma_tendencia
            observaciones_detalladas.append(f"SMA: Tendencia bajista general indicada (SMA {SMA_CORTO_PERIODO}p por debajo de {SMA_LARGO_PERIODO}p). Sugiere que el momentum reciente es negativo.")
    else:
        observaciones_detalladas.append("SMA: No se pudieron evaluar completamente (datos insuficientes o no calculados).")


    # Evaluación RSI
    if pd.notna(ultimo['RSI']):
        rsi_val = ultimo['RSI']
        peso_rsi_extremo = 1.5
        peso_rsi_momentum = 0.5
        if rsi_val < 30:
            puntos_alcistas += peso_rsi_extremo
            confianza_modelo += peso_rsi_extremo
            observaciones_detalladas.append(f"RSI: En zona de SOBREVENTA ({rsi_val:.2f}). Esto A MENUDO sugiere que el precio está 'demasiado bajo' y podría experimentar un rebote alcista. Sin embargo, en tendencias bajistas fuertes, el RSI puede permanecer en sobreventa por periodos prolongados. Buscar confirmación (ej. divergencias, patrones de velas).")
        elif rsi_val > 70:
            puntos_bajistas += peso_rsi_extremo
            confianza_modelo += peso_rsi_extremo
            observaciones_detalladas.append(f"RSI: En zona de SOBRECOMPRA ({rsi_val:.2f}). Esto A MENUDO sugiere que el precio está 'demasiado alto' y podría experimentar una corrección bajista. Sin embargo, en tendencias alcistas fuertes, el RSI puede permanecer en sobrecompra. Buscar confirmación.")
        elif rsi_val > 50: 
            puntos_alcistas += peso_rsi_momentum
            confianza_modelo += peso_rsi_momentum
            observaciones_detalladas.append(f"RSI: ({rsi_val:.2f}) por encima de la línea central de 50, indicando un momentum predominantemente alcista, aunque no extremo.")
        elif rsi_val < 50: 
            puntos_bajistas += peso_rsi_momentum
            confianza_modelo += peso_rsi_momentum
            observaciones_detalladas.append(f"RSI: ({rsi_val:.2f}) por debajo de la línea central de 50, indicando un momentum predominantemente bajista, aunque no extremo.")
    else:
        observaciones_detalladas.append("RSI: No se pudo evaluar (datos insuficientes o no calculado).")


    # Evaluación MACD
    if pd.notna(ultimo['MACD']) and pd.notna(ultimo['MACD_signal']) and pd.notna(anterior['MACD']) and pd.notna(anterior['MACD_signal']):
        peso_macd_cruce = 2.0
        peso_macd_momentum = 1.0
        peso_macd_hist = 0.5
        # Cruces de líneas
        if anterior['MACD'] <= anterior['MACD_signal'] and ultimo['MACD'] > ultimo['MACD_signal']:
            puntos_alcistas += peso_macd_cruce
            confianza_modelo += peso_macd_cruce
            observaciones_detalladas.append("MACD: CRUCE ALCISTA de líneas (MACD sobre Señal). Tradicionalmente una señal de compra o fortalecimiento alcista. Es más fiable si ocurre cerca de la línea cero o desde abajo.")
        elif anterior['MACD'] >= anterior['MACD_signal'] and ultimo['MACD'] < ultimo['MACD_signal']:
            puntos_bajistas += peso_macd_cruce
            confianza_modelo += peso_macd_cruce
            observaciones_detalladas.append("MACD: CRUCE BAJISTA de líneas (MACD bajo Señal). Tradicionalmente una señal de venta o fortalecimiento bajista. Es más fiable si ocurre cerca de la línea cero o desde arriba.")
        # Posición relativa de las líneas (momentum)
        elif ultimo['MACD'] > ultimo['MACD_signal']:
             puntos_alcistas += peso_macd_momentum
             confianza_modelo += peso_macd_momentum
             observaciones_detalladas.append("MACD: Línea MACD por encima de Señal, indicando momentum alcista general. La distancia entre líneas puede indicar la fuerza.")
        else: 
            puntos_bajistas += peso_macd_momentum
            confianza_modelo += peso_macd_momentum
            observaciones_detalladas.append("MACD: Línea MACD por debajo de Señal, indicando momentum bajista general.")
        
        # Histograma
        if len(df_par) > 2: # Necesitamos al menos 3 puntos para ver un cambio en el histograma
            if ultimo['MACD_hist'] > 0 and df_par['MACD_hist'].iloc[-2] <= 0: 
                puntos_alcistas += peso_macd_hist
                confianza_modelo += peso_macd_hist
                observaciones_detalladas.append("MACD Histograma: Cruzó a territorio positivo. Puede confirmar el momentum alcista o anticipar un cruce alcista de líneas.")
            elif ultimo['MACD_hist'] < 0 and df_par['MACD_hist'].iloc[-2] >= 0: 
                puntos_bajistas += peso_macd_hist
                confianza_modelo += peso_macd_hist
                observaciones_detalladas.append("MACD Histograma: Cruzó a territorio negativo. Puede confirmar el momentum bajista o anticipar un cruce bajista de líneas.")
    else:
        observaciones_detalladas.append("MACD: No se pudo evaluar completamente (datos insuficientes o no calculado).")


    # Evaluación Bandas de Bollinger
    if pd.notna(ultimo['Close']) and pd.notna(ultimo['Bollinger_Low']) and pd.notna(ultimo['Bollinger_High']):
        peso_bb = 1.0
        if ultimo['Close'] < ultimo['Bollinger_Low']:
            puntos_alcistas += peso_bb 
            confianza_modelo += peso_bb
            observaciones_detalladas.append("Bandas Bollinger: Precio tocando o por debajo de la Banda Inferior. Esto indica que el precio está en un extremo inferior de su volatilidad reciente y A VECES puede preceder a un rebote alcista hacia la banda media o superior.")
        elif ultimo['Close'] > ultimo['Bollinger_High']:
            puntos_bajistas += peso_bb
            confianza_modelo += peso_bb
            observaciones_detalladas.append("Bandas Bollinger: Precio tocando o por encima de la Banda Superior. Esto indica que el precio está en un extremo superior de su volatilidad reciente y A VECES puede preceder a una corrección bajista hacia la banda media o inferior.")
    else:
        observaciones_detalladas.append("Bandas de Bollinger: No se pudieron evaluar (datos insuficientes o no calculadas).")

    # Evaluación Fibonacci
    if fib_levels and fib_levels.get("retrocesos") and fib_levels.get("direccion_periodo") and pd.notna(ultimo['Close']) and pd.notna(anterior['Low']) and pd.notna(ultimo['High']):
        precio_actual = ultimo['Close']
        peso_fibo_rebote = 0.75 
        # Usar la dirección del periodo calculada en calcular_niveles_fibonacci
        direccion_periodo_fibo_es_alcista = fib_levels["direccion_periodo"] == "alcista"

        for nombre_nivel, valor_nivel in fib_levels["retrocesos"].items():
            if "Extremo" not in nombre_nivel: 
                is_near_level = abs(precio_actual - valor_nivel) / valor_nivel < 0.015 
                
                if direccion_periodo_fibo_es_alcista and anterior['Low'] <= valor_nivel and precio_actual > valor_nivel and is_near_level:
                    puntos_alcistas += peso_fibo_rebote
                    confianza_modelo += peso_fibo_rebote
                    observaciones_detalladas.append(f"Fibonacci: Precio ({precio_actual:.4f}) parece haber REBOTADO AL ALZA desde cerca del nivel de soporte Fibo {nombre_nivel} ({valor_nivel:.4f}) en una tendencia alcista del periodo. Esto puede ser una señal de continuación.")
                    break 
                elif not direccion_periodo_fibo_es_alcista and anterior['High'] >= valor_nivel and precio_actual < valor_nivel and is_near_level:
                    puntos_bajistas += peso_fibo_rebote
                    confianza_modelo += peso_fibo_rebote
                    observaciones_detalladas.append(f"Fibonacci: Precio ({precio_actual:.4f}) parece haber sido RECHAZADO A LA BAJA desde cerca del nivel de resistencia Fibo {nombre_nivel} ({valor_nivel:.4f}) en una tendencia bajista del periodo. Esto puede ser una señal de continuación.")
                    break
    else:
        observaciones_detalladas.append("Niveles Fibonacci: No evaluados para señales de rebote/rechazo en este resumen (o datos insuficientes/no calculados).")


    # Construcción del Resumen
    resumen = f"\n--- Resumen Educativo del Sentimiento Técnico para {par_nombre} (Modelo Ilustrativo) ---\n"
    resumen += f"Análisis basado en velas de {INTERVALO_VELAS} y un histórico de {PERIODO_HISTORICO}.\n"
    resumen += f"Precio de cierre más reciente analizado: {ultimo['Close']:.4f} USDT (vela de {ultimo['Timestamp'].strftime('%Y-%m-%d %H:%M')}).\n\n"
    
    resumen += "**Explicación del Modelo de Puntuación (Solo Ilustrativo):**\n"
    resumen += "  Este script utiliza un sistema de 'puntos' simplificado para agregar las señales de los indicadores.\n"
    resumen += "  Se asignan puntos positivos (alcistas) o negativos (bajistas) según reglas predefinidas para cada indicador.\n"
    resumen += "  ESTE MODELO ES ARBITRARIO Y NO REPRESENTA UNA ESTRATEGIA DE TRADING REAL NI PROBADA.\n"
    resumen += f"  - Puntos Teóricos Alcistas (Potencial Positivo): {puntos_alcistas:.1f}\n"
    resumen += f"  - Puntos Teóricos Bajistas (Potencial Negativo): {puntos_bajistas:.1f}\n\n"
    
    resumen += "**Observaciones Detalladas de los Indicadores (Interpretaciones Educativas):**\n"
    if observaciones_detalladas:
        for obs in observaciones_detalladas:
            resumen += f"  -> {obs}\n"
    else:
        resumen += "  - No se generaron observaciones específicas con la lógica actual o los datos fueron insuficientes.\n"

    # Sección sobre Confluencia y Divergencia
    resumen += "\n**Confluencia y Divergencia de Señales (Concepto Educativo):**\n"
    resumen += "  - Los traders a menudo buscan 'CONFLUENCIA', es decir, múltiples indicadores apuntando en la misma dirección. Esto teóricamente podría aumentar la 'confianza' en una señal (aunque nunca la garantiza).\n"
    resumen += "  - Si los indicadores dan señales MIXTAS o 'DIVERGENTES' (ej. RSI sobrecomprado pero MACD alcista), esto podría indicar incertidumbre en el mercado o la necesidad de un análisis más profundo y cautela.\n"
    resumen += f"  - En este modelo, una diferencia pequeña entre puntos alcistas y bajistas podría reflejar señales mixtas.\n"


    # Lógica para determinar el sentimiento general (más matizada)
    diferencia_puntos = puntos_alcistas - puntos_bajistas
    sentimiento_general_txt = "Neutral o Indefinido (Señales Mixtas o Débiles)"
    # Umbral para considerar una señal más clara. Ajustable.
    umbral_decision_modelo = max(2.0, confianza_modelo * 0.25) # Ej: al menos 2 puntos de diferencia o 25% de la confianza total

    if diferencia_puntos > umbral_decision_modelo: 
        sentimiento_general_txt = "Predominantemente Alcista (según este modelo educativo simplificado)"
    elif diferencia_puntos < -umbral_decision_modelo:
        sentimiento_general_txt = "Predominantemente Bajista (según este modelo educativo simplificado)"
    elif diferencia_puntos > 0: # Diferencia positiva pero no supera el umbral
        sentimiento_general_txt = "Ligeramente Alcista (pero con señales mixtas o sin fuerte convicción del modelo)"
    elif diferencia_puntos < 0: # Diferencia negativa pero no supera el umbral
        sentimiento_general_txt = "Ligeramente Bajista (pero con señales mixtas o sin fuerte convicción del modelo)"
    
    resumen += f"\n**Sentimiento Técnico General (Según este Modelo Educativo Simplificado y sus Ponderaciones): {sentimiento_general_txt}**\n"
    if "Neutral" in sentimiento_general_txt or "Ligeramente" in sentimiento_general_txt or "mixtas" in sentimiento_general_txt:
        resumen += "  Esto puede indicar que los indicadores técnicos analizados no ofrecen una señal clara y unificada en una dirección en el marco de tiempo actual ({INTERVALO_VELAS}).\n"
        resumen += "  En el trading real, esto a menudo llevaría a mayor cautela, a esperar confirmaciones adicionales, o a abstenerse de operar hasta que haya más claridad.\n"

    resumen += "\n**Consideraciones Educativas para Plazos y Capital (Ejemplo Teórico):**\n"
    resumen += f"- **Corto Plazo (basado en velas de {INTERVALO_VELAS}):**\n"
    resumen += f"    - El análisis actual se enfoca en este marco temporal. Las condiciones técnicas en criptomonedas pueden cambiar muy rápidamente, incluso en horas o minutos.\n"
    resumen +=  "    - Una 'sugerencia' alcista a corto plazo podría ser invalidada por una noticia negativa repentina o un cambio en el sentimiento del mercado global. Del mismo modo, una señal bajista podría revertirse.\n"
    resumen +=  "    - El trading a corto plazo (day trading, swing trading) requiere monitoreo constante y una gestión de riesgos muy activa (ej. stop-loss definidos).\n"
    resumen += "- **Mediano Plazo (semanas/meses):**\n"
    resumen +=  "    - Para una perspectiva de mediano plazo, este análisis técnico de {INTERVALO_VELAS} es INSUFICIENTE. Se deberían analizar gráficos con intervalos mayores (ej. '1d' - diario, '1w' - semanal) para identificar tendencias más largas.\n"
    resumen +=  "    - CRUCIALMENTE, el análisis de mediano plazo DEBE incluir un estudio profundo de los **FUNDAMENTOS DEL PROYECTO** (tecnología subyacente, equipo de desarrollo, caso de uso real y adopción, tokenomics del activo, hoja de ruta, comunidad activa, competencia en el sector) y el **CONTEXTO DEL MERCADO** (sentimiento general del mercado cripto, ciclos de mercado, factores macroeconómicos como tasas de interés, inflación, regulaciones). Este script NO cubre estos aspectos fundamentales y críticos.\n"
    
    resumen += "\n- **Sobre Invertir un Capital de $5 (Ejemplo Teórico con Fines Educativos):**\n"
    resumen += "    - **Comisiones y Mínimos de Transacción:** Con un capital tan pequeño como $5 USD, las comisiones de transacción (tanto para comprar como para vender) en la mayoría de los exchanges de criptomonedas probablemente consumirían una parte muy significativa o incluso la totalidad de ese monto. Muchos exchanges tienen comisiones mínimas por operación (ej. $0.10 a $1 o más) o un porcentaje del valor de la operación. Además, puede haber mínimos de compra/venta para ciertos pares.\n"
    resumen += "    - **Ejemplo Ilustrativo de Comisiones:** Si una comisión es de $0.50 por operación, para comprar y luego vender necesitarías $1 en comisiones, lo que es el 20% de $5. Para que una inversión de $5 sea rentable, el activo necesitaría subir más del 20% solo para cubrir costos, sin contar el spread (diferencia entre precio de compra y venta).\n"
    resumen += "    - **Viabilidad Práctica:** Esto hace que la inversión directa con montos tan pequeños sea generalmente impráctica y poco rentable para la mayoría de los activos en exchanges centralizados. Es más un ejercicio teórico para entender cómo se aplican los conceptos.\n"
    resumen += "    - **Propósito de la Mención:** La mención de $5 es para ilustrar que, aunque los principios de análisis técnico se pueden aplicar conceptualmente, la ejecución práctica de una inversión depende de muchos otros factores, incluyendo el capital disponible, los costos de transacción, y la estrategia de gestión de riesgos. El tamaño del capital afecta directamente la viabilidad de ciertas estrategias.\n"
    resumen += "    - **ESTE SCRIPT NO ESTÁ DISEÑADO NI RECOMENDADO PARA REALIZAR OPERACIONES REALES CON DINERO, INDEPENDIENTEMENTE DEL MONTO.** Su propósito es educativo.\n"

    resumen += "\n**Factores Críticos NO Considerados por este Script (Esencial para un Análisis de Inversión Real):**\n"
    resumen += "  Este script es una herramienta técnica básica y omite muchos factores cruciales, incluyendo:\n"
    resumen += "  - **Análisis Fundamental Profundo:** Evaluación del valor intrínseco del proyecto, tecnología, equipo, hoja de ruta, adopción real, tokenomics (distribución, utilidad, inflación/deflación del token), comunidad, competencia, auditorías de seguridad, etc.\n"
    resumen += "  - **Sentimiento General del Mercado Cripto:** El mercado cripto es altamente influenciado por el sentimiento general. Herramientas como el 'Fear & Greed Index', análisis de tendencias en redes sociales, y el flujo de noticias del sector son importantes.\n"
    resumen += "  - **Eventos Macroeconómicos y Geopolíticos:** Tasas de interés globales, inflación, políticas monetarias de bancos centrales, regulaciones gubernamentales, eventos geopolíticos pueden tener un impacto masivo en todos los mercados, incluyendo cripto.\n"
    resumen += "  - **Análisis de Volumen Detallado y Perfiles de Volumen:** El volumen confirma la fuerza de una tendencia. Picos de volumen en rupturas o caídas son significativos. Los perfiles de volumen muestran dónde ha ocurrido la mayor parte del trading, identificando zonas de soporte/resistencia.\n"
    resumen += "  - **Estructura del Mercado Avanzada:** Identificación de soportes y resistencias clave a largo plazo, líneas de tendencia mayores, patrones gráficos complejos (ej. cabeza y hombros, triángulos, cuñas, canales), ondas de Elliott, etc.\n"
    resumen += "  - **Liquidez del Par:** Pares con baja liquidez pueden tener alta volatilidad y spreads grandes, dificultando la entrada y salida de posiciones.\n"
    resumen += "  - **Gestión de Riesgos Personalizada:** Definición de stop-loss, take-profit, tamaño de posición adecuado a la tolerancia al riesgo y al capital total, diversificación de la cartera.\n"
    resumen += "  - **Psicología del Trading:** Control emocional, disciplina, evitar el FOMO (miedo a quedarse fuera) y FUD (miedo, incertidumbre y duda).\n"

    resumen += "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    resumen += "!!! RECORDATORIO FINAL Y ABSOLUTO - ESTE SCRIPT ES PURAMENTE EDUCATIVO Y UNA SIMPLIFICACIÓN EXTREMA !!!\n"
    resumen += "!!! NO TOME NINGUNA DECISIÓN FINANCIERA BASADO EN LA SALIDA DE ESTE SCRIPT. !!!\n"
    resumen += "!!! CONSULTE A UN ASESOR FINANCIERO PROFESIONAL. REALICE SU PROPIA INVESTIGACIÓN EXHAUSTIVA (DYOR). !!!\n"
    resumen += "!!! EL TRADING ES RIESGOSO. USTED ES EL ÚNICO RESPONSABLE DE SUS DECISIONES Y POSIBLES PÉRDIDAS. !!!\n"
    resumen += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    return resumen

# --- Funciones de Gráficos ---

def graficar_datos_tecnicos(df, par, fib_levels=None): 
    """
    Genera y muestra gráficos de los datos técnicos, incluyendo precios, SMAs,
    Bandas de Bollinger, RSI, MACD y niveles de Fibonacci (si se proporcionan).
    """
    if df.empty or len(df) < 2:
        print(f"No hay suficientes datos para graficar para {par}.")
        return

    # Verificar que las columnas necesarias para graficar existen y tienen datos válidos
    columnas_necesarias_plot = ['Close', 'SMA_corto', 'SMA_largo', 'Bollinger_High', 'Bollinger_Low', 'RSI', 'MACD', 'MACD_signal']
    faltan_columnas_o_datos = [
        col for col in columnas_necesarias_plot 
        if col not in df.columns or df[col].isnull().all()
    ]

    if faltan_columnas_o_datos:
        print(f"Faltan datos o columnas completas para graficar algunos indicadores en {par}: {', '.join(faltan_columnas_o_datos)}. El gráfico podría estar incompleto o no generarse.")
        # No retornamos inmediatamente, intentamos graficar lo que se pueda

    print(f"Generando gráficos para {par}...")
    # Crear la figura y los subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 13), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]}) # Un poco más grande
    fig.suptitle(f'Análisis Técnico para {par} (Intervalo: {INTERVALO_VELAS}, Histórico: {PERIODO_HISTORICO})', fontsize=18, fontweight='bold')

    # Subplot 1: Precio, SMAs, Bandas de Bollinger y Fibonacci
    axs[0].plot(df['Timestamp'], df['Close'], label='Precio de Cierre (USDT)', color='dodgerblue', alpha=0.9, linewidth=1.8)
    if 'SMA_corto' in df.columns and not df['SMA_corto'].isnull().all():
        axs[0].plot(df['Timestamp'], df['SMA_corto'], label=f'SMA {SMA_CORTO_PERIODO}p', color='darkorange', linestyle='--', linewidth=1.5)
    if 'SMA_largo' in df.columns and not df['SMA_largo'].isnull().all():
        axs[0].plot(df['Timestamp'], df['SMA_largo'], label=f'SMA {SMA_LARGO_PERIODO}p', color='purple', linestyle='--', linewidth=1.5)
    
    if 'Bollinger_High' in df.columns and not df['Bollinger_High'].isnull().all() and \
       'Bollinger_Low' in df.columns and not df['Bollinger_Low'].isnull().all():
        axs[0].plot(df['Timestamp'], df['Bollinger_High'], label='Bollinger Sup.', color='slategrey', linestyle=':', alpha=0.7, linewidth=1.2)
        axs[0].plot(df['Timestamp'], df['Bollinger_Low'], label='Bollinger Inf.', color='slategrey', linestyle=':', alpha=0.7, linewidth=1.2)
        if 'Bollinger_Mid' in df.columns and not df['Bollinger_Mid'].isnull().all():
             axs[0].plot(df['Timestamp'], df['Bollinger_Mid'], label=f'Bollinger Medio ({BOLLINGER_WINDOW}p SMA)', color='lightgrey', linestyle='-.', alpha=0.8, linewidth=1.2)
        axs[0].fill_between(df['Timestamp'], df['Bollinger_High'], df['Bollinger_Low'], color='whitesmoke', alpha=0.5, label='Rango Bollinger') # Más sutil
    
    # Dibujar Niveles de Fibonacci si están disponibles
    if fib_levels:
        # Marcar los puntos usados para el cálculo de Fibonacci
        axs[0].plot(fib_levels["p_bajo"][1], fib_levels["p_bajo"][0], 'o', color='limegreen', markersize=8, label=f'Mín. Rango Fibo ({fib_levels["p_bajo"][0]:.2f})', zorder=5)
        axs[0].plot(fib_levels["p_alto"][1], fib_levels["p_alto"][0], 'o', color='crimson', markersize=8, label=f'Máx. Rango Fibo ({fib_levels["p_alto"][0]:.2f})', zorder=5)
        
        fibo_colors_retracement = ['mediumseagreen', 'springgreen', 'palegreen', 'lightgreen', 'honeydew']
        fibo_colors_extension = ['lightcoral', 'salmon', 'tomato', 'orangered', 'red']
        
        # Graficar Retrocesos
        for i, (name, val) in enumerate(sorted(fib_levels["retrocesos"].items(), key=lambda item: item[1])):
            if "Extremo" not in name: # No volver a graficar 0% y 100% si son los puntos de rango
                 color_idx = i % len(fibo_colors_retracement)
                 axs[0].axhline(y=val, color=fibo_colors_retracement[color_idx], linestyle='--', linewidth=1, alpha=0.8, label=f'Fib R {name}')
        # Graficar Extensiones
        for i, (name, val) in enumerate(sorted(fib_levels["extensiones"].items(), key=lambda item: item[1])):
            color_idx = i % len(fibo_colors_extension)
            axs[0].axhline(y=val, color=fibo_colors_extension[color_idx], linestyle=':', linewidth=1, alpha=0.8, label=f'{name}') # Nombre ya tiene "Ext"

    axs[0].set_ylabel('Precio (USDT)', fontsize=11)
    axs[0].set_title('Precio, Indicadores Principales y Niveles Fibonacci', fontsize=13, fontweight='bold')
    axs[0].legend(loc='best', fontsize=8) # 'best' para auto-posición, ajustar tamaño
    axs[0].grid(True, linestyle=':', alpha=0.5)
    axs[0].tick_params(axis='y', labelcolor='tab:blue')


    # Subplot 2: RSI
    if 'RSI' in df.columns and not df['RSI'].isnull().all():
        axs[1].plot(df['Timestamp'], df['RSI'], label=f'RSI ({RSI_PERIODO}p)', color='teal', linewidth=1.5)
        axs[1].axhline(70, linestyle='--', color='red', alpha=0.7, linewidth=1, label='Sobrecompra (70)')
        axs[1].axhline(30, linestyle='--', color='green', alpha=0.7, linewidth=1, label='Sobreventa (30)')
        axs[1].axhline(50, linestyle=':', color='grey', alpha=0.6, linewidth=1, label='Neutral (50)')
        axs[1].fill_between(df['Timestamp'], 70, 100, color='mistyrose', alpha=0.3)
        axs[1].fill_between(df['Timestamp'], 0, 30, color='lightcyan', alpha=0.3)
        axs[1].set_ylabel('RSI', fontsize=11)
        axs[1].set_ylim(0, 100) # RSI siempre está entre 0 y 100
        axs[1].set_title(f'Índice de Fuerza Relativa (RSI {RSI_PERIODO}p)', fontsize=13, fontweight='bold')
        axs[1].legend(loc='best', fontsize=9)
        axs[1].grid(True, linestyle=':', alpha=0.5)
        axs[1].tick_params(axis='y', labelcolor='teal')
    else:
        axs[1].text(0.5, 0.5, 'Datos de RSI no disponibles', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12, color='grey')

    # Subplot 3: MACD
    if 'MACD' in df.columns and not df['MACD'].isnull().all() and \
       'MACD_signal' in df.columns and not df['MACD_signal'].isnull().all():
        axs[2].plot(df['Timestamp'], df['MACD'], label='Línea MACD', color='deepskyblue', linewidth=1.5)
        axs[2].plot(df['Timestamp'], df['MACD_signal'], label='Línea de Señal', color='salmon', linestyle='--', linewidth=1.5)
        if 'MACD_hist' in df.columns and not df['MACD_hist'].isnull().all():
            colores_hist = ['mediumseagreen' if val >= 0 else 'lightcoral' for val in df['MACD_hist']]
            # Ajustar el ancho de la barra para que sea relativo al intervalo de tiempo
            if len(df['Timestamp']) > 1:
                # Calcula la diferencia promedio entre timestamps para el ancho
                time_diffs = df['Timestamp'].diff().fillna(pd.Timedelta(seconds=0))
                avg_time_diff = time_diffs.mean()
                bar_width_val = avg_time_diff * 0.8 # 80% del intervalo promedio como ancho
            else: # Fallback si solo hay una vela (no debería ocurrir si len(df) < 2 ya se maneja)
                 bar_width_val = 0.05 # Un valor por defecto pequeño
            axs[2].bar(df['Timestamp'], df['MACD_hist'], label='Histograma MACD', color=colores_hist, alpha=0.5, width=bar_width_val)
        axs[2].axhline(0, linestyle=':', color='grey', alpha=0.6, linewidth=1) # Línea cero para el histograma
        axs[2].set_ylabel('MACD', fontsize=11)
        axs[2].set_title(f'MACD ({MACD_FAST_PERIODO}p, {MACD_SLOW_PERIODO}p, {MACD_SIGN_PERIODO}p)', fontsize=13, fontweight='bold')
        axs[2].legend(loc='best', fontsize=9)
        axs[2].grid(True, linestyle=':', alpha=0.5)
        axs[2].tick_params(axis='y', labelcolor='deepskyblue')
    else:
        axs[2].text(0.5, 0.5, 'Datos de MACD no disponibles', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, fontsize=12, color='grey')

    # Formato común para el eje X (fechas)
    plt.xlabel('Fecha y Hora', fontsize=12, fontweight='bold')
    # Usar mdates para un mejor formateo de las fechas si hay muchos datos
    if len(df) > 50: # Si hay muchos datos, rotar y usar un localizador más inteligente
        axs[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[2].xaxis.get_major_locator()))
    else: # Para menos datos, un formato más simple puede ser suficiente
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right', fontsize=9) 
    
    fig.align_labels() # Alinear etiquetas de los ejes Y
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el título principal y etiquetas del eje x
    # plt.savefig(f"{par}_analisis_tecnico_detallado.png", dpi=200) # Descomentar para guardar el gráfico con alta resolución
    plt.show()
    print(f"Gráficos para {par} mostrados.")
    print("--- GUÍA RÁPIDA PARA INTERPRETAR LOS GRÁFICOS (FINES EDUCATIVOS) ---")
    print("1. Gráfico Superior (Precio y Medias):")
    print("   - Precio (azul): Evolución del precio de cierre.")
    print("   - SMAs (naranja/púrpura): Medias móviles. Cruces entre ellas pueden indicar cambios de tendencia (ej. 'Cruce Dorado').")
    print("   - Bandas de Bollinger (gris): Miden volatilidad. Precio cerca de bandas extremas puede indicar sobrecompra/sobreventa relativa.")
    print("   - Niveles Fibonacci (líneas discontinuas/punteadas de colores): Posibles zonas de soporte o resistencia basadas en el rango de precios del periodo.")
    print("2. Gráfico Medio (RSI):")
    print("   - Línea Verde: RSI. >70 (rojo) = posible sobrecompra. <30 (azul) = posible sobreventa. Cerca de 50 (gris) = neutral.")
    print("3. Gráfico Inferior (MACD):")
    print("   - Línea MACD (cyan) y Línea de Señal (salmón): Cruces entre ellas pueden indicar cambios de momento.")
    print("   - Histograma (barras verdes/rojas): Diferencia entre MACD y Señal. Barras grandes indican fuerte momento.")
    print("¡RECUERDA! Estos son solo indicadores. Un análisis completo requiere más herramientas, contexto y experiencia. NO TOMES DECISIONES DE INVERSIÓN BASADO SOLO EN ESTO.\n")

# --- Función Principal (main) ---
def main():
    """
    Función principal que orquesta el análisis de criptomonedas:
    obtiene datos, calcula indicadores, genera noticias, interpretaciones y gráficos.
    """
    if not client:
        print("Error Crítico: El cliente de Binance no se pudo inicializar. El script no puede continuar.")
        return
        
    # Mensaje de bienvenida y advertencia principal
    print("======================================================================================")
    print("--- SCRIPT DE ANÁLISIS TÉCNICO BÁSICO DE CRIPTOMONEDAS (BINANCE) ---")
    print("======================================================================================")
    print(f"Configuración Actual: Intervalo de Velas: {INTERVALO_VELAS}, Periodo Histórico para Datos: {PERIODO_HISTORICO}")
    print("======================================================================================")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADVERTENCIA EXTREMADAMENTE IMPORTANTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ESTE SCRIPT ES UNA HERRAMIENTA PURAMENTE EDUCATIVA Y DE DEMOSTRACIÓN. !!!")
    print("!!! NO CONSTITUYE ASESORAMIENTO FINANCIERO, LEGAL O DE NINGÚN OTRO TIPO. !!!")
    print("!!! EL TRADING DE CRIPTOMONEDAS, ESPECIALMENTE 'MEME COINS', ES EXTREMADAMENTE VOLÁTIL Y RIESGOSO. !!!")
    print("!!! PUEDES PERDER LA TOTALIDAD DE TU CAPITAL INVERTIDO. !!!")
    print("!!! Las interpretaciones de los indicadores técnicos son simplificadas, subjetivas y NO DEBEN tomarse como consejos de inversión o predicciones certeras. !!!")
    print("!!! ANTES DE TOMAR CUALQUIER DECISIÓN FINANCIERA, REALIZA TU PROPIA INVESTIGACIÓN EXHAUSTIVA (DYOR - Do Your Own Research) Y CONSIDERA CONSULTAR A UN ASESOR FINANCIERO PROFESIONAL REGISTRADO. !!!")
    print("!!! EL USO DE ESTE SCRIPT ES BAJO TU PROPIO Y ÚNICO RIESGO. NO NOS HACEMOS RESPONSABLES DE NINGUNA PÉRDIDA. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")


    if PARES_A_ANALIZAR_MANUALMENTE:
        pares_para_analizar = PARES_A_ANALIZAR_MANUALMENTE
        print(f"Se analizarán los siguientes pares definidos manualmente: {pares_para_analizar}\n")
    else:
        print("No se especificó una lista manual de pares. Se intentará obtener los más populares por volumen (esto puede variar).")
        pares_para_analizar = obtener_pares_populares(NUMERO_DE_PARES_POR_VOLUMEN)
        if not pares_para_analizar:
            print("No se pudieron obtener pares para analizar (ni manuales ni por volumen). Terminando script.")
            return

    for i, par in enumerate(pares_para_analizar):
        print(f"\n==================== ANALIZANDO PAR {i+1}/{len(pares_para_analizar)}: {par} ====================")
        
        # Obtener el activo base para la búsqueda de noticias
        base_asset = None
        try:
            symbol_info = client.get_symbol_info(par)
            if symbol_info:
                base_asset = symbol_info['baseAsset']
        except Exception as e:
            print(f"Advertencia: No se pudo obtener info detallada del símbolo para {par} vía API ({e}). Intentando extracción manual del activo base.")
            # Lógica de fallback mejorada para extraer el activo base
            if par.endswith("USDT"): base_asset = par[:-4]
            elif par.endswith("BUSD"): base_asset = par[:-4] 
            elif par.endswith("FDUSD"): base_asset = par[:-5]
            elif par.endswith("TUSD"): base_asset = par[:-4]
            else: # Intento más genérico
                common_quotes = ["BTC", "ETH", "BNB"]
                for cq in common_quotes:
                    if par.endswith(cq) and len(par) > len(cq):
                        base_asset = par[:-len(cq)]
                        break
                if not base_asset: # Último recurso, tomar los primeros 3 o 4 caracteres
                     base_asset = par[:4] if len(par) > 4 and par[3].isalpha() else par[:3]
            print(f"Activo base inferido manualmente: {base_asset}" if base_asset else "No se pudo inferir el activo base.")


        # Obtener y mostrar noticias
        if not base_asset:
            print(f"No se pudo determinar el activo base para {par}. Saltando sección de noticias para este par.")
        else:
            print(f"Activo base identificado para noticias: {base_asset}")
            noticias = obtener_noticias_relevantes(base_asset)
            print("\n*** Noticias Relevantes Recientes (Contexto General - Fuente: NewsAPI): ***")
            if noticias and not "Servicio de noticias no disponible" in noticias[0] and not "No se encontraron noticias recientes" in noticias[0]:
                for j, noticia in enumerate(noticias):
                    print(f"  {j+1}. {noticia}")
            elif noticias: # Si hay un mensaje de error o "no encontradas"
                print(f"  {noticias[0]}")
            else: # Si la lista de noticias está vacía por alguna razón no manejada
                print("  No se pudo obtener información de noticias.")
            print("-" * 60) 
        
        # Obtener y procesar datos históricos
        df_par_historico = obtener_datos_historicos(par, INTERVALO_VELAS, PERIODO_HISTORICO)

        if not df_par_historico.empty:
            df_par_con_indicadores = calcular_indicadores_tecnicos(df_par_historico.copy()) # Usar .copy() para evitar SettingWithCopyWarning
            
            print(f"\n*** Últimos Datos e Indicadores Técnicos para {par} (Vela de {INTERVALO_VELAS}): ***")
            cols_mostrar = ['Timestamp', 'Close', 'SMA_corto', 'SMA_largo', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']
            # Mostrar solo columnas que existen y no son completamente NaN
            cols_existentes = [c for c in cols_mostrar if c in df_par_con_indicadores.columns and not df_par_con_indicadores[c].isnull().all()]
            
            if not df_par_con_indicadores[cols_existentes].empty:
                 df_display = df_par_con_indicadores[cols_existentes].tail(3).copy() # Mostrar las últimas 3 velas
                 if 'Timestamp' in df_display.columns: # Formatear Timestamp para mejor lectura
                    df_display['Timestamp'] = df_display['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                 print(df_display.to_string()) # .to_string() para mejor formato en consola
            else:
                print("  No hay datos de indicadores suficientes para mostrar en la tabla (podría ser por falta de datos históricos).")

            # Interpretación técnica básica
            print("\n*** Interpretación Técnica Básica de Indicadores (¡NO ES ASESORAMIENTO FINANCIERO!): ***")
            interpretacion_std = analizar_tendencia_basica(df_par_con_indicadores)
            print(f"{interpretacion_std}")

            # Calcular y mostrar niveles Fibonacci
            fib_levels, reporte_fib = calcular_niveles_fibonacci(df_par_con_indicadores)
            if reporte_fib: 
                print("\n*** Niveles de Fibonacci (Basados en el rango del periodo histórico analizado): ***")
                print(reporte_fib)
            else: # Si no se pudo generar el reporte (ej. datos insuficientes)
                print("\n*** Niveles de Fibonacci: ***")
                print("  No se pudieron calcular los niveles de Fibonacci (datos insuficientes o rango no válido).")

            # Generar y mostrar el resumen educativo de "inversión"
            # Esta es la sección que el usuario pidió detallar
            resumen_inv_edu = generar_resumen_educativo_inversion(df_par_con_indicadores, fib_levels, par)
            print(resumen_inv_edu) # Ya incluye los disclaimers necesarios

            # Generar y mostrar gráficos
            graficar_datos_tecnicos(df_par_con_indicadores, par, fib_levels) # Pasar fib_levels al gráfico
        else:
            print(f"No se pudieron obtener o procesar datos técnicos para {par}. No se puede continuar el análisis para este par.")
        
        print(f"--- Fin del Análisis para {par} ---\n")
        if i < len(pares_para_analizar) - 1: # Si no es el último par
            print(">>> Presiona Enter en la consola para continuar con el siguiente par...")
            input() # Espera a que el usuario presione Enter
        else:
            print(">>> Todos los pares seleccionados han sido analizados.")

    print("\n===================================================")
    print("--- ANÁLISIS DE TODOS LOS PARES COMPLETADO ---")
    print("===================================================")
    print("RECUERDA: El trading de criptomonedas implica riesgos significativos. Este script es una herramienta puramente educativa y NO debe ser usado para tomar decisiones de inversión reales.")
    print("Investiga a fondo y considera buscar asesoramiento profesional.")

if __name__ == "__main__":
    # La verificación de claves API ahora se maneja principalmente en la inicialización de los clientes.
    # El script intentará funcionar con los datos públicos si las claves no son válidas o no se proporcionan.
    main()

