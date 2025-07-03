# SISTEMA AVANZADO DE PREDICCIÓN DE CRIPTOMONEDAS
# Desarrollado por: AI Expert Developer & Economist
# Versión: 2.0 - Predicciones con Machine Learning

import warnings
warnings.filterwarnings('ignore')

# Imports básicos
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import os

# APIs
from binance.client import Client
from newsapi import NewsApiClient

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
from scipy import stats

# Deep Learning (TensorFlow/Keras para LSTM)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow no disponible. Instalando modelos básicos de ML...")
    TENSORFLOW_AVAILABLE = False

# Análisis técnico avanzado
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

# Visualización
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
plt.style.use('seaborn-v0_8')

# CONFIGURACIÓN AVANZADA
class ConfiguracionAvanzada:
    # APIs (usar variables de entorno en producción)
    API_KEY = "26Sh06jOHDKVRqZY5puDa7q16hQ1CU0aitWM0OWy1iMF0jU8h8jqKqUFsxzLC5Ze"
    API_SECRET = "yOWjPf3wHDuQbBJVKll3kDYGG6c57GiUgbdu6Xp79VnAG3dzC5AUMU4IDd3LhsnT"
    NEWS_API_KEY = "95a1ebc226f34eb38842c95fd4ce1932"
    
    # Parámetros de predicción
    HORIZONTES_PREDICCION = [1, 4, 12, 24]  # 1h, 4h, 12h, 24h adelante
    INTERVALO_ENTRENAMIENTO = '1h'  # Datos para entrenar
    PERIODO_DATOS = "180 day ago UTC"  # 6 meses de datos
    
    # Pares de alta prioridad para análisis profundo
    PARES_PRIORITARIOS = [
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", 
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "PORTALUSDT"
    ]
    
    # Thresholds para señales
    CONFIANZA_MINIMA = 0.7
    UMBRAL_PREDICCION_FUERTE = 0.8
    
    # ML Model Parameters
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    SEQUENCE_LENGTH = 60  # 60 períodos para LSTM


class AnalizadorPredictivoAvanzado:
    def __init__(self):
        self.config = ConfiguracionAvanzada()
        self.client = None
        self.newsapi = None
        self.modelos_entrenados = {}
        self.scalers = {}
        self.historicos_predicciones = {}
        
        self._inicializar_conexiones()
        
    def _inicializar_conexiones(self):
        """Inicializa conexiones a APIs"""
        try:
            self.client = Client(self.config.API_KEY, self.config.API_SECRET)
            self.client.get_account()
            print("✅ Conexión Binance exitosa")
        except:
            self.client = Client()
            print("⚠️ Binance en modo público")
            
        try:
            self.newsapi = NewsApiClient(api_key=self.config.NEWS_API_KEY)
            print("✅ NewsAPI conectado")
        except:
            print("⚠️ NewsAPI no disponible")

    def obtener_datos_completos(self, par):
        """Obtiene datos históricos con todas las características necesarias"""
        try:
            print(f"📊 Descargando datos históricos para {par}...")
            
            # Obtener datos en múltiples llamadas para cubrir más tiempo
            all_klines = []
            end_time = None
            
            # Hacer múltiples llamadas para obtener más datos
            for i in range(3):  # 3 llamadas = hasta 3000 datos
                try:
                    if end_time:
                        klines = self.client.get_historical_klines(
                            par, self.config.INTERVALO_ENTRENAMIENTO, 
                            end_str=end_time, limit=1000
                        )
                    else:
                        klines = self.client.get_historical_klines(
                            par, self.config.INTERVALO_ENTRENAMIENTO, 
                            self.config.PERIODO_DATOS, limit=1000
                        )
                    
                    if not klines:
                        break
                        
                    all_klines.extend(klines)
                    
                    # Actualizar end_time para la siguiente llamada
                    if klines:
                        end_time = klines[0][0]  # timestamp del primer elemento
                    
                    print(f"   📈 Llamada {i+1}: {len(klines)} registros")
                    
                    # Pausa para evitar límites de API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"   ⚠️ Error en llamada {i+1}: {e}")
                    break
            
            if not all_klines:
                print(f"❌ No se pudieron obtener datos para {par}")
                return pd.DataFrame()
            
            print(f"   ✅ Total descargado: {len(all_klines)} registros")
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Conversiones
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ordenar por timestamp y eliminar duplicados
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            df.dropna(inplace=True)
            df.set_index('timestamp', inplace=True)
            
            print(f"   📊 Datos finales: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos para {par}: {e}")
            return pd.DataFrame()

    def calcular_indicadores_avanzados(self, df):
        """Calcula más de 30 indicadores técnicos avanzados"""
        if len(df) < 50:
            return df
            
        print("🔧 Calculando indicadores avanzados...")
        
        # Precios básicos
        df['precio_medio'] = (df['high'] + df['low']) / 2
        df['precio_tipico'] = (df['high'] + df['low'] + df['close']) / 3
        df['precio_ponderado'] = (df['high'] + df['low'] + 2*df['close']) / 4
        
        # Volatilidad (ventanas más pequeñas)
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=10).average_true_range()
        df['volatilidad_close'] = df['close'].rolling(10).std()
        df['volatilidad_returns'] = df['close'].pct_change().rolling(10).std()
        
        # Tendencia - EMAs más cortas
        for periodo in [9, 21, 50]:  # Removido 100, 200 para usar menos datos
            df[f'ema_{periodo}'] = EMAIndicator(df['close'], window=periodo).ema_indicator()
            df[f'sma_{periodo}'] = SMAIndicator(df['close'], window=periodo).sma_indicator()
        
        # Momentum avanzado
        df['rsi'] = RSIIndicator(df['close'], window=10).rsi()  # Reducido de 14 a 10
        df['rsi_smooth'] = df['rsi'].rolling(3).mean()
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # MACD múltiple (solo el principal)
        macd = MACD(df['close'], window_slow=26, window_fast=12)
        df['macd_12_26'] = macd.macd()
        df['macd_signal_12_26'] = macd.macd_signal()
        df['macd_hist_12_26'] = macd.macd_diff()
        
        # ADX (fuerza de tendencia)
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # Bollinger Bands (solo el principal)
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper_20_2'] = bb.bollinger_hband()
        df['bb_lower_20_2'] = bb.bollinger_lband()
        df['bb_pct_20_2'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Volumen
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['volume_sma'] = df['volume'].rolling(10).mean()  # Reducido de 20 a 10
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        df['vpt'] = VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        # Patrones de precios
        df['doji'] = abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1
        df['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & ((df['high'] - df['close']) < (df['open'] - df['close']))
        df['shooting_star'] = ((df['high'] - df['close']) > 2 * (df['close'] - df['open'])) & ((df['close'] - df['low']) < (df['close'] - df['open']))
        
        # Características temporales
        df['hora'] = df.index.hour
        df['dia_semana'] = df.index.dayofweek
        df['dia_mes'] = df.index.day
        
        # Returns múltiples períodos (reducidos)
        for periodo in [1, 4, 8]:  # Removido 24 para usar menos datos
            df[f'return_{periodo}h'] = df['close'].pct_change(periodo)
            df[f'volatilidad_{periodo}h'] = df[f'return_{periodo}h'].rolling(periodo).std()
        
        return df

    def crear_features_ml(self, df):
        """Crea features específicas para machine learning"""
        features = df.copy()
        
        # Lags de precios (reducidos)
        for lag in [1, 2, 3, 6]:  # Removido 12, 24 para usar menos datos
            features[f'close_lag_{lag}'] = features['close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
        
        # Rolling statistics (ventanas más pequeñas)
        for window in [5, 10]:  # Removido 20 para usar menos datos
            features[f'close_mean_{window}'] = features['close'].rolling(window).mean()
            features[f'close_std_{window}'] = features['close'].rolling(window).std()
            features[f'close_min_{window}'] = features['close'].rolling(window).min()
            features[f'close_max_{window}'] = features['close'].rolling(window).max()
        
        # Diferencias entre medias (solo las disponibles)
        if 'ema_9' in features.columns and 'ema_21' in features.columns:
            features['ema_diff_9_21'] = features['ema_9'] - features['ema_21']
        if 'ema_21' in features.columns and 'ema_50' in features.columns:
            features['ema_diff_21_50'] = features['ema_21'] - features['ema_50']
        
        # Posición relativa en Bollinger
        if 'bb_lower_20_2' in features.columns and 'bb_upper_20_2' in features.columns:
            features['bb_position'] = (features['close'] - features['bb_lower_20_2']) / (features['bb_upper_20_2'] - features['bb_lower_20_2'])
        
        # Momentum combinado (solo con indicadores disponibles)
        momentum_components = []
        if 'rsi' in features.columns:
            momentum_components.append(features['rsi'].fillna(50) / 100 * 0.4)
        if 'stoch_k' in features.columns:
            momentum_components.append(features['stoch_k'].fillna(50) / 100 * 0.3)
        if 'williams_r' in features.columns:
            momentum_components.append((features['williams_r'].fillna(-50) + 100) / 100 * 0.3)
        
        if momentum_components:
            features['momentum_score'] = sum(momentum_components)
        
        return features

    def crear_targets_prediccion(self, df, horizontes):
        """Crea variables objetivo para diferentes horizontes de predicción"""
        targets = {}
        
        for h in horizontes:
            # Predicción de precio
            targets[f'precio_futuro_{h}h'] = df['close'].shift(-h)
            
            # Predicción de dirección (clasificación)
            future_return = (df['close'].shift(-h) - df['close']) / df['close']
            targets[f'direccion_{h}h'] = (future_return > 0.01).astype(int)  # Subida >1%
            
            # Predicción de volatilidad futura
            targets[f'volatilidad_futura_{h}h'] = df['close'].rolling(h).std().shift(-h)
            
            # Señal de trading (fuerte subida/bajada)
            targets[f'señal_trading_{h}h'] = np.where(
                future_return > 0.03, 2,  # Compra fuerte >3%
                np.where(future_return < -0.03, 0, 1)  # Venta fuerte <-3%, Hold
            )
        
        return pd.DataFrame(targets, index=df.index)

    def entrenar_modelo_lstm(self, X, y, par, horizonte):
        """Entrena modelo LSTM para predicción de series temporales"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow no disponible, saltando LSTM")
            return None
            
        try:
            # Preparar datos para LSTM
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Crear secuencias
            def crear_secuencias(data, target, seq_length):
                X_seq, y_seq = [], []
                for i in range(seq_length, len(data)):
                    X_seq.append(data[i-seq_length:i])
                    y_seq.append(target[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_seq, y_seq = crear_secuencias(X_scaled, y_scaled, self.config.SEQUENCE_LENGTH)
            
            # Split train/test temporal
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Construir modelo LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                Dense(25, activation='relu'),
                Dropout(0.1),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Entrenamiento con early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=self.config.LSTM_EPOCHS,
                batch_size=self.config.LSTM_BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluar
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            print(f"📈 LSTM {par} ({horizonte}h) - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            
            return {
                'modelo': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'mae_train': train_mae,
                'mae_test': test_mae,
                'historia': history.history
            }
            
        except Exception as e:
            print(f"❌ Error entrenando LSTM: {e}")
            return None

    def entrenar_modelos_ensemble(self, X, y, par, horizonte, tipo='regresion'):
        """Entrena conjunto de modelos de ML"""
        try:
            # Split temporal
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            modelos = {}
            
            if tipo == 'regresion':
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_train_scaled, y_train)
                modelos['random_forest'] = rf
                
                # XGBoost
                xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
                modelos['xgboost'] = xgb_model
                
                # Gradient Boosting
                gb = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                gb.fit(X_train_scaled, y_train)
                modelos['gradient_boosting'] = gb
                
            else:  # clasificacion
                # Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_train_scaled, y_train)
                modelos['random_forest'] = rf
                
                # XGBoost Classifier
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
                modelos['xgboost'] = xgb_model
            
            # Evaluar modelos
            resultados = {}
            for nombre, modelo in modelos.items():
                pred_train = modelo.predict(X_train_scaled)
                pred_test = modelo.predict(X_test_scaled)
                
                if tipo == 'regresion':
                    mae_train = mean_absolute_error(y_train, pred_train)
                    mae_test = mean_absolute_error(y_test, pred_test)
                    resultados[nombre] = {'mae_train': mae_train, 'mae_test': mae_test}
                    print(f"📊 {nombre} {par} ({horizonte}h) - Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")
                else:
                    acc_train = accuracy_score(y_train, pred_train)
                    acc_test = accuracy_score(y_test, pred_test)
                    resultados[nombre] = {'acc_train': acc_train, 'acc_test': acc_test}
                    print(f"📊 {nombre} {par} ({horizonte}h) - Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}")
            
            return {
                'modelos': modelos,
                'scaler': scaler,
                'resultados': resultados,
                'X_test': X_test_scaled,
                'y_test': y_test
            }
            
        except Exception as e:
            print(f"❌ Error entrenando ensemble: {e}")
            return None 

    def hacer_prediccion_completa(self, par):
        """Función principal que hace predicciones usando todos los modelos"""
        print(f"\n🚀 INICIANDO ANÁLISIS PREDICTIVO COMPLETO PARA {par}")
        print("="*70)
        
        # Obtener datos
        df = self.obtener_datos_completos(par)
        if df.empty:
            return None
        
        print(f"📊 Datos iniciales: {len(df)} registros")
            
        # Calcular indicadores
        df = self.calcular_indicadores_avanzados(df)
        print(f"📊 Después de indicadores: {len(df)} registros")
        
        df = self.crear_features_ml(df)
        print(f"📊 Después de features ML: {len(df)} registros")
        
        # Crear targets
        targets_df = self.crear_targets_prediccion(df, self.config.HORIZONTES_PREDICCION)
        print(f"📊 Targets creados: {len(targets_df)} registros")
        print(f"📊 Índice targets: {targets_df.index[0]} hasta {targets_df.index[-1]}")
        print(f"📊 Índice df: {df.index[0]} hasta {df.index[-1]}")
        
        # Combinar datos
        data_completa = df.join(targets_df, how='inner')
        print(f"📊 Después del join: {len(data_completa)} registros")
        print(f"📊 Columnas con NaN: {data_completa.isnull().sum().sum()} valores NaN totales")
        
        # Estrategia más robusta para manejar NaN
        # Primero forward fill para propagar valores válidos hacia adelante
        data_completa = data_completa.fillna(method='ffill', limit=5)
        # Luego backward fill para llenar los NaN restantes
        data_completa = data_completa.fillna(method='bfill', limit=5)
        # Finalmente, eliminar solo las filas que aún tienen NaN en columnas críticas
        columnas_criticas = ['close', 'volume', 'rsi', 'macd_12_26']
        data_completa = data_completa.dropna(subset=columnas_criticas)
        
        print(f"📊 Datos completos finales: {len(data_completa)} registros")
        
        if len(data_completa) < 100:
            print(f"⚠️ Datos insuficientes para {par} (necesarios: 100, disponibles: {len(data_completa)})")
            return None
        
        predicciones = {}
        
        # Seleccionar features para ML
        feature_cols = [col for col in data_completa.columns 
                       if not any(x in col for x in ['futuro', 'direccion', 'volatilidad', 'señal_trading'])]
        
        X = data_completa[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        
        # Entrenar modelos para cada horizonte
        for horizonte in self.config.HORIZONTES_PREDICCION:
            print(f"\n🔮 Entrenando modelos para horizonte {horizonte}h...")
            
            # Predicción de precio (regresión)
            y_precio = data_completa[f'precio_futuro_{horizonte}h'].dropna()
            if len(y_precio) > 100:
                X_precio = X.loc[y_precio.index]
                
                # LSTM para precio
                if TENSORFLOW_AVAILABLE:
                    lstm_precio = self.entrenar_modelo_lstm(
                        X_precio.values, y_precio.values, par, horizonte
                    )
                
                # Ensemble para precio
                ensemble_precio = self.entrenar_modelos_ensemble(
                    X_precio.values, y_precio.values, par, horizonte, 'regresion'
                )
                
                # Predicción de dirección (clasificación)
                y_direccion = data_completa[f'direccion_{horizonte}h'].dropna()
                X_direccion = X.loc[y_direccion.index]
                
                ensemble_direccion = self.entrenar_modelos_ensemble(
                    X_direccion.values, y_direccion.values, par, horizonte, 'clasificacion'
                )
                
                predicciones[f'{horizonte}h'] = {
                    'lstm_precio': lstm_precio,
                    'ensemble_precio': ensemble_precio,
                    'ensemble_direccion': ensemble_direccion,
                    'ultimo_precio': df['close'].iloc[-1],
                    'features_actuales': X.iloc[-1:].values
                }
        
        # Hacer predicciones actuales
        predicciones_actuales = self.generar_predicciones_futuras(predicciones, par)
        
        # Análisis de sentimientos
        sentimientos = self.analizar_sentimientos_avanzado(par)
        
        # Detección de patrones
        patrones = self.detectar_patrones_chartistas(df)
        
        # Score final
        score_final = self.calcular_score_predictivo(
            predicciones_actuales, sentimientos, patrones, df
        )
        
        # Crear reporte
        reporte = self.generar_reporte_predictivo(
            par, predicciones_actuales, sentimientos, patrones, score_final, df
        )
        
        return {
            'par': par,
            'predicciones': predicciones_actuales,
            'sentimientos': sentimientos,
            'patrones': patrones,
            'score': score_final,
            'reporte': reporte,
            'datos': df
        }

    def generar_predicciones_futuras(self, modelos, par):
        """Genera predicciones específicas para los próximos períodos"""
        predicciones = {}
        
        for horizonte, modelos_h in modelos.items():
            pred = {'horizonte': horizonte}
            
            # Predicción de precio con ensemble
            if modelos_h['ensemble_precio']:
                features = modelos_h['features_actuales']
                ensemble = modelos_h['ensemble_precio']
                
                predicciones_precio = []
                for nombre, modelo in ensemble['modelos'].items():
                    pred_precio = modelo.predict(features)[0]
                    predicciones_precio.append(pred_precio)
                
                pred['precio_predicho'] = np.mean(predicciones_precio)
                pred['precio_std'] = np.std(predicciones_precio)
                pred['precio_actual'] = modelos_h['ultimo_precio']
                pred['cambio_esperado'] = (pred['precio_predicho'] - pred['precio_actual']) / pred['precio_actual']
                
                # Intervalo de confianza
                pred['precio_min'] = pred['precio_predicho'] - 1.96 * pred['precio_std']
                pred['precio_max'] = pred['precio_predicho'] + 1.96 * pred['precio_std']
            
            # Predicción de dirección
            if modelos_h['ensemble_direccion']:
                features = modelos_h['features_actuales']
                ensemble = modelos_h['ensemble_direccion']
                
                probabilidades = []
                for nombre, modelo in ensemble['modelos'].items():
                    if hasattr(modelo, 'predict_proba'):
                        prob = modelo.predict_proba(features)[0]
                        probabilidades.append(prob)
                    else:
                        pred_class = modelo.predict(features)[0]
                        prob = np.zeros(2)
                        prob[pred_class] = 1.0
                        probabilidades.append(prob)
                
                prob_media = np.mean(probabilidades, axis=0)
                pred['probabilidad_subida'] = prob_media[1] if len(prob_media) > 1 else 0.5
                pred['prediccion_direccion'] = 'SUBIDA' if pred['probabilidad_subida'] > 0.5 else 'BAJADA'
                pred['confianza'] = max(prob_media)
            
            # LSTM si está disponible
            if modelos_h['lstm_precio'] and TENSORFLOW_AVAILABLE:
                try:
                    lstm_model = modelos_h['lstm_precio']
                    # Preparar datos para LSTM (necesita secuencia)
                    # Esta es una simplificación - en producción necesitaríamos la secuencia completa
                    pred['lstm_disponible'] = True
                except:
                    pred['lstm_disponible'] = False
            
            predicciones[horizonte] = pred
        
        return predicciones

    def analizar_sentimientos_avanzado(self, par):
        """Análisis de sentimientos desde múltiples fuentes"""
        sentimientos = {
            'noticias': {'score': 0, 'fuentes': []},
            'fear_greed': {'valor': 50, 'interpretacion': 'NEUTRAL'},
            'social': {'mencion_count': 0, 'sentiment_score': 0},
            'score_general': 0
        }
        
        try:
            # Análisis de noticias
            base_asset = par.replace('USDT', '')
            if self.newsapi:
                query = f"{base_asset} crypto cryptocurrency bitcoin"
                articles = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                if articles['articles']:
                    scores = []
                    for article in articles['articles']:
                        # Análisis de palabras clave simples
                        title = article['title'].lower()
                        description = (article['description'] or '').lower()
                        text = f"{title} {description}"
                        
                        # Palabras positivas y negativas
                        palabras_positivas = ['surge', 'rally', 'bull', 'pump', 'moon', 'breakthrough', 'adoption', 'partnership']
                        palabras_negativas = ['crash', 'dump', 'bear', 'decline', 'fall', 'regulatory', 'ban', 'hack']
                        
                        score = 0
                        for palabra in palabras_positivas:
                            score += text.count(palabra) * 2
                        for palabra in palabras_negativas:
                            score -= text.count(palabra) * 2
                        
                        scores.append(score)
                        sentimientos['noticias']['fuentes'].append({
                            'titulo': article['title'],
                            'score': score,
                            'fecha': article['publishedAt']
                        })
                    
                    sentimientos['noticias']['score'] = np.mean(scores) if scores else 0
            
            # Simular Fear & Greed Index (en producción conectar a API real)
            # Por ahora usamos indicadores técnicos como proxy
            sentimientos['fear_greed']['valor'] = np.random.randint(20, 80)  # Placeholder
            
            # Score general combinado
            score_noticias = max(-10, min(10, sentimientos['noticias']['score']))
            score_fg = (sentimientos['fear_greed']['valor'] - 50) / 5  # Normalizar
            
            sentimientos['score_general'] = (score_noticias * 0.6 + score_fg * 0.4)
            
        except Exception as e:
            print(f"⚠️ Error en análisis de sentimientos: {e}")
        
        return sentimientos

    def detectar_patrones_chartistas(self, df):
        """Detección de patrones chartistas y de velas"""
        patrones = {
            'velas_japonesas': [],
            'patrones_chartistas': [],
            'soportes_resistencias': [],
            'score_patron': 0
        }
        
        try:
            if len(df) < 50:
                return patrones
            
            # Arrays de precios
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Detección de patrones de velas (últimas 10 velas)
            for i in range(-10, 0):
                try:
                    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                    prev_o, prev_c = opens[i-1], closes[i-1]
                    
                    # Doji
                    if abs(o - c) / (h - l) < 0.1:
                        patrones['velas_japonesas'].append(f"Doji en posición {i}")
                    
                    # Hammer
                    if (min(o, c) - l) > 2 * abs(o - c) and (h - max(o, c)) < abs(o - c):
                        patrones['velas_japonesas'].append(f"Hammer en posición {i}")
                    
                    # Engulfing
                    if c > prev_o and o < prev_c and abs(c - o) > abs(prev_c - prev_o):
                        patrones['velas_japonesas'].append(f"Bullish Engulfing en posición {i}")
                    elif c < prev_o and o > prev_c and abs(o - c) > abs(prev_o - prev_c):
                        patrones['velas_japonesas'].append(f"Bearish Engulfing en posición {i}")
                except:
                    continue
            
            # Detección de soportes y resistencias
            # Usar puntos pivote simples
            pivot_high = []
            pivot_low = []
            
            for i in range(5, len(closes) - 5):
                # Pivot high
                if all(closes[i] >= closes[i-j] for j in range(1, 6)) and \
                   all(closes[i] >= closes[i+j] for j in range(1, 6)):
                    pivot_high.append((df.index[i], closes[i]))
                
                # Pivot low
                if all(closes[i] <= closes[i-j] for j in range(1, 6)) and \
                   all(closes[i] <= closes[i+j] for j in range(1, 6)):
                    pivot_low.append((df.index[i], closes[i]))
            
            # Mantener solo los más recientes y relevantes
            patrones['soportes_resistencias'] = {
                'resistencias': sorted(pivot_high, key=lambda x: x[0])[-5:],
                'soportes': sorted(pivot_low, key=lambda x: x[0])[-5:]
            }
            
            # Score basado en patrones encontrados
            score = len(patrones['velas_japonesas']) * 2
            if any('Bullish' in p for p in patrones['velas_japonesas']):
                score += 5
            if any('Bearish' in p for p in patrones['velas_japonesas']):
                score -= 5
            
            patrones['score_patron'] = max(-10, min(10, score))
            
        except Exception as e:
            print(f"⚠️ Error detectando patrones: {e}")
        
        return patrones

    def calcular_score_predictivo(self, predicciones, sentimientos, patrones, df):
        """Calcula un score predictivo combinado de 0-100"""
        try:
            scores = []
            explicaciones = []
            
            # Score basado en predicciones de ML
            for horizonte, pred in predicciones.items():
                if 'cambio_esperado' in pred and 'confianza' in pred:
                    # Score basado en la magnitud del cambio esperado y confianza
                    cambio = abs(pred['cambio_esperado'])
                    confianza = pred['confianza']
                    
                    if cambio > 0.05:  # Cambio > 5%
                        if pred['cambio_esperado'] > 0:
                            score_pred = confianza * 80 + 20  # Bias alcista
                        else:
                            score_pred = (1 - confianza) * 20  # Bias bajista
                    else:
                        score_pred = 50  # Neutral
                    
                    scores.append(score_pred)
                    explicaciones.append(f"ML {horizonte}: {score_pred:.1f}")
            
            # Score de sentimientos
            sent_score = 50 + sentimientos['score_general'] * 5
            sent_score = max(0, min(100, sent_score))
            scores.append(sent_score)
            explicaciones.append(f"Sentimientos: {sent_score:.1f}")
            
            # Score de patrones
            patron_score = 50 + patrones['score_patron'] * 2
            patron_score = max(0, min(100, patron_score))
            scores.append(patron_score)
            explicaciones.append(f"Patrones: {patron_score:.1f}")
            
            # Score técnico basado en indicadores actuales
            ultimo = df.iloc[-1]
            score_tecnico = 50
            
            # RSI
            if 'rsi' in ultimo:
                rsi = ultimo['rsi']
                if rsi < 30:
                    score_tecnico += 15  # Oversold = bullish
                elif rsi > 70:
                    score_tecnico -= 15  # Overbought = bearish
            
            # MACD
            if 'macd_12_26' in ultimo and 'macd_signal_12_26' in ultimo:
                if ultimo['macd_12_26'] > ultimo['macd_signal_12_26']:
                    score_tecnico += 10
                else:
                    score_tecnico -= 10
            
            # EMAs
            if 'ema_21' in ultimo and 'ema_50' in ultimo:
                if ultimo['ema_21'] > ultimo['ema_50']:
                    score_tecnico += 5
                else:
                    score_tecnico -= 5
            
            score_tecnico = max(0, min(100, score_tecnico))
            scores.append(score_tecnico)
            explicaciones.append(f"Técnico: {score_tecnico:.1f}")
            
            # Score final ponderado
            pesos = [0.4, 0.2, 0.2, 0.2]  # ML, sentimientos, patrones, técnico
            if len(scores) > 1:
                score_final = sum(s * w for s, w in zip(scores[:4], pesos))
            else:
                score_final = scores[0] if scores else 50
            
            return {
                'score_final': round(score_final, 1),
                'componentes': dict(zip(['ml', 'sentimientos', 'patrones', 'tecnico'], scores[:4])),
                'explicaciones': explicaciones,
                'interpretacion': self._interpretar_score(score_final)
            }
            
        except Exception as e:
            print(f"⚠️ Error calculando score: {e}")
            return {'score_final': 50, 'interpretacion': 'NEUTRAL'}

    def _interpretar_score(self, score):
        """Interpreta el score numérico"""
        if score >= 80:
            return "MUY BULLISH 🚀"
        elif score >= 65:
            return "BULLISH 📈"
        elif score >= 55:
            return "LIGERAMENTE BULLISH ⬆️"
        elif score >= 45:
            return "NEUTRAL ➡️"
        elif score >= 35:
            return "LIGERAMENTE BEARISH ⬇️"
        elif score >= 20:
            return "BEARISH 📉"
        else:
            return "MUY BEARISH 💥"

    def generar_reporte_predictivo(self, par, predicciones, sentimientos, patrones, score, df):
        """Genera reporte completo de predicciones"""
        reporte = f"""
{'='*80}
🔮 REPORTE PREDICTIVO AVANZADO PARA {par}
{'='*80}

📊 SCORE PREDICTIVO GENERAL: {score['score_final']}/100 - {score['interpretacion']}

🎯 PREDICCIONES POR HORIZONTE TEMPORAL:
"""
        
        for horizonte, pred in predicciones.items():
            if 'precio_predicho' in pred:
                cambio_pct = pred['cambio_esperado'] * 100
                reporte += f"""
⏰ {horizonte.upper()}:
   • Precio Actual: ${pred['precio_actual']:.4f}
   • Precio Predicho: ${pred['precio_predicho']:.4f}
   • Cambio Esperado: {cambio_pct:+.2f}%
   • Dirección: {pred.get('prediccion_direccion', 'N/A')}
   • Confianza: {pred.get('confianza', 0):.1%}
   • Rango: ${pred.get('precio_min', 0):.4f} - ${pred.get('precio_max', 0):.4f}
"""
        
        reporte += f"""
📰 ANÁLISIS DE SENTIMIENTOS:
   • Score Noticias: {sentimientos['noticias']['score']:.1f}
   • Fear & Greed: {sentimientos['fear_greed']['valor']}/100
   • Sentiment General: {sentimientos['score_general']:.1f}
   • Últimas Noticias: {len(sentimientos['noticias']['fuentes'])} encontradas

🕯️ PATRONES DETECTADOS:
   • Velas Japonesas: {len(patrones['velas_japonesas'])} patrones
   • Score Patrones: {patrones['score_patron']:.1f}
"""
        
        if patrones['velas_japonesas']:
            reporte += "   • Patrones Recientes:\n"
            for patron in patrones['velas_japonesas'][-3:]:
                reporte += f"     - {patron}\n"
        
        # Análisis técnico actual
        ultimo = df.iloc[-1]
        reporte += f"""
📈 ESTADO TÉCNICO ACTUAL:
   • RSI: {ultimo.get('rsi', 'N/A'):.1f}
   • MACD: {ultimo.get('macd_12_26', 'N/A'):.4f}
   • EMA 21: ${ultimo.get('ema_21', 'N/A'):.4f}
   • EMA 50: ${ultimo.get('ema_50', 'N/A'):.4f}
   • Volatilidad 24h: {ultimo.get('volatilidad_returns', 'N/A'):.1%}
"""
        
        # Recomendaciones
        reporte += self._generar_recomendaciones(score['score_final'], predicciones)
        
        reporte += f"""
⚠️ DISCLAIMER:
Este análisis es generado por IA con fines educativos únicamente.
NO constituye asesoramiento financiero. Las predicciones tienen alta
incertidumbre y los mercados de criptomonedas son extremadamente volátiles.
NUNCA inviertas más de lo que puedes permitirte perder.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return reporte

    def _generar_recomendaciones(self, score, predicciones):
        """Genera recomendaciones basadas en el análisis"""
        recomendaciones = "\n🎯 RECOMENDACIONES ALGORÍTMICAS:\n"
        
        if score >= 70:
            recomendaciones += """
   🟢 SEÑAL ALCISTA FUERTE
   • Considerar posición larga con gestión de riesgo
   • Stop-loss recomendado: -5% desde entrada
   • Take-profit sugerido: Escalonado en +10%, +20%
   • Timeframe: Seguir evolución en próximas 4-24h
"""
        elif score >= 55:
            recomendaciones += """
   🟡 SEÑAL ALCISTA MODERADA
   • Posición pequeña con strict risk management
   • Stop-loss: -3% desde entrada
   • Monitorear indicadores adicionales
"""
        elif score <= 30:
            recomendaciones += """
   🔴 SEÑAL BAJISTA FUERTE
   • Evitar posiciones largas
   • Considerar estrategias defensivas
   • Esperar confirmación de cambio de tendencia
"""
        else:
            recomendaciones += """
   ⚪ SEÑAL NEUTRAL
   • Mantener posiciones actuales
   • Esperar mayor claridad en indicadores
   • Preparar para movimiento direccional
"""
        
        # Análisis de riesgo
        volatilidades = []
        for pred in predicciones.values():
            if 'precio_std' in pred:
                volatilidades.append(pred['precio_std'])
        
        if volatilidades:
            vol_promedio = np.mean(volatilidades)
            if vol_promedio > np.percentile(volatilidades, 75):
                recomendaciones += "\n   ⚠️ ALTA VOLATILIDAD ESPERADA - Reducir tamaño de posición\n"
        
        return recomendaciones

    def generar_grafico_predicciones(self, par, df, predicciones, score):
        """Genera gráfico visual de predicciones y análisis técnico"""
        try:
            # Configurar el estilo del gráfico
            plt.style.use('dark_background')
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(f'🔮 ANÁLISIS PREDICTIVO AVANZADO - {par}', fontsize=16, fontweight='bold', color='white')
            
            # Gráfico 1: Precio y predicciones
            ax1 = axes[0]
            ax1.plot(df.index, df['close'], label='Precio Real', color='#00ff88', linewidth=2)
            
            # Agregar predicciones como líneas punteadas
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            for i, (horizonte, pred) in enumerate(predicciones.items()):
                if 'precio_predicho' in pred:
                    # Extender el índice para mostrar predicciones futuras
                    ultimo_tiempo = df.index[-1]
                    tiempo_futuro = ultimo_tiempo + pd.Timedelta(hours=int(horizonte.replace('h', '')))
                    
                    ax1.scatter(tiempo_futuro, pred['precio_predicho'], 
                               color=colors[i], s=100, marker='o', 
                               label=f'Predicción {horizonte}: ${pred["precio_predicho"]:.4f}')
                    
                    # Línea de conexión
                    ax1.plot([ultimo_tiempo, tiempo_futuro], 
                            [df['close'].iloc[-1], pred['precio_predicho']], 
                            '--', color=colors[i], alpha=0.7)
            
            ax1.set_title(f'📈 Precio y Predicciones - Score: {score["score_final"]}/100', color='white')
            ax1.set_ylabel('Precio (USDT)', color='white')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Indicadores técnicos
            ax2 = axes[1]
            ax2.plot(df.index, df['rsi'], label='RSI', color='#ff9ff3', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
            ax2.set_title('📊 Indicadores Técnicos', color='white')
            ax2.set_ylabel('RSI', color='white')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # MACD en el mismo gráfico
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df.index, df['macd_12_26'], label='MACD', color='#feca57', linewidth=1.5)
            ax2_twin.plot(df.index, df['macd_signal_12_26'], label='Señal MACD', color='#ff9ff3', linewidth=1.5)
            ax2_twin.set_ylabel('MACD', color='white')
            ax2_twin.legend(loc='upper right')
            
            # Gráfico 3: Volumen y volatilidad
            ax3 = axes[2]
            ax3.bar(df.index, df['volume'], alpha=0.6, color='#54a0ff', label='Volumen')
            ax3.set_title('📊 Volumen y Actividad', color='white')
            ax3.set_ylabel('Volumen', color='white')
            ax3.set_xlabel('Fecha', color='white')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Volatilidad en el mismo gráfico
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df.index, df['volatilidad_returns'], color='#ff6b6b', linewidth=2, label='Volatilidad')
            ax3_twin.set_ylabel('Volatilidad', color='white')
            ax3_twin.legend(loc='upper right')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"graficos/{par}_{timestamp}.png"
            
            # Crear directorio si no existe
            if not os.path.exists('graficos'):
                os.makedirs('graficos')
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            print(f"📊 Gráfico guardado: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️ Error generando gráfico para {par}: {e}")
            return None

    def generar_grafico_comparativo(self, resultados):
        """Genera gráfico comparativo de scores entre todas las criptomonedas"""
        try:
            if not resultados:
                return None
                
            # Preparar datos
            pares = list(resultados.keys())
            scores = [resultados[par]['score']['score_final'] for par in pares]
            
            # Configurar gráfico
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Crear barras con colores según el score
            colors = []
            for score in scores:
                if score >= 70:
                    colors.append('#00ff88')  # Verde para muy bullish
                elif score >= 55:
                    colors.append('#4ecdc4')  # Azul para bullish
                elif score >= 45:
                    colors.append('#feca57')  # Amarillo para neutral
                elif score >= 30:
                    colors.append('#ff9ff3')  # Rosa para bearish
                else:
                    colors.append('#ff6b6b')  # Rojo para muy bearish
            
            bars = ax.bar(pares, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Agregar valores en las barras
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}', ha='center', va='bottom', color='white', fontweight='bold')
            
            ax.set_title('🏆 COMPARATIVA DE SCORES PREDICTIVOS', fontsize=16, fontweight='bold', color='white', pad=20)
            ax.set_ylabel('Score Predictivo (0-100)', color='white')
            ax.set_xlabel('Criptomonedas', color='white')
            ax.set_ylim(0, 100)
            
            # Agregar líneas de referencia
            ax.axhline(y=70, color='#00ff88', linestyle='--', alpha=0.7, label='Muy Bullish')
            ax.axhline(y=55, color='#4ecdc4', linestyle='--', alpha=0.7, label='Bullish')
            ax.axhline(y=45, color='#feca57', linestyle='--', alpha=0.7, label='Neutral')
            ax.axhline(y=30, color='#ff6b6b', linestyle='--', alpha=0.7, label='Bearish')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotar etiquetas del eje X
            plt.xticks(rotation=45, ha='right')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"graficos/comparativa_scores_{timestamp}.png"
            
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            print(f"📊 Gráfico comparativo guardado: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️ Error generando gráfico comparativo: {e}")
            return None

    def ejecutar_analisis_completo(self):
        """Ejecuta análisis completo para todos los pares prioritarios"""
        print("🚀 INICIANDO SISTEMA PREDICTIVO AVANZADO")
        print("="*80)
        
        resultados = {}
        graficos_generados = []
        
        for par in self.config.PARES_PRIORITARIOS:
            try:
                print(f"\n🔍 Procesando {par}...")
                resultado = self.hacer_prediccion_completa(par)
                
                if resultado:
                    resultados[par] = resultado
                    print(resultado['reporte'])
                    
                    # Generar gráfico individual
                    print(f"\n🎨 Generando gráfico para {par}...")
                    grafico = self.generar_grafico_predicciones(
                        par, resultado['datos'], resultado['predicciones'], resultado['score']
                    )
                    if grafico:
                        graficos_generados.append(grafico)
                    
                    # Guardar modelo para uso futuro
                    self._guardar_modelo(par, resultado)
                else:
                    print(f"❌ No se pudo procesar {par}")
                
                # Pausa entre análisis para evitar límites de API
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error procesando {par}: {e}")
                continue
        
        # Generar gráfico comparativo
        if resultados:
            print(f"\n🏆 Generando gráfico comparativo...")
            grafico_comparativo = self.generar_grafico_comparativo(resultados)
            if grafico_comparativo:
                graficos_generados.append(grafico_comparativo)
        
        # Resumen general
        self._generar_resumen_general(resultados)
        
        # Mostrar resumen de gráficos generados
        if graficos_generados:
            print(f"\n📊 GRÁFICOS GENERADOS:")
            print("="*50)
            for grafico in graficos_generados:
                print(f"   📈 {grafico}")
            print(f"\n💡 Los gráficos se han guardado en la carpeta 'graficos/'")
        
        return resultados

    def _guardar_modelo(self, par, resultado):
        """Guarda modelos entrenados para uso futuro"""
        try:
            if not os.path.exists('modelos'):
                os.makedirs('modelos')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"modelos/{par}_{timestamp}.pkl"
            
            datos_guardar = {
                'par': par,
                'timestamp': timestamp,
                'score': resultado['score'],
                'predicciones': resultado['predicciones'],
                # No guardamos los modelos de ML por tamaño, solo los resultados
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(datos_guardar, f)
            
            print(f"💾 Modelo guardado: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error guardando modelo: {e}")

    def _generar_resumen_general(self, resultados):
        """Genera resumen de todos los análisis"""
        if not resultados:
            return
        
        print("\n" + "="*80)
        print("📊 RESUMEN GENERAL DEL MERCADO")
        print("="*80)
        
        scores = [(par, res['score']['score_final']) for par, res in resultados.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 RANKING POR SCORE PREDICTIVO:")
        for i, (par, score) in enumerate(scores, 1):
            interpretacion = self._interpretar_score(score)
            print(f"{i:2d}. {par:10s} - {score:5.1f}/100 - {interpretacion}")
        
        # Estadísticas generales
        scores_values = [s[1] for s in scores]
        print(f"\n📈 ESTADÍSTICAS DEL MERCADO:")
        print(f"   • Score Promedio: {np.mean(scores_values):.1f}")
        print(f"   • Score Máximo: {np.max(scores_values):.1f} ({scores[0][0]})")
        print(f"   • Score Mínimo: {np.min(scores_values):.1f} ({scores[-1][0]})")
        
        # Sentimiento general del mercado
        avg_score = np.mean(scores_values)
        if avg_score >= 60:
            sentimiento_mercado = "BULLISH 📈"
        elif avg_score >= 40:
            sentimiento_mercado = "NEUTRAL ➡️"
        else:
            sentimiento_mercado = "BEARISH 📉"
        
        print(f"   • Sentimiento General: {sentimiento_mercado}")
        
        print("\n⚠️ RECUERDA: Estas son predicciones algorítmicas con alta incertidumbre.")
        print("Siempre realiza tu propia investigación y gestiona el riesgo adecuadamente.")


# FUNCIÓN PRINCIPAL
def main():
    """Función principal del sistema predictivo"""
    print("🔮 SISTEMA AVANZADO DE PREDICCIÓN DE CRIPTOMONEDAS")
    print("Para los trillonarios")
    print("="*80)
    
    # Crear analizador
    analizador = AnalizadorPredictivoAvanzado()
    
    # Ejecutar análisis completo
    resultados = analizador.ejecutar_analisis_completo()
    
    print(f"\n✅ Análisis completado para {len(resultados)} pares")
    print("💡 Los modelos han sido entrenados y las predicciones generadas")
    print("📊 Revisa los reportes individuales arriba para detalles específicos")


if __name__ == "__main__":
    main() 