# FEATURE ENGINEERING AVANZADO MULTI-ASSET
"""
Sistema avanzado de feature engineering para predicción de criptomonedas
Incluye correlaciones multi-asset, regime detection, y factores macro
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Análisis estadístico
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
import ta

# APIs
import yfinance as yf
from binance.client import Client
import requests

# Interfaces
from core.interfaces import IFeatureEngineer
from analysis.macro_analyzer import MacroEconomicAnalyzer
from analysis.sentiment_analyzer import AdvancedSentimentAnalyzer

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer(IFeatureEngineer):
    """Feature Engineer avanzado con capacidades multi-asset"""
    
    def __init__(self, config):
        self.config = config
        self.crypto_universe = [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT',
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'AVAXUSDT', 'LINKUSDT',
            'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT'
        ]
        
        # Activos tradicionales para correlaciones
        self.traditional_assets = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'GLD': 'Gold',
            'TLT': 'Treasury Bonds',
            'VIX': 'Volatility Index',
            'DXY': 'Dollar Index'
        }
        
        # Cache para correlaciones
        self.correlation_cache = {}
        self.cache_ttl = 3600  # 1 hora
        
    def create_multi_asset_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Crea features basadas en múltiples activos"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Correlaciones dinámicas con otros cryptos
            crypto_correlations = self._calculate_crypto_correlations(symbol, data)
            features = pd.concat([features, crypto_correlations], axis=1)
            
            # 2. Correlaciones con activos tradicionales
            traditional_correlations = self._calculate_traditional_correlations(symbol, data)
            features = pd.concat([features, traditional_correlations], axis=1)
            
            # 3. Dominancia de Bitcoin
            btc_dominance = self._calculate_btc_dominance_features(symbol, data)
            features = pd.concat([features, btc_dominance], axis=1)
            
            # 4. Índices de crypto market
            market_indices = self._calculate_market_indices(data)
            features = pd.concat([features, market_indices], axis=1)
            
            # 5. Cross-asset momentum
            cross_momentum = self._calculate_cross_asset_momentum(symbol, data)
            features = pd.concat([features, cross_momentum], axis=1)
            
            logger.info(f"✅ Features multi-asset creadas: {len(features.columns)} características")
            return features
            
        except Exception as e:
            logger.error(f"Error creando features multi-asset: {e}")
            return pd.DataFrame(index=data.index)
    
    def create_regime_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detecta regímenes de mercado y crea features"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Régimen de volatilidad
            volatility_regime = self._detect_volatility_regime(data)
            features['volatility_regime'] = volatility_regime
            
            # 2. Régimen de tendencia
            trend_regime = self._detect_trend_regime(data)
            features['trend_regime'] = trend_regime
            
            # 3. Régimen de liquidez
            liquidity_regime = self._detect_liquidity_regime(data)
            features['liquidity_regime'] = liquidity_regime
            
            # 4. Régimen de correlación
            correlation_regime = self._detect_correlation_regime(data)
            features['correlation_regime'] = correlation_regime
            
            # 5. Régimen compuesto
            features['market_regime'] = self._calculate_composite_regime(features)
            
            # 6. Transiciones de régimen
            features['regime_transition'] = features['market_regime'].diff().abs()
            
            logger.info("✅ Features de detección de régimen creadas")
            return features
            
        except Exception as e:
            logger.error(f"Error en detección de régimen: {e}")
            return pd.DataFrame(index=data.index)
    
    def create_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea features basadas en factores macroeconómicos"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Datos de tasas de interés
            interest_rates = self._get_interest_rate_features(data)
            features = pd.concat([features, interest_rates], axis=1)
            
            # 2. Indicadores de inflación
            inflation_features = self._get_inflation_features(data)
            features = pd.concat([features, inflation_features], axis=1)
            
            # 3. Datos de empleo
            employment_features = self._get_employment_features(data)
            features = pd.concat([features, employment_features], axis=1)
            
            # 4. Índices de miedo y codicia
            fear_greed_features = self._get_fear_greed_features(data)
            features = pd.concat([features, fear_greed_features], axis=1)
            
            # 5. Datos de DeFi
            defi_features = self._get_defi_features(data)
            features = pd.concat([features, defi_features], axis=1)
            
            logger.info("✅ Features macroeconómicas creadas")
            return features
            
        except Exception as e:
            logger.error(f"Error creando features macro: {e}")
            return pd.DataFrame(index=data.index)
    
    def create_alternative_data_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Crea features basadas en datos alternativos"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Google Trends
            google_trends = self._get_google_trends_features(symbol, data)
            features = pd.concat([features, google_trends], axis=1)
            
            # 2. GitHub activity
            github_activity = self._get_github_activity_features(symbol, data)
            features = pd.concat([features, github_activity], axis=1)
            
            # 3. Social media sentiment
            social_sentiment = self._get_social_sentiment_features(symbol, data)
            features = pd.concat([features, social_sentiment], axis=1)
            
            # 4. On-chain metrics
            onchain_metrics = self._get_onchain_metrics_features(symbol, data)
            features = pd.concat([features, onchain_metrics], axis=1)
            
            # 5. Exchange flows
            exchange_flows = self._get_exchange_flow_features(symbol, data)
            features = pd.concat([features, exchange_flows], axis=1)
            
            logger.info("✅ Features de datos alternativos creadas")
            return features
            
        except Exception as e:
            logger.error(f"Error creando features alternativas: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_crypto_correlations(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula correlaciones dinámicas con otros cryptos"""
        
        correlations = pd.DataFrame(index=data.index)
        
        try:
            # Lista de otros cryptos para correlación
            other_cryptos = [crypto for crypto in self.crypto_universe if crypto != symbol]
            
            for crypto in other_cryptos[:5]:  # Top 5 para evitar sobrecarga
                try:
                    # Obtener datos del crypto
                    crypto_data = self._get_crypto_data(crypto, data.index[0], data.index[-1])
                    
                    if not crypto_data.empty:
                        # Correlación rolling
                        window = 24  # 24 períodos
                        crypto_returns = crypto_data['close'].pct_change()
                        symbol_returns = data['close'].pct_change()
                        
                        # Alinear datos
                        aligned_data = pd.concat([symbol_returns, crypto_returns], axis=1, join='inner')
                        aligned_data.columns = ['symbol', 'crypto']
                        
                        # Correlación rolling
                        rolling_corr = aligned_data['symbol'].rolling(window).corr(aligned_data['crypto'])
                        
                        correlations[f'corr_{crypto.replace("USDT", "").lower()}'] = rolling_corr.reindex(data.index).fillna(0)
                        
                except Exception as e:
                    logger.warning(f"Error calculando correlación con {crypto}: {e}")
                    continue
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error en correlaciones crypto: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_traditional_correlations(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula correlaciones con activos tradicionales"""
        
        correlations = pd.DataFrame(index=data.index)
        
        try:
            symbol_returns = data['close'].pct_change()
            
            for ticker, name in self.traditional_assets.items():
                try:
                    # Obtener datos tradicionales
                    traditional_data = self._get_traditional_data(ticker, data.index[0], data.index[-1])
                    
                    if not traditional_data.empty:
                        traditional_returns = traditional_data['close'].pct_change()
                        
                        # Alinear datos
                        aligned_data = pd.concat([symbol_returns, traditional_returns], axis=1, join='inner')
                        aligned_data.columns = ['crypto', 'traditional']
                        
                        # Correlación rolling
                        window = 24
                        rolling_corr = aligned_data['crypto'].rolling(window).corr(aligned_data['traditional'])
                        
                        correlations[f'corr_{ticker.lower()}'] = rolling_corr.reindex(data.index).fillna(0)
                        
                except Exception as e:
                    logger.warning(f"Error calculando correlación con {ticker}: {e}")
                    continue
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error en correlaciones tradicionales: {e}")
            return pd.DataFrame(index=data.index)
    
    def _detect_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detecta régimen de volatilidad"""
        
        try:
            # Calcular volatilidad realizada
            returns = data['close'].pct_change()
            volatility = returns.rolling(24).std()
            
            # Cuantiles para clasificar regímenes
            low_threshold = volatility.quantile(0.33)
            high_threshold = volatility.quantile(0.67)
            
            # Clasificar regímenes
            regime = pd.Series(index=data.index, dtype=int)
            regime[volatility <= low_threshold] = 0  # Baja volatilidad
            regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 1  # Media volatilidad
            regime[volatility > high_threshold] = 2  # Alta volatilidad
            
            return regime.fillna(1)
            
        except Exception as e:
            logger.error(f"Error detectando régimen de volatilidad: {e}")
            return pd.Series(index=data.index, dtype=int).fillna(1)
    
    def _detect_trend_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detecta régimen de tendencia"""
        
        try:
            # Calcular múltiples EMAs
            ema_short = data['close'].ewm(span=12).mean()
            ema_medium = data['close'].ewm(span=26).mean()
            ema_long = data['close'].ewm(span=50).mean()
            
            # Régimen basado en orden de EMAs
            regime = pd.Series(index=data.index, dtype=int)
            
            # Tendencia alcista
            bullish = (ema_short > ema_medium) & (ema_medium > ema_long)
            # Tendencia bajista
            bearish = (ema_short < ema_medium) & (ema_medium < ema_long)
            
            regime[bullish] = 2  # Alcista
            regime[bearish] = 0  # Bajista
            regime[~(bullish | bearish)] = 1  # Lateral
            
            return regime.fillna(1)
            
        except Exception as e:
            logger.error(f"Error detectando régimen de tendencia: {e}")
            return pd.Series(index=data.index, dtype=int).fillna(1)
    
    def _detect_liquidity_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detecta régimen de liquidez"""
        
        try:
            # Usar volumen como proxy de liquidez
            volume_ma = data['volume'].rolling(24).mean()
            volume_ratio = data['volume'] / volume_ma
            
            # Clasificar regímenes
            low_threshold = volume_ratio.quantile(0.33)
            high_threshold = volume_ratio.quantile(0.67)
            
            regime = pd.Series(index=data.index, dtype=int)
            regime[volume_ratio <= low_threshold] = 0  # Baja liquidez
            regime[(volume_ratio > low_threshold) & (volume_ratio <= high_threshold)] = 1  # Media liquidez
            regime[volume_ratio > high_threshold] = 2  # Alta liquidez
            
            return regime.fillna(1)
            
        except Exception as e:
            logger.error(f"Error detectando régimen de liquidez: {e}")
            return pd.Series(index=data.index, dtype=int).fillna(1)
    
    def _get_crypto_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Obtiene datos de otro crypto"""
        # Implementación simplificada - en producción usar cache
        try:
            # Usar Binance API o cache
            client = Client()
            klines = client.get_historical_klines(
                symbol, '1h', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['close'] = pd.to_numeric(df['close'])
                df.set_index('timestamp', inplace=True)
                
                return df[['close']]
            
        except Exception as e:
            logger.warning(f"Error obteniendo datos para {symbol}: {e}")
        
        return pd.DataFrame()
    
    def _get_traditional_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Obtiene datos de activos tradicionales"""
        try:
            # Usar Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
            if not data.empty:
                return data[['Close']].rename(columns={'Close': 'close'})
        except Exception as e:
            logger.warning(f"Error obteniendo datos tradicionales para {ticker}: {e}")
        
        return pd.DataFrame()
    
    # Métodos adicionales para otras features...
    def _calculate_btc_dominance_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula features de dominancia de Bitcoin"""
        features = pd.DataFrame(index=data.index)
        
        if symbol != 'BTCUSDT':
            try:
                # Obtener datos de BTC
                btc_data = self._get_crypto_data('BTCUSDT', data.index[0], data.index[-1])
                
                if not btc_data.empty:
                    # Ratio del precio vs BTC
                    btc_ratio = data['close'] / btc_data['close']
                    features['btc_ratio'] = btc_ratio
                    features['btc_ratio_ma'] = btc_ratio.rolling(24).mean()
                    features['btc_ratio_std'] = btc_ratio.rolling(24).std()
                    
            except Exception as e:
                logger.warning(f"Error calculando dominancia BTC: {e}")
        
        return features
    
    def _calculate_market_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula índices de mercado crypto"""
        features = pd.DataFrame(index=data.index)
        
        # Simulación de índices - en producción usar APIs reales
        features['crypto_market_cap_index'] = 100 + np.random.normal(0, 5, len(data))
        features['altcoin_index'] = 100 + np.random.normal(0, 8, len(data))
        features['defi_index'] = 100 + np.random.normal(0, 12, len(data))
        
        return features 