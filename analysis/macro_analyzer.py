# ANALIZADOR MACROECONÓMICO AVANZADO
"""
Sistema completo de análisis macroeconómico para trading de criptomonedas
Integra datos de FED, inflación, correlaciones inter-mercados y factores macro
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# APIs y datos externos
import requests
import yfinance as yf
from fredapi import Fred
import pandas_datareader.data as web
from concurrent.futures import ThreadPoolExecutor, as_completed

# Análisis estadístico
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Interfaces
from core.interfaces import IMacroAnalyzer
from config.system_config import SystemConfig

logger = logging.getLogger(__name__)

class MacroEconomicAnalyzer(IMacroAnalyzer):
    """Analizador macroeconómico completo"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.fred_api_key = config.api.fred_api_key
        self.fred = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora
        
        # Inicializar FRED API
        if self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                logger.info("FRED API inicializada correctamente")
            except Exception as e:
                logger.warning(f"Error inicializando FRED API: {e}")
        
        # Símbolos de mercados tradicionales
        self.traditional_markets = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'GLD': 'Gold',
            'TLT': 'US 20Y Treasury',
            'DXY': 'US Dollar Index',
            'VIX': 'Volatility Index',
            'USO': 'Oil',
            'EURUSD=X': 'EUR/USD',
            'JPYUSD=X': 'JPY/USD'
        }
        
        # Indicadores FRED principales
        self.fred_indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'CPILFESL': 'Core CPI',
            'UNRATE': 'Unemployment Rate',
            'GDP': 'Gross Domestic Product',
            'DEXUSEU': 'USD/EUR Exchange Rate',
            'BAMLH0A0HYM2': 'High Yield Credit Spread',
            'T10Y2Y': 'Treasury 10Y-2Y Spread',
            'VIXCLS': 'VIX Volatility Index',
            'WALCL': 'Fed Balance Sheet',
            'M2SL': 'M2 Money Supply',
            'PAYEMS': 'Non-farm Payrolls'
        }
    
    def get_fed_rates(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """
        Obtiene datos de tasas de la FED
        
        Args:
            start_date: Fecha de inicio
            
        Returns:
            DataFrame con datos de tasas FED
        """
        try:
            cache_key = f"fed_rates_{start_date}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            if not self.fred:
                logger.warning("FRED API no disponible")
                return pd.DataFrame()
            
            # Obtener diferentes tasas
            rates_data = {}
            
            rate_series = {
                'FEDFUNDS': 'Fed Funds Rate',
                'TB3MS': '3-Month Treasury',
                'TB6MS': '6-Month Treasury',
                'GS1': '1-Year Treasury',
                'GS2': '2-Year Treasury',
                'GS5': '5-Year Treasury',
                'GS10': '10-Year Treasury',
                'GS30': '30-Year Treasury',
                'MORTGAGE30US': '30-Year Mortgage'
            }
            
            for series_id, name in rate_series.items():
                try:
                    data = self.fred.get_series(series_id, start=start_date)
                    rates_data[name] = data
                    logger.debug(f"Descargado {name}: {len(data)} registros")
                except Exception as e:
                    logger.warning(f"Error descargando {name}: {e}")
                    continue
            
            if not rates_data:
                return pd.DataFrame()
            
            # Combinar datos
            df_rates = pd.DataFrame(rates_data)
            df_rates.index = pd.to_datetime(df_rates.index)
            
            # Forward fill para datos faltantes
            df_rates = df_rates.fillna(method='ffill')
            
            # Calcular spreads importantes
            if 'Fed Funds Rate' in df_rates.columns and '10-Year Treasury' in df_rates.columns:
                df_rates['Fed_10Y_Spread'] = df_rates['10-Year Treasury'] - df_rates['Fed Funds Rate']
            
            if '10-Year Treasury' in df_rates.columns and '2-Year Treasury' in df_rates.columns:
                df_rates['10Y_2Y_Spread'] = df_rates['10-Year Treasury'] - df_rates['2-Year Treasury']
            
            # Cache
            self._cache_data(cache_key, df_rates)
            
            logger.info(f"Datos FED obtenidos: {len(df_rates)} registros desde {df_rates.index[0]}")
            return df_rates
            
        except Exception as e:
            logger.error(f"Error obteniendo datos FED: {e}")
            return pd.DataFrame()
    
    def get_inflation_data(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """
        Obtiene datos de inflación
        
        Args:
            start_date: Fecha de inicio
            
        Returns:
            DataFrame con datos de inflación
        """
        try:
            cache_key = f"inflation_{start_date}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            if not self.fred:
                logger.warning("FRED API no disponible")
                return pd.DataFrame()
            
            # Series de inflación
            inflation_series = {
                'CPIAUCSL': 'CPI All Items',
                'CPILFESL': 'Core CPI',
                'CPIUFDSL': 'CPI Food',
                'CPIENGSL': 'CPI Energy',
                'CPIMEDSL': 'CPI Medical',
                'CPITRNSL': 'CPI Transportation',
                'CPIHOSSL': 'CPI Housing',
                'PCEPILFE': 'PCE Core',
                'PCEPI': 'PCE All Items'
            }
            
            inflation_data = {}
            
            for series_id, name in inflation_series.items():
                try:
                    data = self.fred.get_series(series_id, start=start_date)
                    inflation_data[name] = data
                    logger.debug(f"Descargado {name}: {len(data)} registros")
                except Exception as e:
                    logger.warning(f"Error descargando {name}: {e}")
                    continue
            
            if not inflation_data:
                return pd.DataFrame()
            
            # Combinar datos
            df_inflation = pd.DataFrame(inflation_data)
            df_inflation.index = pd.to_datetime(df_inflation.index)
            
            # Calcular tasas de cambio YoY
            for col in df_inflation.columns:
                if col in df_inflation.columns:
                    df_inflation[f'{col}_YoY'] = df_inflation[col].pct_change(periods=12) * 100
            
            # Forward fill
            df_inflation = df_inflation.fillna(method='ffill')
            
            # Cache
            self._cache_data(cache_key, df_inflation)
            
            logger.info(f"Datos inflación obtenidos: {len(df_inflation)} registros")
            return df_inflation
            
        except Exception as e:
            logger.error(f"Error obteniendo datos inflación: {e}")
            return pd.DataFrame()
    
    def get_market_correlations(self, symbols: List[str], 
                              period: str = "2y",
                              include_crypto: bool = True) -> pd.DataFrame:
        """
        Obtiene correlaciones con mercados tradicionales
        
        Args:
            symbols: Lista de símbolos crypto
            period: Período de datos
            include_crypto: Incluir correlaciones crypto-crypto
            
        Returns:
            Matriz de correlaciones
        """
        try:
            cache_key = f"correlations_{hash(str(symbols))}_{period}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            all_data = {}
            
            # Obtener datos de mercados tradicionales
            logger.info("Descargando datos de mercados tradicionales...")
            traditional_data = self._download_traditional_markets(period)
            all_data.update(traditional_data)
            
            # Obtener datos de crypto
            if include_crypto:
                logger.info("Descargando datos de criptomonedas...")
                crypto_data = self._download_crypto_data(symbols, period)
                all_data.update(crypto_data)
            
            if len(all_data) < 2:
                logger.warning("Datos insuficientes para correlaciones")
                return pd.DataFrame()
            
            # Combinar todos los datos
            df_combined = pd.DataFrame(all_data)
            
            # Calcular retornos diarios
            returns = df_combined.pct_change().dropna()
            
            # Calcular correlaciones
            correlation_matrix = returns.corr()
            
            # Cache
            self._cache_data(cache_key, correlation_matrix)
            
            logger.info(f"Matriz de correlaciones calculada: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculando correlaciones: {e}")
            return pd.DataFrame()
    
    def analyze_economic_calendar(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analiza eventos del calendario económico
        
        Args:
            lookback_days: Días hacia atrás para analizar
            
        Returns:
            Análisis de eventos económicos
        """
        try:
            # Simulación de calendario económico (en producción usar API real)
            events = self._simulate_economic_calendar(lookback_days)
            
            analysis = {
                'total_events': len(events),
                'high_impact_events': len([e for e in events if e['impact'] == 'high']),
                'fed_related_events': len([e for e in events if 'fed' in e['description'].lower()]),
                'inflation_events': len([e for e in events if 'inflation' in e['description'].lower() or 'cpi' in e['description'].lower()]),
                'employment_events': len([e for e in events if 'employment' in e['description'].lower() or 'nfp' in e['description'].lower()]),
                'upcoming_events': self._get_upcoming_events(),
                'market_impact_assessment': self._assess_market_impact(events)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando calendario económico: {e}")
            return {'error': str(e)}
    
    def analyze_crypto_macro_regime(self, crypto_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza el régimen macroeconómico para crypto
        
        Args:
            crypto_data: Datos de criptomonedas
            
        Returns:
            Análisis de régimen macro
        """
        try:
            # Obtener datos macro
            fed_data = self.get_fed_rates()
            inflation_data = self.get_inflation_data()
            correlations = self.get_market_correlations(['BTC', 'ETH'])
            
            regime_analysis = {
                'monetary_policy_stance': self._analyze_monetary_policy(fed_data),
                'inflation_regime': self._analyze_inflation_regime(inflation_data),
                'risk_sentiment': self._analyze_risk_sentiment(correlations),
                'dollar_strength': self._analyze_dollar_strength(),
                'liquidity_conditions': self._analyze_liquidity_conditions(fed_data),
                'crypto_correlation_regime': self._analyze_crypto_correlations(correlations),
                'overall_regime': 'neutral'  # Será calculado
            }
            
            # Determinar régimen general
            regime_analysis['overall_regime'] = self._determine_overall_regime(regime_analysis)
            
            # Implications para crypto
            regime_analysis['crypto_implications'] = self._generate_crypto_implications(regime_analysis)
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error analizando régimen macro: {e}")
            return {'error': str(e)}
    
    def get_macro_factors_pca(self, lookback_months: int = 24) -> Dict[str, Any]:
        """
        Análisis PCA de factores macroeconómicos
        
        Args:
            lookback_months: Meses de datos históricos
            
        Returns:
            Análisis PCA de factores macro
        """
        try:
            # Obtener datos macro
            start_date = (datetime.now() - timedelta(days=lookback_months * 30)).strftime('%Y-%m-%d')
            
            fed_data = self.get_fed_rates(start_date)
            inflation_data = self.get_inflation_data(start_date)
            market_data = self._download_traditional_markets("2y")
            
            # Combinar datos
            all_macro_data = pd.concat([
                fed_data.resample('D').last(),
                inflation_data.resample('D').last(),
                pd.DataFrame(market_data).resample('D').last()
            ], axis=1)
            
            # Limpiar datos
            all_macro_data = all_macro_data.dropna()
            
            if len(all_macro_data) < 50:
                logger.warning("Datos insuficientes para PCA")
                return {'error': 'Datos insuficientes'}
            
            # Calcular retornos
            returns = all_macro_data.pct_change().dropna()
            
            # Estandarizar
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns)
            
            # PCA
            pca = PCA()
            pca_result = pca.fit_transform(returns_scaled)
            
            # Análisis de componentes
            n_components = min(5, len(returns.columns))
            
            analysis = {
                'explained_variance_ratio': pca.explained_variance_ratio_[:n_components].tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_[:n_components]).tolist(),
                'components': {},
                'factor_loadings': {},
                'factor_interpretation': {}
            }
            
            # Cargas de factores
            for i in range(n_components):
                component_name = f'Factor_{i+1}'
                analysis['components'][component_name] = pca.components_[i][:len(returns.columns)].tolist()
                
                # Interpretación de factores
                loadings = dict(zip(returns.columns, pca.components_[i][:len(returns.columns)]))
                analysis['factor_loadings'][component_name] = loadings
                analysis['factor_interpretation'][component_name] = self._interpret_pca_factor(loadings)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en PCA macro: {e}")
            return {'error': str(e)}
    
    def _download_traditional_markets(self, period: str) -> Dict[str, pd.Series]:
        """Descarga datos de mercados tradicionales"""
        market_data = {}
        
        def download_symbol(symbol):
            try:
                if symbol == 'DXY':
                    # DXY no está en yfinance, usar proxy
                    ticker = yf.Ticker('UUP')  # Dollar ETF
                else:
                    ticker = yf.Ticker(symbol)
                
                hist = ticker.history(period=period)
                if not hist.empty:
                    return symbol, hist['Close']
                return symbol, None
            except Exception as e:
                logger.warning(f"Error descargando {symbol}: {e}")
                return symbol, None
        
        # Descargar en paralelo
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_symbol, symbol) for symbol in self.traditional_markets.keys()]
            
            for future in as_completed(futures):
                symbol, data = future.result()
                if data is not None:
                    market_data[symbol] = data
        
        return market_data
    
    def _download_crypto_data(self, symbols: List[str], period: str) -> Dict[str, pd.Series]:
        """Descarga datos de criptomonedas"""
        crypto_data = {}
        
        def download_crypto(symbol):
            try:
                # Asegurar formato correcto para yfinance
                if not symbol.endswith('-USD'):
                    yahoo_symbol = f"{symbol}-USD"
                else:
                    yahoo_symbol = symbol
                
                ticker = yf.Ticker(yahoo_symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    return symbol, hist['Close']
                return symbol, None
            except Exception as e:
                logger.warning(f"Error descargando crypto {symbol}: {e}")
                return symbol, None
        
        # Descargar en paralelo
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_crypto, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                symbol, data = future.result()
                if data is not None:
                    crypto_data[symbol] = data
        
        return crypto_data
    
    def _analyze_monetary_policy(self, fed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza stance de política monetaria"""
        if fed_data.empty or 'Fed Funds Rate' not in fed_data.columns:
            return {'stance': 'unknown', 'trend': 'unknown'}
        
        current_rate = fed_data['Fed Funds Rate'].iloc[-1]
        rate_6m_ago = fed_data['Fed Funds Rate'].iloc[-126] if len(fed_data) > 126 else fed_data['Fed Funds Rate'].iloc[0]
        
        rate_change = current_rate - rate_6m_ago
        
        if rate_change > 0.5:
            stance = 'hawkish'
        elif rate_change < -0.5:
            stance = 'dovish'
        else:
            stance = 'neutral'
        
        # Analizar tendencia
        recent_trend = fed_data['Fed Funds Rate'].iloc[-20:].diff().mean()
        trend = 'rising' if recent_trend > 0.01 else 'falling' if recent_trend < -0.01 else 'stable'
        
        return {
            'stance': stance,
            'trend': trend,
            'current_rate': current_rate,
            'rate_change_6m': rate_change,
            'confidence': 0.8
        }
    
    def _analyze_inflation_regime(self, inflation_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza régimen de inflación"""
        if inflation_data.empty:
            return {'regime': 'unknown'}
        
        # Usar CPI YoY si está disponible
        if 'CPI All Items_YoY' in inflation_data.columns:
            current_inflation = inflation_data['CPI All Items_YoY'].iloc[-1]
            avg_inflation = inflation_data['CPI All Items_YoY'].iloc[-12:].mean()
        else:
            return {'regime': 'unknown'}
        
        if current_inflation > 4:
            regime = 'high_inflation'
        elif current_inflation > 2.5:
            regime = 'moderate_inflation'
        elif current_inflation > 0:
            regime = 'low_inflation'
        else:
            regime = 'deflation'
        
        # Tendencia
        inflation_trend = inflation_data['CPI All Items_YoY'].iloc[-6:].diff().mean()
        trend = 'rising' if inflation_trend > 0.1 else 'falling' if inflation_trend < -0.1 else 'stable'
        
        return {
            'regime': regime,
            'trend': trend,
            'current_inflation': current_inflation,
            'avg_inflation_12m': avg_inflation,
            'confidence': 0.7
        }
    
    def _analyze_risk_sentiment(self, correlations: pd.DataFrame) -> Dict[str, Any]:
        """Analiza sentimiento de riesgo basado en correlaciones"""
        if correlations.empty:
            return {'sentiment': 'unknown'}
        
        # Buscar correlación BTC-SPY como proxy de risk-on/risk-off
        btc_spy_corr = 0
        if 'BTC' in correlations.index and 'SPY' in correlations.columns:
            btc_spy_corr = correlations.loc['BTC', 'SPY']
        elif 'BTC-USD' in correlations.index and 'SPY' in correlations.columns:
            btc_spy_corr = correlations.loc['BTC-USD', 'SPY']
        
        # Correlación alta = risk-on, baja = risk-off
        if btc_spy_corr > 0.5:
            sentiment = 'risk_on'
        elif btc_spy_corr < 0.2:
            sentiment = 'risk_off'
        else:
            sentiment = 'mixed'
        
        return {
            'sentiment': sentiment,
            'btc_spy_correlation': btc_spy_corr,
            'confidence': 0.6
        }
    
    def _analyze_dollar_strength(self) -> Dict[str, Any]:
        """Analiza fortaleza del dólar"""
        # Implementación simplificada
        return {
            'strength': 'neutral',
            'trend': 'stable',
            'confidence': 0.5
        }
    
    def _analyze_liquidity_conditions(self, fed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza condiciones de liquidez"""
        if fed_data.empty:
            return {'conditions': 'unknown'}
        
        # Usar spreads como proxy de liquidez
        if 'Fed_10Y_Spread' in fed_data.columns:
            current_spread = fed_data['Fed_10Y_Spread'].iloc[-1]
            if current_spread > 2:
                conditions = 'tight'
            elif current_spread > 0.5:
                conditions = 'neutral'
            else:
                conditions = 'loose'
        else:
            conditions = 'unknown'
        
        return {
            'conditions': conditions,
            'confidence': 0.6
        }
    
    def _analyze_crypto_correlations(self, correlations: pd.DataFrame) -> Dict[str, Any]:
        """Analiza correlaciones crypto"""
        if correlations.empty:
            return {'regime': 'unknown'}
        
        # Analizar correlaciones promedio
        crypto_symbols = [col for col in correlations.columns if any(c in col.upper() for c in ['BTC', 'ETH'])]
        
        if len(crypto_symbols) < 2:
            return {'regime': 'unknown'}
        
        crypto_corr_matrix = correlations.loc[crypto_symbols, crypto_symbols]
        avg_correlation = crypto_corr_matrix.values[np.triu_indices_from(crypto_corr_matrix.values, k=1)].mean()
        
        if avg_correlation > 0.8:
            regime = 'high_correlation'
        elif avg_correlation > 0.5:
            regime = 'moderate_correlation'
        else:
            regime = 'low_correlation'
        
        return {
            'regime': regime,
            'avg_correlation': avg_correlation,
            'confidence': 0.7
        }
    
    def _determine_overall_regime(self, regime_analysis: Dict[str, Any]) -> str:
        """Determina régimen general"""
        # Lógica simplificada para combinar regímenes
        monetary_stance = regime_analysis.get('monetary_policy_stance', {}).get('stance', 'unknown')
        inflation_regime = regime_analysis.get('inflation_regime', {}).get('regime', 'unknown')
        risk_sentiment = regime_analysis.get('risk_sentiment', {}).get('sentiment', 'unknown')
        
        # Scoring simple
        score = 0
        if monetary_stance == 'dovish':
            score += 1
        elif monetary_stance == 'hawkish':
            score -= 1
            
        if risk_sentiment == 'risk_on':
            score += 1
        elif risk_sentiment == 'risk_off':
            score -= 1
        
        if score > 0:
            return 'crypto_favorable'
        elif score < 0:
            return 'crypto_unfavorable'
        else:
            return 'neutral'
    
    def _generate_crypto_implications(self, regime_analysis: Dict[str, Any]) -> List[str]:
        """Genera implicaciones para crypto"""
        implications = []
        
        overall_regime = regime_analysis.get('overall_regime', 'neutral')
        
        if overall_regime == 'crypto_favorable':
            implications.append("Entorno macro favorable para criptomonedas")
            implications.append("Política monetaria expansiva puede impulsar activos de riesgo")
        elif overall_regime == 'crypto_unfavorable':
            implications.append("Entorno macro desafiante para criptomonedas")
            implications.append("Política monetaria restrictiva puede presionar activos de riesgo")
        else:
            implications.append("Entorno macro neutral para criptomonedas")
            implications.append("Otros factores pueden dominar la acción del precio")
        
        return implications
    
    def _simulate_economic_calendar(self, lookback_days: int) -> List[Dict[str, Any]]:
        """Simula calendario económico (placeholder)"""
        # En producción, usar API real de calendario económico
        events = [
            {'date': datetime.now() - timedelta(days=i), 'description': 'FOMC Meeting', 'impact': 'high'}
            for i in range(0, lookback_days, 30)
        ]
        return events
    
    def _get_upcoming_events(self) -> List[Dict[str, Any]]:
        """Obtiene eventos próximos (placeholder)"""
        return [
            {'date': datetime.now() + timedelta(days=7), 'description': 'CPI Release', 'impact': 'high'},
            {'date': datetime.now() + timedelta(days=14), 'description': 'FOMC Meeting', 'impact': 'high'}
        ]
    
    def _assess_market_impact(self, events: List[Dict[str, Any]]) -> str:
        """Evalúa impacto de mercado"""
        high_impact_count = len([e for e in events if e['impact'] == 'high'])
        
        if high_impact_count > 3:
            return 'high_volatility_expected'
        elif high_impact_count > 1:
            return 'moderate_volatility_expected'
        else:
            return 'low_volatility_expected'
    
    def _interpret_pca_factor(self, loadings: Dict[str, float]) -> str:
        """Interpreta factor PCA"""
        # Encontrar variables con mayor carga
        sorted_loadings = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        top_variables = sorted_loadings[:3]
        
        # Interpretación simple basada en variables dominantes
        if any('Fed' in var or 'Treasury' in var for var, _ in top_variables):
            return 'Interest Rate Factor'
        elif any('CPI' in var or 'inflation' in var.lower() for var, _ in top_variables):
            return 'Inflation Factor'
        elif any('SPY' in var or 'QQQ' in var for var, _ in top_variables):
            return 'Equity Market Factor'
        else:
            return 'Mixed Factor'
    
    def _is_cached(self, key: str) -> bool:
        """Verifica si los datos están en cache"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Guarda datos en cache"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        } 