# ORQUESTADOR DE DATOS AVANZADO
"""
Sistema de orquestación de datos con múltiples fuentes
Incluye validación de calidad, caching, y fallback automático
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

# APIs
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import yfinance as yf
import requests
import ccxt

# Análisis de datos
from scipy import stats
from sklearn.preprocessing import StandardScaler
import ta

# Interfaces
from core.interfaces import IDataProvider, MarketData, DataQuality
from core.cache_manager import CacheManager
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

class DataSource(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO_FINANCE = "yahoo_finance"
    COINGECKO = "coingecko"
    MESSARI = "messari"
    ALPHA_VANTAGE = "alpha_vantage"

@dataclass
class DataQualityMetrics:
    """Métricas de calidad de datos"""
    completeness: float  # % de datos sin NaN
    consistency: float   # Consistencia temporal
    accuracy: float      # Precisión vs otras fuentes
    timeliness: float    # Actualidad de los datos
    reliability: float   # Confiabilidad histórica
    overall_score: float # Score general
    
    def __post_init__(self):
        self.overall_score = np.mean([
            self.completeness, self.consistency, 
            self.accuracy, self.timeliness, self.reliability
        ])

@dataclass
class DataSourceConfig:
    """Configuración de fuente de datos"""
    source: DataSource
    enabled: bool = True
    priority: int = 1  # 1 = más alta prioridad
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests por minuto
    timeout: int = 30
    max_retries: int = 3
    backup_sources: List[DataSource] = field(default_factory=list)

class DataQualityValidator:
    """Validador de calidad de datos"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% de datos completos
            'consistency': 0.90,   # 90% consistencia
            'accuracy': 0.85,      # 85% precisión
            'timeliness': 0.80,    # 80% actualidad
            'reliability': 0.75    # 75% confiabilidad
        }
    
    def validate_data(self, data: pd.DataFrame, 
                     source: DataSource,
                     reference_data: Optional[pd.DataFrame] = None) -> DataQualityMetrics:
        """
        Valida la calidad de los datos
        
        Args:
            data: DataFrame con datos OHLCV
            source: Fuente de datos
            reference_data: Datos de referencia para comparación
            
        Returns:
            DataQualityMetrics con métricas de calidad
        """
        try:
            # Completeness - % de datos sin NaN
            completeness = self._calculate_completeness(data)
            
            # Consistency - Consistencia temporal y lógica
            consistency = self._calculate_consistency(data)
            
            # Accuracy - Precisión vs datos de referencia
            accuracy = self._calculate_accuracy(data, reference_data) if reference_data is not None else 0.8
            
            # Timeliness - Actualidad de los datos
            timeliness = self._calculate_timeliness(data)
            
            # Reliability - Confiabilidad histórica de la fuente
            reliability = self._get_source_reliability(source)
            
            metrics = DataQualityMetrics(
                completeness=completeness,
                consistency=consistency,
                accuracy=accuracy,
                timeliness=timeliness,
                reliability=reliability,
                overall_score=0  # Se calcula automáticamente
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validando calidad de datos: {e}")
            return DataQualityMetrics(0, 0, 0, 0, 0, 0)
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calcula completeness de los datos"""
        if data.empty:
            return 0.0
        
        # Verificar columnas OHLCV
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in data.columns]
        
        if not available_columns:
            return 0.0
        
        # % de datos sin NaN
        completeness_scores = []
        for col in available_columns:
            non_null_pct = data[col].notna().sum() / len(data)
            completeness_scores.append(non_null_pct)
        
        return np.mean(completeness_scores)
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calcula consistencia de los datos"""
        if len(data) < 2:
            return 0.0
        
        consistency_scores = []
        
        # 1. Verificar que high >= low
        if 'high' in data.columns and 'low' in data.columns:
            valid_high_low = (data['high'] >= data['low']).sum() / len(data)
            consistency_scores.append(valid_high_low)
        
        # 2. Verificar que open, close estén entre high y low
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            valid_open = ((data['open'] >= data['low']) & (data['open'] <= data['high'])).sum() / len(data)
            valid_close = ((data['close'] >= data['low']) & (data['close'] <= data['high'])).sum() / len(data)
            consistency_scores.extend([valid_open, valid_close])
        
        # 3. Verificar que volume >= 0
        if 'volume' in data.columns:
            valid_volume = (data['volume'] >= 0).sum() / len(data)
            consistency_scores.append(valid_volume)
        
        # 4. Verificar continuidad temporal
        if hasattr(data.index, 'to_series'):
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                # Verificar que los intervalos sean consistentes
                mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else timedelta(hours=1)
                consistent_intervals = (time_diffs == mode_diff).sum() / len(time_diffs)
                consistency_scores.append(consistent_intervals)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_accuracy(self, data: pd.DataFrame, reference_data: pd.DataFrame) -> float:
        """Calcula precisión vs datos de referencia"""
        if reference_data is None or data.empty or reference_data.empty:
            return 0.8  # Score por defecto
        
        try:
            # Alinear datos por timestamp
            aligned_data = data.align(reference_data, join='inner')
            if aligned_data[0].empty:
                return 0.8
            
            accuracy_scores = []
            
            # Comparar precios de cierre
            if 'close' in aligned_data[0].columns and 'close' in aligned_data[1].columns:
                close_corr = aligned_data[0]['close'].corr(aligned_data[1]['close'])
                if not np.isnan(close_corr):
                    accuracy_scores.append(abs(close_corr))
            
            # Comparar volúmenes
            if 'volume' in aligned_data[0].columns and 'volume' in aligned_data[1].columns:
                volume_corr = aligned_data[0]['volume'].corr(aligned_data[1]['volume'])
                if not np.isnan(volume_corr):
                    accuracy_scores.append(abs(volume_corr))
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.8
            
        except Exception as e:
            logger.warning(f"Error calculando precisión: {e}")
            return 0.8
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calcula actualidad de los datos"""
        if data.empty:
            return 0.0
        
        try:
            # Obtener timestamp más reciente
            latest_timestamp = data.index.max()
            current_time = datetime.now()
            
            # Si no hay timezone info, asumir UTC
            if latest_timestamp.tz is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=current_time.tzinfo)
            
            # Calcular diferencia
            time_diff = current_time - latest_timestamp
            hours_behind = time_diff.total_seconds() / 3600
            
            # Score basado en qué tan recientes son los datos
            if hours_behind <= 1:
                return 1.0
            elif hours_behind <= 6:
                return 0.8
            elif hours_behind <= 24:
                return 0.6
            elif hours_behind <= 168:  # 1 semana
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Error calculando actualidad: {e}")
            return 0.5
    
    def _get_source_reliability(self, source: DataSource) -> float:
        """Obtiene confiabilidad histórica de la fuente"""
        # Scores basados en experiencia y disponibilidad
        reliability_scores = {
            DataSource.BINANCE: 0.95,
            DataSource.COINBASE: 0.90,
            DataSource.KRAKEN: 0.85,
            DataSource.YAHOO_FINANCE: 0.80,
            DataSource.COINGECKO: 0.75,
            DataSource.MESSARI: 0.70,
            DataSource.ALPHA_VANTAGE: 0.65
        }
        
        return reliability_scores.get(source, 0.5)
    
    def get_data_quality_level(self, metrics: DataQualityMetrics) -> DataQuality:
        """Determina el nivel de calidad de los datos"""
        if metrics.overall_score >= 0.9:
            return DataQuality.EXCELLENT
        elif metrics.overall_score >= 0.8:
            return DataQuality.GOOD
        elif metrics.overall_score >= 0.6:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

class BinanceDataProvider(IDataProvider):
    """Proveedor de datos de Binance"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa cliente de Binance"""
        try:
            config_manager = get_config_manager()
            api_config = config_manager.get_api_config()
            
            if api_config.binance_api_key and api_config.binance_api_secret:
                self.client = BinanceClient(
                    api_config.binance_api_key,
                    api_config.binance_api_secret,
                    testnet=api_config.binance_testnet
                )
                # Verificar conectividad
                self.client.get_server_time()
                logger.info("✅ Cliente Binance autenticado")
            else:
                self.client = BinanceClient()
                logger.info("⚠️ Cliente Binance en modo público")
                
        except Exception as e:
            logger.error(f"Error inicializando cliente Binance: {e}")
            self.client = BinanceClient()
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> MarketData:
        """Obtiene datos históricos de Binance"""
        try:
            # Convertir timeframe
            interval = self._convert_timeframe(timeframe)
            
            # Obtener datos
            klines = self.client.get_historical_klines(
                symbol, interval, start_date, end_date
            )
            
            if not klines:
                raise Exception("No se obtuvieron datos")
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Procesar datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convertir a numérico
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Crear MarketData
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=df[numeric_columns],
                timestamp=datetime.now(),
                metadata={
                    'source': DataSource.BINANCE.value,
                    'records_count': len(df),
                    'start_date': df.index.min(),
                    'end_date': df.index.max()
                }
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Binance: {e}")
            raise
    
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Obtiene datos en tiempo real"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                'symbol': symbol,
                'price': float(ticker['price']),
                'timestamp': datetime.now(),
                'source': DataSource.BINANCE.value
            }
        except Exception as e:
            logger.error(f"Error obteniendo datos en tiempo real: {e}")
            raise
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQuality:
        """Valida calidad de datos"""
        validator = DataQualityValidator()
        metrics = validator.validate_data(data, DataSource.BINANCE)
        return validator.get_data_quality_level(metrics)
    
    def get_available_symbols(self) -> List[str]:
        """Obtiene símbolos disponibles"""
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return symbols
        except Exception as e:
            logger.error(f"Error obteniendo símbolos: {e}")
            return []
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convierte timeframe a formato Binance"""
        conversion_map = {
            '1m': BinanceClient.KLINE_INTERVAL_1MINUTE,
            '5m': BinanceClient.KLINE_INTERVAL_5MINUTE,
            '15m': BinanceClient.KLINE_INTERVAL_15MINUTE,
            '30m': BinanceClient.KLINE_INTERVAL_30MINUTE,
            '1h': BinanceClient.KLINE_INTERVAL_1HOUR,
            '4h': BinanceClient.KLINE_INTERVAL_4HOUR,
            '1d': BinanceClient.KLINE_INTERVAL_1DAY,
            '1w': BinanceClient.KLINE_INTERVAL_1WEEK
        }
        return conversion_map.get(timeframe, BinanceClient.KLINE_INTERVAL_1HOUR)

class YahooFinanceDataProvider(IDataProvider):
    """Proveedor de datos de Yahoo Finance"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> MarketData:
        """Obtiene datos históricos de Yahoo Finance"""
        try:
            # Convertir símbolo crypto a formato Yahoo
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            
            # Obtener datos
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if df.empty:
                raise Exception("No se obtuvieron datos")
            
            # Normalizar columnas
            df.columns = [col.lower() for col in df.columns]
            
            # Crear MarketData
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=df,
                timestamp=datetime.now(),
                metadata={
                    'source': DataSource.YAHOO_FINANCE.value,
                    'records_count': len(df),
                    'start_date': df.index.min(),
                    'end_date': df.index.max()
                }
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Yahoo Finance: {e}")
            raise
    
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Obtiene datos en tiempo real"""
        try:
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'timestamp': datetime.now(),
                'source': DataSource.YAHOO_FINANCE.value
            }
        except Exception as e:
            logger.error(f"Error obteniendo datos en tiempo real: {e}")
            raise
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQuality:
        """Valida calidad de datos"""
        validator = DataQualityValidator()
        metrics = validator.validate_data(data, DataSource.YAHOO_FINANCE)
        return validator.get_data_quality_level(metrics)
    
    def get_available_symbols(self) -> List[str]:
        """Obtiene símbolos disponibles"""
        # Yahoo Finance no tiene API pública para listar todos los símbolos
        # Retornar lista común de criptomonedas
        return ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD']
    
    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """Convierte símbolo a formato Yahoo Finance"""
        # Convertir de formato Binance a Yahoo
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}-USD"
        elif symbol.endswith('BTC'):
            base = symbol[:-3]
            return f"{base}-BTC"
        else:
            return symbol

class DataOrchestrator:
    """Orquestador principal de datos"""
    
    def __init__(self):
        self.providers: Dict[DataSource, IDataProvider] = {}
        self.cache_manager = CacheManager()
        self.quality_validator = DataQualityValidator()
        self.config = get_config_manager()
        
        # Inicializar proveedores
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa proveedores de datos"""
        
        # Configuraciones por defecto
        configs = {
            DataSource.BINANCE: DataSourceConfig(
                source=DataSource.BINANCE,
                priority=1,
                rate_limit=1200,  # 1200 requests/min
                backup_sources=[DataSource.YAHOO_FINANCE]
            ),
            DataSource.YAHOO_FINANCE: DataSourceConfig(
                source=DataSource.YAHOO_FINANCE,
                priority=2,
                rate_limit=60,
                backup_sources=[DataSource.BINANCE]
            )
        }
        
        # Inicializar proveedores
        for source, config in configs.items():
            try:
                if source == DataSource.BINANCE:
                    self.providers[source] = BinanceDataProvider(config)
                elif source == DataSource.YAHOO_FINANCE:
                    self.providers[source] = YahooFinanceDataProvider(config)
                
                logger.info(f"✅ Proveedor {source.value} inicializado")
                
            except Exception as e:
                logger.error(f"❌ Error inicializando proveedor {source.value}: {e}")
    
    def get_best_data(self, symbol: str, timeframe: str, 
                     start_date: str, end_date: str,
                     min_quality: DataQuality = DataQuality.GOOD) -> MarketData:
        """
        Obtiene los mejores datos disponibles
        
        Args:
            symbol: Símbolo del activo
            timeframe: Marco temporal
            start_date: Fecha de inicio
            end_date: Fecha de fin
            min_quality: Calidad mínima requerida
            
        Returns:
            MarketData con los mejores datos disponibles
        """
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(symbol, timeframe, start_date, end_date)
        
        # Intentar obtener desde cache
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"📦 Datos obtenidos desde cache: {symbol}")
            return cached_data
        
        # Obtener datos de múltiples fuentes
        data_sources = []
        
        # Ordenar proveedores por prioridad
        sorted_providers = sorted(
            self.providers.items(), 
            key=lambda x: self._get_provider_priority(x[0])
        )
        
        for source, provider in sorted_providers:
            try:
                logger.info(f"🔍 Obteniendo datos de {source.value} para {symbol}")
                
                market_data = provider.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                
                # Validar calidad
                quality_metrics = self.quality_validator.validate_data(
                    market_data.data, source
                )
                
                quality_level = self.quality_validator.get_data_quality_level(quality_metrics)
                
                logger.info(f"📊 Calidad de datos {source.value}: {quality_level.value} "
                           f"(Score: {quality_metrics.overall_score:.2f})")
                
                data_sources.append({
                    'source': source,
                    'data': market_data,
                    'quality_metrics': quality_metrics,
                    'quality_level': quality_level
                })
                
                # Si encontramos datos de calidad suficiente, usar esos
                if quality_level.value in [DataQuality.EXCELLENT.value, DataQuality.GOOD.value]:
                    if quality_level.value == min_quality.value or quality_metrics.overall_score >= 0.8:
                        break
                        
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo datos de {source.value}: {e}")
                continue
        
        if not data_sources:
            raise Exception("No se pudieron obtener datos de ninguna fuente")
        
        # Seleccionar la mejor fuente
        best_source = max(data_sources, key=lambda x: x['quality_metrics'].overall_score)
        
        # Enriquecer con metadatos de calidad
        best_data = best_source['data']
        best_data.metadata.update({
            'quality_metrics': best_source['quality_metrics'],
            'quality_level': best_source['quality_level'].value,
            'alternative_sources': len(data_sources) - 1
        })
        
        # Guardar en cache
        self.cache_manager.set(cache_key, best_data, ttl=3600)  # 1 hora
        
        logger.info(f"✅ Mejores datos seleccionados de {best_source['source'].value} "
                   f"para {symbol} (Score: {best_source['quality_metrics'].overall_score:.2f})")
        
        return best_data
    
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Obtiene datos en tiempo real con fallback"""
        
        # Intentar fuentes por prioridad
        for source, provider in self.providers.items():
            try:
                return provider.get_real_time_data(symbol)
            except Exception as e:
                logger.warning(f"Error obteniendo datos en tiempo real de {source.value}: {e}")
                continue
        
        raise Exception("No se pudieron obtener datos en tiempo real")
    
    def _generate_cache_key(self, symbol: str, timeframe: str, 
                           start_date: str, end_date: str) -> str:
        """Genera clave de cache"""
        key_data = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_provider_priority(self, source: DataSource) -> int:
        """Obtiene prioridad del proveedor"""
        priorities = {
            DataSource.BINANCE: 1,
            DataSource.YAHOO_FINANCE: 2,
            DataSource.COINBASE: 3,
            DataSource.KRAKEN: 4
        }
        return priorities.get(source, 10)
    
    def get_data_quality_report(self, symbol: str, timeframe: str,
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """Genera reporte de calidad de datos"""
        
        report = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': f"{start_date} to {end_date}",
            'sources_evaluated': [],
            'recommendation': None,
            'timestamp': datetime.now()
        }
        
        # Evaluar cada fuente
        for source, provider in self.providers.items():
            try:
                market_data = provider.get_historical_data(
                    symbol, timeframe, start_date, end_date
                )
                
                quality_metrics = self.quality_validator.validate_data(
                    market_data.data, source
                )
                
                quality_level = self.quality_validator.get_data_quality_level(quality_metrics)
                
                source_report = {
                    'source': source.value,
                    'quality_level': quality_level.value,
                    'quality_metrics': {
                        'completeness': quality_metrics.completeness,
                        'consistency': quality_metrics.consistency,
                        'accuracy': quality_metrics.accuracy,
                        'timeliness': quality_metrics.timeliness,
                        'reliability': quality_metrics.reliability,
                        'overall_score': quality_metrics.overall_score
                    },
                    'record_count': len(market_data.data),
                    'date_range': {
                        'start': market_data.data.index.min().isoformat(),
                        'end': market_data.data.index.max().isoformat()
                    }
                }
                
                report['sources_evaluated'].append(source_report)
                
            except Exception as e:
                logger.warning(f"Error evaluando fuente {source.value}: {e}")
                continue
        
        # Generar recomendación
        if report['sources_evaluated']:
            best_source = max(
                report['sources_evaluated'],
                key=lambda x: x['quality_metrics']['overall_score']
            )
            
            report['recommendation'] = {
                'best_source': best_source['source'],
                'quality_level': best_source['quality_level'],
                'overall_score': best_source['quality_metrics']['overall_score'],
                'reason': f"Mejor score de calidad: {best_source['quality_metrics']['overall_score']:.2f}"
            }
        
        return report

# Singleton para acceso global
_data_orchestrator = None

def get_data_orchestrator() -> DataOrchestrator:
    """Obtiene el orquestador de datos (singleton)"""
    global _data_orchestrator
    if _data_orchestrator is None:
        _data_orchestrator = DataOrchestrator()
    return _data_orchestrator 