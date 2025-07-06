# INTERFACES PRINCIPALES DEL SISTEMA
"""
Interfaces y contratos principales del sistema de predicción
Siguiendo principios SOLID para permitir inyección de dependencias
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

# Tipos de datos básicos
PriceData = pd.DataFrame
Features = pd.DataFrame
Predictions = Dict[str, Any]
ModelMetrics = Dict[str, float]
BacktestResults = Dict[str, Any]

@dataclass
class MarketData:
    """Estructura de datos de mercado"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PredictionResult:
    """Resultado de predicción"""
    symbol: str
    horizon: str
    predicted_price: float
    confidence: float
    probability_up: float
    price_range: Tuple[float, float]
    timestamp: datetime
    model_used: str
    features_importance: Dict[str, float]

@dataclass
class RiskMetrics:
    """Métricas de riesgo"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float

@dataclass
class TradeSignal:
    """Señal de trading"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: List[str]
    timestamp: datetime

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

# ========== INTERFACES PRINCIPALES ==========

class IDataProvider(ABC):
    """Interface para proveedores de datos"""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> MarketData:
        """Obtiene datos históricos"""
        pass
    
    @abstractmethod
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Obtiene datos en tiempo real"""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: pd.DataFrame) -> DataQuality:
        """Valida calidad de datos"""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Obtiene símbolos disponibles"""
        pass

class ITechnicalAnalyzer(ABC):
    """Interface para análisis técnico"""
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Features:
        """Calcula indicadores técnicos"""
        pass
    
    @abstractmethod
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detecta patrones chartistas"""
        pass
    
    @abstractmethod
    def identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identifica soportes y resistencias"""
        pass
    
    @abstractmethod
    def calculate_volatility_regime(self, data: pd.DataFrame) -> str:
        """Calcula régimen de volatilidad"""
        pass

class IFeatureEngineer(ABC):
    """Interface para ingeniería de características"""
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> Features:
        """Crea características para ML"""
        pass
    
    @abstractmethod
    def select_features(self, features: Features, target: pd.Series) -> Features:
        """Selecciona características relevantes"""
        pass
    
    @abstractmethod
    def transform_features(self, features: Features) -> Features:
        """Transforma características"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, features: Features, target: pd.Series) -> Dict[str, float]:
        """Obtiene importancia de características"""
        pass

class IMLModel(ABC):
    """Interface para modelos de ML"""
    
    @abstractmethod
    def train(self, X: Features, y: pd.Series) -> ModelMetrics:
        """Entrena el modelo"""
        pass
    
    @abstractmethod
    def predict(self, X: Features) -> np.ndarray:
        """Hace predicciones"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Features) -> np.ndarray:
        """Predice probabilidades"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características"""
        pass
    
    @abstractmethod
    def validate(self, X: Features, y: pd.Series) -> ModelMetrics:
        """Valida el modelo"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Guarda el modelo"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Carga el modelo"""
        pass

class IEnsembleModel(ABC):
    """Interface para modelos ensemble"""
    
    @abstractmethod
    def add_model(self, model: IMLModel, weight: float = 1.0) -> None:
        """Añade modelo al ensemble"""
        pass
    
    @abstractmethod
    def train_ensemble(self, X: Features, y: pd.Series) -> ModelMetrics:
        """Entrena el ensemble"""
        pass
    
    @abstractmethod
    def predict_ensemble(self, X: Features) -> Tuple[np.ndarray, np.ndarray]:
        """Predicción ensemble con incertidumbre"""
        pass

class IRiskManager(ABC):
    """Interface para gestión de riesgo"""
    
    @abstractmethod
    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calcula Value at Risk"""
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calcula Expected Shortfall"""
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, weights: np.ndarray, 
                               cov_matrix: np.ndarray) -> float:
        """Calcula riesgo del portfolio"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal_strength: float, 
                              account_balance: float, risk_per_trade: float) -> float:
        """Calcula tamaño de posición"""
        pass
    
    @abstractmethod
    def calculate_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calcula correlaciones"""
        pass
    
    @abstractmethod
    def detect_regime_change(self, data: pd.DataFrame) -> str:
        """Detecta cambio de régimen"""
        pass

class IBacktester(ABC):
    """Interface para backtesting"""
    
    @abstractmethod
    def run_backtest(self, strategy: Any, data: pd.DataFrame, 
                    initial_capital: float) -> BacktestResults:
        """Ejecuta backtest"""
        pass
    
    @abstractmethod
    def calculate_metrics(self, returns: pd.Series, 
                         benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calcula métricas de performance"""
        pass
    
    @abstractmethod
    def generate_report(self, results: BacktestResults) -> str:
        """Genera reporte de backtest"""
        pass

class ISentimentAnalyzer(ABC):
    """Interface para análisis de sentimientos"""
    
    @abstractmethod
    def analyze_news_sentiment(self, symbol: str, 
                             lookback_days: int = 7) -> Dict[str, float]:
        """Analiza sentimiento de noticias"""
        pass
    
    @abstractmethod
    def analyze_social_sentiment(self, symbol: str, 
                               lookback_days: int = 1) -> Dict[str, float]:
        """Analiza sentimiento de redes sociales"""
        pass
    
    @abstractmethod
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Obtiene índice de miedo y codicia"""
        pass

class IMacroAnalyzer(ABC):
    """Interface para análisis macroeconómico"""
    
    @abstractmethod
    def get_fed_rates(self) -> pd.DataFrame:
        """Obtiene tasas de la FED"""
        pass
    
    @abstractmethod
    def get_inflation_data(self) -> pd.DataFrame:
        """Obtiene datos de inflación"""
        pass
    
    @abstractmethod
    def get_market_correlations(self, symbols: List[str]) -> pd.DataFrame:
        """Obtiene correlaciones con mercados tradicionales"""
        pass
    
    @abstractmethod
    def analyze_economic_calendar(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Analiza calendario económico"""
        pass

class IOnChainAnalyzer(ABC):
    """Interface para análisis on-chain"""
    
    @abstractmethod
    def get_nvt_ratio(self, symbol: str) -> float:
        """Obtiene ratio NVT"""
        pass
    
    @abstractmethod
    def get_mvrv_ratio(self, symbol: str) -> float:
        """Obtiene ratio MVRV"""
        pass
    
    @abstractmethod
    def get_active_addresses(self, symbol: str) -> pd.DataFrame:
        """Obtiene direcciones activas"""
        pass
    
    @abstractmethod
    def get_whale_movements(self, symbol: str) -> pd.DataFrame:
        """Obtiene movimientos de ballenas"""
        pass
    
    @abstractmethod
    def get_exchange_flows(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Obtiene flujos de exchanges"""
        pass

class IPortfolioOptimizer(ABC):
    """Interface para optimización de portfolio"""
    
    @abstractmethod
    def optimize_weights(self, expected_returns: pd.Series, 
                        cov_matrix: pd.DataFrame, 
                        risk_tolerance: float) -> np.ndarray:
        """Optimiza pesos del portfolio"""
        pass
    
    @abstractmethod
    def calculate_efficient_frontier(self, expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcula frontera eficiente"""
        pass
    
    @abstractmethod
    def rebalance_portfolio(self, current_weights: np.ndarray, 
                          target_weights: np.ndarray, 
                          transaction_costs: float) -> np.ndarray:
        """Rebalancea el portfolio"""
        pass

class INotificationManager(ABC):
    """Interface para gestión de notificaciones"""
    
    @abstractmethod
    def send_signal_notification(self, signal: TradeSignal) -> None:
        """Envía notificación de señal"""
        pass
    
    @abstractmethod
    def send_risk_alert(self, alert: Dict[str, Any]) -> None:
        """Envía alerta de riesgo"""
        pass
    
    @abstractmethod
    def send_system_status(self, status: Dict[str, Any]) -> None:
        """Envía estado del sistema"""
        pass

class ICacheManager(ABC):
    """Interface para gestión de cache"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Establece valor en cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Elimina valor del cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Limpia todo el cache"""
        pass

class IDatabase(ABC):
    """Interface para base de datos"""
    
    @abstractmethod
    def save_prediction(self, prediction: PredictionResult) -> None:
        """Guarda predicción"""
        pass
    
    @abstractmethod
    def get_predictions(self, symbol: str, 
                       start_date: datetime, 
                       end_date: datetime) -> List[PredictionResult]:
        """Obtiene predicciones históricas"""
        pass
    
    @abstractmethod
    def save_model_metrics(self, model_name: str, 
                          metrics: ModelMetrics) -> None:
        """Guarda métricas del modelo"""
        pass
    
    @abstractmethod
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Obtiene performance del modelo"""
        pass 