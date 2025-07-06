# SISTEMA DE CONFIGURACIÓN AVANZADO
"""
Sistema de configuración robusto con validación y variables de entorno
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field
from enum import Enum
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class ModelType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    REINFORCEMENT = "reinforcement"

@dataclass
class APIConfig:
    """Configuración de APIs"""
    binance_api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    binance_api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    news_api_key: str = field(default_factory=lambda: os.getenv('NEWS_API_KEY', ''))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY', ''))
    fred_api_key: str = field(default_factory=lambda: os.getenv('FRED_API_KEY', ''))
    coinmetrics_key: str = field(default_factory=lambda: os.getenv('COINMETRICS_KEY', ''))
    
    def __post_init__(self):
        if not self.binance_api_key:
            logger.warning("BINANCE_API_KEY no configurada - usando modo público")

@dataclass
class TradingConfig:
    """Configuración de trading"""
    pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "AVAXUSDT", "LINKUSDT"
    ])
    timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 72])
    max_position_risk: float = 0.02  # 2% del capital por posición
    max_portfolio_risk: float = 0.10  # 10% del capital total
    min_confidence_threshold: float = 0.70
    stop_loss_pct: float = 0.03  # 3%
    take_profit_pct: float = 0.06  # 6%
    
@dataclass
class MLConfig:
    """Configuración de Machine Learning"""
    models_to_train: List[ModelType] = field(default_factory=lambda: [
        ModelType.LSTM, ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.ENSEMBLE
    ])
    sequence_length: int = 60
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    feature_selection_threshold: float = 0.001
    early_stopping_patience: int = 15
    learning_rate: float = 0.001
    regularization_strength: float = 0.01
    
@dataclass
class RiskConfig:
    """Configuración de gestión de riesgo"""
    var_confidence_level: float = 0.95
    var_lookback_days: int = 252
    max_drawdown_threshold: float = 0.15
    correlation_threshold: float = 0.7
    volatility_lookback: int = 30
    kelly_criterion_enabled: bool = True
    risk_parity_enabled: bool = True
    
@dataclass
class BacktestConfig:
    """Configuración de backtesting"""
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    benchmark: str = "BTCUSDT"
    rebalance_frequency: str = "daily"
    
class SystemConfig(BaseModel):
    """Configuración principal del sistema"""
    api: APIConfig = Field(default_factory=APIConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Configuración de sistema
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hora
    parallel_processing: bool = True
    max_workers: int = 4
    log_level: str = "INFO"
    save_models: bool = True
    save_predictions: bool = True
    
    # Configuración de base de datos
    db_url: str = "sqlite:///crypto_predictions.db"
    db_pool_size: int = 5
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @classmethod
    def from_file(cls, config_path: str = "config/config.json") -> 'SystemConfig':
        """Carga configuración desde archivo"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return cls(**config_data)
            else:
                logger.warning(f"Archivo de configuración {config_path} no encontrado. Usando configuración por defecto.")
                return cls()
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return cls()
    
    def save_to_file(self, config_path: str = "config/config.json"):
        """Guarda configuración a archivo"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.dict(), f, indent=2, default=str)
            logger.info(f"Configuración guardada en {config_path}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}") 