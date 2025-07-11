# CONFIGURACIÓN EXTENDIDA PARA HORIZONTES LARGOS
"""
Configuración avanzada con horizontes de predicción extendidos
Para trading a corto, mediano y largo plazo
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import timedelta

@dataclass
class ExtendedPredictionConfig:
    """Configuración extendida para predicciones multi-horizonte"""
    
    # Horizontes extendidos (en horas)
    SHORT_TERM_HORIZONS: List[int] = field(default_factory=lambda: [1, 4, 12, 24])  # 1h-24h
    MEDIUM_TERM_HORIZONS: List[int] = field(default_factory=lambda: [72, 168, 336])  # 3d-14d
    LONG_TERM_HORIZONS: List[int] = field(default_factory=lambda: [720, 2160, 4320])  # 30d-180d
    
    # Intervalos de datos por horizonte
    TIMEFRAMES_BY_HORIZON: Dict[str, str] = field(default_factory=lambda: {
        'short_term': '1h',    # Para predicciones intraday
        'medium_term': '4h',   # Para predicciones semanales
        'long_term': '1d'      # Para predicciones mensuales
    })
    
    # Períodos de datos históricos por horizonte
    HISTORICAL_PERIODS: Dict[str, str] = field(default_factory=lambda: {
        'short_term': '30 day ago UTC',     # 30 días para short-term
        'medium_term': '90 day ago UTC',    # 90 días para medium-term
        'long_term': '365 day ago UTC'      # 1 año para long-term
    })
    
    # Modelos específicos por horizonte
    MODELS_BY_HORIZON: Dict[str, List[str]] = field(default_factory=lambda: {
        'short_term': ['LSTM', 'XGBoost', 'RandomForest'],
        'medium_term': ['Transformer', 'LSTM', 'XGBoost', 'LinearModel'],
        'long_term': ['LinearModel', 'ARIMA', 'Prophet', 'MacroModel']
    })
    
    # Pesos de features por horizonte
    FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'short_term': {
            'technical': 0.7,
            'sentiment': 0.2,
            'macro': 0.1
        },
        'medium_term': {
            'technical': 0.4,
            'sentiment': 0.3,
            'macro': 0.3
        },
        'long_term': {
            'technical': 0.2,
            'sentiment': 0.2,
            'macro': 0.6
        }
    })
    
    # Umbrales de confianza por horizonte
    CONFIDENCE_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'short_term': 0.75,
        'medium_term': 0.65,
        'long_term': 0.55
    })
    
    @property
    def all_horizons(self) -> List[int]:
        """Retorna todos los horizontes combinados"""
        return self.SHORT_TERM_HORIZONS + self.MEDIUM_TERM_HORIZONS + self.LONG_TERM_HORIZONS
    
    def get_horizon_category(self, horizon: int) -> str:
        """Determina la categoría de un horizonte"""
        if horizon in self.SHORT_TERM_HORIZONS:
            return 'short_term'
        elif horizon in self.MEDIUM_TERM_HORIZONS:
            return 'medium_term'
        elif horizon in self.LONG_TERM_HORIZONS:
            return 'long_term'
        else:
            return 'unknown'
    
    def get_optimal_timeframe(self, horizon: int) -> str:
        """Retorna el timeframe óptimo para un horizonte"""
        category = self.get_horizon_category(horizon)
        return self.TIMEFRAMES_BY_HORIZON.get(category, '1h')
    
    def get_historical_period(self, horizon: int) -> str:
        """Retorna el período histórico para un horizonte"""
        category = self.get_horizon_category(horizon)
        return self.HISTORICAL_PERIODS.get(category, '30 day ago UTC') 