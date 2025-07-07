# SISTEMA ROBUSTO DE GESTIÓN DE RIESGO
"""
Sistema avanzado de gestión de riesgo con circuit breakers, límites operacionales
y controles de riesgo en tiempo real para trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Análisis financiero
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Interfaces
from core.interfaces import IRiskManager, RiskMetrics, TradeSignal
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Niveles de riesgo"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitBreakerState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"      # Operación normal
    OPEN = "open"          # Bloqueado
    HALF_OPEN = "half_open"  # Prueba limitada

@dataclass
class RiskEvent:
    """Evento de riesgo"""
    timestamp: datetime
    event_type: str
    severity: RiskLevel
    description: str
    symbol: Optional[str] = None
    position_size: Optional[float] = None
    risk_metric: Optional[float] = None
    action_taken: Optional[str] = None

@dataclass
class CircuitBreakerConfig:
    """Configuración de circuit breaker"""
    failure_threshold: int = 5  # Fallos antes de abrir
    recovery_timeout: int = 300  # Segundos antes de intentar half-open
    success_threshold: int = 3  # Éxitos para cerrar
    max_failure_rate: float = 0.5  # Tasa máxima de fallos
    time_window: int = 60  # Ventana de tiempo en segundos

@dataclass
class PositionLimits:
    """Límites de posición"""
    max_position_size: float = 0.02  # 2% del capital
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    max_correlation_exposure: float = 0.3  # 30%
    max_sector_exposure: float = 0.4  # 40%
    min_time_between_trades: int = 300  # 5 minutos

@dataclass
class RiskMetrics:
    """Métricas de riesgo actualizadas"""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    portfolio_correlation: float = 0.0
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    leverage_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW

class CircuitBreaker:
    """Circuit breaker para control de riesgo"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.failure_times = deque(maxlen=config.failure_threshold * 2)
        self.lock = threading.Lock()
        
    def call(self, func, *args, **kwargs):
        """Ejecuta función con circuit breaker"""
        
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} cambiado a HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} está OPEN")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} cambiado a CLOSED")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Determina si debe intentar reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    def _record_success(self):
        """Registra éxito"""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
    
    def _record_failure(self):
        """Registra fallo"""
        with self.lock:
            current_time = datetime.now()
            self.failure_times.append(current_time)
            self.last_failure_time = current_time
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} cambiado a OPEN (fallo en half-open)")
            
            # Calcular tasa de fallos en ventana de tiempo
            cutoff_time = current_time - timedelta(seconds=self.config.time_window)
            recent_failures = [t for t in self.failure_times if t > cutoff_time]
            
            if len(recent_failures) >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker {self.name} ABIERTO por exceso de fallos")
    
    def get_state(self) -> CircuitBreakerState:
        """Obtiene estado actual"""
        return self.state
    
    def force_open(self):
        """Fuerza apertura del circuit breaker"""
        with self.lock:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = datetime.now()
            logger.warning(f"Circuit breaker {self.name} FORZADO A OPEN")
    
    def force_close(self):
        """Fuerza cierre del circuit breaker"""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} FORZADO A CLOSED")

class RobustRiskManager:
    """Gestor de riesgo robusto con controles avanzados"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.risk_limits = self.config.get_risk_limits()
        self.trading_limits = self.config.get_trading_limits()
        
        # Estado del sistema
        self.current_positions: Dict[str, Dict] = {}
        self.risk_events: List[RiskEvent] = []
        self.portfolio_metrics = RiskMetrics()
        
        # Circuit breakers
        self.circuit_breakers = {
            'position_size': CircuitBreaker('position_size', CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=600
            )),
            'portfolio_risk': CircuitBreaker('portfolio_risk', CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=900
            )),
            'drawdown': CircuitBreaker('drawdown', CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=1800
            )),
            'correlation': CircuitBreaker('correlation', CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=300
            ))
        }
        
        # Límites dinámicos
        self.dynamic_limits = {
            'max_position_size': self.risk_limits.max_position_size,
            'max_portfolio_var': self.risk_limits.var_limit_daily,
            'max_correlation': self.risk_limits.max_correlation_exposure
        }
        
        # Thread pool para cálculos asíncronos
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Historial de precios para cálculos
        self.price_history = {}
        
        logger.info("✅ Sistema robusto de gestión de riesgo inicializado")
    
    def validate_trade_signal(self, signal: TradeSignal, 
                            current_portfolio: Dict[str, float],
                            market_data: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valida señal de trading con controles de riesgo
        
        Args:
            signal: Señal de trading
            current_portfolio: Portfolio actual
            market_data: Datos de mercado
            
        Returns:
            Tuple con (es_válida, razón, métricas_riesgo)
        """
        
        try:
            # Validaciones básicas
            basic_validation = self._validate_basic_limits(signal, current_portfolio)
            if not basic_validation[0]:
                return basic_validation
            
            # Validaciones con circuit breakers
            cb_validation = self._validate_with_circuit_breakers(signal, current_portfolio, market_data)
            if not cb_validation[0]:
                return cb_validation
            
            # Validaciones de riesgo de portfolio
            portfolio_validation = self._validate_portfolio_risk(signal, current_portfolio, market_data)
            if not portfolio_validation[0]:
                return portfolio_validation
            
            # Validaciones de correlación
            correlation_validation = self._validate_correlation_risk(signal, current_portfolio, market_data)
            if not correlation_validation[0]:
                return correlation_validation
            
            # Validaciones de liquidez
            liquidity_validation = self._validate_liquidity_risk(signal, market_data)
            if not liquidity_validation[0]:
                return liquidity_validation
            
            # Calcular métricas de riesgo finales
            risk_metrics = self._calculate_trade_risk_metrics(signal, current_portfolio, market_data)
            
            # Registrar evento
            self._record_risk_event(
                RiskEvent(
                    timestamp=datetime.now(),
                    event_type="TRADE_VALIDATED",
                    severity=RiskLevel.LOW,
                    description=f"Señal validada: {signal.symbol} {signal.action}",
                    symbol=signal.symbol,
                    position_size=signal.position_size,
                    risk_metric=risk_metrics.get('expected_risk', 0)
                )
            )
            
            return True, "Señal validada correctamente", risk_metrics
            
        except Exception as e:
            logger.error(f"Error validando señal: {e}")
            return False, f"Error de validación: {str(e)}", {}
    
    def _validate_basic_limits(self, signal: TradeSignal, 
                             current_portfolio: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """Valida límites básicos"""
        
        # Validar tamaño de posición
        if signal.position_size > self.trading_limits.max_order_size:
            return False, f"Tamaño de orden excede límite: {signal.position_size} > {self.trading_limits.max_order_size}", {}
        
        if signal.position_size < self.trading_limits.min_order_size:
            return False, f"Tamaño de orden menor al mínimo: {signal.position_size} < {self.trading_limits.min_order_size}", {}
        
        # Validar número de posiciones
        current_positions = len([p for p in current_portfolio.values() if p != 0])
        if current_positions >= self.risk_limits.max_total_positions:
            return False, f"Máximo número de posiciones alcanzado: {current_positions}", {}
        
        # Validar posiciones por símbolo
        current_symbol_positions = sum(1 for symbol, pos in current_portfolio.items() 
                                     if symbol.startswith(signal.symbol.split('USDT')[0]) and pos != 0)
        if current_symbol_positions >= self.risk_limits.max_positions_per_asset:
            return False, f"Máximo de posiciones por activo alcanzado: {signal.symbol}", {}
        
        return True, "Límites básicos OK", {}
    
    def _validate_with_circuit_breakers(self, signal: TradeSignal, 
                                      current_portfolio: Dict[str, float],
                                      market_data: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Valida con circuit breakers"""
        
        try:
            # Validar tamaño de posición
            self.circuit_breakers['position_size'].call(
                self._check_position_size_limit, signal.position_size
            )
            
            # Validar riesgo de portfolio
            portfolio_value = sum(abs(v) for v in current_portfolio.values())
            self.circuit_breakers['portfolio_risk'].call(
                self._check_portfolio_risk_limit, signal.position_size, portfolio_value
            )
            
            # Validar drawdown
            if hasattr(self, 'current_drawdown'):
                self.circuit_breakers['drawdown'].call(
                    self._check_drawdown_limit, self.current_drawdown
                )
            
            return True, "Circuit breakers OK", {}
            
        except Exception as e:
            return False, f"Circuit breaker activado: {str(e)}", {}
    
    def _check_position_size_limit(self, position_size: float):
        """Verifica límite de tamaño de posición"""
        max_allowed = self.dynamic_limits['max_position_size']
        if position_size > max_allowed:
            raise Exception(f"Posición excede límite dinámico: {position_size} > {max_allowed}")
    
    def _check_portfolio_risk_limit(self, position_size: float, portfolio_value: float):
        """Verifica límite de riesgo de portfolio"""
        if portfolio_value == 0:
            return
        
        position_ratio = position_size / portfolio_value
        if position_ratio > self.risk_limits.max_portfolio_risk:
            raise Exception(f"Riesgo de portfolio excede límite: {position_ratio:.1%} > {self.risk_limits.max_portfolio_risk:.1%}")
    
    def _check_drawdown_limit(self, current_drawdown: float):
        """Verifica límite de drawdown"""
        if abs(current_drawdown) > self.risk_limits.max_drawdown_stop:
            raise Exception(f"Drawdown excede límite: {abs(current_drawdown):.1%} > {self.risk_limits.max_drawdown_stop:.1%}")
    
    def _validate_portfolio_risk(self, signal: TradeSignal, 
                               current_portfolio: Dict[str, float],
                               market_data: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Valida riesgo de portfolio"""
        
        try:
            # Simular portfolio con nueva posición
            simulated_portfolio = current_portfolio.copy()
            simulated_portfolio[signal.symbol] = simulated_portfolio.get(signal.symbol, 0) + signal.position_size
            
            # Calcular VaR del portfolio simulado
            portfolio_var = self._calculate_portfolio_var(simulated_portfolio, market_data)
            
            if portfolio_var > self.risk_limits.var_limit_daily:
                return False, f"VaR de portfolio excede límite: {portfolio_var:.1%} > {self.risk_limits.var_limit_daily:.1%}", {}
            
            # Calcular concentración
            total_portfolio_value = sum(abs(v) for v in simulated_portfolio.values())
            if total_portfolio_value > 0:
                concentration = max(abs(v) / total_portfolio_value for v in simulated_portfolio.values())
                if concentration > self.risk_limits.max_position_size:
                    return False, f"Concentración excede límite: {concentration:.1%} > {self.risk_limits.max_position_size:.1%}", {}
            
            return True, "Riesgo de portfolio OK", {"portfolio_var": portfolio_var}
            
        except Exception as e:
            logger.error(f"Error validando riesgo de portfolio: {e}")
            return False, f"Error en validación de portfolio: {str(e)}", {}
    
    def _validate_correlation_risk(self, signal: TradeSignal, 
                                 current_portfolio: Dict[str, float],
                                 market_data: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Valida riesgo de correlación"""
        
        try:
            # Obtener símbolos del portfolio actual
            portfolio_symbols = [symbol for symbol, pos in current_portfolio.items() if pos != 0]
            
            if not portfolio_symbols:
                return True, "No hay correlaciones que validar", {}
            
            # Calcular correlaciones
            correlations = self._calculate_symbol_correlations(signal.symbol, portfolio_symbols, market_data)
            
            # Verificar correlaciones altas
            high_correlations = [corr for corr in correlations.values() if abs(corr) > self.risk_limits.max_correlation_exposure]
            
            if high_correlations:
                max_correlation = max(high_correlations, key=abs)
                return False, f"Correlación alta detectada: {max_correlation:.2f} > {self.risk_limits.max_correlation_exposure:.2f}", {}
            
            return True, "Riesgo de correlación OK", {"correlations": correlations}
            
        except Exception as e:
            logger.warning(f"Error validando correlaciones: {e}")
            return True, "Validación de correlación omitida", {}
    
    def _validate_liquidity_risk(self, signal: TradeSignal, market_data: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Valida riesgo de liquidez"""
        
        try:
            if 'volume' not in market_data.columns:
                return True, "Datos de volumen no disponibles", {}
            
            # Calcular volumen promedio
            recent_volume = market_data['volume'].tail(20).mean()
            
            # Estimar impacto de mercado
            position_volume_ratio = signal.position_size / recent_volume if recent_volume > 0 else 0
            
            # Límite de impacto de mercado
            if position_volume_ratio > self.trading_limits.market_impact_threshold:
                return False, f"Impacto de mercado alto: {position_volume_ratio:.1%} > {self.trading_limits.market_impact_threshold:.1%}", {}
            
            return True, "Riesgo de liquidez OK", {"market_impact": position_volume_ratio}
            
        except Exception as e:
            logger.warning(f"Error validando liquidez: {e}")
            return True, "Validación de liquidez omitida", {}
    
    def _calculate_trade_risk_metrics(self, signal: TradeSignal, 
                                    current_portfolio: Dict[str, float],
                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas de riesgo para el trade"""
        
        metrics = {}
        
        try:
            # Volatilidad del activo
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                metrics['asset_volatility'] = returns.std() * np.sqrt(252)  # Anualizada
            
            # Riesgo esperado
            asset_vol = metrics.get('asset_volatility', 0.2)
            position_ratio = signal.position_size / max(sum(abs(v) for v in current_portfolio.values()), signal.position_size)
            metrics['expected_risk'] = asset_vol * position_ratio
            
            # Sharpe ratio esperado
            if signal.confidence > 0:
                expected_return = signal.confidence * 0.1  # Estimación simple
                metrics['expected_sharpe'] = expected_return / max(asset_vol, 0.01)
            
            # Kelly fraction
            if 'expected_sharpe' in metrics:
                metrics['kelly_fraction'] = min(metrics['expected_sharpe'] / 4, 0.25)  # Conservador
            
        except Exception as e:
            logger.warning(f"Error calculando métricas de riesgo: {e}")
        
        return metrics
    
    def _calculate_portfolio_var(self, portfolio: Dict[str, float], 
                               market_data: pd.DataFrame, 
                               confidence_level: float = 0.95) -> float:
        """Calcula VaR del portfolio"""
        
        try:
            if not portfolio or 'close' not in market_data.columns:
                return 0.0
            
            # Calcular returns
            returns = market_data['close'].pct_change().dropna()
            
            # Para simplificar, asumir que todas las posiciones tienen la misma volatilidad
            # En un sistema real, necesitaríamos datos de múltiples activos
            
            total_exposure = sum(abs(v) for v in portfolio.values())
            if total_exposure == 0:
                return 0.0
            
            # VaR simple basado en volatilidad histórica
            volatility = returns.std()
            var_multiplier = stats.norm.ppf(1 - confidence_level)
            portfolio_var = abs(var_multiplier * volatility * np.sqrt(total_exposure))
            
            return portfolio_var
            
        except Exception as e:
            logger.error(f"Error calculando VaR: {e}")
            return 0.0
    
    def _calculate_symbol_correlations(self, symbol: str, 
                                     portfolio_symbols: List[str], 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """Calcula correlaciones entre símbolos"""
        
        correlations = {}
        
        try:
            # Para simplificar, usar datos sintéticos
            # En un sistema real, necesitaríamos datos de múltiples activos
            
            for portfolio_symbol in portfolio_symbols:
                if portfolio_symbol == symbol:
                    continue
                
                # Correlación estimada basada en similitud de símbolos
                base_symbol = symbol.replace('USDT', '').replace('BTC', '')
                portfolio_base = portfolio_symbol.replace('USDT', '').replace('BTC', '')
                
                # Correlaciones sintéticas basadas en categorías
                if base_symbol in ['BTC', 'ETH'] and portfolio_base in ['BTC', 'ETH']:
                    correlations[portfolio_symbol] = 0.8  # Alta correlación entre majors
                elif base_symbol == portfolio_base:
                    correlations[portfolio_symbol] = 1.0  # Mismo activo
                else:
                    correlations[portfolio_symbol] = 0.3  # Correlación moderada por defecto
            
        except Exception as e:
            logger.error(f"Error calculando correlaciones: {e}")
        
        return correlations
    
    def update_portfolio_metrics(self, portfolio: Dict[str, float], 
                               market_data: pd.DataFrame,
                               benchmark_data: Optional[pd.DataFrame] = None):
        """Actualiza métricas de riesgo del portfolio"""
        
        try:
            # Calcular métricas básicas
            self.portfolio_metrics.var_95 = self._calculate_portfolio_var(portfolio, market_data, 0.95)
            self.portfolio_metrics.var_99 = self._calculate_portfolio_var(portfolio, market_data, 0.99)
            
            # Calcular volatilidad
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                self.portfolio_metrics.volatility = returns.std() * np.sqrt(252)
            
            # Calcular drawdown
            if 'close' in market_data.columns:
                cumulative_returns = (1 + market_data['close'].pct_change()).cumprod()
                peak = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - peak) / peak
                self.portfolio_metrics.max_drawdown = drawdown.min()
                self.portfolio_metrics.current_drawdown = drawdown.iloc[-1]
            
            # Determinar nivel de riesgo
            self.portfolio_metrics.risk_level = self._determine_risk_level()
            
            # Ajustar límites dinámicos
            self._adjust_dynamic_limits()
            
        except Exception as e:
            logger.error(f"Error actualizando métricas: {e}")
    
    def _determine_risk_level(self) -> RiskLevel:
        """Determina nivel de riesgo actual"""
        
        risk_scores = []
        
        # Score basado en VaR
        if self.portfolio_metrics.var_95 > 0.1:  # 10%
            risk_scores.append(4)
        elif self.portfolio_metrics.var_95 > 0.05:  # 5%
            risk_scores.append(3)
        elif self.portfolio_metrics.var_95 > 0.02:  # 2%
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Score basado en drawdown
        if abs(self.portfolio_metrics.current_drawdown) > 0.2:  # 20%
            risk_scores.append(4)
        elif abs(self.portfolio_metrics.current_drawdown) > 0.1:  # 10%
            risk_scores.append(3)
        elif abs(self.portfolio_metrics.current_drawdown) > 0.05:  # 5%
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Score basado en volatilidad
        if self.portfolio_metrics.volatility > 0.8:  # 80%
            risk_scores.append(4)
        elif self.portfolio_metrics.volatility > 0.5:  # 50%
            risk_scores.append(3)
        elif self.portfolio_metrics.volatility > 0.3:  # 30%
            risk_scores.append(2)
        else:
            risk_scores.append(1)
        
        # Determinar nivel general
        avg_score = np.mean(risk_scores)
        
        if avg_score >= 3.5:
            return RiskLevel.CRITICAL
        elif avg_score >= 2.5:
            return RiskLevel.HIGH
        elif avg_score >= 1.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _adjust_dynamic_limits(self):
        """Ajusta límites dinámicos basado en condiciones de mercado"""
        
        risk_level = self.portfolio_metrics.risk_level
        
        # Ajustar límites basado en nivel de riesgo
        if risk_level == RiskLevel.CRITICAL:
            self.dynamic_limits['max_position_size'] = self.risk_limits.max_position_size * 0.5
            self.dynamic_limits['max_portfolio_var'] = self.risk_limits.var_limit_daily * 0.5
        elif risk_level == RiskLevel.HIGH:
            self.dynamic_limits['max_position_size'] = self.risk_limits.max_position_size * 0.7
            self.dynamic_limits['max_portfolio_var'] = self.risk_limits.var_limit_daily * 0.7
        elif risk_level == RiskLevel.MEDIUM:
            self.dynamic_limits['max_position_size'] = self.risk_limits.max_position_size * 0.9
            self.dynamic_limits['max_portfolio_var'] = self.risk_limits.var_limit_daily * 0.9
        else:
            self.dynamic_limits['max_position_size'] = self.risk_limits.max_position_size
            self.dynamic_limits['max_portfolio_var'] = self.risk_limits.var_limit_daily
        
        logger.info(f"Límites dinámicos ajustados para riesgo {risk_level.value}")
    
    def _record_risk_event(self, event: RiskEvent):
        """Registra evento de riesgo"""
        self.risk_events.append(event)
        
        # Mantener solo los últimos 1000 eventos
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-1000:]
        
        # Log del evento
        logger.info(f"Evento de riesgo: {event.event_type} - {event.description}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Genera reporte de riesgo"""
        
        return {
            'timestamp': datetime.now(),
            'portfolio_metrics': {
                'var_95': self.portfolio_metrics.var_95,
                'var_99': self.portfolio_metrics.var_99,
                'expected_shortfall': self.portfolio_metrics.expected_shortfall,
                'max_drawdown': self.portfolio_metrics.max_drawdown,
                'current_drawdown': self.portfolio_metrics.current_drawdown,
                'volatility': self.portfolio_metrics.volatility,
                'risk_level': self.portfolio_metrics.risk_level.value
            },
            'circuit_breakers': {
                name: breaker.get_state().value 
                for name, breaker in self.circuit_breakers.items()
            },
            'dynamic_limits': self.dynamic_limits,
            'recent_events': [
                {
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'severity': event.severity.value,
                    'description': event.description
                }
                for event in self.risk_events[-10:]  # Últimos 10 eventos
            ]
        }
    
    def emergency_stop(self, reason: str):
        """Detiene todas las operaciones por emergencia"""
        
        logger.error(f"🚨 PARADA DE EMERGENCIA: {reason}")
        
        # Abrir todos los circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker.force_open()
        
        # Registrar evento crítico
        self._record_risk_event(
            RiskEvent(
                timestamp=datetime.now(),
                event_type="EMERGENCY_STOP",
                severity=RiskLevel.CRITICAL,
                description=f"Parada de emergencia: {reason}"
            )
        )
        
        # Aquí se podrían cerrar posiciones automáticamente
        # (implementar según necesidades específicas)
    
    def reset_circuit_breakers(self):
        """Resetea todos los circuit breakers"""
        
        for name, breaker in self.circuit_breakers.items():
            breaker.force_close()
            logger.info(f"Circuit breaker {name} reseteado")
    
    def get_current_limits(self) -> Dict[str, float]:
        """Obtiene límites actuales"""
        return self.dynamic_limits.copy()

# Singleton para acceso global
_risk_manager = None

def get_risk_manager() -> RobustRiskManager:
    """Obtiene el gestor de riesgo (singleton)"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RobustRiskManager()
    return _risk_manager 