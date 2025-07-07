# SISTEMA DE BACKTESTING REALISTA
"""
Sistema avanzado de backtesting con costos realistas, market impact dinámico,
y slippage sofisticado para resultados más precisos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Análisis financiero
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Interfaces
from core.interfaces import IBacktester, TradeSignal, RiskMetrics
from core.robust_risk_manager import get_risk_manager
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Tipos de orden"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Lados de orden"""
    BUY = "buy"
    SELL = "sell"

class ExecutionQuality(Enum):
    """Calidad de ejecución"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class MarketConditions:
    """Condiciones de mercado"""
    volatility: float
    spread_bps: float
    volume_ratio: float  # Volumen actual vs promedio
    market_impact_factor: float
    liquidity_score: float
    time_of_day_factor: float

@dataclass
class TradingCosts:
    """Costos de trading"""
    commission: float
    spread_cost: float
    market_impact: float
    slippage: float
    financing_cost: float
    total_cost: float

@dataclass
class ExecutionReport:
    """Reporte de ejecución"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_quantity: float
    executed_quantity: float
    avg_price: float
    execution_time: datetime
    execution_quality: ExecutionQuality
    costs: TradingCosts
    market_conditions: MarketConditions
    fill_ratio: float

@dataclass
class RealisticTrade:
    """Trade realista con todos los costos"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_costs: TradingCosts
    exit_costs: Optional[TradingCosts]
    unrealized_pnl: float
    realized_pnl: Optional[float]
    max_favorable_excursion: float
    max_adverse_excursion: float
    holding_period: Optional[timedelta]
    execution_reports: List[ExecutionReport]

class MarketImpactModel:
    """Modelo de impacto de mercado"""
    
    def __init__(self):
        self.impact_parameters = {
            'permanent_impact_factor': 0.1,  # Factor de impacto permanente
            'temporary_impact_factor': 0.5,  # Factor de impacto temporal
            'volume_power': 0.6,  # Exponente para volumen
            'volatility_scaling': 1.5,  # Escalamiento por volatilidad
            'liquidity_adjustment': 0.8  # Ajuste por liquidez
        }
    
    def calculate_market_impact(self, order_size: float, market_data: pd.Series,
                              market_conditions: MarketConditions) -> Tuple[float, float]:
        """
        Calcula impacto de mercado permanente y temporal
        
        Args:
            order_size: Tamaño de la orden
            market_data: Datos de mercado recientes
            market_conditions: Condiciones actuales del mercado
            
        Returns:
            Tuple con (impacto_permanente, impacto_temporal)
        """
        
        try:
            # Obtener volumen promedio
            avg_volume = market_data.get('volume', 1000000)
            current_price = market_data.get('close', 100)
            
            # Calcular participación en volumen
            volume_participation = (order_size / current_price) / avg_volume
            
            # Impacto base
            base_impact = volume_participation ** self.impact_parameters['volume_power']
            
            # Ajustes por condiciones de mercado
            volatility_adj = (market_conditions.volatility / 0.3) ** self.impact_parameters['volatility_scaling']
            liquidity_adj = (1 / market_conditions.liquidity_score) ** self.impact_parameters['liquidity_adjustment']
            
            # Impacto permanente (afecta el precio de manera duradera)
            permanent_impact = (base_impact * volatility_adj * liquidity_adj * 
                              self.impact_parameters['permanent_impact_factor'])
            
            # Impacto temporal (se revierte después de la ejecución)
            temporary_impact = (base_impact * volatility_adj * 
                              self.impact_parameters['temporary_impact_factor'])
            
            return permanent_impact, temporary_impact
            
        except Exception as e:
            logger.error(f"Error calculando impacto de mercado: {e}")
            return 0.001, 0.002  # Valores por defecto conservadores

class SlippageModel:
    """Modelo sofisticado de slippage"""
    
    def __init__(self):
        self.slippage_parameters = {
            'base_slippage': 0.0005,  # 0.05% base
            'volatility_multiplier': 2.0,
            'volume_factor': 1.5,
            'spread_factor': 0.5,
            'time_decay': 0.1,  # Decay por demora en ejecución
            'market_hours_factor': 0.8  # Menor slippage en horas de mercado activo
        }
    
    def calculate_slippage(self, order_type: OrderType, order_size: float,
                          market_conditions: MarketConditions,
                          execution_delay: float = 0) -> float:
        """
        Calcula slippage realista basado en múltiples factores
        
        Args:
            order_type: Tipo de orden
            order_size: Tamaño de la orden
            market_conditions: Condiciones del mercado
            execution_delay: Demora en ejecución (segundos)
            
        Returns:
            Slippage como porcentaje
        """
        
        try:
            base_slippage = self.slippage_parameters['base_slippage']
            
            # Ajuste por tipo de orden
            if order_type == OrderType.MARKET:
                type_multiplier = 1.0
            elif order_type == OrderType.LIMIT:
                type_multiplier = 0.3  # Menor slippage para limit orders
            else:
                type_multiplier = 0.8
            
            # Ajuste por volatilidad
            vol_adjustment = (market_conditions.volatility / 0.3) ** self.slippage_parameters['volatility_multiplier']
            
            # Ajuste por volumen/liquidez
            volume_adjustment = (1 / market_conditions.volume_ratio) ** self.slippage_parameters['volume_factor']
            
            # Ajuste por spread
            spread_adjustment = 1 + (market_conditions.spread_bps / 10) * self.slippage_parameters['spread_factor']
            
            # Ajuste por demora en ejecución
            time_adjustment = 1 + (execution_delay / 60) * self.slippage_parameters['time_decay']
            
            # Ajuste por hora del día
            time_factor = market_conditions.time_of_day_factor * self.slippage_parameters['market_hours_factor']
            
            # Slippage final
            total_slippage = (base_slippage * type_multiplier * vol_adjustment * 
                            volume_adjustment * spread_adjustment * time_adjustment * time_factor)
            
            # Limitar slippage máximo
            return min(total_slippage, 0.05)  # Máximo 5%
            
        except Exception as e:
            logger.error(f"Error calculando slippage: {e}")
            return self.slippage_parameters['base_slippage']

class CommissionModel:
    """Modelo de comisiones dinámico"""
    
    def __init__(self):
        self.commission_tiers = {
            'maker': {
                0: 0.001,      # 0.1% para volumen bajo
                100000: 0.0008, # 0.08% para volumen medio
                1000000: 0.0006, # 0.06% para volumen alto
                10000000: 0.0004 # 0.04% para volumen muy alto
            },
            'taker': {
                0: 0.001,      # 0.1%
                100000: 0.001,  # 0.1%
                1000000: 0.0008, # 0.08%
                10000000: 0.0006 # 0.06%
            }
        }
        
        self.volume_tracking = {}  # Tracking de volumen por usuario
    
    def calculate_commission(self, order_size: float, is_maker: bool = False,
                           user_id: str = "default") -> float:
        """Calcula comisión basada en volumen histórico"""
        
        # Obtener volumen histórico del usuario
        historical_volume = self.volume_tracking.get(user_id, 0)
        
        # Seleccionar tier apropiado
        commission_schedule = self.commission_tiers['maker' if is_maker else 'taker']
        
        # Encontrar tier correspondiente
        commission_rate = commission_schedule[0]
        for volume_threshold in sorted(commission_schedule.keys(), reverse=True):
            if historical_volume >= volume_threshold:
                commission_rate = commission_schedule[volume_threshold]
                break
        
        # Actualizar volumen histórico
        self.volume_tracking[user_id] = historical_volume + order_size
        
        return commission_rate * order_size

class OrderExecutionEngine:
    """Motor de ejecución de órdenes realista"""
    
    def __init__(self):
        self.market_impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()
        self.commission_model = CommissionModel()
        
        # Estado del libro de órdenes simulado
        self.order_book_state = {}
        
    def execute_order(self, signal: TradeSignal, market_data: pd.DataFrame,
                     current_time: datetime) -> ExecutionReport:
        """
        Ejecuta una orden con modeling realista
        
        Args:
            signal: Señal de trading
            market_data: Datos de mercado
            current_time: Tiempo actual de ejecución
            
        Returns:
            ExecutionReport con detalles de ejecución
        """
        
        try:
            # Analizar condiciones de mercado
            market_conditions = self._analyze_market_conditions(market_data, current_time)
            
            # Determinar tipo de orden
            order_type = self._determine_order_type(signal, market_conditions)
            
            # Calcular costos de trading
            costs = self._calculate_trading_costs(
                signal, market_data.iloc[-1], market_conditions, order_type
            )
            
            # Simular ejecución
            execution_result = self._simulate_execution(
                signal, market_data.iloc[-1], market_conditions, order_type, costs
            )
            
            # Crear reporte de ejecución
            execution_report = ExecutionReport(
                order_id=f"order_{int(current_time.timestamp())}",
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.action == 'BUY' else OrderSide.SELL,
                order_type=order_type,
                requested_quantity=signal.position_size,
                executed_quantity=execution_result['executed_quantity'],
                avg_price=execution_result['avg_price'],
                execution_time=current_time,
                execution_quality=execution_result['quality'],
                costs=costs,
                market_conditions=market_conditions,
                fill_ratio=execution_result['executed_quantity'] / signal.position_size
            )
            
            return execution_report
            
        except Exception as e:
            logger.error(f"Error ejecutando orden: {e}")
            # Retornar ejecución básica en caso de error
            return self._create_fallback_execution_report(signal, current_time)
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame, 
                                 current_time: datetime) -> MarketConditions:
        """Analiza condiciones actuales del mercado"""
        
        try:
            recent_data = market_data.tail(20)
            
            # Calcular volatilidad
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Anualizada
            
            # Estimar spread (simplificado)
            spread_bps = max(0.5, volatility * 100)  # Spread basado en volatilidad
            
            # Calcular ratio de volumen
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Factor de impacto de mercado
            market_impact_factor = 1.0 / max(0.1, volume_ratio)
            
            # Score de liquidez (basado en volumen y spread)
            liquidity_score = min(1.0, volume_ratio / (1 + spread_bps / 10))
            
            # Factor de hora del día
            hour = current_time.hour
            if 9 <= hour <= 16:  # Horas activas
                time_of_day_factor = 1.0
            elif 17 <= hour <= 23 or 0 <= hour <= 8:  # Horas menos activas
                time_of_day_factor = 0.7
            else:  # Horas de muy baja actividad
                time_of_day_factor = 0.5
            
            return MarketConditions(
                volatility=volatility,
                spread_bps=spread_bps,
                volume_ratio=volume_ratio,
                market_impact_factor=market_impact_factor,
                liquidity_score=liquidity_score,
                time_of_day_factor=time_of_day_factor
            )
            
        except Exception as e:
            logger.error(f"Error analizando condiciones de mercado: {e}")
            # Condiciones por defecto
            return MarketConditions(
                volatility=0.3,
                spread_bps=1.0,
                volume_ratio=1.0,
                market_impact_factor=1.0,
                liquidity_score=0.8,
                time_of_day_factor=1.0
            )
    
    def _determine_order_type(self, signal: TradeSignal, 
                            market_conditions: MarketConditions) -> OrderType:
        """Determina el tipo de orden óptimo"""
        
        # Usar market orders para señales de alta confianza o condiciones de alta liquidez
        if signal.confidence > 0.8 or market_conditions.liquidity_score > 0.9:
            return OrderType.MARKET
        
        # Usar limit orders para condiciones normales
        return OrderType.LIMIT
    
    def _calculate_trading_costs(self, signal: TradeSignal, market_data: pd.Series,
                               market_conditions: MarketConditions, 
                               order_type: OrderType) -> TradingCosts:
        """Calcula todos los costos de trading"""
        
        try:
            # Comisión
            is_maker = order_type == OrderType.LIMIT
            commission = self.commission_model.calculate_commission(
                signal.position_size, is_maker
            )
            
            # Spread cost
            spread_cost = (market_conditions.spread_bps / 10000) * signal.position_size
            
            # Market impact
            permanent_impact, temporary_impact = self.market_impact_model.calculate_market_impact(
                signal.position_size, market_data, market_conditions
            )
            market_impact = (permanent_impact + temporary_impact) * signal.position_size
            
            # Slippage
            slippage = self.slippage_model.calculate_slippage(
                order_type, signal.position_size, market_conditions
            ) * signal.position_size
            
            # Financing cost (para posiciones mantenidas durante la noche)
            financing_cost = 0.0  # Se calculará en el backtester principal
            
            total_cost = commission + spread_cost + market_impact + slippage + financing_cost
            
            return TradingCosts(
                commission=commission,
                spread_cost=spread_cost,
                market_impact=market_impact,
                slippage=slippage,
                financing_cost=financing_cost,
                total_cost=total_cost
            )
            
        except Exception as e:
            logger.error(f"Error calculando costos: {e}")
            # Costos conservadores por defecto
            default_cost = signal.position_size * 0.002  # 0.2%
            return TradingCosts(
                commission=default_cost * 0.5,
                spread_cost=default_cost * 0.2,
                market_impact=default_cost * 0.2,
                slippage=default_cost * 0.1,
                financing_cost=0.0,
                total_cost=default_cost
            )
    
    def _simulate_execution(self, signal: TradeSignal, market_data: pd.Series,
                          market_conditions: MarketConditions, order_type: OrderType,
                          costs: TradingCosts) -> Dict[str, Any]:
        """Simula la ejecución de la orden"""
        
        try:
            current_price = market_data['close']
            
            # Determinar precio de ejecución
            if order_type == OrderType.MARKET:
                # Market order: precio actual + slippage
                execution_price = current_price * (1 + costs.slippage / signal.position_size)
                fill_probability = 1.0  # Market orders se ejecutan completamente
                
            elif order_type == OrderType.LIMIT:
                # Limit order: mejor precio pero menor probabilidad de ejecución
                if signal.action == 'BUY':
                    execution_price = current_price * 0.999  # Slightly better price
                else:
                    execution_price = current_price * 1.001
                
                # Probabilidad de ejecución basada en condiciones de mercado
                fill_probability = min(0.95, market_conditions.liquidity_score * 1.2)
            
            else:
                execution_price = current_price
                fill_probability = 0.8
            
            # Determinar cantidad ejecutada
            if np.random.random() < fill_probability:
                executed_quantity = signal.position_size
                quality = ExecutionQuality.EXCELLENT if fill_probability > 0.9 else ExecutionQuality.GOOD
            else:
                # Ejecución parcial
                executed_quantity = signal.position_size * np.random.uniform(0.3, 0.8)
                quality = ExecutionQuality.FAIR
            
            return {
                'executed_quantity': executed_quantity,
                'avg_price': execution_price,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Error simulando ejecución: {e}")
            return {
                'executed_quantity': signal.position_size,
                'avg_price': market_data['close'],
                'quality': ExecutionQuality.FAIR
            }
    
    def _create_fallback_execution_report(self, signal: TradeSignal, 
                                        current_time: datetime) -> ExecutionReport:
        """Crea reporte de ejecución básico como fallback"""
        
        basic_cost = signal.position_size * 0.001
        
        return ExecutionReport(
            order_id=f"fallback_{int(current_time.timestamp())}",
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.action == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            requested_quantity=signal.position_size,
            executed_quantity=signal.position_size,
            avg_price=100.0,  # Precio placeholder
            execution_time=current_time,
            execution_quality=ExecutionQuality.FAIR,
            costs=TradingCosts(basic_cost, 0, 0, 0, 0, basic_cost),
            market_conditions=MarketConditions(0.3, 1.0, 1.0, 1.0, 0.8, 1.0),
            fill_ratio=1.0
        )

class RealisticBacktester(IBacktester):
    """Backtester realista con costos y ejecución sofisticados"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.risk_manager = get_risk_manager()
        self.execution_engine = OrderExecutionEngine()
        
        # Estado del backtest
        self.trades: List[RealisticTrade] = []
        self.open_positions: Dict[str, RealisticTrade] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.execution_reports: List[ExecutionReport] = []
        
        # Métricas de performance
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_market_impact = 0.0
        
        logger.info("✅ Backtester realista inicializado")
    
    def run_backtest(self, strategy, data: pd.DataFrame, 
                    initial_capital: float) -> Dict[str, Any]:
        """
        Ejecuta backtest completo con modeling realista
        
        Args:
            strategy: Estrategia de trading
            data: Datos históricos
            initial_capital: Capital inicial
            
        Returns:
            Resultados detallados del backtest
        """
        
        try:
            logger.info(f"🚀 Iniciando backtest realista con capital: ${initial_capital:,.2f}")
            
            # Inicializar
            self._initialize_backtest(initial_capital)
            
            # Generar señales
            signals = strategy.generate_signals(data)
            logger.info(f"📊 Generadas {len(signals)} señales")
            
            # Procesar cada señal
            for i, signal in enumerate(signals):
                try:
                    # Obtener datos de mercado para este momento
                    if signal.timestamp not in data.index:
                        continue
                    
                    current_data = data.loc[:signal.timestamp].tail(50)  # Datos históricos hasta ahora
                    
                    # Validar señal con risk manager
                    portfolio_state = self._get_current_portfolio_state()
                    is_valid, reason, risk_metrics = self.risk_manager.validate_trade_signal(
                        signal, portfolio_state, current_data
                    )
                    
                    if not is_valid:
                        logger.debug(f"Señal rechazada: {reason}")
                        continue
                    
                    # Ejecutar orden
                    execution_report = self.execution_engine.execute_order(
                        signal, current_data, signal.timestamp
                    )
                    
                    self.execution_reports.append(execution_report)
                    
                    # Procesar ejecución
                    self._process_execution(execution_report, current_data.iloc[-1])
                    
                    # Actualizar portfolio
                    self._update_portfolio_state(current_data.iloc[-1])
                    
                    if i % 100 == 0:
                        logger.info(f"📈 Procesadas {i}/{len(signals)} señales")
                        
                except Exception as e:
                    logger.error(f"Error procesando señal {i}: {e}")
                    continue
            
            # Cerrar posiciones abiertas al final
            self._close_all_positions(data.iloc[-1])
            
            # Calcular métricas finales
            results = self._calculate_backtest_results(data, initial_capital)
            
            logger.info(f"✅ Backtest completado. Retorno total: {results['total_return']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en backtest: {e}")
            raise
    
    def _initialize_backtest(self, initial_capital: float):
        """Inicializa el backtest"""
        self.trades = []
        self.open_positions = {}
        self.portfolio_history = []
        self.execution_reports = []
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_market_impact = 0.0
        self.cash = initial_capital
        self.initial_capital = initial_capital
    
    def _get_current_portfolio_state(self) -> Dict[str, float]:
        """Obtiene estado actual del portfolio"""
        portfolio = {}
        
        for symbol, trade in self.open_positions.items():
            if trade.side == OrderSide.BUY:
                portfolio[symbol] = trade.quantity
            else:
                portfolio[symbol] = -trade.quantity
        
        return portfolio
    
    def _process_execution(self, execution_report: ExecutionReport, market_data: pd.Series):
        """Procesa la ejecución de una orden"""
        
        try:
            symbol = execution_report.symbol
            
            # Actualizar costos totales
            self.total_commission_paid += execution_report.costs.commission
            self.total_slippage_cost += execution_report.costs.slippage
            self.total_market_impact += execution_report.costs.market_impact
            
            if execution_report.side == OrderSide.BUY:
                # Abrir posición long o cerrar posición short
                if symbol in self.open_positions:
                    existing_trade = self.open_positions[symbol]
                    if existing_trade.side == OrderSide.SELL:
                        # Cerrar posición short
                        self._close_position(symbol, execution_report, market_data)
                    else:
                        # Aumentar posición long
                        self._increase_position(symbol, execution_report)
                else:
                    # Nueva posición long
                    self._open_position(execution_report, market_data)
            
            else:  # SELL
                # Abrir posición short o cerrar posición long
                if symbol in self.open_positions:
                    existing_trade = self.open_positions[symbol]
                    if existing_trade.side == OrderSide.BUY:
                        # Cerrar posición long
                        self._close_position(symbol, execution_report, market_data)
                    else:
                        # Aumentar posición short
                        self._increase_position(symbol, execution_report)
                else:
                    # Nueva posición short
                    self._open_position(execution_report, market_data)
            
        except Exception as e:
            logger.error(f"Error procesando ejecución: {e}")
    
    def _open_position(self, execution_report: ExecutionReport, market_data: pd.Series):
        """Abre nueva posición"""
        
        trade = RealisticTrade(
            entry_time=execution_report.execution_time,
            exit_time=None,
            symbol=execution_report.symbol,
            side=execution_report.side,
            entry_price=execution_report.avg_price,
            exit_price=None,
            quantity=execution_report.executed_quantity,
            entry_costs=execution_report.costs,
            exit_costs=None,
            unrealized_pnl=0.0,
            realized_pnl=None,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0,
            holding_period=None,
            execution_reports=[execution_report]
        )
        
        self.open_positions[execution_report.symbol] = trade
        
        # Actualizar cash
        total_cost = execution_report.executed_quantity * execution_report.avg_price + execution_report.costs.total_cost
        if execution_report.side == OrderSide.BUY:
            self.cash -= total_cost
        else:
            self.cash += total_cost
    
    def _close_position(self, symbol: str, execution_report: ExecutionReport, market_data: pd.Series):
        """Cierra posición existente"""
        
        if symbol not in self.open_positions:
            return
        
        existing_trade = self.open_positions[symbol]
        
        # Calcular PnL realizado
        if existing_trade.side == OrderSide.BUY:
            # Cerrando posición long con venta
            realized_pnl = ((execution_report.avg_price - existing_trade.entry_price) * 
                          min(existing_trade.quantity, execution_report.executed_quantity))
        else:
            # Cerrando posición short con compra
            realized_pnl = ((existing_trade.entry_price - execution_report.avg_price) * 
                          min(existing_trade.quantity, execution_report.executed_quantity))
        
        # Restar costos
        total_costs = existing_trade.entry_costs.total_cost + execution_report.costs.total_cost
        realized_pnl -= total_costs
        
        # Completar el trade
        existing_trade.exit_time = execution_report.execution_time
        existing_trade.exit_price = execution_report.avg_price
        existing_trade.exit_costs = execution_report.costs
        existing_trade.realized_pnl = realized_pnl
        existing_trade.holding_period = existing_trade.exit_time - existing_trade.entry_time
        existing_trade.execution_reports.append(execution_report)
        
        # Mover a trades completados
        self.trades.append(existing_trade)
        del self.open_positions[symbol]
        
        # Actualizar cash
        if execution_report.side == OrderSide.SELL:
            proceeds = execution_report.executed_quantity * execution_report.avg_price - execution_report.costs.total_cost
            self.cash += proceeds
        else:
            cost = execution_report.executed_quantity * execution_report.avg_price + execution_report.costs.total_cost
            self.cash -= cost
    
    def _increase_position(self, symbol: str, execution_report: ExecutionReport):
        """Aumenta posición existente"""
        
        if symbol not in self.open_positions:
            return
        
        existing_trade = self.open_positions[symbol]
        
        # Calcular nuevo precio promedio
        total_quantity = existing_trade.quantity + execution_report.executed_quantity
        total_cost = (existing_trade.quantity * existing_trade.entry_price + 
                     execution_report.executed_quantity * execution_report.avg_price)
        
        existing_trade.entry_price = total_cost / total_quantity
        existing_trade.quantity = total_quantity
        existing_trade.execution_reports.append(execution_report)
        
        # Actualizar costos
        existing_trade.entry_costs.total_cost += execution_report.costs.total_cost
    
    def _close_all_positions(self, market_data: pd.Series):
        """Cierra todas las posiciones abiertas al final del backtest"""
        
        for symbol, trade in list(self.open_positions.items()):
            # Simular cierre al precio de mercado
            current_price = market_data.get('close', trade.entry_price)
            
            # Crear execution report sintético para el cierre
            close_execution = ExecutionReport(
                order_id=f"close_{symbol}_{int(trade.entry_time.timestamp())}",
                symbol=symbol,
                side=OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.MARKET,
                requested_quantity=trade.quantity,
                executed_quantity=trade.quantity,
                avg_price=current_price,
                execution_time=datetime.now(),
                execution_quality=ExecutionQuality.FAIR,
                costs=TradingCosts(0.001 * trade.quantity, 0, 0, 0, 0, 0.001 * trade.quantity),
                market_conditions=MarketConditions(0.3, 1.0, 1.0, 1.0, 0.8, 1.0),
                fill_ratio=1.0
            )
            
            self._close_position(symbol, close_execution, market_data)
    
    def _update_portfolio_state(self, market_data: pd.Series):
        """Actualiza estado del portfolio"""
        
        # Calcular valor total del portfolio
        total_value = self.cash
        
        for symbol, trade in self.open_positions.items():
            current_price = market_data.get('close', trade.entry_price)
            
            if trade.side == OrderSide.BUY:
                position_value = trade.quantity * current_price
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                position_value = trade.quantity * trade.entry_price  # Short position
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
            
            total_value += position_value
            trade.unrealized_pnl = unrealized_pnl - trade.entry_costs.total_cost
            
            # Actualizar MAE y MFE
            if unrealized_pnl > trade.max_favorable_excursion:
                trade.max_favorable_excursion = unrealized_pnl
            if unrealized_pnl < trade.max_adverse_excursion:
                trade.max_adverse_excursion = unrealized_pnl
        
        # Guardar en historial
        self.portfolio_history.append({
            'timestamp': market_data.name,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'open_positions': len(self.open_positions)
        })
    
    def _calculate_backtest_results(self, data: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Calcula resultados completos del backtest"""
        
        try:
            # Métricas básicas
            final_value = self.portfolio_history[-1]['total_value'] if self.portfolio_history else initial_capital
            total_return = (final_value - initial_capital) / initial_capital
            
            # Análisis de trades
            winning_trades = [t for t in self.trades if t.realized_pnl and t.realized_pnl > 0]
            losing_trades = [t for t in self.trades if t.realized_pnl and t.realized_pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
            
            # Costos totales
            total_costs = self.total_commission_paid + self.total_slippage_cost + self.total_market_impact
            cost_ratio = total_costs / initial_capital
            
            # Métricas de ejecución
            execution_quality_dist = {}
            for report in self.execution_reports:
                quality = report.execution_quality.value
                execution_quality_dist[quality] = execution_quality_dist.get(quality, 0) + 1
            
            # Análisis temporal
            holding_periods = [t.holding_period.total_seconds() / 3600 for t in self.trades 
                             if t.holding_period]
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            return {
                'total_return': total_return,
                'final_value': final_value,
                'initial_capital': initial_capital,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
                'total_costs': total_costs,
                'cost_ratio': cost_ratio,
                'commission_paid': self.total_commission_paid,
                'slippage_cost': self.total_slippage_cost,
                'market_impact': self.total_market_impact,
                'execution_quality': execution_quality_dist,
                'avg_holding_period_hours': avg_holding_period,
                'portfolio_history': self.portfolio_history,
                'completed_trades': [self._trade_to_dict(t) for t in self.trades],
                'execution_reports': [self._execution_report_to_dict(r) for r in self.execution_reports]
            }
            
        except Exception as e:
            logger.error(f"Error calculando resultados: {e}")
            return {
                'total_return': 0,
                'error': str(e)
            }
    
    def _trade_to_dict(self, trade: RealisticTrade) -> Dict[str, Any]:
        """Convierte trade a diccionario"""
        return {
            'symbol': trade.symbol,
            'side': trade.side.value,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'realized_pnl': trade.realized_pnl,
            'holding_period_hours': trade.holding_period.total_seconds() / 3600 if trade.holding_period else None,
            'entry_costs': trade.entry_costs.total_cost,
            'exit_costs': trade.exit_costs.total_cost if trade.exit_costs else 0
        }
    
    def _execution_report_to_dict(self, report: ExecutionReport) -> Dict[str, Any]:
        """Convierte execution report a diccionario"""
        return {
            'symbol': report.symbol,
            'side': report.side.value,
            'order_type': report.order_type.value,
            'execution_time': report.execution_time.isoformat(),
            'avg_price': report.avg_price,
            'executed_quantity': report.executed_quantity,
            'fill_ratio': report.fill_ratio,
            'execution_quality': report.execution_quality.value,
            'total_costs': report.costs.total_cost
        }

# Singleton para acceso global
_realistic_backtester = None

def get_realistic_backtester() -> RealisticBacktester:
    """Obtiene el backtester realista (singleton)"""
    global _realistic_backtester
    if _realistic_backtester is None:
        _realistic_backtester = RealisticBacktester()
    return _realistic_backtester 