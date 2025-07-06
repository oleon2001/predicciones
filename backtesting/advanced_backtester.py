# FRAMEWORK AVANZADO DE BACKTESTING
"""
Sistema completo de backtesting para estrategias de trading cuantitativo
Incluye métricas institucionales, análisis de drawdown, y optimización de parámetros
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Análisis financiero
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Interfaces
from core.interfaces import IBacktester, BacktestResults, TradeSignal, RiskMetrics
from core.risk_manager import AdvancedRiskManager
from config.system_config import SystemConfig, BacktestConfig

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Representación de una operación"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration: timedelta
    max_adverse_excursion: float  # MAE
    max_favorable_excursion: float  # MFE
    
    @property
    def is_winning(self) -> bool:
        return self.pnl > 0

@dataclass
class PerformanceMetrics:
    """Métricas de performance completas"""
    # Retornos
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Riesgo
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # Trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Ratios avanzados
    omega_ratio: float
    tail_ratio: float
    skewness: float
    kurtosis: float
    
    # Benchmark
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

class Strategy(ABC):
    """Clase base para estrategias"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Genera señales de trading"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Nombre de la estrategia"""
        pass

class AdvancedBacktester(IBacktester):
    """Backtester avanzado con métricas institucionales"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_manager = AdvancedRiskManager(config)
        self.trades = []
        self.portfolio_history = []
        self.positions = {}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        self.benchmark_returns = None
        
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame, 
                    initial_capital: float) -> BacktestResults:
        """
        Ejecuta backtest completo
        
        Args:
            strategy: Estrategia a testear
            data: Datos históricos
            initial_capital: Capital inicial
            
        Returns:
            Resultados completos del backtest
        """
        try:
            logger.info(f"Iniciando backtest para {strategy.get_name()}")
            
            # Inicializar
            self._initialize_backtest(initial_capital)
            
            # Generar señales
            signals = strategy.generate_signals(data)
            logger.info(f"Generadas {len(signals)} señales")
            
            # Ejecutar trades
            self._execute_trades(signals, data)
            
            # Calcular métricas
            portfolio_returns = self._calculate_portfolio_returns()
            metrics = self.calculate_metrics(portfolio_returns, self.benchmark_returns)
            
            # Análisis de trades
            trade_analysis = self._analyze_trades()
            
            # Crear resultados
            results = BacktestResults(
                strategy_name=strategy.get_name(),
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_capital=initial_capital,
                final_capital=self.total_value,
                total_return=(self.total_value - initial_capital) / initial_capital,
                metrics=metrics,
                trades=self.trades,
                portfolio_history=self.portfolio_history,
                trade_analysis=trade_analysis,
                benchmark_return=self.benchmark_returns.iloc[-1] if self.benchmark_returns is not None else 0
            )
            
            logger.info(f"Backtest completado. Retorno total: {results.total_return:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en backtest: {e}")
            raise
    
    def _initialize_backtest(self, initial_capital: float):
        """Inicializa el backtest"""
        self.trades = []
        self.portfolio_history = []
        self.positions = {}
        self.cash = initial_capital
        self.total_value = initial_capital
        
    def _execute_trades(self, signals: List[TradeSignal], data: pd.DataFrame):
        """Ejecuta las operaciones basadas en señales"""
        for signal in signals:
            try:
                if signal.timestamp not in data.index:
                    continue
                
                current_price = data.loc[signal.timestamp, 'close']
                
                if signal.action == 'BUY':
                    self._execute_buy(signal, current_price, signal.timestamp)
                elif signal.action == 'SELL':
                    self._execute_sell(signal, current_price, signal.timestamp)
                
                # Actualizar portfolio
                self._update_portfolio_value(data.loc[signal.timestamp])
                
            except Exception as e:
                logger.warning(f"Error ejecutando señal: {e}")
                continue
    
    def _execute_buy(self, signal: TradeSignal, price: float, timestamp: datetime):
        """Ejecuta compra"""
        # Aplicar slippage
        execution_price = price * (1 + self.config.slippage_rate)
        
        # Calcular cantidad
        if signal.position_size > 0:
            quantity = signal.position_size / execution_price
        else:
            quantity = (self.cash * 0.95) / execution_price  # 95% del cash disponible
        
        # Verificar suficiente cash
        total_cost = quantity * execution_price
        commission = total_cost * self.config.commission_rate
        
        if total_cost + commission > self.cash:
            logger.warning(f"Insuficiente cash para compra: {total_cost + commission} > {self.cash}")
            return
        
        # Ejecutar
        self.cash -= (total_cost + commission)
        
        if signal.symbol in self.positions:
            self.positions[signal.symbol] += quantity
        else:
            self.positions[signal.symbol] = quantity
        
        logger.debug(f"Compra ejecutada: {quantity:.6f} {signal.symbol} a {execution_price:.4f}")
    
    def _execute_sell(self, signal: TradeSignal, price: float, timestamp: datetime):
        """Ejecuta venta"""
        if signal.symbol not in self.positions or self.positions[signal.symbol] <= 0:
            logger.warning(f"No hay posición para vender: {signal.symbol}")
            return
        
        # Aplicar slippage
        execution_price = price * (1 - self.config.slippage_rate)
        
        # Calcular cantidad a vender
        if signal.position_size > 0:
            quantity = min(signal.position_size / execution_price, self.positions[signal.symbol])
        else:
            quantity = self.positions[signal.symbol]  # Vender todo
        
        # Ejecutar
        total_proceeds = quantity * execution_price
        commission = total_proceeds * self.config.commission_rate
        
        self.cash += (total_proceeds - commission)
        self.positions[signal.symbol] -= quantity
        
        # Crear trade record
        trade = Trade(
            entry_time=timestamp,  # Simplificado
            exit_time=timestamp,
            entry_price=execution_price,
            exit_price=execution_price,
            quantity=quantity,
            side='long',
            pnl=0,  # Calcular después
            pnl_pct=0,
            commission=commission,
            slippage=abs(price - execution_price),
            duration=timedelta(0),
            max_adverse_excursion=0,
            max_favorable_excursion=0
        )
        
        self.trades.append(trade)
        
        logger.debug(f"Venta ejecutada: {quantity:.6f} {signal.symbol} a {execution_price:.4f}")
    
    def _update_portfolio_value(self, market_data: pd.Series):
        """Actualiza el valor del portfolio"""
        portfolio_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                current_price = market_data.get('close', 0)
                portfolio_value += quantity * current_price
        
        self.total_value = portfolio_value
        
        # Guardar en historial
        self.portfolio_history.append({
            'timestamp': market_data.name,
            'total_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash
        })
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calcula serie de retornos del portfolio"""
        if not self.portfolio_history:
            return pd.Series([0])
        
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('timestamp', inplace=True)
        
        # Calcular retornos
        returns = df['total_value'].pct_change().dropna()
        
        return returns
    
    def calculate_metrics(self, returns: pd.Series, 
                         benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """
        Calcula métricas comprehensivas de performance
        
        Args:
            returns: Serie de retornos de la estrategia
            benchmark_returns: Serie de retornos del benchmark
            
        Returns:
            Métricas completas de performance
        """
        try:
            if len(returns) < 2:
                logger.warning("Datos insuficientes para métricas confiables")
                return self._default_metrics()
            
            # Limpiar datos
            returns_clean = returns.dropna()
            
            # Métricas básicas
            total_return = (1 + returns_clean).prod() - 1
            annual_return = (1 + returns_clean.mean()) ** 252 - 1
            annual_volatility = returns_clean.std() * np.sqrt(252)
            
            # Ratios de riesgo-retorno
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Downside deviation
            downside_returns = returns_clean[returns_clean < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown
            cumulative_returns = (1 + returns_clean).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # Duración del drawdown
            drawdown_duration = self._calculate_drawdown_duration(drawdown)
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # VaR y ES
            var_95 = self.risk_manager.calculate_var(returns_clean, 0.95)
            var_99 = self.risk_manager.calculate_var(returns_clean, 0.99)
            expected_shortfall = self.risk_manager.calculate_expected_shortfall(returns_clean, 0.95)
            
            # Análisis de trades
            trade_stats = self._calculate_trade_statistics()
            
            # Momentos estadísticos
            skewness = stats.skew(returns_clean)
            kurtosis = stats.kurtosis(returns_clean)
            
            # Ratios avanzados
            omega_ratio = self._calculate_omega_ratio(returns_clean)
            tail_ratio = self._calculate_tail_ratio(returns_clean)
            
            # Métricas vs benchmark
            alpha, beta, information_ratio, tracking_error = 0, 1, 0, 0
            if benchmark_returns is not None:
                alpha, beta, information_ratio, tracking_error = self._calculate_benchmark_metrics(
                    returns_clean, benchmark_returns
                )
            
            return PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                annual_volatility=annual_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=drawdown_duration,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades'],
                win_rate=trade_stats['win_rate'],
                avg_win=trade_stats['avg_win'],
                avg_loss=trade_stats['avg_loss'],
                profit_factor=trade_stats['profit_factor'],
                omega_ratio=omega_ratio,
                tail_ratio=tail_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return self._default_metrics()
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calcula duración máxima del drawdown"""
        try:
            in_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            return max(drawdown_periods) if drawdown_periods else 0
            
        except Exception as e:
            logger.error(f"Error calculando duración drawdown: {e}")
            return 0
    
    def _calculate_trade_statistics(self) -> Dict[str, float]:
        """Calcula estadísticas de trading"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        winning_trades = [t for t in self.trades if t.is_winning]
        losing_trades = [t for t in self.trades if not t.is_winning]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calcula Omega ratio"""
        try:
            returns_above = returns[returns > threshold]
            returns_below = returns[returns <= threshold]
            
            gains = returns_above.sum()
            losses = abs(returns_below.sum())
            
            return gains / losses if losses > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculando Omega ratio: {e}")
            return 0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calcula Tail ratio"""
        try:
            p95 = returns.quantile(0.95)
            p5 = returns.quantile(0.05)
            
            return abs(p95 / p5) if p5 != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculando Tail ratio: {e}")
            return 0
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Tuple[float, float, float, float]:
        """Calcula métricas vs benchmark"""
        try:
            # Alinear series
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 10:
                return 0, 1, 0, 0
            
            returns_aligned = returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                benchmark_aligned, returns_aligned
            )
            
            beta = slope
            alpha = intercept * 252  # Anualizado
            
            # Information ratio
            active_returns = returns_aligned - benchmark_aligned
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
            
            # Tracking error
            tracking_error = active_returns.std() * np.sqrt(252)
            
            return alpha, beta, information_ratio, tracking_error
            
        except Exception as e:
            logger.error(f"Error calculando métricas benchmark: {e}")
            return 0, 1, 0, 0
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Análisis detallado de trades"""
        if not self.trades:
            return {'message': 'No hay trades para analizar'}
        
        # Análisis por duración
        durations = [t.duration.total_seconds() / 3600 for t in self.trades]  # En horas
        
        # Análisis por PnL
        pnls = [t.pnl for t in self.trades]
        
        # Análisis consecutivo
        consecutive_wins = self._analyze_consecutive_trades(True)
        consecutive_losses = self._analyze_consecutive_trades(False)
        
        return {
            'avg_duration_hours': np.mean(durations),
            'median_duration_hours': np.median(durations),
            'avg_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'pnl_distribution': {
                'q25': np.percentile(pnls, 25),
                'q50': np.percentile(pnls, 50),
                'q75': np.percentile(pnls, 75),
                'q90': np.percentile(pnls, 90),
                'q99': np.percentile(pnls, 99)
            }
        }
    
    def _analyze_consecutive_trades(self, winning: bool) -> int:
        """Analiza trades consecutivos"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.is_winning == winning:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _default_metrics(self) -> PerformanceMetrics:
        """Métricas por defecto cuando hay errores"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, annual_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            var_95=0, var_99=0, expected_shortfall=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            omega_ratio=0, tail_ratio=0, skewness=0, kurtosis=0,
            alpha=0, beta=1, information_ratio=0, tracking_error=0
        )
    
    def generate_report(self, results: BacktestResults) -> str:
        """Genera reporte completo del backtest"""
        try:
            report = f"""
{'='*80}
📊 REPORTE DE BACKTESTING AVANZADO
{'='*80}

🔍 INFORMACIÓN GENERAL:
   • Estrategia: {results.strategy_name}
   • Período: {results.start_date.strftime('%Y-%m-%d')} a {results.end_date.strftime('%Y-%m-%d')}
   • Capital Inicial: ${results.initial_capital:,.2f}
   • Capital Final: ${results.final_capital:,.2f}
   • Retorno Total: {results.total_return:.2%}

📈 MÉTRICAS DE PERFORMANCE:
   • Retorno Anual: {results.metrics.annual_return:.2%}
   • Volatilidad Anual: {results.metrics.annual_volatility:.2%}
   • Sharpe Ratio: {results.metrics.sharpe_ratio:.3f}
   • Sortino Ratio: {results.metrics.sortino_ratio:.3f}
   • Calmar Ratio: {results.metrics.calmar_ratio:.3f}

⚠️ MÉTRICAS DE RIESGO:
   • Máximo Drawdown: {results.metrics.max_drawdown:.2%}
   • Duración DD (días): {results.metrics.max_drawdown_duration}
   • VaR 95%: {results.metrics.var_95:.2%}
   • VaR 99%: {results.metrics.var_99:.2%}
   • Expected Shortfall: {results.metrics.expected_shortfall:.2%}

🎯 ESTADÍSTICAS DE TRADING:
   • Total Trades: {results.metrics.total_trades}
   • Tasa de Éxito: {results.metrics.win_rate:.1%}
   • Trades Ganadores: {results.metrics.winning_trades}
   • Trades Perdedores: {results.metrics.losing_trades}
   • Ganancia Promedio: {results.metrics.avg_win:.2%}
   • Pérdida Promedio: {results.metrics.avg_loss:.2%}
   • Profit Factor: {results.metrics.profit_factor:.3f}

📊 RATIOS AVANZADOS:
   • Omega Ratio: {results.metrics.omega_ratio:.3f}
   • Tail Ratio: {results.metrics.tail_ratio:.3f}
   • Skewness: {results.metrics.skewness:.3f}
   • Kurtosis: {results.metrics.kurtosis:.3f}

📉 VS BENCHMARK:
   • Alpha: {results.metrics.alpha:.2%}
   • Beta: {results.metrics.beta:.3f}
   • Information Ratio: {results.metrics.information_ratio:.3f}
   • Tracking Error: {results.metrics.tracking_error:.2%}

⚠️ DISCLAIMER:
Este backtest es una simulación histórica con fines educativos.
Los resultados pasados no garantizan resultados futuros.
Considera costos de transacción, slippage y liquidez en trading real.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            return f"Error generando reporte: {e}"
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """Genera gráficos de resultados"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Resultados Backtest - {results.strategy_name}', fontsize=16)
            
            # Gráfico 1: Equity curve
            portfolio_df = pd.DataFrame(results.portfolio_history)
            portfolio_df.set_index('timestamp', inplace=True)
            
            axes[0, 0].plot(portfolio_df.index, portfolio_df['total_value'], 
                          label='Portfolio Value', color='blue', linewidth=2)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gráfico 2: Drawdown
            returns = portfolio_df['total_value'].pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            
            axes[0, 1].fill_between(drawdown.index, drawdown, 0, 
                                  color='red', alpha=0.3, label='Drawdown')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gráfico 3: Distribución de retornos
            axes[1, 0].hist(returns, bins=50, alpha=0.7, color='green', density=True)
            axes[1, 0].set_title('Distribución de Retornos')
            axes[1, 0].set_xlabel('Retorno')
            axes[1, 0].set_ylabel('Densidad')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Gráfico 4: Rolling Sharpe
            rolling_sharpe = returns.rolling(window=60).mean() / returns.rolling(window=60).std() * np.sqrt(252)
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe, 
                          color='purple', linewidth=2, label='Rolling Sharpe (60d)')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Rolling Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráficos guardados en {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error generando gráficos: {e}")

# ============= OPTIMIZACIÓN DE PARÁMETROS =============

class ParameterOptimizer:
    """Optimizador de parámetros para estrategias"""
    
    def __init__(self, backtester: AdvancedBacktester):
        self.backtester = backtester
        
    def optimize_parameters(self, strategy_class: type, 
                          data: pd.DataFrame,
                          param_ranges: Dict[str, Tuple[float, float]],
                          optimization_metric: str = 'sharpe_ratio',
                          max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimiza parámetros de estrategia
        
        Args:
            strategy_class: Clase de estrategia
            data: Datos históricos
            param_ranges: Rangos de parámetros
            optimization_metric: Métrica a optimizar
            max_iterations: Máximo de iteraciones
            
        Returns:
            Parámetros óptimos y resultados
        """
        try:
            logger.info(f"Iniciando optimización de parámetros para {strategy_class.__name__}")
            
            # Definir función objetivo
            def objective(params):
                try:
                    # Crear estrategia con parámetros
                    param_dict = dict(zip(param_ranges.keys(), params))
                    strategy = strategy_class(**param_dict)
                    
                    # Ejecutar backtest
                    results = self.backtester.run_backtest(strategy, data, 100000)
                    
                    # Obtener métrica a optimizar
                    metric_value = getattr(results.metrics, optimization_metric)
                    
                    # Maximizar (minimizar negativo)
                    return -metric_value
                    
                except Exception as e:
                    logger.warning(f"Error en optimización: {e}")
                    return float('inf')
            
            # Configurar optimización
            bounds = list(param_ranges.values())
            initial_guess = [np.mean(bound) for bound in bounds]
            
            # Optimizar
            result = minimize(
                objective,
                initial_guess,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations}
            )
            
            # Resultados
            optimal_params = dict(zip(param_ranges.keys(), result.x))
            
            # Ejecutar backtest final con parámetros óptimos
            final_strategy = strategy_class(**optimal_params)
            final_results = self.backtester.run_backtest(final_strategy, data, 100000)
            
            return {
                'optimal_parameters': optimal_params,
                'optimization_success': result.success,
                'final_metric_value': -result.fun,
                'iterations': result.nit,
                'backtest_results': final_results
            }
            
        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            raise 