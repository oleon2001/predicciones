# SISTEMA AVANZADO DE GESTIÓN DE RIESGO
"""
Sistema completo de gestión de riesgo para trading cuantitativo
Implementa VaR, Expected Shortfall, Position Sizing, Correlaciones y más
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from core.interfaces import IRiskManager, RiskMetrics
from config.system_config import SystemConfig

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class PortfolioRisk:
    """Métricas de riesgo del portfolio"""
    total_risk: float
    individual_risks: Dict[str, float]
    correlation_matrix: pd.DataFrame
    diversification_ratio: float
    concentration_risk: float
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float

class AdvancedRiskManager(IRiskManager):
    """Gestor avanzado de riesgo"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.risk_config = config.risk
        self.regime_cache = {}
        self.correlation_cache = {}
        self.volatility_cache = {}
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Value at Risk usando múltiples métodos
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza (0.95 = 95%)
            
        Returns:
            VaR como valor positivo (pérdida esperada)
        """
        try:
            if len(returns) < 30:
                logger.warning("Datos insuficientes para VaR confiable")
                return 0.0
                
            # Limpiar datos
            returns_clean = returns.dropna()
            
            # Método 1: VaR Histórico
            var_historical = -np.percentile(returns_clean, (1 - confidence_level) * 100)
            
            # Método 2: VaR Paramétrico (distribución normal)
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            var_parametric = -(mean_return + stats.norm.ppf(1 - confidence_level) * std_return)
            
            # Método 3: VaR con distribución t-Student
            try:
                df_fitted, loc_fitted, scale_fitted = stats.t.fit(returns_clean)
                var_t_student = -(loc_fitted + stats.t.ppf(1 - confidence_level, df_fitted) * scale_fitted)
            except:
                var_t_student = var_parametric
                
            # Método 4: VaR con EWMA (Exponentially Weighted Moving Average)
            lambda_ewma = 0.94
            ewma_var = self._calculate_ewma_variance(returns_clean, lambda_ewma)
            var_ewma = -(mean_return + stats.norm.ppf(1 - confidence_level) * np.sqrt(ewma_var))
            
            # Combinar métodos (ponderación)
            var_combined = (
                0.4 * var_historical +
                0.2 * var_parametric +
                0.2 * var_t_student +
                0.2 * var_ewma
            )
            
            return max(0, var_combined)
            
        except Exception as e:
            logger.error(f"Error calculando VaR: {e}")
            return 0.05  # VaR conservador del 5%
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Expected Shortfall (CVaR)
        
        Args:
            returns: Serie de retornos
            confidence_level: Nivel de confianza
            
        Returns:
            Expected Shortfall como valor positivo
        """
        try:
            if len(returns) < 30:
                return 0.0
                
            returns_clean = returns.dropna()
            var_threshold = -np.percentile(returns_clean, (1 - confidence_level) * 100)
            
            # Calcular promedio de pérdidas que exceden VaR
            tail_losses = returns_clean[returns_clean <= -var_threshold]
            
            if len(tail_losses) == 0:
                return var_threshold
                
            expected_shortfall = -tail_losses.mean()
            return max(0, expected_shortfall)
            
        except Exception as e:
            logger.error(f"Error calculando Expected Shortfall: {e}")
            return 0.07  # ES conservador del 7%
    
    def calculate_portfolio_risk(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Calcula riesgo del portfolio usando matriz de covarianza
        
        Args:
            weights: Pesos del portfolio
            cov_matrix: Matriz de covarianza
            
        Returns:
            Volatilidad del portfolio (anualizada)
        """
        try:
            if len(weights) != cov_matrix.shape[0]:
                raise ValueError("Dimensiones incompatibles")
                
            # Normalizar pesos
            weights = weights / np.sum(weights)
            
            # Calcular volatilidad del portfolio
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Anualizar (asumiendo datos diarios)
            portfolio_risk_annual = portfolio_risk * np.sqrt(252)
            
            return portfolio_risk_annual
            
        except Exception as e:
            logger.error(f"Error calculando riesgo del portfolio: {e}")
            return 0.2  # Riesgo conservador del 20%
    
    def calculate_position_size(self, signal_strength: float, 
                              account_balance: float, 
                              risk_per_trade: float,
                              volatility: float = None,
                              kelly_fraction: float = None) -> float:
        """
        Calcula tamaño de posición usando múltiples métodos
        
        Args:
            signal_strength: Fuerza de la señal (0-1)
            account_balance: Balance de la cuenta
            risk_per_trade: Riesgo por operación (0-1)
            volatility: Volatilidad del activo
            kelly_fraction: Fracción de Kelly
            
        Returns:
            Tamaño de posición en unidades monetarias
        """
        try:
            # Método 1: Risk-based Position Sizing
            risk_amount = account_balance * risk_per_trade
            
            # Método 2: Volatility-adjusted Position Sizing
            if volatility is not None:
                volatility_adjustment = min(1.0, 0.2 / volatility)  # Target 20% volatility
                risk_amount *= volatility_adjustment
            
            # Método 3: Kelly Criterion
            if kelly_fraction is not None and self.risk_config.kelly_criterion_enabled:
                kelly_size = account_balance * kelly_fraction
                risk_amount = min(risk_amount, kelly_size)
            
            # Método 4: Signal Strength Adjustment
            signal_adjusted_size = risk_amount * signal_strength
            
            # Aplicar límites de seguridad
            max_position = account_balance * self.config.trading.max_position_risk
            position_size = min(signal_adjusted_size, max_position)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculando tamaño de posición: {e}")
            return account_balance * 0.01  # 1% conservador
    
    def calculate_correlations(self, returns: pd.DataFrame, 
                             method: str = "pearson",
                             min_periods: int = 30) -> pd.DataFrame:
        """
        Calcula matriz de correlaciones con diferentes métodos
        
        Args:
            returns: DataFrame con retornos de múltiples activos
            method: Método de correlación (pearson, spearman, kendall)
            min_periods: Períodos mínimos requeridos
            
        Returns:
            Matriz de correlaciones
        """
        try:
            # Limpiar datos
            returns_clean = returns.dropna()
            
            if len(returns_clean) < min_periods:
                logger.warning(f"Datos insuficientes para correlaciones confiables: {len(returns_clean)}")
                return pd.DataFrame(np.eye(len(returns.columns)), 
                                  index=returns.columns, 
                                  columns=returns.columns)
            
            # Calcular correlaciones
            if method == "pearson":
                corr_matrix = returns_clean.corr(method='pearson')
            elif method == "spearman":
                corr_matrix = returns_clean.corr(method='spearman')
            elif method == "kendall":
                corr_matrix = returns_clean.corr(method='kendall')
            else:
                # Correlación robusta usando método DCC (Dynamic Conditional Correlation)
                corr_matrix = self._calculate_dcc_correlation(returns_clean)
            
            # Validar matriz
            if corr_matrix.isnull().any().any():
                logger.warning("Matriz de correlaciones contiene NaN")
                corr_matrix = corr_matrix.fillna(0)
                np.fill_diagonal(corr_matrix.values, 1.0)
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error calculando correlaciones: {e}")
            return pd.DataFrame(np.eye(len(returns.columns)), 
                              index=returns.columns, 
                              columns=returns.columns)
    
    def detect_regime_change(self, data: pd.DataFrame, 
                           lookback_window: int = 60) -> Tuple[str, float]:
        """
        Detecta cambio de régimen de mercado usando múltiples indicadores
        
        Args:
            data: DataFrame con datos OHLCV
            lookback_window: Ventana de análisis
            
        Returns:
            Tupla con (régimen, confianza)
        """
        try:
            if len(data) < lookback_window:
                return MarketRegime.SIDEWAYS.value, 0.5
            
            recent_data = data.tail(lookback_window)
            
            # Indicador 1: Tendencia de precios
            prices = recent_data['close']
            price_trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
            # Indicador 2: Volatilidad
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Anualizada
            
            # Indicador 3: Momentum
            momentum = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
            
            # Indicador 4: Volumen
            volume_ma = recent_data['volume'].rolling(20).mean()
            volume_trend = recent_data['volume'].iloc[-5:].mean() / volume_ma.iloc[-5:].mean()
            
            # Indicador 5: Drawdown
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            max_drawdown = drawdown.min()
            
            # Clasificación de régimen
            regime_scores = {
                MarketRegime.BULL.value: 0,
                MarketRegime.BEAR.value: 0,
                MarketRegime.SIDEWAYS.value: 0,
                MarketRegime.VOLATILE.value: 0
            }
            
            # Scoring basado en indicadores
            if price_trend > 0.1 and momentum > 0.05:
                regime_scores[MarketRegime.BULL.value] += 2
            elif price_trend < -0.1 and momentum < -0.05:
                regime_scores[MarketRegime.BEAR.value] += 2
            else:
                regime_scores[MarketRegime.SIDEWAYS.value] += 1
                
            if volatility > 0.4:
                regime_scores[MarketRegime.VOLATILE.value] += 2
            elif volatility < 0.2:
                regime_scores[MarketRegime.SIDEWAYS.value] += 1
                
            if volume_trend > 1.5:
                regime_scores[MarketRegime.BULL.value] += 1
            elif volume_trend < 0.8:
                regime_scores[MarketRegime.BEAR.value] += 1
                
            if max_drawdown < -0.2:
                regime_scores[MarketRegime.BEAR.value] += 2
                
            # Determinar régimen dominante
            dominant_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[dominant_regime] / sum(regime_scores.values())
            
            return dominant_regime, confidence
            
        except Exception as e:
            logger.error(f"Error detectando cambio de régimen: {e}")
            return MarketRegime.SIDEWAYS.value, 0.5
    
    def calculate_diversification_ratio(self, weights: np.ndarray, 
                                      volatilities: np.ndarray, 
                                      cov_matrix: np.ndarray) -> float:
        """Calcula ratio de diversificación del portfolio"""
        try:
            # Volatilidad ponderada de componentes individuales
            weighted_avg_vol = np.sum(weights * volatilities)
            
            # Volatilidad del portfolio
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Ratio de diversificación
            diversification_ratio = weighted_avg_vol / portfolio_vol
            
            return diversification_ratio
            
        except Exception as e:
            logger.error(f"Error calculando ratio de diversificación: {e}")
            return 1.0
    
    def calculate_kelly_fraction(self, win_rate: float, 
                               avg_win: float, 
                               avg_loss: float) -> float:
        """
        Calcula fracción de Kelly para position sizing
        
        Args:
            win_rate: Tasa de éxito (0-1)
            avg_win: Ganancia promedio
            avg_loss: Pérdida promedio
            
        Returns:
            Fracción de Kelly
        """
        try:
            if avg_loss <= 0:
                return 0.0
                
            # Fórmula de Kelly
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
            
            # Aplicar límites de seguridad
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Máximo 25%
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculando fracción de Kelly: {e}")
            return 0.02  # 2% conservador
    
    def generate_risk_report(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Genera reporte completo de riesgo
        
        Args:
            portfolio_data: Diccionario con datos de cada activo
            
        Returns:
            Reporte completo de riesgo
        """
        try:
            report = {
                'timestamp': datetime.now(),
                'individual_risks': {},
                'portfolio_risk': {},
                'correlations': {},
                'regime_analysis': {},
                'recommendations': []
            }
            
            # Calcular métricas individuales
            for symbol, data in portfolio_data.items():
                returns = data['close'].pct_change().dropna()
                
                report['individual_risks'][symbol] = {
                    'var_95': self.calculate_var(returns, 0.95),
                    'var_99': self.calculate_var(returns, 0.99),
                    'expected_shortfall': self.calculate_expected_shortfall(returns, 0.95),
                    'volatility': returns.std() * np.sqrt(252),
                    'max_drawdown': self._calculate_max_drawdown(data['close']),
                    'regime': self.detect_regime_change(data)[0]
                }
            
            # Calcular correlaciones
            all_returns = pd.DataFrame()
            for symbol, data in portfolio_data.items():
                all_returns[symbol] = data['close'].pct_change()
            
            report['correlations'] = self.calculate_correlations(all_returns).to_dict()
            
            # Generar recomendaciones
            report['recommendations'] = self._generate_risk_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte de riesgo: {e}")
            return {'error': str(e)}
    
    def _calculate_ewma_variance(self, returns: pd.Series, lambda_ewma: float) -> float:
        """Calcula varianza usando EWMA"""
        returns_squared = returns ** 2
        ewma_var = returns_squared.iloc[0]
        
        for i in range(1, len(returns_squared)):
            ewma_var = lambda_ewma * ewma_var + (1 - lambda_ewma) * returns_squared.iloc[i]
        
        return ewma_var
    
    def _calculate_dcc_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calcula correlación DCC (simplified)"""
        # Implementación simplificada - en producción usar arch library
        return returns.corr(method='pearson')
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calcula máximo drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _generate_risk_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis de riesgo"""
        recommendations = []
        
        # Analizar VaR alto
        high_var_assets = []
        for symbol, metrics in report['individual_risks'].items():
            if metrics['var_95'] > 0.05:  # VaR > 5%
                high_var_assets.append(symbol)
        
        if high_var_assets:
            recommendations.append(f"Reducir exposición en activos de alto riesgo: {', '.join(high_var_assets)}")
        
        # Analizar correlaciones altas
        correlation_matrix = pd.DataFrame(report['correlations'])
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        if high_corr_pairs:
            recommendations.append("Considerar diversificar - correlaciones altas detectadas")
        
        # Analizar regímenes
        bear_regimes = [symbol for symbol, metrics in report['individual_risks'].items() 
                       if metrics['regime'] == 'bear']
        
        if len(bear_regimes) > len(report['individual_risks']) * 0.5:
            recommendations.append("Mercado en régimen bajista - considerar estrategias defensivas")
        
        return recommendations 