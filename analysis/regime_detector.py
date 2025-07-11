# DETECTOR DE REGÍMENES DE MERCADO AVANZADO
"""
Sistema avanzado de detección de regímenes de mercado para criptomonedas
Identifica automáticamente bull/bear markets, consolidación, y cambios de régimen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ML y estadísticas
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import ta

# Regímenes de mercado
class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class RegimeSignal:
    """Señal de cambio de régimen"""
    regime: MarketRegime
    confidence: float
    timestamp: datetime
    duration_hours: int
    supporting_indicators: List[str]
    risk_level: str  # 'low', 'medium', 'high'

class AdvancedRegimeDetector:
    """Detector avanzado de regímenes de mercado"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.regime_model = GaussianMixture(n_components=5, random_state=42)
        self.current_regime = MarketRegime.CONSOLIDATION
        self.regime_history = []
        self.confidence_threshold = 0.7
        
        # Parámetros de detección
        self.lookback_window = 168  # 7 días en horas
        self.volatility_window = 24
        self.trend_window = 48
        
        # Umbrales para clasificación
        self.thresholds = {
            'bull_threshold': 0.02,      # 2% cambio positivo
            'bear_threshold': -0.02,     # 2% cambio negativo
            'high_vol_threshold': 0.05,  # 5% volatilidad
            'low_vol_threshold': 0.01,   # 1% volatilidad
            'consolidation_range': 0.03  # 3% rango lateral
        }
        
        logger.info("✅ Detector de regímenes inicializado")
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeSignal:
        """Detecta el régimen actual del mercado"""
        
        try:
            # 1. Preparar features para detección
            features = self._prepare_regime_features(data)
            
            # 2. Aplicar modelo de detección
            regime_probs = self._calculate_regime_probabilities(features)
            
            # 3. Determinar régimen dominante
            dominant_regime = self._determine_dominant_regime(regime_probs, data)
            
            # 4. Calcular confianza
            confidence = self._calculate_regime_confidence(regime_probs, features)
            
            # 5. Validar con indicadores adicionales
            supporting_indicators = self._validate_regime_with_indicators(dominant_regime, data)
            
            # 6. Determinar nivel de riesgo
            risk_level = self._assess_regime_risk(dominant_regime, features)
            
            # 7. Calcular duración estimada
            duration = self._estimate_regime_duration(dominant_regime, data)
            
            # Crear señal de régimen
            regime_signal = RegimeSignal(
                regime=dominant_regime,
                confidence=confidence,
                timestamp=datetime.now(),
                duration_hours=duration,
                supporting_indicators=supporting_indicators,
                risk_level=risk_level
            )
            
            # Actualizar historial
            self._update_regime_history(regime_signal)
            
            logger.info(f"🔍 Régimen detectado: {dominant_regime.value} (confianza: {confidence:.2f})")
            
            return regime_signal
            
        except Exception as e:
            logger.error(f"Error detectando régimen: {e}")
            # Retornar régimen neutral por defecto
            return RegimeSignal(
                regime=MarketRegime.CONSOLIDATION,
                confidence=0.5,
                timestamp=datetime.now(),
                duration_hours=24,
                supporting_indicators=[],
                risk_level='medium'
            )
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para detección de régimen"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Features de precio
            features['price_change_1h'] = data['close'].pct_change(1)
            features['price_change_4h'] = data['close'].pct_change(4)
            features['price_change_24h'] = data['close'].pct_change(24)
            features['price_change_7d'] = data['close'].pct_change(168)
            
            # 2. Features de volatilidad
            features['volatility_1h'] = data['close'].pct_change().rolling(self.volatility_window).std()
            features['volatility_4h'] = data['close'].pct_change(4).rolling(self.volatility_window).std()
            features['volatility_24h'] = data['close'].pct_change(24).rolling(self.volatility_window).std()
            
            # 3. Features de tendencia
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            ema_50 = data['close'].ewm(span=50).mean()
            
            features['trend_12_26'] = (ema_12 - ema_26) / ema_26
            features['trend_26_50'] = (ema_26 - ema_50) / ema_50
            features['trend_slope'] = (data['close'] - data['close'].shift(24)) / data['close'].shift(24)
            
            # 4. Features de momentum
            rsi = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
            features['rsi'] = rsi
            features['rsi_divergence'] = rsi - rsi.rolling(24).mean()
            
            macd = ta.trend.MACD(data['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_histogram'] = macd.macd_diff()
            
            # 5. Features de volumen
            features['volume_change'] = data['volume'].pct_change()
            features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(24).mean()
            
            # Volume Price Trend
            vpt = ta.volume.VolumePriceTrendIndicator(data['close'], data['volume'])
            features['vpt'] = vpt.volume_price_trend()
            features['vpt_change'] = features['vpt'].pct_change()
            
            # 6. Features de rango
            features['high_low_range'] = (data['high'] - data['low']) / data['close']
            features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
            # 7. Features de Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            features['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            features['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            # 8. Features de anomalías
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            price_anomaly = isolation_forest.fit_predict(data['close'].values.reshape(-1, 1))
            features['price_anomaly'] = price_anomaly
            
            # Limpiar datos
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparando features de régimen: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_regime_probabilities(self, features: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Calcula probabilidades de cada régimen"""
        
        try:
            # Usar últimos datos para detección
            recent_features = features.tail(self.lookback_window)
            
            if len(recent_features) < 24:
                # Datos insuficientes, retornar distribución uniforme
                return {regime: 0.2 for regime in MarketRegime}
            
            # Normalizar features
            feature_values = recent_features.select_dtypes(include=[np.number]).values
            if feature_values.shape[1] == 0:
                return {regime: 0.2 for regime in MarketRegime}
            
            scaled_features = self.scaler.fit_transform(feature_values)
            
            # Aplicar PCA para reducir dimensionalidad
            pca = PCA(n_components=min(5, scaled_features.shape[1]))
            reduced_features = pca.fit_transform(scaled_features)
            
            # Entrenar modelo de mezcla gaussiana
            self.regime_model.fit(reduced_features)
            
            # Predecir régimen para último punto
            last_point = reduced_features[-1:].reshape(1, -1)
            regime_probs = self.regime_model.predict_proba(last_point)[0]
            
            # Mapear probabilidades a regímenes
            regime_mapping = list(MarketRegime)[:len(regime_probs)]
            return {regime: prob for regime, prob in zip(regime_mapping, regime_probs)}
            
        except Exception as e:
            logger.error(f"Error calculando probabilidades de régimen: {e}")
            return {regime: 0.2 for regime in MarketRegime}
    
    def _determine_dominant_regime(self, regime_probs: Dict[MarketRegime, float], data: pd.DataFrame) -> MarketRegime:
        """Determina el régimen dominante usando probabilidades y reglas"""
        
        try:
            # Obtener últimos datos
            recent_data = data.tail(self.trend_window)
            
            # Calcular métricas clave
            price_change_24h = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-24]) / recent_data['close'].iloc[-24]
            price_change_7d = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            volatility = recent_data['close'].pct_change().std()
            
            # Reglas heurísticas
            if price_change_7d > self.thresholds['bull_threshold'] and price_change_24h > 0:
                if volatility < self.thresholds['high_vol_threshold']:
                    return MarketRegime.BULL_MARKET
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            elif price_change_7d < self.thresholds['bear_threshold'] and price_change_24h < 0:
                if volatility < self.thresholds['high_vol_threshold']:
                    return MarketRegime.BEAR_MARKET
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            elif abs(price_change_7d) < self.thresholds['consolidation_range']:
                if volatility < self.thresholds['low_vol_threshold']:
                    return MarketRegime.LOW_VOLATILITY
                else:
                    return MarketRegime.CONSOLIDATION
            
            # Si no se cumple ninguna regla, usar probabilidades
            return max(regime_probs.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error determinando régimen dominante: {e}")
            return MarketRegime.CONSOLIDATION
    
    def _calculate_regime_confidence(self, regime_probs: Dict[MarketRegime, float], features: pd.DataFrame) -> float:
        """Calcula confianza en la detección de régimen"""
        
        try:
            # Confianza basada en probabilidad máxima
            max_prob = max(regime_probs.values())
            
            # Penalizar si hay probabilidades similares (ambigüedad)
            sorted_probs = sorted(regime_probs.values(), reverse=True)
            prob_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.5
            
            # Confianza basada en consistencia de features
            feature_consistency = self._calculate_feature_consistency(features)
            
            # Combinar métricas
            confidence = (max_prob + prob_gap + feature_consistency) / 3
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def _calculate_feature_consistency(self, features: pd.DataFrame) -> float:
        """Calcula consistencia entre features"""
        
        try:
            # Usar últimos datos
            recent_features = features.tail(24)
            
            # Calcular correlaciones entre features
            correlations = recent_features.corr()
            avg_correlation = correlations.values[np.triu_indices_from(correlations.values, k=1)].mean()
            
            return max(0, min(1, avg_correlation + 0.5))
            
        except Exception as e:
            logger.error(f"Error calculando consistencia: {e}")
            return 0.5
    
    def _validate_regime_with_indicators(self, regime: MarketRegime, data: pd.DataFrame) -> List[str]:
        """Valida régimen con indicadores técnicos adicionales"""
        
        supporting_indicators = []
        
        try:
            recent_data = data.tail(48)
            
            # RSI
            rsi = ta.momentum.RSIIndicator(recent_data['close']).rsi().iloc[-1]
            if regime == MarketRegime.BULL_MARKET and rsi > 50:
                supporting_indicators.append('RSI_bullish')
            elif regime == MarketRegime.BEAR_MARKET and rsi < 50:
                supporting_indicators.append('RSI_bearish')
            
            # MACD
            macd = ta.trend.MACD(recent_data['close'])
            macd_line = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            
            if regime == MarketRegime.BULL_MARKET and macd_line > macd_signal:
                supporting_indicators.append('MACD_bullish')
            elif regime == MarketRegime.BEAR_MARKET and macd_line < macd_signal:
                supporting_indicators.append('MACD_bearish')
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(recent_data['close'])
            bb_pos = (recent_data['close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
            
            if regime == MarketRegime.HIGH_VOLATILITY and bb_pos > 0.8:
                supporting_indicators.append('BB_high_volatility')
            elif regime == MarketRegime.LOW_VOLATILITY and 0.2 < bb_pos < 0.8:
                supporting_indicators.append('BB_low_volatility')
            
            # Volume
            volume_ratio = recent_data['volume'].iloc[-1] / recent_data['volume'].mean()
            if volume_ratio > 1.5:
                supporting_indicators.append('Volume_high')
            elif volume_ratio < 0.5:
                supporting_indicators.append('Volume_low')
            
            return supporting_indicators
            
        except Exception as e:
            logger.error(f"Error validando con indicadores: {e}")
            return []
    
    def _assess_regime_risk(self, regime: MarketRegime, features: pd.DataFrame) -> str:
        """Evalúa el nivel de riesgo del régimen actual"""
        
        try:
            # Mapeo básico de riesgo por régimen
            risk_mapping = {
                MarketRegime.BULL_MARKET: 'medium',
                MarketRegime.BEAR_MARKET: 'high',
                MarketRegime.CONSOLIDATION: 'low',
                MarketRegime.HIGH_VOLATILITY: 'high',
                MarketRegime.LOW_VOLATILITY: 'low',
                MarketRegime.BREAKOUT: 'high',
                MarketRegime.REVERSAL: 'high',
                MarketRegime.ACCUMULATION: 'medium',
                MarketRegime.DISTRIBUTION: 'medium'
            }
            
            base_risk = risk_mapping.get(regime, 'medium')
            
            # Ajustar según volatilidad actual
            if len(features) > 0:
                recent_volatility = features['volatility_24h'].tail(24).mean()
                if recent_volatility > 0.05:  # Alta volatilidad
                    if base_risk == 'low':
                        return 'medium'
                    elif base_risk == 'medium':
                        return 'high'
                elif recent_volatility < 0.01:  # Baja volatilidad
                    if base_risk == 'high':
                        return 'medium'
                    elif base_risk == 'medium':
                        return 'low'
            
            return base_risk
            
        except Exception as e:
            logger.error(f"Error evaluando riesgo: {e}")
            return 'medium'
    
    def _estimate_regime_duration(self, regime: MarketRegime, data: pd.DataFrame) -> int:
        """Estima duración del régimen en horas"""
        
        try:
            # Duraciones históricas promedio (en horas)
            typical_durations = {
                MarketRegime.BULL_MARKET: 336,      # 14 días
                MarketRegime.BEAR_MARKET: 504,      # 21 días
                MarketRegime.CONSOLIDATION: 168,    # 7 días
                MarketRegime.HIGH_VOLATILITY: 72,   # 3 días
                MarketRegime.LOW_VOLATILITY: 120,   # 5 días
                MarketRegime.BREAKOUT: 24,          # 1 día
                MarketRegime.REVERSAL: 48,          # 2 días
                MarketRegime.ACCUMULATION: 240,     # 10 días
                MarketRegime.DISTRIBUTION: 192      # 8 días
            }
            
            base_duration = typical_durations.get(regime, 168)
            
            # Ajustar según volatilidad (más volatilidad = menor duración)
            if len(data) > 24:
                recent_volatility = data['close'].pct_change().tail(24).std()
                if recent_volatility > 0.05:
                    base_duration = int(base_duration * 0.7)
                elif recent_volatility < 0.01:
                    base_duration = int(base_duration * 1.3)
            
            return base_duration
            
        except Exception as e:
            logger.error(f"Error estimando duración: {e}")
            return 168
    
    def _update_regime_history(self, regime_signal: RegimeSignal):
        """Actualiza historial de regímenes"""
        
        try:
            # Mantener solo últimos 30 regímenes
            if len(self.regime_history) >= 30:
                self.regime_history.pop(0)
            
            self.regime_history.append(regime_signal)
            self.current_regime = regime_signal.regime
            
        except Exception as e:
            logger.error(f"Error actualizando historial: {e}")
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Retorna análisis completo del régimen actual"""
        
        return {
            'current_regime': self.current_regime.value,
            'regime_history': [r.regime.value for r in self.regime_history[-10:]],
            'avg_confidence': np.mean([r.confidence for r in self.regime_history[-5:]]) if self.regime_history else 0.5,
            'regime_transitions': len(set(r.regime for r in self.regime_history[-10:])),
            'risk_levels': [r.risk_level for r in self.regime_history[-5:]]
        } 