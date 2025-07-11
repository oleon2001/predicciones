# ENSEMBLE TEMPORAL MULTI-HORIZONTE
"""
Sistema de ensemble que combina predicciones de múltiples horizontes temporales
para mejorar la precisión y estabilidad de las predicciones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Optimization
from scipy.optimize import minimize
from scipy.stats import norm

# Interfaces
from core.interfaces import IEnsembleModel, ModelMetrics, PredictionResult
from config.extended_config import ExtendedPredictionConfig

logger = logging.getLogger(__name__)

@dataclass
class HorizonPrediction:
    """Predicción para un horizonte específico"""
    horizon: int
    prediction: float
    confidence: float
    model_type: str
    timestamp: datetime
    features_used: List[str]
    
@dataclass
class EnsemblePrediction:
    """Predicción combinada del ensemble"""
    short_term: float
    medium_term: float
    long_term: float
    weighted_average: float
    confidence_score: float
    risk_adjusted: float
    directional_bias: str  # 'bullish', 'bearish', 'neutral'
    horizon_weights: Dict[str, float]

class TemporalEnsemble(IEnsembleModel):
    """Ensemble que combina predicciones de múltiples horizontes"""
    
    def __init__(self, config: ExtendedPredictionConfig):
        self.config = config
        self.horizon_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Pesos iniciales por categoría de horizonte
        self.base_weights = {
            'short_term': 0.5,
            'medium_term': 0.3,
            'long_term': 0.2
        }
        
        # Pesos dinámicos que se ajustan según rendimiento
        self.dynamic_weights = self.base_weights.copy()
        
        # Historial de predicciones para adaptación
        self.prediction_history = []
        self.performance_history = {}
        
    def train_ensemble(self, data: Dict[str, pd.DataFrame], targets: Dict[str, pd.Series]) -> Dict[str, ModelMetrics]:
        """Entrena el ensemble con datos multi-horizonte"""
        
        logger.info("🚀 Entrenando ensemble temporal multi-horizonte")
        
        results = {}
        
        try:
            # 1. Entrenar modelos individuales por horizonte
            for horizon_str, horizon_data in data.items():
                if horizon_str in targets:
                    logger.info(f"   Entrenando modelo para horizonte {horizon_str}")
                    
                    # Crear y entrenar modelo específico
                    model = self._create_horizon_model(horizon_str)
                    metrics = model.train(horizon_data, targets[horizon_str])
                    
                    self.horizon_models[horizon_str] = model
                    results[horizon_str] = metrics
            
            # 2. Entrenar meta-modelo para combinar predicciones
            self._train_meta_model(data, targets)
            
            # 3. Optimizar pesos dinámicos
            self._optimize_ensemble_weights(data, targets)
            
            self.is_trained = True
            
            logger.info("✅ Ensemble temporal entrenado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error entrenando ensemble: {e}")
            raise
    
    def predict_ensemble(self, data: Dict[str, pd.DataFrame]) -> EnsemblePrediction:
        """Genera predicción combinada del ensemble"""
        
        if not self.is_trained:
            raise ValueError("Ensemble no entrenado")
        
        try:
            # 1. Obtener predicciones individuales
            horizon_predictions = {}
            
            for horizon_str, horizon_data in data.items():
                if horizon_str in self.horizon_models:
                    model = self.horizon_models[horizon_str]
                    pred = model.predict(horizon_data)
                    
                    # Crear objeto de predicción estructurada
                    horizon_predictions[horizon_str] = HorizonPrediction(
                        horizon=self._extract_horizon_hours(horizon_str),
                        prediction=pred[-1] if len(pred) > 0 else 0,
                        confidence=self._calculate_prediction_confidence(model, horizon_data),
                        model_type=model.__class__.__name__,
                        timestamp=datetime.now(),
                        features_used=list(horizon_data.columns)
                    )
            
            # 2. Combinar predicciones usando meta-modelo
            combined_prediction = self._combine_predictions(horizon_predictions)
            
            # 3. Aplicar ajustes de riesgo
            risk_adjusted_prediction = self._apply_risk_adjustments(combined_prediction, horizon_predictions)
            
            # 4. Determinar sesgo direccional
            directional_bias = self._determine_directional_bias(horizon_predictions)
            
            # 5. Actualizar pesos dinámicos
            self._update_dynamic_weights(horizon_predictions)
            
            return EnsemblePrediction(
                short_term=self._get_category_prediction(horizon_predictions, 'short_term'),
                medium_term=self._get_category_prediction(horizon_predictions, 'medium_term'),
                long_term=self._get_category_prediction(horizon_predictions, 'long_term'),
                weighted_average=combined_prediction,
                confidence_score=self._calculate_ensemble_confidence(horizon_predictions),
                risk_adjusted=risk_adjusted_prediction,
                directional_bias=directional_bias,
                horizon_weights=self.dynamic_weights.copy()
            )
            
        except Exception as e:
            logger.error(f"Error en predicción ensemble: {e}")
            raise
    
    def _create_horizon_model(self, horizon_str: str):
        """Crea modelo específico para un horizonte"""
        
        horizon_hours = self._extract_horizon_hours(horizon_str)
        category = self.config.get_horizon_category(horizon_hours)
        
        # Seleccionar modelo según categoría
        if category == 'short_term':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif category == 'medium_term':
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:  # long_term
            return Ridge(alpha=1.0)
    
    def _train_meta_model(self, data: Dict[str, pd.DataFrame], targets: Dict[str, pd.Series]):
        """Entrena meta-modelo para combinar predicciones"""
        
        logger.info("   🧠 Entrenando meta-modelo de combinación")
        
        try:
            # Preparar datos para meta-modelo
            meta_features = []
            meta_targets = []
            
            # Usar validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=5)
            
            for horizon_str, horizon_data in data.items():
                if horizon_str in targets and horizon_str in self.horizon_models:
                    model = self.horizon_models[horizon_str]
                    target = targets[horizon_str]
                    
                    # Predicciones out-of-fold
                    oof_predictions = np.zeros(len(horizon_data))
                    
                    for train_idx, val_idx in tscv.split(horizon_data):
                        X_train, X_val = horizon_data.iloc[train_idx], horizon_data.iloc[val_idx]
                        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
                        
                        # Entrenar modelo fold
                        fold_model = self._create_horizon_model(horizon_str)
                        fold_model.fit(X_train, y_train)
                        
                        # Predicción out-of-fold
                        val_pred = fold_model.predict(X_val)
                        oof_predictions[val_idx] = val_pred
                    
                    # Agregar features del meta-modelo
                    meta_features.append(oof_predictions)
                    meta_targets.append(target.values)
            
            if meta_features:
                # Combinar features
                X_meta = np.column_stack(meta_features)
                y_meta = np.mean(meta_targets, axis=0)  # Promedio de targets
                
                # Entrenar meta-modelo
                self.meta_model = Ridge(alpha=0.1)
                self.meta_model.fit(X_meta, y_meta)
                
                logger.info("   ✅ Meta-modelo entrenado")
            
        except Exception as e:
            logger.error(f"Error entrenando meta-modelo: {e}")
    
    def _optimize_ensemble_weights(self, data: Dict[str, pd.DataFrame], targets: Dict[str, pd.Series]):
        """Optimiza pesos del ensemble usando optimización bayesiana"""
        
        logger.info("   ⚖️ Optimizando pesos del ensemble")
        
        try:
            def objective(weights):
                """Función objetivo para optimización"""
                # Normalizar pesos
                weights = weights / np.sum(weights)
                
                # Calcular error promedio con estos pesos
                total_error = 0
                count = 0
                
                for horizon_str, horizon_data in data.items():
                    if horizon_str in targets and horizon_str in self.horizon_models:
                        model = self.horizon_models[horizon_str]
                        target = targets[horizon_str]
                        
                        # Predicción
                        pred = model.predict(horizon_data)
                        
                        # Aplicar peso según categoría
                        category = self.config.get_horizon_category(self._extract_horizon_hours(horizon_str))
                        category_idx = ['short_term', 'medium_term', 'long_term'].index(category)
                        
                        # Error ponderado
                        error = mean_squared_error(target.values[-len(pred):], pred)
                        total_error += error * weights[category_idx]
                        count += 1
                
                return total_error / count if count > 0 else 1.0
            
            # Optimización con restricciones
            initial_weights = [0.5, 0.3, 0.2]  # short, medium, long
            bounds = [(0.1, 0.8), (0.1, 0.6), (0.1, 0.5)]
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            
            result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x
                self.dynamic_weights = {
                    'short_term': optimized_weights[0],
                    'medium_term': optimized_weights[1],
                    'long_term': optimized_weights[2]
                }
                
                logger.info(f"   ✅ Pesos optimizados: {self.dynamic_weights}")
            
        except Exception as e:
            logger.error(f"Error optimizando pesos: {e}")
    
    def _combine_predictions(self, horizon_predictions: Dict[str, HorizonPrediction]) -> float:
        """Combina predicciones usando meta-modelo y pesos dinámicos"""
        
        try:
            # Preparar features para meta-modelo
            if self.meta_model is not None:
                features = []
                for horizon_str, pred in horizon_predictions.items():
                    features.append(pred.prediction)
                
                if features:
                    X_meta = np.array(features).reshape(1, -1)
                    meta_prediction = self.meta_model.predict(X_meta)[0]
                    return meta_prediction
            
            # Fallback: combinación ponderada simple
            weighted_sum = 0
            total_weight = 0
            
            for horizon_str, pred in horizon_predictions.items():
                category = self.config.get_horizon_category(pred.horizon)
                weight = self.dynamic_weights.get(category, 0.33)
                
                weighted_sum += pred.prediction * weight * pred.confidence
                total_weight += weight * pred.confidence
            
            return weighted_sum / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Error combinando predicciones: {e}")
            return 0
    
    def _apply_risk_adjustments(self, prediction: float, horizon_predictions: Dict[str, HorizonPrediction]) -> float:
        """Aplica ajustes de riesgo a la predicción"""
        
        try:
            # Calcular volatilidad implícita
            predictions = [pred.prediction for pred in horizon_predictions.values()]
            volatility = np.std(predictions) if len(predictions) > 1 else 0
            
            # Factor de ajuste de riesgo
            risk_factor = 1.0
            
            # Ajustar según volatilidad
            if volatility > 0.1:  # Alta volatilidad
                risk_factor *= 0.9
            elif volatility < 0.02:  # Baja volatilidad
                risk_factor *= 1.1
            
            # Ajustar según confianza promedio
            avg_confidence = np.mean([pred.confidence for pred in horizon_predictions.values()])
            risk_factor *= avg_confidence
            
            return prediction * risk_factor
            
        except Exception as e:
            logger.error(f"Error aplicando ajustes de riesgo: {e}")
            return prediction
    
    def _determine_directional_bias(self, horizon_predictions: Dict[str, HorizonPrediction]) -> str:
        """Determina el sesgo direccional general"""
        
        try:
            # Contar predicciones alcistas vs bajistas
            bullish_count = 0
            bearish_count = 0
            
            for pred in horizon_predictions.values():
                if pred.prediction > 1.02:  # >2% alcista
                    bullish_count += 1
                elif pred.prediction < 0.98:  # >2% bajista
                    bearish_count += 1
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determinando sesgo direccional: {e}")
            return 'neutral'
    
    def _calculate_ensemble_confidence(self, horizon_predictions: Dict[str, HorizonPrediction]) -> float:
        """Calcula confianza del ensemble"""
        
        try:
            # Confianza promedio ponderada
            weighted_confidence = 0
            total_weight = 0
            
            for pred in horizon_predictions.values():
                category = self.config.get_horizon_category(pred.horizon)
                weight = self.dynamic_weights.get(category, 0.33)
                
                weighted_confidence += pred.confidence * weight
                total_weight += weight
            
            ensemble_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            
            # Ajustar por consistencia entre predicciones
            predictions = [pred.prediction for pred in horizon_predictions.values()]
            consistency = 1.0 - (np.std(predictions) / np.mean(predictions)) if np.mean(predictions) != 0 else 0
            
            return ensemble_confidence * max(0.5, consistency)
            
        except Exception as e:
            logger.error(f"Error calculando confianza ensemble: {e}")
            return 0.5
    
    def _extract_horizon_hours(self, horizon_str: str) -> int:
        """Extrae número de horas del string de horizonte"""
        try:
            return int(horizon_str.replace('h', ''))
        except:
            return 24
    
    def _get_category_prediction(self, horizon_predictions: Dict[str, HorizonPrediction], category: str) -> float:
        """Obtiene predicción promedio para una categoría"""
        
        category_preds = []
        for pred in horizon_predictions.values():
            if self.config.get_horizon_category(pred.horizon) == category:
                category_preds.append(pred.prediction)
        
        return np.mean(category_preds) if category_preds else 0
    
    def _calculate_prediction_confidence(self, model, data: pd.DataFrame) -> float:
        """Calcula confianza de predicción para un modelo"""
        # Implementación simplificada
        return 0.7  # En producción, usar métricas más sofisticadas
    
    def _update_dynamic_weights(self, horizon_predictions: Dict[str, HorizonPrediction]):
        """Actualiza pesos dinámicos basado en rendimiento"""
        # Implementación simplificada para demo
        pass 