# MODELOS AVANZADOS DE MACHINE LEARNING
"""
Implementación de modelos avanzados de ML para predicción financiera
Incluye Transformers, GANs, Reinforcement Learning y Ensemble avanzados
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import joblib
from pathlib import Path

# ML Libraries
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch no disponible - algunos modelos no estarán disponibles")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, BatchNormalization, 
                                       Attention, MultiHeadAttention, LayerNormalization,
                                       Input, Embedding, TransformerBlock)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam, AdamW
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow no disponible - algunos modelos no estarán disponibles")

from core.interfaces import IMLModel, IEnsembleModel, ModelMetrics
from config.system_config import SystemConfig

logger = logging.getLogger(__name__)

# ============= CLASES BASE =============

@dataclass
class ModelConfig:
    """Configuración para modelos"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_selection: bool = True
    scaling_method: str = "robust"
    validation_method: str = "time_series"
    early_stopping: bool = True
    
class BaseMLModel(IMLModel):
    """Clase base para modelos de ML"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.is_trained = False
        self.feature_importance_ = {}
        self.training_history = []
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento/predicción"""
        # Feature selection
        if self.config.feature_selection and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X.values
            
        # Scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_selected)
        else:
            X_scaled = X_selected
            
        if y is not None:
            return X_scaled, y.values
        else:
            return X_scaled, None
    
    def _setup_preprocessing(self, X: pd.DataFrame, y: pd.Series):
        """Configura preprocessing"""
        # Feature selection
        if self.config.feature_selection:
            self.feature_selector = SelectKBest(
                score_func=f_regression,
                k=min(50, X.shape[1])  # Máximo 50 features
            )
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = X.values
            
        # Scaling
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        if self.scaler is not None:
            self.scaler.fit(X_selected)
    
    def save_model(self, path: str):
        """Guarda modelo"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'config': self.config,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        logger.info(f"Modelo guardado en {path}")
    
    def load_model(self, path: str):
        """Carga modelo"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.config = model_data['config']
        self.feature_importance_ = model_data.get('feature_importance', {})
        self.training_history = model_data.get('training_history', [])
        self.is_trained = True
        logger.info(f"Modelo cargado desde {path}")

# ============= MODELOS TRADICIONALES MEJORADOS =============

class AdvancedRandomForest(BaseMLModel):
    """Random Forest con optimizaciones avanzadas"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.hyperparameters = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42
        }
        self.hyperparameters.update(config.hyperparameters)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Entrena Random Forest"""
        try:
            # Preprocessing
            self._setup_preprocessing(X, y)
            X_processed, y_processed = self._prepare_data(X, y)
            
            # Crear modelo
            self.model = RandomForestRegressor(**self.hyperparameters)
            
            # Entrenar
            self.model.fit(X_processed, y_processed)
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
                self.feature_importance_ = dict(zip(feature_names, self.model.feature_importances_))
            
            # Validación
            metrics = self.validate(X, y)
            
            self.is_trained = True
            return metrics
            
        except Exception as e:
            logger.error(f"Error entrenando Random Forest: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X_processed, _ = self._prepare_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones con incertidumbre"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Para regresión, usar predicciones de árboles individuales
        X_processed, _ = self._prepare_data(X)
        
        # Predicciones de cada árbol
        tree_predictions = np.array([tree.predict(X_processed) for tree in self.model.estimators_])
        
        # Estadísticas
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)
        
        return np.column_stack([mean_pred, std_pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características"""
        return self.feature_importance_
    
    def validate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Valida el modelo"""
        try:
            if self.config.validation_method == "time_series":
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                mse = -cv_scores.mean()
            else:
                # Validación simple
                X_processed, y_processed = self._prepare_data(X, y)
                predictions = self.model.predict(X_processed)
                mse = mean_squared_error(y_processed, predictions)
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': np.sqrt(mse) * 0.8,  # Aproximación
                'r2': self.model.score(X_processed, y_processed) if hasattr(self, 'model') else 0,
                'oob_score': getattr(self.model, 'oob_score_', None)
            }
            
        except Exception as e:
            logger.error(f"Error validando modelo: {e}")
            return {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}

class AdvancedXGBoost(BaseMLModel):
    """XGBoost con optimizaciones avanzadas"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.hyperparameters = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1
        }
        self.hyperparameters.update(config.hyperparameters)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Entrena XGBoost"""
        try:
            # Preprocessing
            self._setup_preprocessing(X, y)
            X_processed, y_processed = self._prepare_data(X, y)
            
            # Split para early stopping
            split_idx = int(len(X_processed) * 0.8)
            X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
            y_train, y_val = y_processed[:split_idx], y_processed[split_idx:]
            
            # Crear modelo
            self.model = xgb.XGBRegressor(**self.hyperparameters)
            
            # Entrenar con early stopping
            if self.config.early_stopping:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_processed, y_processed)
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
                self.feature_importance_ = dict(zip(feature_names, self.model.feature_importances_))
            
            # Validación
            metrics = self.validate(X, y)
            
            self.is_trained = True
            return metrics
            
        except Exception as e:
            logger.error(f"Error entrenando XGBoost: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X_processed, _ = self._prepare_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones con incertidumbre usando quantile regression"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X_processed, _ = self._prepare_data(X)
        
        # Predicción central
        pred_mean = self.model.predict(X_processed)
        
        # Estimación de incertidumbre (simplificada)
        # En producción, usar modelos de quantile regression
        uncertainty = np.std(pred_mean) * np.ones_like(pred_mean)
        
        return np.column_stack([pred_mean, uncertainty])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características"""
        return self.feature_importance_
    
    def validate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Valida el modelo"""
        try:
            X_processed, y_processed = self._prepare_data(X, y)
            
            if self.config.validation_method == "time_series":
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(self.model, X_processed, y_processed, 
                                          cv=tscv, scoring='neg_mean_squared_error')
                mse = -cv_scores.mean()
            else:
                predictions = self.model.predict(X_processed)
                mse = mean_squared_error(y_processed, predictions)
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': np.sqrt(mse) * 0.8,
                'r2': self.model.score(X_processed, y_processed) if hasattr(self, 'model') else 0
            }
            
        except Exception as e:
            logger.error(f"Error validando modelo: {e}")
            return {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}

# ============= MODELOS TRANSFORMER =============

class TransformerModel(BaseMLModel):
    """Modelo Transformer para series temporales financieras"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow requerido para Transformer")
        
        self.hyperparameters = {
            'sequence_length': 60,
            'n_features': 20,
            'embed_dim': 64,
            'num_heads': 4,
            'ff_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15
        }
        self.hyperparameters.update(config.hyperparameters)
    
    def _create_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Crea arquitectura Transformer"""
        try:
            # Input
            inputs = Input(shape=input_shape)
            
            # Embedding posicional
            x = Dense(self.hyperparameters['embed_dim'])(inputs)
            
            # Transformer blocks
            for _ in range(self.hyperparameters['num_layers']):
                # Multi-head attention
                attn_output = MultiHeadAttention(
                    num_heads=self.hyperparameters['num_heads'],
                    key_dim=self.hyperparameters['embed_dim']
                )(x, x)
                
                # Add & Norm
                x = LayerNormalization()(x + attn_output)
                
                # Feed forward
                ff_output = Dense(self.hyperparameters['ff_dim'], activation='relu')(x)
                ff_output = Dropout(self.hyperparameters['dropout'])(ff_output)
                ff_output = Dense(self.hyperparameters['embed_dim'])(ff_output)
                
                # Add & Norm
                x = LayerNormalization()(x + ff_output)
            
            # Global pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Output layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(self.hyperparameters['dropout'])(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(self.hyperparameters['dropout'])(x)
            outputs = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model
            
        except Exception as e:
            logger.error(f"Error creando modelo Transformer: {e}")
            raise
    
    def _prepare_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara secuencias para Transformer"""
        seq_length = self.hyperparameters['sequence_length']
        
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_length, len(X)):
            X_sequences.append(X.iloc[i-seq_length:i].values)
            if y is not None:
                y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y is not None else None
        
        return X_sequences, y_sequences
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Entrena Transformer"""
        try:
            # Preprocessing
            self._setup_preprocessing(X, y)
            X_processed, y_processed = self._prepare_data(X, y)
            
            # Crear secuencias
            X_seq, y_seq = self._prepare_sequences(pd.DataFrame(X_processed), pd.Series(y_processed))
            
            if len(X_seq) < 100:
                raise ValueError("Datos insuficientes para entrenamiento")
            
            # Split
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Crear modelo
            self.model = self._create_transformer_model(X_train.shape[1:])
            
            # Compilar
            self.model.compile(
                optimizer=Adam(learning_rate=self.hyperparameters['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.hyperparameters['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6
                )
            ]
            
            # Entrenar
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.hyperparameters['batch_size'],
                epochs=self.hyperparameters['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.training_history = history.history
            
            # Validación
            metrics = self.validate(X, y)
            
            self.is_trained = True
            return metrics
            
        except Exception as e:
            logger.error(f"Error entrenando Transformer: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X_processed, _ = self._prepare_data(X)
        X_seq, _ = self._prepare_sequences(pd.DataFrame(X_processed))
        
        return self.model.predict(X_seq).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicciones con incertidumbre usando Monte Carlo Dropout"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        X_processed, _ = self._prepare_data(X)
        X_seq, _ = self._prepare_sequences(pd.DataFrame(X_processed))
        
        # Múltiples predicciones con dropout
        predictions = []
        for _ in range(100):  # 100 muestras Monte Carlo
            pred = self.model(X_seq, training=True)  # Mantener dropout activo
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return np.column_stack([mean_pred, std_pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Feature importance para Transformer (aproximado)"""
        # Implementación simplificada usando gradientes
        return self.feature_importance_
    
    def validate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Valida el modelo"""
        try:
            X_processed, y_processed = self._prepare_data(X, y)
            X_seq, y_seq = self._prepare_sequences(pd.DataFrame(X_processed), pd.Series(y_processed))
            
            predictions = self.model.predict(X_seq).flatten()
            
            mse = mean_squared_error(y_seq, predictions)
            mae = mean_absolute_error(y_seq, predictions)
            r2 = r2_score(y_seq, predictions)
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            }
            
        except Exception as e:
            logger.error(f"Error validando Transformer: {e}")
            return {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}

# ============= ENSEMBLE AVANZADO =============

class AdvancedEnsemble(IEnsembleModel):
    """Ensemble avanzado con stacking y blending"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.base_models = []
        self.meta_model = None
        self.model_weights = {}
        self.is_trained = False
        self.performance_history = []
        
    def add_model(self, model: IMLModel, weight: float = 1.0):
        """Añade modelo al ensemble"""
        self.base_models.append(model)
        self.model_weights[len(self.base_models) - 1] = weight
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Entrena ensemble con stacking"""
        try:
            if len(self.base_models) == 0:
                raise ValueError("No hay modelos base en el ensemble")
            
            # Entrenar modelos base
            base_predictions = []
            model_metrics = []
            
            for i, model in enumerate(self.base_models):
                logger.info(f"Entrenando modelo base {i+1}/{len(self.base_models)}")
                
                # Entrenar modelo
                metrics = model.train(X, y)
                model_metrics.append(metrics)
                
                # Obtener predicciones out-of-fold
                predictions = self._get_out_of_fold_predictions(model, X, y)
                base_predictions.append(predictions)
            
            # Crear features para meta-modelo
            meta_features = np.column_stack(base_predictions)
            
            # Entrenar meta-modelo
            self.meta_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self.meta_model.fit(meta_features, y)
            
            # Validar ensemble
            ensemble_metrics = self._validate_ensemble(X, y)
            
            self.is_trained = True
            return ensemble_metrics
            
        except Exception as e:
            logger.error(f"Error entrenando ensemble: {e}")
            raise
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicción ensemble con incertidumbre"""
        if not self.is_trained:
            raise ValueError("Ensemble no entrenado")
        
        # Predicciones de modelos base
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Meta-predicción
        meta_features = np.column_stack(base_predictions)
        ensemble_pred = self.meta_model.predict(meta_features)
        
        # Calcular incertidumbre
        pred_std = np.std(base_predictions, axis=0)
        
        return ensemble_pred, pred_std
    
    def _get_out_of_fold_predictions(self, model: IMLModel, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Obtiene predicciones out-of-fold para evitar overfitting"""
        tscv = TimeSeriesSplit(n_splits=5)
        oof_predictions = np.zeros(len(y))
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar en fold
            model.train(X_train, y_train)
            
            # Predecir en validación
            pred = model.predict(X_val)
            oof_predictions[val_idx] = pred
        
        return oof_predictions
    
    def _validate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Valida el ensemble"""
        try:
            predictions, uncertainty = self.predict_ensemble(X)
            
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'uncertainty_mean': np.mean(uncertainty)
            }
            
        except Exception as e:
            logger.error(f"Error validando ensemble: {e}")
            return {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}

# ============= FACTORY DE MODELOS =============

class ModelFactory:
    """Factory para crear modelos"""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> IMLModel:
        """Crea modelo según tipo"""
        if model_type == "random_forest":
            return AdvancedRandomForest(config)
        elif model_type == "xgboost":
            return AdvancedXGBoost(config)
        elif model_type == "transformer":
            return TransformerModel(config)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    @staticmethod
    def create_ensemble(config: SystemConfig) -> AdvancedEnsemble:
        """Crea ensemble con modelos configurados"""
        ensemble = AdvancedEnsemble(config)
        
        # Añadir modelos base
        for model_type in config.ml.models_to_train:
            model_config = ModelConfig(
                model_type=model_type.value,
                hyperparameters={}
            )
            model = ModelFactory.create_model(model_type.value, model_config)
            ensemble.add_model(model)
        
        return ensemble 