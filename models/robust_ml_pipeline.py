# PIPELINE ML ROBUSTO
"""
Pipeline de Machine Learning robusto con walk-forward analysis,
validación temporal, y prevención de overfitting para trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb

# Feature engineering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import ta

# Interfaces
from core.interfaces import IMLModel, Features, ModelMetrics
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuración de walk-forward analysis"""
    initial_train_size: int = 1000  # Tamaño inicial de entrenamiento
    step_size: int = 100  # Tamaño del paso
    test_size: int = 50  # Tamaño del conjunto de prueba
    max_train_size: Optional[int] = 5000  # Tamaño máximo de entrenamiento
    purged_samples: int = 5  # Muestras a purgar entre train/test
    embargo_period: int = 24  # Período de embargo (horas)

@dataclass
class FeatureEngineeringConfig:
    """Configuración de feature engineering"""
    include_technical_indicators: bool = True
    include_price_features: bool = True
    include_volume_features: bool = True
    include_volatility_features: bool = True
    include_momentum_features: bool = True
    include_statistical_features: bool = True
    include_regime_features: bool = True
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    max_features: int = 100
    feature_selection_method: str = "mutual_info"  # 'mutual_info', 'f_test', 'variance'

@dataclass
class ModelValidationResult:
    """Resultado de validación de modelo"""
    model_name: str
    validation_type: str
    start_date: datetime
    end_date: datetime
    n_samples: int
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    predictions: pd.Series
    actuals: pd.Series
    overfitting_score: float
    stability_score: float

class FeatureEngineer:
    """Ingeniero de características avanzado"""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.feature_names = []
        self.scalers = {}
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características avanzadas para ML
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con características engineered
        """
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # 1. Características de precio
            if self.config.include_price_features:
                features = pd.concat([features, self._create_price_features(data)], axis=1)
            
            # 2. Indicadores técnicos
            if self.config.include_technical_indicators:
                features = pd.concat([features, self._create_technical_features(data)], axis=1)
            
            # 3. Características de volumen
            if self.config.include_volume_features:
                features = pd.concat([features, self._create_volume_features(data)], axis=1)
            
            # 4. Características de volatilidad
            if self.config.include_volatility_features:
                features = pd.concat([features, self._create_volatility_features(data)], axis=1)
            
            # 5. Características de momentum
            if self.config.include_momentum_features:
                features = pd.concat([features, self._create_momentum_features(data)], axis=1)
            
            # 6. Características estadísticas
            if self.config.include_statistical_features:
                features = pd.concat([features, self._create_statistical_features(data)], axis=1)
            
            # 7. Características de régimen
            if self.config.include_regime_features:
                features = pd.concat([features, self._create_regime_features(data)], axis=1)
            
            # 8. Características temporales
            features = pd.concat([features, self._create_temporal_features(data)], axis=1)
            
            # 9. Características de interacción
            features = pd.concat([features, self._create_interaction_features(features)], axis=1)
            
            # Limpiar y preparar
            features = self._clean_features(features)
            
            # Selección de características
            if len(features.columns) > self.config.max_features:
                features = self._select_features(features, data['close'])
            
            self.feature_names = list(features.columns)
            
            logger.info(f"✅ Features engineered: {len(features.columns)} características creadas")
            
            return features
            
        except Exception as e:
            logger.error(f"Error en feature engineering: {e}")
            return pd.DataFrame(index=data.index)
    
    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en precios"""
        
        features = pd.DataFrame(index=data.index)
        
        # Returns en múltiples períodos
        for period in self.config.lookback_periods:
            features[f'return_{period}'] = data['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(data['close'] / data['close'].shift(period))
        
        # Precios relativos
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['price_range'] = (data['high'] - data['low']) / data['close']
        
        # Moving averages ratios
        for period in [5, 10, 20, 50]:
            if len(data) > period:
                ma = data['close'].rolling(period).mean()
                features[f'price_ma_ratio_{period}'] = data['close'] / ma
                features[f'ma_slope_{period}'] = ma.pct_change(period)
        
        return features
    
    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea indicadores técnicos"""
        
        features = pd.DataFrame(index=data.index)
        
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(data['close'])
            features['rsi'] = rsi.rsi()
            features['rsi_normalized'] = (features['rsi'] - 50) / 50
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            features['bb_upper'] = bb.bollinger_hband()
            features['bb_lower'] = bb.bollinger_lband()
            features['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            features['williams_r'] = ta.momentum.WilliamsRIndicator(data['high'], data['low'], data['close']).williams_r()
            
            # ADX
            features['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx()
            
        except Exception as e:
            logger.warning(f"Error creando indicadores técnicos: {e}")
        
        return features
    
    def _create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características basadas en volumen"""
        
        features = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            return features
        
        # Volume ratios
        for period in [5, 10, 20]:
            if len(data) > period:
                avg_volume = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / avg_volume
        
        # Price-Volume relationship
        features['price_volume'] = data['close'] * data['volume']
        features['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(data['close'], data['volume']).volume_price_trend()
        
        # On-Balance Volume
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
        features['obv_change'] = features['obv'].pct_change()
        
        return features
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características de volatilidad"""
        
        features = pd.DataFrame(index=data.index)
        
        # Realized volatility
        returns = data['close'].pct_change()
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'volatility_rank_{period}'] = features[f'volatility_{period}'].rolling(252).rank(pct=True)
        
        # True Range
        features['true_range'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
        
        # GARCH-like features
        features['volatility_momentum'] = features['volatility_10'] / features['volatility_20']
        
        # Volatility regimes
        vol_20 = returns.rolling(20).std()
        features['vol_regime'] = (vol_20 > vol_20.rolling(252).quantile(0.8)).astype(int)
        
        return features
    
    def _create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características de momentum"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # Acceleration
        momentum_10 = data['close'] / data['close'].shift(10) - 1
        features['momentum_acceleration'] = momentum_10 - momentum_10.shift(10)
        
        # Momentum strength
        features['momentum_strength'] = np.where(
            features['momentum_10'] > 0, 
            features['momentum_10'] / features['volatility_10'], 
            features['momentum_10'] / features['volatility_10']
        )
        
        return features
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características estadísticas"""
        
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # Momentos estadísticos
        for period in [20, 50]:
            if len(data) > period:
                rolling_returns = returns.rolling(period)
                features[f'skewness_{period}'] = rolling_returns.skew()
                features[f'kurtosis_{period}'] = rolling_returns.kurt()
        
        # Percentiles
        for period in [20, 50]:
            for percentile in [25, 75]:
                features[f'price_percentile_{percentile}_{period}'] = data['close'].rolling(period).rank(pct=True)
        
        # Drawdown
        peak = data['close'].expanding().max()
        features['drawdown'] = (data['close'] - peak) / peak
        features['underwater_duration'] = (features['drawdown'] < 0).astype(int).groupby((features['drawdown'] >= 0).cumsum()).cumsum()
        
        return features
    
    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características de régimen de mercado"""
        
        features = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # Régimen de volatilidad
        vol_20 = returns.rolling(20).std()
        vol_percentile = vol_20.rolling(252).rank(pct=True)
        features['vol_regime_low'] = (vol_percentile < 0.33).astype(int)
        features['vol_regime_high'] = (vol_percentile > 0.67).astype(int)
        
        # Régimen de tendencia
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        features['trend_regime'] = (sma_20 > sma_50).astype(int)
        
        # Régimen de momentum
        momentum_20 = data['close'] / data['close'].shift(20) - 1
        momentum_percentile = momentum_20.rolling(252).rank(pct=True)
        features['momentum_regime'] = (momentum_percentile > 0.5).astype(int)
        
        return features
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características temporales"""
        
        features = pd.DataFrame(index=data.index)
        
        # Características cíclicas
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['month'] = data.index.month
        
        # Encoding cíclico
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Crea características de interacción"""
        
        interaction_features = pd.DataFrame(index=features.index)
        
        # Seleccionar algunas características principales para interacciones
        main_features = ['return_10', 'rsi', 'volatility_20', 'momentum_20']
        available_features = [f for f in main_features if f in features.columns]
        
        # Interacciones multiplicativas
        for i, feat1 in enumerate(available_features):
            for feat2 in available_features[i+1:]:
                if feat1 in features.columns and feat2 in features.columns:
                    interaction_features[f'{feat1}_x_{feat2}'] = features[feat1] * features[feat2]
        
        return interaction_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Limpia características"""
        
        # Remover características con demasiados NaN
        nan_threshold = 0.3
        features = features.loc[:, features.isnull().mean() < nan_threshold]
        
        # Remover características constantes
        features = features.loc[:, features.std() > 1e-6]
        
        # Remover outliers extremos
        for col in features.select_dtypes(include=[np.number]).columns:
            q99 = features[col].quantile(0.99)
            q01 = features[col].quantile(0.01)
            features[col] = features[col].clip(q01, q99)
        
        # Forward fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Selecciona mejores características"""
        
        try:
            # Alinear features y target
            aligned_data = features.align(target, join='inner')
            X_aligned = aligned_data[0]
            y_aligned = aligned_data[1]
            
            # Remover NaN
            valid_idx = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
            X_clean = X_aligned[valid_idx]
            y_clean = y_aligned[valid_idx]
            
            if len(X_clean) == 0:
                return features.iloc[:, :self.config.max_features]
            
            # Selección de características
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
            else:
                selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
            
            X_selected = selector.fit_transform(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()]
            
            return features[selected_features]
            
        except Exception as e:
            logger.warning(f"Error en selección de características: {e}")
            return features.iloc[:, :self.config.max_features]

class WalkForwardValidator:
    """Validador con walk-forward analysis"""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        
    def generate_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Genera splits para walk-forward analysis
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            Lista de tuples (train_index, test_index)
        """
        
        splits = []
        data_length = len(data)
        
        # Comenzar desde el tamaño inicial de entrenamiento
        start_idx = 0
        train_end_idx = self.config.initial_train_size
        
        while train_end_idx + self.config.test_size <= data_length:
            # Índices de entrenamiento
            train_start_idx = max(0, train_end_idx - (self.config.max_train_size or train_end_idx))
            
            # Período de purga entre entrenamiento y test
            test_start_idx = train_end_idx + self.config.purged_samples
            test_end_idx = test_start_idx + self.config.test_size
            
            if test_end_idx <= data_length:
                train_index = data.index[train_start_idx:train_end_idx]
                test_index = data.index[test_start_idx:test_end_idx]
                
                splits.append((train_index, test_index))
            
            # Avanzar
            train_end_idx += self.config.step_size
        
        logger.info(f"✅ Generados {len(splits)} splits para walk-forward analysis")
        
        return splits
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]) -> List[ModelValidationResult]:
        """
        Valida modelo usando walk-forward analysis
        
        Args:
            model: Modelo a validar
            X: Características
            y: Target
            splits: Splits de tiempo
            
        Returns:
            Lista de resultados de validación
        """
        
        results = []
        
        for i, (train_index, test_index) in enumerate(splits):
            try:
                # Datos de entrenamiento
                X_train = X.loc[train_index]
                y_train = y.loc[train_index]
                
                # Datos de test
                X_test = X.loc[test_index]
                y_test = y.loc[test_index]
                
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                mse = mean_squared_error(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
                
                # Importancia de características
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                
                # Scores de overfitting y estabilidad
                overfitting_score = self._calculate_overfitting_score(model, X_train, y_train, X_test, y_test)
                stability_score = self._calculate_stability_score(results, feature_importance)
                
                # Crear resultado
                result = ModelValidationResult(
                    model_name=type(model).__name__,
                    validation_type="walk_forward",
                    start_date=test_index[0],
                    end_date=test_index[-1],
                    n_samples=len(y_test),
                    metrics={
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse),
                        'correlation': correlation,
                        'r2': 1 - mse / np.var(y_test) if np.var(y_test) > 0 else 0
                    },
                    feature_importance=feature_importance,
                    predictions=pd.Series(y_pred, index=test_index),
                    actuals=y_test,
                    overfitting_score=overfitting_score,
                    stability_score=stability_score
                )
                
                results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"📊 Validación {i+1}/{len(splits)} completada")
                    
            except Exception as e:
                logger.error(f"Error en validación {i}: {e}")
                continue
        
        return results
    
    def _calculate_overfitting_score(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Calcula score de overfitting"""
        
        try:
            # Predicciones en train y test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Errores
            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)
            
            # Score de overfitting (0 = no overfitting, 1 = mucho overfitting)
            if train_error == 0:
                return 1.0
            
            overfitting_score = max(0, (test_error - train_error) / train_error)
            return min(1.0, overfitting_score)
            
        except Exception as e:
            logger.warning(f"Error calculando overfitting score: {e}")
            return 0.5
    
    def _calculate_stability_score(self, previous_results: List[ModelValidationResult],
                                 current_importance: Dict[str, float]) -> float:
        """Calcula score de estabilidad de características"""
        
        if not previous_results or not current_importance:
            return 1.0
        
        try:
            # Obtener importancia de resultado anterior
            prev_importance = previous_results[-1].feature_importance
            
            if not prev_importance:
                return 1.0
            
            # Características comunes
            common_features = set(current_importance.keys()) & set(prev_importance.keys())
            
            if not common_features:
                return 0.0
            
            # Calcular correlación de importancias
            current_vals = [current_importance[f] for f in common_features]
            prev_vals = [prev_importance[f] for f in common_features]
            
            correlation = np.corrcoef(current_vals, prev_vals)[0, 1]
            
            return max(0, correlation) if not np.isnan(correlation) else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculando stability score: {e}")
            return 0.5

class RobustMLPipeline:
    """Pipeline de ML robusto con validación temporal"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.ml_config = self.config.get_ml_config()
        
        # Configuraciones
        self.feature_config = FeatureEngineeringConfig()
        self.walkforward_config = WalkForwardConfig()
        
        # Componentes
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.validator = WalkForwardValidator(self.walkforward_config)
        
        # Estado
        self.trained_models = {}
        self.validation_results = {}
        self.feature_importance_history = []
        
        logger.info("✅ Pipeline ML robusto inicializado")
    
    def prepare_data(self, data: pd.DataFrame, target_horizons: List[int] = [1, 4, 12, 24]) -> Dict[str, pd.DataFrame]:
        """
        Prepara datos para entrenamiento
        
        Args:
            data: Datos OHLCV
            target_horizons: Horizontes de predicción en horas
            
        Returns:
            Diccionario con features y targets
        """
        
        logger.info("🔧 Preparando datos para ML")
        
        # Crear características
        features = self.feature_engineer.engineer_features(data)
        
        # Crear targets para diferentes horizontes
        targets = {}
        for horizon in target_horizons:
            # Target de retorno futuro
            future_return = data['close'].shift(-horizon) / data['close'] - 1
            targets[f'return_{horizon}h'] = future_return
            
            # Target de dirección (clasificación)
            targets[f'direction_{horizon}h'] = (future_return > 0).astype(int)
            
            # Target de volatilidad futura
            targets[f'volatility_{horizon}h'] = data['close'].pct_change().rolling(horizon).std().shift(-horizon)
        
        # Alinear features y targets
        result = {'features': features}
        
        for target_name, target_series in targets.items():
            aligned_data = features.align(target_series, join='inner')
            result[target_name] = {
                'features': aligned_data[0],
                'target': aligned_data[1]
            }
        
        logger.info(f"✅ Datos preparados: {len(features.columns)} features, {len(targets)} targets")
        
        return result
    
    def train_and_validate_models(self, prepared_data: Dict[str, Any]) -> Dict[str, List[ModelValidationResult]]:
        """
        Entrena y valida modelos con walk-forward analysis
        
        Args:
            prepared_data: Datos preparados
            
        Returns:
            Resultados de validación por target
        """
        
        logger.info("🚀 Iniciando entrenamiento y validación con walk-forward analysis")
        
        all_results = {}
        
        # Modelos a probar
        models_to_test = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        # Procesar cada target
        for target_name, target_data in prepared_data.items():
            if target_name == 'features':
                continue
            
            logger.info(f"📊 Procesando target: {target_name}")
            
            X = target_data['features']
            y = target_data['target']
            
            # Remover NaN
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            if len(X_clean) < self.walkforward_config.initial_train_size * 2:
                logger.warning(f"Insuficientes datos para {target_name}")
                continue
            
            # Generar splits
            splits = self.validator.generate_splits(X_clean)
            
            target_results = {}
            
            # Probar cada modelo
            for model_name, model in models_to_test.items():
                try:
                    logger.info(f"  🤖 Validando {model_name}")
                    
                    # Validación walk-forward
                    validation_results = self.validator.validate_model(model, X_clean, y_clean, splits)
                    
                    if validation_results:
                        target_results[model_name] = validation_results
                        
                        # Métricas promedio
                        avg_metrics = self._calculate_average_metrics(validation_results)
                        logger.info(f"    📈 {model_name} - R²: {avg_metrics['r2']:.3f}, "
                                  f"Correlation: {avg_metrics['correlation']:.3f}, "
                                  f"Overfitting: {avg_metrics['overfitting_score']:.3f}")
                        
                except Exception as e:
                    logger.error(f"Error validando {model_name} para {target_name}: {e}")
                    continue
            
            all_results[target_name] = target_results
        
        self.validation_results = all_results
        
        logger.info("✅ Validación walk-forward completada")
        
        return all_results
    
    def _calculate_average_metrics(self, results: List[ModelValidationResult]) -> Dict[str, float]:
        """Calcula métricas promedio"""
        
        metrics = ['mse', 'mae', 'rmse', 'correlation', 'r2']
        avg_metrics = {}
        
        for metric in metrics:
            values = [r.metrics.get(metric, 0) for r in results]
            avg_metrics[metric] = np.mean(values)
        
        # Métricas adicionales
        avg_metrics['overfitting_score'] = np.mean([r.overfitting_score for r in results])
        avg_metrics['stability_score'] = np.mean([r.stability_score for r in results])
        
        return avg_metrics
    
    def select_best_models(self) -> Dict[str, str]:
        """Selecciona mejores modelos por target"""
        
        best_models = {}
        
        for target_name, target_results in self.validation_results.items():
            best_score = -np.inf
            best_model = None
            
            for model_name, results in target_results.items():
                # Score compuesto considerando performance y robustez
                avg_metrics = self._calculate_average_metrics(results)
                
                score = (avg_metrics['correlation'] * 0.4 +  # Performance
                        (1 - avg_metrics['overfitting_score']) * 0.3 +  # Anti-overfitting
                        avg_metrics['stability_score'] * 0.3)  # Estabilidad
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            best_models[target_name] = best_model
            logger.info(f"🏆 Mejor modelo para {target_name}: {best_model} (score: {best_score:.3f})")
        
        return best_models
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Genera reporte de performance detallado"""
        
        report = {
            'timestamp': datetime.now(),
            'total_targets': len(self.validation_results),
            'total_models_tested': 0,
            'best_models': self.select_best_models(),
            'target_summaries': {},
            'overall_metrics': {},
            'stability_analysis': self._analyze_stability(),
            'overfitting_analysis': self._analyze_overfitting()
        }
        
        # Análisis por target
        for target_name, target_results in self.validation_results.items():
            report['total_models_tested'] += len(target_results)
            
            target_summary = {
                'models_tested': list(target_results.keys()),
                'best_model': report['best_models'].get(target_name),
                'model_performance': {}
            }
            
            for model_name, results in target_results.items():
                avg_metrics = self._calculate_average_metrics(results)
                target_summary['model_performance'][model_name] = avg_metrics
            
            report['target_summaries'][target_name] = target_summary
        
        # Métricas generales
        all_correlations = []
        all_overfitting = []
        all_stability = []
        
        for target_results in self.validation_results.values():
            for results in target_results.values():
                avg_metrics = self._calculate_average_metrics(results)
                all_correlations.append(avg_metrics['correlation'])
                all_overfitting.append(avg_metrics['overfitting_score'])
                all_stability.append(avg_metrics['stability_score'])
        
        if all_correlations:
            report['overall_metrics'] = {
                'avg_correlation': np.mean(all_correlations),
                'avg_overfitting': np.mean(all_overfitting),
                'avg_stability': np.mean(all_stability),
                'correlation_std': np.std(all_correlations)
            }
        
        return report
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """Analiza estabilidad temporal de los modelos"""
        
        stability_analysis = {
            'feature_importance_drift': {},
            'performance_consistency': {},
            'model_reliability': {}
        }
        
        # Analizar drift de importancia de características
        for target_name, target_results in self.validation_results.items():
            for model_name, results in target_results.items():
                # Calcular variabilidad de importancia de características
                importance_vars = []
                feature_names = set()
                
                for result in results:
                    feature_names.update(result.feature_importance.keys())
                
                for feature in feature_names:
                    importances = [r.feature_importance.get(feature, 0) for r in results]
                    if importances:
                        importance_vars.append(np.std(importances))
                
                key = f"{target_name}_{model_name}"
                stability_analysis['feature_importance_drift'][key] = np.mean(importance_vars) if importance_vars else 0
        
        return stability_analysis
    
    def _analyze_overfitting(self) -> Dict[str, Any]:
        """Analiza patrones de overfitting"""
        
        overfitting_analysis = {
            'high_overfitting_models': [],
            'overfitting_trends': {},
            'recommendations': []
        }
        
        # Identificar modelos con alto overfitting
        for target_name, target_results in self.validation_results.items():
            for model_name, results in target_results.items():
                avg_overfitting = np.mean([r.overfitting_score for r in results])
                
                if avg_overfitting > 0.3:  # Threshold de overfitting
                    overfitting_analysis['high_overfitting_models'].append({
                        'target': target_name,
                        'model': model_name,
                        'overfitting_score': avg_overfitting
                    })
        
        # Recomendaciones
        if overfitting_analysis['high_overfitting_models']:
            overfitting_analysis['recommendations'].extend([
                "Considerar regularización adicional",
                "Reducir complejidad del modelo",
                "Aumentar tamaño del conjunto de entrenamiento",
                "Implementar early stopping"
            ])
        
        return overfitting_analysis

# Singleton para acceso global
_robust_ml_pipeline = None

def get_robust_ml_pipeline() -> RobustMLPipeline:
    """Obtiene el pipeline ML robusto (singleton)"""
    global _robust_ml_pipeline
    if _robust_ml_pipeline is None:
        _robust_ml_pipeline = RobustMLPipeline()
    return _robust_ml_pipeline 