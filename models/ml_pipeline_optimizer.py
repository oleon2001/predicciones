# OPTIMIZADOR DE PIPELINE DE MACHINE LEARNING
"""
Sistema avanzado de optimización de pipeline ML
Incluye feature selection, hyperparameter tuning, ensemble optimization y AutoML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import joblib
import json

# ML Libraries
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit,
    cross_val_score, validation_curve
)
from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV, SelectFromModel,
    f_regression, mutual_info_regression
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
import xgboost as xgb
import lightgbm as lgb

# Optimization libraries
from scipy.optimize import minimize, differential_evolution
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from optuna import create_study, Trial
import optuna.samplers as samplers

from core.interfaces import IMLModel, ModelMetrics
from config.system_config import SystemConfig
from core.monitoring_system import PerformanceMonitor, MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    best_score: float
    best_params: Dict[str, Any]
    optimization_time: float
    iterations: int
    history: List[Dict[str, Any]]
    model_path: Optional[str] = None

@dataclass
class FeatureSelectionResult:
    """Resultado de selección de features"""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_method: str
    n_features_selected: int
    improvement_score: float

@dataclass
class PipelineConfig:
    """Configuración del pipeline"""
    feature_selection_methods: List[str] = None
    hyperparameter_methods: List[str] = None
    ensemble_methods: List[str] = None
    cv_folds: int = 5
    n_jobs: int = -1
    max_optimization_time: int = 3600  # 1 hora
    early_stopping_rounds: int = 50

class FeatureSelector:
    """Selector de características avanzado"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.selection_methods = {
            'univariate': self._univariate_selection,
            'recursive': self._recursive_selection,
            'model_based': self._model_based_selection,
            'correlation': self._correlation_selection,
            'mutual_info': self._mutual_info_selection,
            'stability': self._stability_selection,
            'ensemble': self._ensemble_selection
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'ensemble') -> FeatureSelectionResult:
        """Selecciona características"""
        logger.info(f"Iniciando selección de features con método: {method}")
        
        if method not in self.selection_methods:
            raise ValueError(f"Método no soportado: {method}")
        
        start_time = time.time()
        
        # Ejecutar selección
        selected_features, feature_scores = self.selection_methods[method](X, y)
        
        # Evaluar mejora
        improvement_score = self._evaluate_improvement(X, y, selected_features)
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method=method,
            n_features_selected=len(selected_features),
            improvement_score=improvement_score
        )
        
        optimization_time = time.time() - start_time
        logger.info(f"Selección completada en {optimization_time:.2f}s. "
                   f"Features: {len(selected_features)}, Mejora: {improvement_score:.4f}")
        
        return result
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección univariada"""
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        
        # Obtener scores
        scores = dict(zip(X.columns, selector.scores_))
        
        # Seleccionar mejores features (top 50% o máximo 50)
        n_features = min(len(X.columns) // 2, 50)
        selected_features = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n_features]
        
        return selected_features, scores
    
    def _recursive_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección recursiva"""
        # Usar RandomForest como estimador base
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # RFE con validación cruzada
        selector = RFECV(estimator, cv=3, scoring='neg_mean_squared_error', n_jobs=self.config.n_jobs)
        selector.fit(X, y)
        
        # Obtener features seleccionadas
        selected_features = X.columns[selector.support_].tolist()
        
        # Crear scores basados en ranking
        feature_scores = {}
        for i, feature in enumerate(X.columns):
            # Invertir ranking para que mayor score = mejor feature
            feature_scores[feature] = 1.0 / selector.ranking_[i]
        
        return selected_features, feature_scores
    
    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección basada en modelo"""
        # Usar Lasso para feature selection
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X, y)
        
        # Seleccionar features con coeficientes no nulos
        selector = SelectFromModel(lasso, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Usar coeficientes absolutos como scores
        feature_scores = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        return selected_features, feature_scores
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección por correlación"""
        # Calcular correlación con target
        correlations = X.corrwith(y).abs()
        
        # Remover features altamente correlacionadas entre sí
        selected_features = []
        feature_scores = correlations.to_dict()
        
        # Ordenar por correlación con target
        sorted_features = correlations.sort_values(ascending=False)
        
        for feature in sorted_features.index:
            if not selected_features:
                selected_features.append(feature)
            else:
                # Verificar correlación con features ya seleccionadas
                max_corr = X[selected_features].corrwith(X[feature]).abs().max()
                if max_corr < 0.8:  # Umbral de correlación
                    selected_features.append(feature)
        
        return selected_features, feature_scores
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección por información mutua"""
        mi_scores = mutual_info_regression(X, y, random_state=42)
        feature_scores = dict(zip(X.columns, mi_scores))
        
        # Seleccionar top 50% o máximo 50 features
        n_features = min(len(X.columns) // 2, 50)
        selected_features = sorted(feature_scores.keys(), 
                                 key=lambda x: feature_scores[x], 
                                 reverse=True)[:n_features]
        
        return selected_features, feature_scores
    
    def _stability_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección por estabilidad"""
        n_bootstrap = 50
        stability_scores = {feature: 0 for feature in X.columns}
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Selección con Lasso
            lasso = Lasso(alpha=0.01, random_state=42)
            lasso.fit(X_boot, y_boot)
            
            # Contar selecciones
            for i, feature in enumerate(X.columns):
                if abs(lasso.coef_[i]) > 1e-6:
                    stability_scores[feature] += 1
        
        # Normalizar scores
        for feature in stability_scores:
            stability_scores[feature] /= n_bootstrap
        
        # Seleccionar features con estabilidad > 0.6
        selected_features = [f for f, score in stability_scores.items() if score > 0.6]
        
        return selected_features, stability_scores
    
    def _ensemble_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Selección ensemble combinando múltiples métodos"""
        methods = ['univariate', 'recursive', 'model_based', 'correlation', 'mutual_info']
        
        all_results = {}
        for method in methods:
            try:
                selected, scores = self.selection_methods[method](X, y)
                all_results[method] = (selected, scores)
            except Exception as e:
                logger.warning(f"Error en método {method}: {e}")
        
        # Combinar resultados
        feature_votes = {feature: 0 for feature in X.columns}
        feature_scores = {feature: 0.0 for feature in X.columns}
        
        for method, (selected, scores) in all_results.items():
            for feature in selected:
                feature_votes[feature] += 1
            
            # Normalizar scores del método
            max_score = max(scores.values()) if scores.values() else 1.0
            for feature, score in scores.items():
                feature_scores[feature] += score / max_score
        
        # Seleccionar features con más votos
        min_votes = len(all_results) // 2  # Mayoría
        selected_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        
        return selected_features, feature_scores
    
    def _evaluate_improvement(self, X: pd.DataFrame, y: pd.Series, 
                            selected_features: List[str]) -> float:
        """Evalúa mejora de selección"""
        if len(selected_features) == 0:
            return 0.0
        
        # Evaluar con modelo simple
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        
        # Score con todas las features
        cv_scores_all = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        score_all = -cv_scores_all.mean()
        
        # Score con features seleccionadas
        cv_scores_selected = cross_val_score(model, X[selected_features], y, cv=3, scoring='neg_mean_squared_error')
        score_selected = -cv_scores_selected.mean()
        
        # Calcular mejora (menor MSE es mejor)
        improvement = (score_all - score_selected) / score_all if score_all > 0 else 0.0
        
        return improvement

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros avanzado"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.optimizers = {
            'grid_search': self._grid_search,
            'random_search': self._random_search,
            'bayesian': self._bayesian_optimization,
            'optuna': self._optuna_optimization,
            'differential_evolution': self._differential_evolution
        }
    
    def optimize(self, model: IMLModel, X: pd.DataFrame, y: pd.Series,
                param_space: Dict[str, Any], method: str = 'bayesian') -> OptimizationResult:
        """Optimiza hiperparámetros"""
        logger.info(f"Iniciando optimización con método: {method}")
        
        if method not in self.optimizers:
            raise ValueError(f"Método no soportado: {method}")
        
        start_time = time.time()
        
        # Ejecutar optimización
        best_params, best_score, history = self.optimizers[method](model, X, y, param_space)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_score=best_score,
            best_params=best_params,
            optimization_time=optimization_time,
            iterations=len(history),
            history=history
        )
        
        logger.info(f"Optimización completada en {optimization_time:.2f}s. "
                   f"Mejor score: {best_score:.4f}")
        
        return result
    
    def _grid_search(self, model: IMLModel, X: pd.DataFrame, y: pd.Series,
                    param_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Grid search"""
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        grid_search = GridSearchCV(
            model, param_space, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs, return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        # Extraer historial
        history = []
        for i in range(len(grid_search.cv_results_['mean_test_score'])):
            history.append({
                'params': grid_search.cv_results_['params'][i],
                'score': -grid_search.cv_results_['mean_test_score'][i]
            })
        
        return grid_search.best_params_, -grid_search.best_score_, history
    
    def _bayesian_optimization(self, model: IMLModel, X: pd.DataFrame, y: pd.Series,
                             param_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Optimización bayesiana usando hyperopt"""
        
        def objective(params):
            # Convertir parámetros
            converted_params = self._convert_params(params, param_space)
            
            # Entrenar modelo
            model.set_params(**converted_params)
            cv_scores = cross_val_score(model, X, y, cv=self.config.cv_folds, 
                                      scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
            
            return {'loss': score, 'status': STATUS_OK, 'eval_time': time.time()}
        
        # Convertir param_space a hyperopt format
        hyperopt_space = self._convert_to_hyperopt_space(param_space)
        
        # Ejecutar optimización
        trials = Trials()
        best = fmin(fn=objective, space=hyperopt_space, algo=tpe.suggest,
                   max_evals=100, trials=trials)
        
        # Extraer resultados
        best_params = self._convert_params(best, param_space)
        best_score = min([trial['result']['loss'] for trial in trials.trials])
        
        history = []
        for trial in trials.trials:
            history.append({
                'params': trial['misc']['vals'],
                'score': trial['result']['loss']
            })
        
        return best_params, best_score, history
    
    def _optuna_optimization(self, model: IMLModel, X: pd.DataFrame, y: pd.Series,
                           param_space: Dict[str, Any]) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Optimización con Optuna"""
        
        def objective(trial: Trial):
            # Sugerir parámetros
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            
            # Entrenar modelo
            model.set_params(**params)
            cv_scores = cross_val_score(model, X, y, cv=self.config.cv_folds, 
                                      scoring='neg_mean_squared_error')
            return -cv_scores.mean()
        
        # Crear estudio
        study = create_study(direction='minimize', sampler=samplers.TPESampler())
        study.optimize(objective, n_trials=100, timeout=self.config.max_optimization_time)
        
        # Extraer resultados
        best_params = study.best_params
        best_score = study.best_value
        
        history = []
        for trial in study.trials:
            history.append({
                'params': trial.params,
                'score': trial.value
            })
        
        return best_params, best_score, history
    
    def _convert_to_hyperopt_space(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte param_space a formato hyperopt"""
        hyperopt_space = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                hyperopt_space[param_name] = hp.quniform(
                    param_name, param_config['low'], param_config['high'], 1
                )
            elif param_config['type'] == 'float':
                hyperopt_space[param_name] = hp.uniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'categorical':
                hyperopt_space[param_name] = hp.choice(param_name, param_config['choices'])
        
        return hyperopt_space
    
    def _convert_params(self, params: Dict[str, Any], param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte parámetros a tipos correctos"""
        converted = {}
        
        for param_name, value in params.items():
            if param_name in param_space:
                if param_space[param_name]['type'] == 'int':
                    converted[param_name] = int(value)
                else:
                    converted[param_name] = value
            else:
                converted[param_name] = value
        
        return converted

class MLPipelineOptimizer:
    """Optimizador principal del pipeline ML"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.pipeline_config = PipelineConfig()
        self.feature_selector = FeatureSelector(self.pipeline_config)
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.pipeline_config)
        self.performance_monitor = PerformanceMonitor(MetricsCollector())
    
    def optimize_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                         models: List[IMLModel],
                         optimization_budget: int = 3600) -> Dict[str, Any]:
        """Optimiza pipeline completo"""
        logger.info("Iniciando optimización completa del pipeline")
        
        results = {
            'feature_selection': None,
            'model_optimization': {},
            'ensemble_optimization': None,
            'final_performance': {},
            'optimization_time': 0
        }
        
        start_time = time.time()
        
        with self.performance_monitor.timer("pipeline_optimization"):
            # 1. Selección de features
            logger.info("Paso 1: Selección de características")
            feature_result = self.feature_selector.select_features(X, y, method='ensemble')
            results['feature_selection'] = feature_result
            
            # Usar features seleccionadas
            X_selected = X[feature_result.selected_features]
            
            # 2. Optimización de modelos individuales
            logger.info("Paso 2: Optimización de modelos individuales")
            model_results = {}
            
            for model in models:
                model_name = model.__class__.__name__
                logger.info(f"Optimizando {model_name}")
                
                # Definir espacio de parámetros
                param_space = self._get_param_space(model)
                
                # Optimizar
                optimization_result = self.hyperparameter_optimizer.optimize(
                    model, X_selected, y, param_space, method='optuna'
                )
                
                model_results[model_name] = optimization_result
            
            results['model_optimization'] = model_results
            
            # 3. Optimización de ensemble
            logger.info("Paso 3: Optimización de ensemble")
            ensemble_result = self._optimize_ensemble(X_selected, y, models, model_results)
            results['ensemble_optimization'] = ensemble_result
            
            # 4. Evaluación final
            logger.info("Paso 4: Evaluación final")
            final_performance = self._evaluate_final_performance(X_selected, y, ensemble_result)
            results['final_performance'] = final_performance
        
        results['optimization_time'] = time.time() - start_time
        
        logger.info(f"Optimización completada en {results['optimization_time']:.2f}s")
        
        return results
    
    def _get_param_space(self, model: IMLModel) -> Dict[str, Any]:
        """Obtiene espacio de parámetros para modelo"""
        model_name = model.__class__.__name__
        
        param_spaces = {
            'RandomForestRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'XGBRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'LGBMRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'num_leaves': {'type': 'int', 'low': 20, 'high': 100}
            }
        }
        
        return param_spaces.get(model_name, {})
    
    def _optimize_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                          models: List[IMLModel], 
                          model_results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """Optimiza ensemble de modelos"""
        # Configurar modelos con mejores parámetros
        optimized_models = []
        
        for model in models:
            model_name = model.__class__.__name__
            if model_name in model_results:
                best_params = model_results[model_name].best_params
                model.set_params(**best_params)
            optimized_models.append(model)
        
        # Optimizar pesos del ensemble
        def ensemble_objective(weights):
            # Normalizar pesos
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Obtener predicciones
            predictions = np.zeros(len(y))
            cv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model_preds = []
                for model in optimized_models:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    model_preds.append(pred)
                
                # Ensemble prediction
                ensemble_pred = np.average(model_preds, weights=weights, axis=0)
                predictions[test_idx] = ensemble_pred
            
            # Calcular error
            mse = mean_squared_error(y, predictions)
            return mse
        
        # Optimizar pesos
        n_models = len(optimized_models)
        initial_weights = np.ones(n_models) / n_models
        
        result = minimize(
            ensemble_objective,
            initial_weights,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            bounds=[(0, 1) for _ in range(n_models)]
        )
        
        optimal_weights = result.x
        best_score = result.fun
        
        return {
            'optimal_weights': optimal_weights.tolist(),
            'best_score': best_score,
            'model_names': [model.__class__.__name__ for model in optimized_models]
        }
    
    def _evaluate_final_performance(self, X: pd.DataFrame, y: pd.Series, 
                                  ensemble_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa performance final"""
        # Crear ensemble con pesos optimizados
        # Esto es una simplificación - en producción sería más complejo
        
        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simulación de predicción ensemble
            # En implementación real, usaríamos los modelos optimizados
            dummy_pred = np.mean(y_train) * np.ones(len(y_test))
            score = mean_squared_error(y_test, dummy_pred)
            cv_scores.append(score)
        
        return {
            'cv_mse': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }

def main():
    """Función principal para testing"""
    config = SystemConfig()
    optimizer = MLPipelineOptimizer(config)
    
    # Generar datos de ejemplo
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(np.random.randn(1000))
    
    # Crear modelos de ejemplo
    models = [
        RandomForestRegressor(random_state=42),
        xgb.XGBRegressor(random_state=42),
        lgb.LGBMRegressor(random_state=42, verbose=-1)
    ]
    
    # Optimizar pipeline
    results = optimizer.optimize_pipeline(X, y, models, optimization_budget=300)
    
    print("=== RESULTADOS DE OPTIMIZACIÓN ===")
    print(f"Tiempo total: {results['optimization_time']:.2f}s")
    print(f"Features seleccionadas: {results['feature_selection'].n_features_selected}")
    print(f"Mejora de features: {results['feature_selection'].improvement_score:.4f}")
    print(f"Modelos optimizados: {len(results['model_optimization'])}")
    
    return results

if __name__ == "__main__":
    main() 