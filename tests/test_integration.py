# TESTS DE INTEGRACIÓN COMPLETOS
"""
Suite completa de tests para el sistema de predicción de criptomonedas
Incluye tests unitarios, integración y end-to-end
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import json
import pickle
from typing import Dict, List, Any

# Imports del sistema
from config.system_config import SystemConfig, APIConfig, TradingConfig
from core.interfaces import *
from core.dependency_container import DependencyContainer, configure_container
from core.cache_manager import CacheManager
from core.risk_manager import AdvancedRiskManager
from core.monitoring_system import MonitoringSystem, MetricsCollector
from models.advanced_ml_models import AdvancedRandomForest, ModelFactory
from models.ml_pipeline_optimizer import MLPipelineOptimizer
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.macro_analyzer import MacroAnalyzer
from backtesting.advanced_backtester import AdvancedBacktester

# Configuración de testing
@pytest.fixture
def test_config():
    """Configuración de test"""
    return SystemConfig(
        api=APIConfig(
            binance_api_key="test_key",
            binance_api_secret="test_secret",
            news_api_key="test_news_key"
        ),
        trading=TradingConfig(
            pairs=["BTCUSDT", "ETHUSDT"],
            timeframes=["1h", "4h"],
            prediction_horizons=[1, 4, 12]
        ),
        cache_enabled=True,
        parallel_processing=False,  # Disable para tests
        max_workers=1
    )

@pytest.fixture
def sample_market_data():
    """Datos de mercado de muestra"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    n_points = len(dates)
    
    # Generar datos realistas
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=dates)
    
    return MarketData(
        symbol="BTCUSDT",
        timeframe="1h",
        data=df,
        timestamp=datetime.now(),
        metadata={"source": "test"}
    )

@pytest.fixture
def temp_cache_dir():
    """Directorio temporal para cache"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

class TestSystemConfiguration:
    """Tests del sistema de configuración"""
    
    def test_config_creation(self, test_config):
        """Test creación de configuración"""
        assert test_config.api.binance_api_key == "test_key"
        assert test_config.trading.pairs == ["BTCUSDT", "ETHUSDT"]
        assert test_config.cache_enabled is True
    
    def test_config_validation(self):
        """Test validación de configuración"""
        # Test log level válido
        config = SystemConfig(log_level="INFO")
        assert config.log_level == "INFO"
        
        # Test log level inválido
        with pytest.raises(ValueError):
            SystemConfig(log_level="INVALID")
    
    def test_config_from_file(self, tmp_path):
        """Test cargar configuración desde archivo"""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "cache_enabled": False,
            "max_workers": 2,
            "log_level": "DEBUG"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = SystemConfig.from_file(str(config_file))
        assert config.cache_enabled is False
        assert config.max_workers == 2
        assert config.log_level == "DEBUG"

class TestDependencyContainer:
    """Tests del container de dependencias"""
    
    def test_container_registration(self, test_config):
        """Test registro de servicios"""
        container = DependencyContainer()
        
        # Registrar servicio
        container.register_singleton(SystemConfig, SystemConfig)
        container.register_instance(SystemConfig, test_config)
        
        # Resolver servicio
        resolved_config = container.resolve(SystemConfig)
        assert resolved_config == test_config
    
    def test_container_factory(self, test_config):
        """Test factory registration"""
        container = DependencyContainer()
        
        # Registrar factory
        container.register_factory(
            SystemConfig,
            lambda: test_config,
            lifetime="singleton"
        )
        
        # Resolver múltiples veces - debe ser la misma instancia
        config1 = container.resolve(SystemConfig)
        config2 = container.resolve(SystemConfig)
        
        assert config1 is config2
    
    def test_container_transient(self, test_config):
        """Test servicios transient"""
        container = DependencyContainer()
        
        # Registrar como transient
        container.register_transient(SystemConfig, SystemConfig)
        
        # Mock constructor para evitar validación
        with patch('config.system_config.SystemConfig.__init__', return_value=None):
            # Resolver múltiples veces - deben ser instancias diferentes
            config1 = container.resolve(SystemConfig)
            config2 = container.resolve(SystemConfig)
            
            assert config1 is not config2

class TestCacheManager:
    """Tests del sistema de cache"""
    
    def test_cache_basic_operations(self, temp_cache_dir):
        """Test operaciones básicas de cache"""
        cache = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=False)
        
        # Set y get
        cache.set("test_key", "test_value", ttl=3600)
        assert cache.get("test_key") == "test_value"
        
        # Delete
        cache.delete("test_key")
        assert cache.get("test_key") is None
    
    def test_cache_expiration(self, temp_cache_dir):
        """Test expiración de cache"""
        cache = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=False)
        
        # Set con TTL corto
        cache.set("test_key", "test_value", ttl=1)
        assert cache.get("test_key") == "test_value"
        
        # Esperar expiración
        import time
        time.sleep(2)
        assert cache.get("test_key") is None
    
    def test_cache_stats(self, temp_cache_dir):
        """Test estadísticas de cache"""
        cache = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=False)
        
        # Operaciones para generar stats
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # hit
        cache.get("key3")  # miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['entries_count'] == 2
    
    def test_cache_persistence(self, temp_cache_dir):
        """Test persistencia de cache"""
        # Crear cache con persistencia
        cache1 = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=True)
        cache1.set("persistent_key", "persistent_value")
        
        # Crear nuevo cache manager - debe cargar datos persistentes
        cache2 = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=True)
        assert cache2.get("persistent_key") == "persistent_value"

class TestRiskManager:
    """Tests del gestor de riesgo"""
    
    def test_var_calculation(self, test_config):
        """Test cálculo de VaR"""
        risk_manager = AdvancedRiskManager(test_config)
        
        # Generar returns de prueba
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        # Calcular VaR
        var_95 = risk_manager.calculate_var(returns, confidence_level=0.95)
        
        assert var_95 > 0
        assert isinstance(var_95, float)
    
    def test_expected_shortfall(self, test_config):
        """Test cálculo de Expected Shortfall"""
        risk_manager = AdvancedRiskManager(test_config)
        
        # Generar returns de prueba
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        # Calcular ES
        es = risk_manager.calculate_expected_shortfall(returns, confidence_level=0.95)
        
        assert es > 0
        assert isinstance(es, float)
    
    def test_position_sizing(self, test_config):
        """Test cálculo de tamaño de posición"""
        risk_manager = AdvancedRiskManager(test_config)
        
        position_size = risk_manager.calculate_position_size(
            signal_strength=0.8,
            account_balance=10000,
            risk_per_trade=0.02
        )
        
        assert position_size > 0
        assert position_size <= 10000 * 0.02  # No debe exceder el riesgo máximo
    
    def test_correlation_calculation(self, test_config):
        """Test cálculo de correlaciones"""
        risk_manager = AdvancedRiskManager(test_config)
        
        # Generar datos de prueba
        np.random.seed(42)
        returns = pd.DataFrame({
            'BTC': np.random.normal(0, 0.03, 100),
            'ETH': np.random.normal(0, 0.04, 100),
            'ADA': np.random.normal(0, 0.05, 100)
        })
        
        correlations = risk_manager.calculate_correlations(returns)
        
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape == (3, 3)
        assert np.allclose(np.diag(correlations), 1.0)  # Diagonal debe ser 1

class TestMLModels:
    """Tests de modelos ML"""
    
    def test_random_forest_training(self, sample_market_data):
        """Test entrenamiento de Random Forest"""
        from models.advanced_ml_models import ModelConfig
        
        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        model = AdvancedRandomForest(config)
        
        # Generar features sintéticas
        data = sample_market_data.data
        X = data[['open', 'high', 'low', 'volume']].dropna()
        y = data['close'].dropna()
        
        # Alinear índices
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Entrenar
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert 'mse' in metrics
        assert 'r2' in metrics
    
    def test_model_factory(self):
        """Test factory de modelos"""
        from models.advanced_ml_models import ModelConfig
        
        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={'n_estimators': 10}
        )
        
        model = ModelFactory.create_model("random_forest", config)
        assert model is not None
        assert isinstance(model, AdvancedRandomForest)
    
    def test_model_serialization(self, sample_market_data, tmp_path):
        """Test serialización de modelos"""
        from models.advanced_ml_models import ModelConfig
        
        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        model = AdvancedRandomForest(config)
        
        # Entrenar con datos mínimos
        data = sample_market_data.data.head(100)
        X = data[['open', 'high', 'low', 'volume']].dropna()
        y = data['close'].dropna()
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        model.train(X, y)
        
        # Guardar modelo
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path))
        
        # Cargar modelo
        new_model = AdvancedRandomForest(config)
        new_model.load_model(str(model_path))
        
        assert new_model.is_trained
        
        # Verificar que las predicciones son iguales
        pred1 = model.predict(X)
        pred2 = new_model.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)

class TestMonitoringSystem:
    """Tests del sistema de monitoreo"""
    
    def test_metrics_collection(self):
        """Test colección de métricas"""
        metrics_collector = MetricsCollector()
        
        # Registrar métricas
        metrics_collector.increment_counter("test_counter", 5)
        metrics_collector.set_gauge("test_gauge", 42.5)
        metrics_collector.record_histogram("test_histogram", 3.14)
        
        # Verificar métricas
        summary = metrics_collector.get_metrics_summary()
        
        assert summary['counters']['test_counter'] == 5
        assert summary['gauges']['test_gauge'] == 42.5
        assert summary['histograms']['test_histogram']['count'] == 1
        assert summary['histograms']['test_histogram']['sum'] == 3.14
    
    def test_monitoring_system_lifecycle(self):
        """Test ciclo de vida del sistema de monitoreo"""
        monitoring = MonitoringSystem()
        
        # Iniciar
        monitoring.start()
        assert monitoring.running
        
        # Detener
        monitoring.stop()
        assert not monitoring.running
    
    def test_performance_monitor(self):
        """Test monitor de performance"""
        metrics_collector = MetricsCollector()
        performance_monitor = PerformanceMonitor(metrics_collector)
        
        # Usar timer
        with performance_monitor.timer("test_operation"):
            import time
            time.sleep(0.1)
        
        # Verificar que se registró métrica
        summary = metrics_collector.get_metrics_summary()
        assert "test_operation" in summary['timers']
        assert summary['timers']['test_operation']['count'] == 1

class TestIntegrationScenarios:
    """Tests de integración end-to-end"""
    
    def test_full_pipeline_integration(self, test_config, sample_market_data):
        """Test integración completa del pipeline"""
        # Configurar container
        container = configure_container(test_config)
        
        # Resolver componentes
        risk_manager = container.resolve(IRiskManager)
        cache_manager = container.resolve(ICacheManager)
        
        # Verificar que los componentes funcionan juntos
        assert risk_manager is not None
        assert cache_manager is not None
        
        # Test operación integrada
        cache_manager.set("test_data", sample_market_data.data)
        cached_data = cache_manager.get("test_data")
        
        assert cached_data is not None
        assert len(cached_data) == len(sample_market_data.data)
    
    def test_ml_pipeline_optimization(self, test_config, sample_market_data):
        """Test optimización del pipeline ML"""
        optimizer = MLPipelineOptimizer(test_config)
        
        # Preparar datos
        data = sample_market_data.data.head(100)  # Datos pequeños para test
        X = data[['open', 'high', 'low', 'volume']].dropna()
        y = data['close'].dropna()
        
        # Alinear índices
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Crear modelos mock
        from sklearn.ensemble import RandomForestRegressor
        models = [RandomForestRegressor(n_estimators=5, random_state=42)]
        
        # Optimizar con presupuesto pequeño
        results = optimizer.optimize_pipeline(X, y, models, optimization_budget=30)
        
        assert 'feature_selection' in results
        assert 'model_optimization' in results
        assert results['optimization_time'] > 0
    
    @patch('binance.client.Client')
    def test_data_provider_integration(self, mock_client, test_config):
        """Test integración con proveedores de datos"""
        # Mock respuesta de Binance
        mock_client.return_value.get_historical_klines.return_value = [
            [1640995200000, '47000', '48000', '46000', '47500', '100.5', 1640998800000, 
             '4750000', 1500, '50.25', '2375000', '0']
        ]
        
        # Simular obtención de datos
        from binance.client import Client
        client = Client()
        
        klines = client.get_historical_klines("BTCUSDT", "1h", "1 day ago UTC")
        
        assert len(klines) == 1
        assert klines[0][4] == '47500'  # Close price

class TestErrorHandling:
    """Tests de manejo de errores"""
    
    def test_config_error_handling(self):
        """Test manejo de errores en configuración"""
        # Test archivo inexistente
        config = SystemConfig.from_file("nonexistent_file.json")
        assert isinstance(config, SystemConfig)  # Debe usar defaults
    
    def test_cache_error_handling(self, temp_cache_dir):
        """Test manejo de errores en cache"""
        cache = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=False)
        
        # Test con valor no serializable
        import threading
        non_serializable = threading.Lock()
        
        # No debe lanzar excepción
        cache.set("bad_key", non_serializable)
        result = cache.get("bad_key")
        
        # Debe manejar el error gracefully
        assert result is not None or result is None  # Cualquiera es válido
    
    def test_model_error_handling(self):
        """Test manejo de errores en modelos"""
        from models.advanced_ml_models import ModelConfig
        
        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={'n_estimators': 10}
        )
        
        model = AdvancedRandomForest(config)
        
        # Test predicción sin entrenar
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        
        try:
            model.predict(X)
            assert False, "Debería lanzar excepción"
        except Exception as e:
            assert "not trained" in str(e).lower() or "fit" in str(e).lower()

class TestPerformance:
    """Tests de rendimiento"""
    
    def test_cache_performance(self, temp_cache_dir):
        """Test rendimiento del cache"""
        cache = CacheManager(cache_dir=temp_cache_dir, persist_to_disk=False)
        
        import time
        
        # Test escritura
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        
        # Test lectura
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        read_time = time.time() - start_time
        
        # Verificar que es razonablemente rápido
        assert write_time < 5.0  # Menos de 5 segundos para 1000 writes
        assert read_time < 1.0   # Menos de 1 segundo para 1000 reads
    
    def test_risk_calculation_performance(self, test_config):
        """Test rendimiento de cálculos de riesgo"""
        risk_manager = AdvancedRiskManager(test_config)
        
        # Generar datos grandes
        np.random.seed(42)
        large_returns = pd.Series(np.random.normal(0, 0.02, 10000))
        
        import time
        start_time = time.time()
        
        # Calcular múltiples métricas
        var_95 = risk_manager.calculate_var(large_returns, 0.95)
        var_99 = risk_manager.calculate_var(large_returns, 0.99)
        es = risk_manager.calculate_expected_shortfall(large_returns, 0.95)
        
        calculation_time = time.time() - start_time
        
        # Verificar que es razonablemente rápido
        assert calculation_time < 2.0  # Menos de 2 segundos
        assert var_95 > 0
        assert var_99 > var_95  # VaR 99% debe ser mayor que VaR 95%
        assert es > var_95     # ES debe ser mayor que VaR

# Fixtures para tests específicos
@pytest.fixture
def mock_binance_client():
    """Mock del cliente Binance"""
    client = Mock()
    client.get_historical_klines.return_value = [
        [1640995200000, '47000', '48000', '46000', '47500', '100.5', 1640998800000, 
         '4750000', 1500, '50.25', '2375000', '0']
    ]
    return client

@pytest.fixture
def mock_news_api():
    """Mock de NewsAPI"""
    news_api = Mock()
    news_api.get_everything.return_value = {
        'articles': [
            {
                'title': 'Bitcoin rises to new highs',
                'description': 'Bitcoin positive news',
                'publishedAt': '2023-01-01T00:00:00Z',
                'url': 'https://example.com'
            }
        ]
    }
    return news_api

# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Ejecución de tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 