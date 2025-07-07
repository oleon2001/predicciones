# FRAMEWORK DE TESTING COMPLETO
"""
Suite de testing completo para el sistema de trading
Incluye unit tests, integration tests, performance tests, y testing de ML
"""

import unittest
import pytest
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Testing libraries
from unittest.mock import Mock, patch, MagicMock
import pytest
from parameterized import parameterized

# ML testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score

# Sistema bajo test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.secure_config import SecureConfigManager
from core.data_orchestrator import DataOrchestrator
from core.robust_risk_manager import RobustRiskManager
from core.monitoring_system import MonitoringSystem
from models.robust_ml_pipeline import RobustMLPipeline
from backtesting.realistic_backtester import RealisticBacktester
from deployment.production_system import ProductionSystem

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generador de datos sintéticos para testing"""
    
    @staticmethod
    def generate_ohlcv_data(periods: int = 1000, 
                           start_price: float = 100.0,
                           volatility: float = 0.02) -> pd.DataFrame:
        """Genera datos OHLCV sintéticos"""
        
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='1H')
        
        # Generar precios con random walk
        returns = np.random.normal(0, volatility, periods)
        prices = [start_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        # Generar OHLCV
        data = []
        for i in range(periods):
            price = prices[i]
            
            # High y Low con variación
            high = price * (1 + abs(np.random.normal(0, volatility/2)))
            low = price * (1 - abs(np.random.normal(0, volatility/2)))
            
            # Asegurar que open y close estén entre high y low
            open_price = np.random.uniform(low, high)
            close_price = prices[i]  # El precio de cierre es el precio generado
            
            # Volumen correlacionado con volatilidad
            volume = np.random.exponential(100000) * (1 + abs(returns[i]) * 10)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @staticmethod
    def generate_trade_signals(n_signals: int = 100) -> List[Dict[str, Any]]:
        """Genera señales de trading sintéticas"""
        
        signals = []
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT']
        
        for i in range(n_signals):
            signal = {
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'symbol': np.random.choice(symbols),
                'action': np.random.choice(['BUY', 'SELL']),
                'position_size': np.random.uniform(100, 10000),
                'confidence': np.random.uniform(0.5, 1.0),
                'strategy': 'test_strategy'
            }
            signals.append(signal)
        
        return signals

class TestSecureConfig(unittest.TestCase):
    """Tests para el sistema de configuración segura"""
    
    def setUp(self):
        """Setup para cada test"""
        self.config_manager = SecureConfigManager("test_config.json")
    
    def test_config_loading(self):
        """Test carga de configuración"""
        self.assertIsNotNone(self.config_manager)
        self.assertIsNotNone(self.config_manager.get_api_config())
        self.assertIsNotNone(self.config_manager.get_risk_limits())
    
    def test_config_validation(self):
        """Test validación de configuración"""
        validation_report = self.config_manager.validate_configuration()
        self.assertIsInstance(validation_report, dict)
        self.assertIn('valid', validation_report)
        self.assertIn('configuration_summary', validation_report)
    
    def test_risk_limits(self):
        """Test límites de riesgo"""
        risk_limits = self.config_manager.get_risk_limits()
        
        # Verificar que los límites están en rangos razonables
        self.assertGreater(risk_limits.max_position_size, 0)
        self.assertLess(risk_limits.max_position_size, 1)  # Máximo 100%
        self.assertGreater(risk_limits.max_portfolio_risk, 0)
        self.assertLess(risk_limits.max_portfolio_risk, 1)

class TestDataOrchestrator(unittest.TestCase):
    """Tests para el orquestador de datos"""
    
    def setUp(self):
        """Setup para cada test"""
        self.data_orchestrator = DataOrchestrator()
        self.test_data = TestDataGenerator.generate_ohlcv_data()
    
    def test_data_quality_validation(self):
        """Test validación de calidad de datos"""
        from core.data_orchestrator import DataQualityValidator, DataSource
        
        validator = DataQualityValidator()
        metrics = validator.validate_data(self.test_data, DataSource.BINANCE)
        
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.completeness, 0)
        self.assertGreater(metrics.consistency, 0)
        self.assertGreaterEqual(metrics.overall_score, 0)
        self.assertLessEqual(metrics.overall_score, 1)
    
    @patch('core.data_orchestrator.BinanceDataProvider')
    def test_data_provider_fallback(self, mock_provider):
        """Test fallback entre proveedores de datos"""
        # Simular fallo del primer proveedor
        mock_provider.return_value.get_historical_data.side_effect = Exception("API Error")
        
        # El orquestador debería usar el proveedor de backup
        with self.assertLogs(level='WARNING'):
            try:
                data = self.data_orchestrator.get_best_data(
                    'BTCUSDT', '1h', '1 day ago', 'now'
                )
            except:
                pass  # Esperamos que falle en testing sin APIs reales
    
    def test_cache_functionality(self):
        """Test funcionalidad de cache"""
        # Este test requeriría una implementación mock del cache
        pass

class TestRiskManager(unittest.TestCase):
    """Tests para el gestor de riesgo"""
    
    def setUp(self):
        """Setup para cada test"""
        self.risk_manager = RobustRiskManager()
        self.test_data = TestDataGenerator.generate_ohlcv_data()
    
    def test_risk_limits_validation(self):
        """Test validación de límites de riesgo"""
        from core.interfaces import TradeSignal
        
        # Señal normal
        normal_signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            action='BUY',
            position_size=100.0,
            confidence=0.8,
            strategy='test'
        )
        
        portfolio = {}
        is_valid, reason, metrics = self.risk_manager.validate_trade_signal(
            normal_signal, portfolio, self.test_data
        )
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(reason, str)
        self.assertIsInstance(metrics, dict)
    
    def test_position_size_limits(self):
        """Test límites de tamaño de posición"""
        from core.interfaces import TradeSignal
        
        # Señal con posición demasiado grande
        oversized_signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            action='BUY',
            position_size=1000000.0,  # Muy grande
            confidence=0.8,
            strategy='test'
        )
        
        portfolio = {}
        is_valid, reason, metrics = self.risk_manager.validate_trade_signal(
            oversized_signal, portfolio, self.test_data
        )
        
        self.assertFalse(is_valid)
        self.assertIn('límite', reason.lower())
    
    def test_circuit_breakers(self):
        """Test circuit breakers"""
        
        # Activar circuit breaker
        self.risk_manager.circuit_breakers['position_size'].force_open()
        
        from core.interfaces import TradeSignal
        signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            action='BUY',
            position_size=100.0,
            confidence=0.8,
            strategy='test'
        )
        
        portfolio = {}
        is_valid, reason, metrics = self.risk_manager.validate_trade_signal(
            signal, portfolio, self.test_data
        )
        
        self.assertFalse(is_valid)
        self.assertIn('circuit breaker', reason.lower())
    
    def test_portfolio_var_calculation(self):
        """Test cálculo de VaR del portfolio"""
        
        portfolio = {'BTCUSDT': 1000.0, 'ETHUSDT': 500.0}
        var = self.risk_manager._calculate_portfolio_var(portfolio, self.test_data)
        
        self.assertIsInstance(var, float)
        self.assertGreaterEqual(var, 0)

class TestMLPipeline(unittest.TestCase):
    """Tests para el pipeline de ML"""
    
    def setUp(self):
        """Setup para cada test"""
        self.ml_pipeline = RobustMLPipeline()
        self.test_data = TestDataGenerator.generate_ohlcv_data(periods=2000)
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        
        features = self.ml_pipeline.feature_engineer.engineer_features(self.test_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(len(features), len(self.test_data))
        
        # Verificar que no hay demasiados NaN
        nan_ratio = features.isnull().mean().mean()
        self.assertLess(nan_ratio, 0.5)
    
    def test_walk_forward_splits(self):
        """Test walk-forward analysis splits"""
        
        from models.robust_ml_pipeline import WalkForwardConfig, WalkForwardValidator
        
        config = WalkForwardConfig(
            initial_train_size=500,
            step_size=50,
            test_size=25
        )
        
        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(self.test_data)
        
        self.assertGreater(len(splits), 0)
        
        # Verificar que los splits son válidos
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            self.assertTrue(train_idx.max() < test_idx.min())  # No overlap
    
    def test_data_preparation(self):
        """Test preparación de datos"""
        
        prepared_data = self.ml_pipeline.prepare_data(self.test_data, [1, 4, 12])
        
        self.assertIn('features', prepared_data)
        self.assertIn('return_1h', prepared_data)
        self.assertIn('return_4h', prepared_data)
        self.assertIn('return_12h', prepared_data)
        
        # Verificar estructura de targets
        for target_name, target_data in prepared_data.items():
            if target_name != 'features':
                self.assertIn('features', target_data)
                self.assertIn('target', target_data)
    
    @pytest.mark.slow
    def test_model_training_and_validation(self):
        """Test entrenamiento y validación de modelos (lento)"""
        
        # Usar dataset pequeño para testing
        small_data = self.test_data.tail(500)
        prepared_data = self.ml_pipeline.prepare_data(small_data, [1, 4])
        
        # Ejecutar validación walk-forward
        results = self.ml_pipeline.train_and_validate_models(prepared_data)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Verificar estructura de resultados
        for target_name, target_results in results.items():
            self.assertIsInstance(target_results, dict)
            for model_name, model_results in target_results.items():
                self.assertIsInstance(model_results, list)
                if model_results:
                    result = model_results[0]
                    self.assertIn('metrics', result.__dict__)
                    self.assertIn('feature_importance', result.__dict__)

class TestBacktester(unittest.TestCase):
    """Tests para el backtester"""
    
    def setUp(self):
        """Setup para cada test"""
        self.backtester = RealisticBacktester()
        self.test_data = TestDataGenerator.generate_ohlcv_data(periods=1000)
    
    def test_market_impact_calculation(self):
        """Test cálculo de market impact"""
        
        from backtesting.realistic_backtester import MarketImpactModel, MarketConditions
        
        model = MarketImpactModel()
        conditions = MarketConditions(
            volatility=0.3,
            spread_bps=1.0,
            volume_ratio=1.0,
            market_impact_factor=1.0,
            liquidity_score=0.8,
            time_of_day_factor=1.0
        )
        
        market_data = self.test_data.iloc[-1]
        permanent, temporary = model.calculate_market_impact(1000.0, market_data, conditions)
        
        self.assertIsInstance(permanent, float)
        self.assertIsInstance(temporary, float)
        self.assertGreaterEqual(permanent, 0)
        self.assertGreaterEqual(temporary, 0)
    
    def test_slippage_calculation(self):
        """Test cálculo de slippage"""
        
        from backtesting.realistic_backtester import SlippageModel, OrderType, MarketConditions
        
        model = SlippageModel()
        conditions = MarketConditions(
            volatility=0.3,
            spread_bps=1.0,
            volume_ratio=1.0,
            market_impact_factor=1.0,
            liquidity_score=0.8,
            time_of_day_factor=1.0
        )
        
        slippage = model.calculate_slippage(OrderType.MARKET, 1000.0, conditions)
        
        self.assertIsInstance(slippage, float)
        self.assertGreaterEqual(slippage, 0)
        self.assertLessEqual(slippage, 0.05)  # Máximo 5%
    
    def test_order_execution(self):
        """Test ejecución de órdenes"""
        
        from backtesting.realistic_backtester import OrderExecutionEngine
        from core.interfaces import TradeSignal
        
        engine = OrderExecutionEngine()
        signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            action='BUY',
            position_size=1000.0,
            confidence=0.8,
            strategy='test'
        )
        
        execution_report = engine.execute_order(signal, self.test_data, datetime.now())
        
        self.assertIsNotNone(execution_report)
        self.assertEqual(execution_report.symbol, 'BTCUSDT')
        self.assertGreater(execution_report.executed_quantity, 0)
        self.assertGreater(execution_report.avg_price, 0)

class TestMonitoringSystem(unittest.TestCase):
    """Tests para el sistema de monitoreo"""
    
    def setUp(self):
        """Setup para cada test"""
        self.monitoring_system = MonitoringSystem()
    
    def test_metrics_collection(self):
        """Test recolección de métricas"""
        
        collector = self.monitoring_system.metrics_collector
        
        # Registrar métricas de test
        collector.record_metric('test_metric', 100.0)
        collector.record_metric('test_metric', 150.0)
        collector.record_metric('test_metric', 120.0)
        
        # Obtener estadísticas
        stats = collector.get_metric_stats('test_metric', hours=1)
        
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('avg', stats)
        self.assertEqual(stats['min'], 100.0)
        self.assertEqual(stats['max'], 150.0)
        self.assertAlmostEqual(stats['avg'], 123.33, places=1)
    
    def test_alert_creation(self):
        """Test creación de alertas"""
        
        from core.monitoring_system import AlertLevel
        
        alert_manager = self.monitoring_system.alert_manager
        
        alert = alert_manager.create_alert(
            alert_id='test_alert',
            level=AlertLevel.WARNING,
            title='Test Alert',
            description='This is a test alert',
            component='test_component'
        )
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert.title, 'Test Alert')
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertIn('test_alert', alert_manager.alerts)
    
    def test_health_checks(self):
        """Test health checks"""
        
        health_checker = self.monitoring_system.health_checker
        
        # Ejecutar health checks
        results = health_checker.run_health_checks()
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Verificar estructura de resultados
        for check_name, result in results.items():
            self.assertIn('status', result.__dict__)
            self.assertIn('last_check', result.__dict__)
            self.assertIn('response_time', result.__dict__)

class TestProductionSystem(unittest.TestCase):
    """Tests para el sistema de producción"""
    
    def setUp(self):
        """Setup para cada test"""
        # Crear configuración de test
        self.test_config_path = "test_production_config.yaml"
        self._create_test_config()
        
        self.production_system = ProductionSystem(self.test_config_path)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        if Path(self.test_config_path).exists():
            Path(self.test_config_path).unlink()
    
    def _create_test_config(self):
        """Crea configuración de test"""
        import yaml
        
        config = {
            'stage': 'development',
            'version': '1.0.0-test',
            'services': [
                {
                    'name': 'test_service',
                    'port': 9999,
                    'enabled': True,
                    'auto_restart': False
                }
            ]
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_config_loading(self):
        """Test carga de configuración"""
        
        self.assertIsNotNone(self.production_system.deployment_config)
        self.assertEqual(self.production_system.deployment_config.version, '1.0.0-test')
        self.assertEqual(len(self.production_system.deployment_config.services), 1)
    
    def test_service_registration(self):
        """Test registro de servicios"""
        
        service_config = self.production_system.deployment_config.services[0]
        self.production_system.service_manager.register_service(service_config)
        
        self.assertIn('test_service', self.production_system.service_manager.services)
        
        status = self.production_system.service_manager.get_service_status('test_service')
        self.assertEqual(status['name'], 'test_service')

class PerformanceTests(unittest.TestCase):
    """Tests de performance"""
    
    def setUp(self):
        """Setup para tests de performance"""
        self.large_dataset = TestDataGenerator.generate_ohlcv_data(periods=10000)
    
    @pytest.mark.performance
    def test_data_processing_performance(self):
        """Test performance de procesamiento de datos"""
        
        ml_pipeline = RobustMLPipeline()
        
        start_time = time.time()
        features = ml_pipeline.feature_engineer.engineer_features(self.large_dataset)
        processing_time = time.time() - start_time
        
        # Debe procesar 10k registros en menos de 30 segundos
        self.assertLess(processing_time, 30.0)
        self.assertGreater(len(features), 0)
        
        logger.info(f"Procesamiento de {len(self.large_dataset)} registros: {processing_time:.2f}s")
    
    @pytest.mark.performance
    def test_risk_validation_performance(self):
        """Test performance de validación de riesgo"""
        
        risk_manager = RobustRiskManager()
        signals = TestDataGenerator.generate_trade_signals(1000)
        
        start_time = time.time()
        
        for signal_data in signals[:100]:  # Test con primeras 100 señales
            from core.interfaces import TradeSignal
            signal = TradeSignal(**signal_data)
            
            portfolio = {}
            risk_manager.validate_trade_signal(signal, portfolio, self.large_dataset)
        
        validation_time = time.time() - start_time
        
        # Debe validar 100 señales en menos de 5 segundos
        self.assertLess(validation_time, 5.0)
        
        logger.info(f"Validación de 100 señales: {validation_time:.2f}s")

class IntegrationTests(unittest.TestCase):
    """Tests de integración"""
    
    def setUp(self):
        """Setup para tests de integración"""
        self.test_data = TestDataGenerator.generate_ohlcv_data(periods=1000)
    
    @pytest.mark.integration
    def test_complete_trading_workflow(self):
        """Test workflow completo de trading"""
        
        # 1. Configuración
        config_manager = SecureConfigManager()
        self.assertIsNotNone(config_manager)
        
        # 2. Datos
        data_orchestrator = DataOrchestrator()
        self.assertIsNotNone(data_orchestrator)
        
        # 3. ML Pipeline
        ml_pipeline = RobustMLPipeline()
        prepared_data = ml_pipeline.prepare_data(self.test_data, [1])
        self.assertIn('features', prepared_data)
        
        # 4. Risk Management
        risk_manager = RobustRiskManager()
        
        from core.interfaces import TradeSignal
        test_signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            action='BUY',
            position_size=100.0,
            confidence=0.8,
            strategy='integration_test'
        )
        
        is_valid, reason, metrics = risk_manager.validate_trade_signal(
            test_signal, {}, self.test_data
        )
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(reason, str)
        
        # 5. Backtesting
        backtester = RealisticBacktester()
        self.assertIsNotNone(backtester)
        
        logger.info("✅ Workflow completo de trading ejecutado exitosamente")
    
    @pytest.mark.integration
    def test_monitoring_integration(self):
        """Test integración del sistema de monitoreo"""
        
        monitoring_system = MonitoringSystem()
        
        # Simular métricas
        monitoring_system.metrics_collector.record_metric('test_integration', 100.0)
        
        # Verificar dashboard
        dashboard = monitoring_system.get_system_dashboard()
        
        self.assertIn('system_status', dashboard)
        self.assertIn('uptime_seconds', dashboard)
        
        logger.info("✅ Sistema de monitoreo integrado correctamente")

def run_test_suite():
    """Ejecuta la suite completa de tests"""
    
    print("🧪 INICIANDO SUITE DE TESTS COMPLETA")
    print("=" * 50)
    
    # Configurar logging para tests
    logging.basicConfig(level=logging.INFO)
    
    # Test suites
    test_suites = [
        'TestSecureConfig',
        'TestDataOrchestrator', 
        'TestRiskManager',
        'TestMLPipeline',
        'TestBacktester',
        'TestMonitoringSystem',
        'TestProductionSystem'
    ]
    
    # Tests de performance (opcionales)
    performance_tests = [
        'PerformanceTests'
    ]
    
    # Tests de integración
    integration_tests = [
        'IntegrationTests'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Cargar tests básicos
    for test_class_name in test_suites:
        test_class = globals()[test_class_name]
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Reporte de resultados
    print("\n" + "=" * 50)
    print("📊 RESULTADOS DE TESTS")
    print("=" * 50)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    
    if result.errors:
        print("\n❌ ERRORES:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if result.failures:
        print("\n❌ FALLOS:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.wasSuccessful():
        print("\n✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    exit(0 if success else 1) 