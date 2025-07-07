# SISTEMA DE CONFIGURACIÓN SEGURO
"""
Sistema de configuración seguro con variables de entorno y gestión de secretos
Elimina las API keys hard-coded y mejora la seguridad del sistema
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import keyring
from cryptography.fernet import Fernet
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SecureAPIConfig:
    """Configuración segura para APIs"""
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    news_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    
    # Configuración de API
    binance_testnet: bool = True
    rate_limit_buffer: float = 0.1
    max_retries: int = 3
    timeout: int = 30

@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "trader"
    password: Optional[str] = None
    ssl_mode: str = "require"
    connection_pool_size: int = 10
    max_overflow: int = 5

@dataclass
class RedisConfig:
    """Configuración de Redis para caching"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 50

@dataclass
class RiskLimits:
    """Límites de riesgo operacional"""
    max_position_size: float = 0.02  # 2% del capital
    max_portfolio_risk: float = 0.10  # 10% VaR
    max_correlation_exposure: float = 0.30  # 30% correlación
    max_drawdown_stop: float = 0.10  # 10% stop loss
    var_limit_daily: float = 0.05  # 5% VaR diario
    max_positions_per_asset: int = 1
    max_total_positions: int = 10
    min_time_between_trades: int = 300  # 5 minutos

@dataclass
class TradingLimits:
    """Límites de trading"""
    min_order_size: float = 10.0  # USD
    max_order_size: float = 10000.0  # USD
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    max_orders_per_day: int = 500
    slippage_threshold: float = 0.005  # 0.5%
    market_impact_threshold: float = 0.001  # 0.1%

@dataclass
class MonitoringConfig:
    """Configuración de monitoreo"""
    log_level: str = "INFO"
    metrics_interval: int = 60  # segundos
    health_check_interval: int = 30  # segundos
    alert_channels: Dict[str, str] = field(default_factory=dict)
    performance_tracking: bool = True
    system_metrics: bool = True

@dataclass
class MLConfig:
    """Configuración de Machine Learning"""
    model_retrain_interval: int = 168  # horas (1 semana)
    min_training_samples: int = 1000
    validation_split: float = 0.2
    walk_forward_window: int = 252  # días
    feature_selection_threshold: float = 0.05
    hyperparameter_tuning_trials: int = 100
    ensemble_weights_adjustment: bool = True

class SecureConfigManager:
    """Gestor de configuración seguro"""
    
    def __init__(self, config_path: str = "config/production.json"):
        self.config_path = Path(config_path)
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self._load_config()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Obtiene o crea clave de encriptación"""
        try:
            key = keyring.get_password("trading_system", "encryption_key")
            if key:
                return base64.urlsafe_b64decode(key.encode())
            else:
                # Crear nueva clave
                key = Fernet.generate_key()
                keyring.set_password("trading_system", "encryption_key", 
                                   base64.urlsafe_b64encode(key).decode())
                return key
        except Exception as e:
            logger.warning(f"Error con keyring: {e}. Usando clave temporal.")
            return Fernet.generate_key()
    
    def _load_config(self):
        """Carga configuración desde múltiples fuentes"""
        # 1. Configuración base
        self.api_config = SecureAPIConfig()
        self.db_config = DatabaseConfig()
        self.redis_config = RedisConfig()
        self.risk_limits = RiskLimits()
        self.trading_limits = TradingLimits()
        self.monitoring_config = MonitoringConfig()
        self.ml_config = MLConfig()
        
        # 2. Cargar desde archivo de configuración
        self._load_from_file()
        
        # 3. Cargar desde variables de entorno (prioritario)
        self._load_from_environment()
        
        # 4. Cargar secretos desde keyring
        self._load_secrets()
    
    def _load_from_file(self):
        """Carga configuración desde archivo JSON"""
        if not self.config_path.exists():
            logger.info(f"Archivo de configuración no encontrado: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Actualizar configuraciones
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api_config, key):
                        setattr(self.api_config, key, value)
            
            if 'database' in config_data:
                for key, value in config_data['database'].items():
                    if hasattr(self.db_config, key):
                        setattr(self.db_config, key, value)
            
            if 'redis' in config_data:
                for key, value in config_data['redis'].items():
                    if hasattr(self.redis_config, key):
                        setattr(self.redis_config, key, value)
            
            if 'risk_limits' in config_data:
                for key, value in config_data['risk_limits'].items():
                    if hasattr(self.risk_limits, key):
                        setattr(self.risk_limits, key, value)
            
            if 'trading_limits' in config_data:
                for key, value in config_data['trading_limits'].items():
                    if hasattr(self.trading_limits, key):
                        setattr(self.trading_limits, key, value)
            
            if 'monitoring' in config_data:
                for key, value in config_data['monitoring'].items():
                    if hasattr(self.monitoring_config, key):
                        setattr(self.monitoring_config, key, value)
            
            if 'ml' in config_data:
                for key, value in config_data['ml'].items():
                    if hasattr(self.ml_config, key):
                        setattr(self.ml_config, key, value)
            
            logger.info(f"Configuración cargada desde: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
    
    def _load_from_environment(self):
        """Carga configuración desde variables de entorno"""
        
        # API Configuration
        self.api_config.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.api_config.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        self.api_config.news_api_key = os.getenv('NEWS_API_KEY')
        self.api_config.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Environment settings
        if os.getenv('BINANCE_TESTNET'):
            self.api_config.binance_testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Database
        if os.getenv('DATABASE_URL'):
            # Parse DATABASE_URL
            import urllib.parse
            url = urllib.parse.urlparse(os.getenv('DATABASE_URL'))
            self.db_config.host = url.hostname
            self.db_config.port = url.port
            self.db_config.database = url.path[1:]  # Remove leading /
            self.db_config.username = url.username
            self.db_config.password = url.password
        
        # Redis
        if os.getenv('REDIS_URL'):
            import urllib.parse
            url = urllib.parse.urlparse(os.getenv('REDIS_URL'))
            self.redis_config.host = url.hostname
            self.redis_config.port = url.port
            self.redis_config.password = url.password
        
        # Risk limits from environment
        for attr in ['max_position_size', 'max_portfolio_risk', 'max_drawdown_stop']:
            env_val = os.getenv(f'RISK_{attr.upper()}')
            if env_val:
                setattr(self.risk_limits, attr, float(env_val))
        
        # Trading limits
        for attr in ['min_order_size', 'max_order_size', 'max_orders_per_day']:
            env_val = os.getenv(f'TRADING_{attr.upper()}')
            if env_val:
                setattr(self.trading_limits, attr, float(env_val) if '.' in env_val else int(env_val))
        
        # Monitoring
        if os.getenv('LOG_LEVEL'):
            self.monitoring_config.log_level = os.getenv('LOG_LEVEL')
        
        logger.info("Configuración cargada desde variables de entorno")
    
    def _load_secrets(self):
        """Carga secretos desde keyring"""
        try:
            # Intentar cargar secretos desde keyring
            if not self.api_config.binance_api_key:
                key = keyring.get_password("trading_system", "binance_api_key")
                if key:
                    self.api_config.binance_api_key = self._decrypt_secret(key)
            
            if not self.api_config.binance_api_secret:
                secret = keyring.get_password("trading_system", "binance_api_secret")
                if secret:
                    self.api_config.binance_api_secret = self._decrypt_secret(secret)
            
            if not self.db_config.password:
                password = keyring.get_password("trading_system", "db_password")
                if password:
                    self.db_config.password = self._decrypt_secret(password)
            
            if not self.redis_config.password:
                password = keyring.get_password("trading_system", "redis_password")
                if password:
                    self.redis_config.password = self._decrypt_secret(password)
            
        except Exception as e:
            logger.warning(f"Error cargando secretos: {e}")
    
    def _encrypt_secret(self, secret: str) -> str:
        """Encripta un secreto"""
        return self.fernet.encrypt(secret.encode()).decode()
    
    def _decrypt_secret(self, encrypted_secret: str) -> str:
        """Desencripta un secreto"""
        return self.fernet.decrypt(encrypted_secret.encode()).decode()
    
    def store_secret(self, key: str, secret: str):
        """Almacena un secreto de forma segura"""
        try:
            encrypted_secret = self._encrypt_secret(secret)
            keyring.set_password("trading_system", key, encrypted_secret)
            logger.info(f"Secreto almacenado: {key}")
        except Exception as e:
            logger.error(f"Error almacenando secreto {key}: {e}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Valida la configuración y retorna un reporte"""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'configuration_summary': {}
        }
        
        # Validar API keys
        if not self.api_config.binance_api_key:
            validation_report['errors'].append("Binance API key no configurada")
            validation_report['valid'] = False
        
        if not self.api_config.binance_api_secret:
            validation_report['errors'].append("Binance API secret no configurado")
            validation_report['valid'] = False
        
        # Validar límites de riesgo
        if self.risk_limits.max_position_size > 0.1:
            validation_report['warnings'].append("Max position size muy alto (>10%)")
        
        if self.risk_limits.max_portfolio_risk > 0.2:
            validation_report['warnings'].append("Max portfolio risk muy alto (>20%)")
        
        # Validar configuración de base de datos
        if self.db_config.password is None:
            validation_report['warnings'].append("Database password no configurada")
        
        # Resumen de configuración
        validation_report['configuration_summary'] = {
            'environment': 'testnet' if self.api_config.binance_testnet else 'production',
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                'max_drawdown_stop': self.risk_limits.max_drawdown_stop
            },
            'trading_limits': {
                'max_order_size': self.trading_limits.max_order_size,
                'max_orders_per_day': self.trading_limits.max_orders_per_day
            },
            'monitoring': {
                'log_level': self.monitoring_config.log_level
            }
        }
        
        return validation_report
    
    def get_api_config(self) -> SecureAPIConfig:
        """Retorna configuración de API"""
        return self.api_config
    
    def get_risk_limits(self) -> RiskLimits:
        """Retorna límites de riesgo"""
        return self.risk_limits
    
    def get_trading_limits(self) -> TradingLimits:
        """Retorna límites de trading"""
        return self.trading_limits
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Retorna configuración de monitoreo"""
        return self.monitoring_config
    
    def get_ml_config(self) -> MLConfig:
        """Retorna configuración de ML"""
        return self.ml_config
    
    def export_config_template(self, path: str = "config/production_template.json"):
        """Exporta template de configuración"""
        template = {
            "api": {
                "binance_testnet": True,
                "rate_limit_buffer": 0.1,
                "max_retries": 3,
                "timeout": 30
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "trading_system",
                "username": "trader",
                "ssl_mode": "require"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "ssl": False
            },
            "risk_limits": {
                "max_position_size": 0.02,
                "max_portfolio_risk": 0.10,
                "max_correlation_exposure": 0.30,
                "max_drawdown_stop": 0.10,
                "var_limit_daily": 0.05
            },
            "trading_limits": {
                "min_order_size": 10.0,
                "max_order_size": 10000.0,
                "max_orders_per_day": 500,
                "slippage_threshold": 0.005
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval": 60,
                "health_check_interval": 30,
                "performance_tracking": True
            },
            "ml": {
                "model_retrain_interval": 168,
                "min_training_samples": 1000,
                "validation_split": 0.2,
                "walk_forward_window": 252
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Template de configuración exportado a: {path}")

# Singleton para acceso global
_config_manager = None

def get_config_manager() -> SecureConfigManager:
    """Obtiene el gestor de configuración (singleton)"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecureConfigManager()
    return _config_manager

def initialize_config(config_path: str = None):
    """Inicializa el gestor de configuración"""
    global _config_manager
    if config_path:
        _config_manager = SecureConfigManager(config_path)
    else:
        _config_manager = SecureConfigManager()
    return _config_manager 