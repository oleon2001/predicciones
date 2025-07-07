#!/usr/bin/env python3
"""
SCRIPT DE MIGRACIÓN DE CONFIGURACIÓN
Migra de configuración hard-coded a sistema seguro
"""

import os
import json
import logging
from pathlib import Path
from secure_config import SecureConfigManager

logger = logging.getLogger(__name__)

class ConfigMigrator:
    """Migrador de configuración"""
    
    def __init__(self):
        self.legacy_config = self._extract_legacy_config()
        self.config_manager = SecureConfigManager()
    
    def _extract_legacy_config(self) -> dict:
        """Extrae configuración legacy del código"""
        legacy_config = {}
        
        # Extraer de prediccion_avanzada.py
        try:
            from prediccion_avanzada import ConfiguracionAvanzada
            config = ConfiguracionAvanzada()
            
            legacy_config['api'] = {
                'binance_api_key': config.API_KEY,
                'binance_api_secret': config.API_SECRET,
                'news_api_key': config.NEWS_API_KEY,
                'binance_testnet': False  # Asumir producción
            }
            
            legacy_config['trading'] = {
                'prediction_horizons': config.HORIZONTES_PREDICCION,
                'training_interval': config.INTERVALO_ENTRENAMIENTO,
                'data_period': config.PERIODO_DATOS,
                'priority_pairs': config.PARES_PRIORITARIOS,
                'min_confidence': config.CONFIANZA_MINIMA,
                'strong_prediction_threshold': config.UMBRAL_PREDICCION_FUERTE
            }
            
            legacy_config['ml'] = {
                'lstm_epochs': config.LSTM_EPOCHS,
                'lstm_batch_size': config.LSTM_BATCH_SIZE,
                'sequence_length': config.SEQUENCE_LENGTH
            }
            
            logger.info("Configuración legacy extraída exitosamente")
            
        except ImportError:
            logger.warning("No se pudo importar configuración legacy")
            legacy_config = self._create_default_legacy_config()
        
        return legacy_config
    
    def _create_default_legacy_config(self) -> dict:
        """Crea configuración legacy por defecto"""
        return {
            'api': {
                'binance_api_key': '',
                'binance_api_secret': '',
                'news_api_key': '',
                'binance_testnet': True
            },
            'trading': {
                'prediction_horizons': [1, 4, 12, 24],
                'training_interval': '1h',
                'data_period': '180 day ago UTC',
                'priority_pairs': [
                    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"
                ],
                'min_confidence': 0.7,
                'strong_prediction_threshold': 0.8
            },
            'ml': {
                'lstm_epochs': 50,
                'lstm_batch_size': 32,
                'sequence_length': 60
            }
        }
    
    def migrate_to_secure_config(self):
        """Migra a configuración segura"""
        
        print("🔄 INICIANDO MIGRACIÓN DE CONFIGURACIÓN")
        print("="*50)
        
        # 1. Crear archivo de configuración
        self._create_production_config()
        
        # 2. Migrar secretos a keyring
        self._migrate_secrets()
        
        # 3. Crear archivo .env template
        self._create_env_template()
        
        # 4. Crear script de configuración inicial
        self._create_setup_script()
        
        # 5. Validar configuración migrada
        self._validate_migration()
        
        print("\n✅ MIGRACIÓN COMPLETADA")
        print("="*50)
        print("📋 PRÓXIMOS PASOS:")
        print("1. Ejecutar: python config/setup_config.py")
        print("2. Configurar variables de entorno en .env")
        print("3. Ejecutar: python config/test_config.py")
    
    def _create_production_config(self):
        """Crea archivo de configuración de producción"""
        
        production_config = {
            "api": {
                "binance_testnet": True,  # Empezar en testnet
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
                "var_limit_daily": 0.05,
                "max_positions_per_asset": 1,
                "max_total_positions": 10,
                "min_time_between_trades": 300
            },
            "trading_limits": {
                "min_order_size": 10.0,
                "max_order_size": 1000.0,  # Conservador inicialmente
                "max_orders_per_minute": 5,
                "max_orders_per_hour": 50,
                "max_orders_per_day": 200,
                "slippage_threshold": 0.005,
                "market_impact_threshold": 0.001
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval": 60,
                "health_check_interval": 30,
                "performance_tracking": True,
                "system_metrics": True
            },
            "ml": {
                "model_retrain_interval": 168,
                "min_training_samples": 1000,
                "validation_split": 0.2,
                "walk_forward_window": 252,
                "feature_selection_threshold": 0.05,
                "hyperparameter_tuning_trials": 100,
                "ensemble_weights_adjustment": True
            }
        }
        
        # Incorporar configuración legacy
        if 'ml' in self.legacy_config:
            production_config['ml'].update({
                'lstm_epochs': self.legacy_config['ml'].get('lstm_epochs', 50),
                'lstm_batch_size': self.legacy_config['ml'].get('lstm_batch_size', 32),
                'sequence_length': self.legacy_config['ml'].get('sequence_length', 60)
            })
        
        # Guardar configuración
        config_path = Path("config/production.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(production_config, f, indent=2)
        
        print(f"✅ Archivo de configuración creado: {config_path}")
    
    def _migrate_secrets(self):
        """Migra secretos a keyring"""
        
        if 'api' in self.legacy_config:
            api_config = self.legacy_config['api']
            
            # Migrar API keys si están presentes
            if api_config.get('binance_api_key') and api_config['binance_api_key'] != '':
                print("🔐 Migrando Binance API key...")
                # No almacenar las keys reales del código por seguridad
                print("⚠️  API keys detectadas en código - deben configurarse manualmente")
            
            if api_config.get('news_api_key') and api_config['news_api_key'] != '':
                print("🔐 Migrando News API key...")
                # No almacenar las keys reales del código por seguridad
                print("⚠️  News API key detectada en código - debe configurarse manualmente")
        
        print("✅ Secretos preparados para migración manual")
    
    def _create_env_template(self):
        """Crea template de variables de entorno"""
        
        env_template = """# CONFIGURACIÓN DE ENTORNO - SISTEMA DE TRADING
# Copia este archivo a .env y configura los valores

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Binance API (NUNCA COMPARTIR ESTAS KEYS)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# News API
NEWS_API_KEY=your_news_api_key_here

# FRED API (Federal Reserve Economic Data)
FRED_API_KEY=your_fred_api_key_here

# Environment
BINANCE_TESTNET=true
ENVIRONMENT=development

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# PostgreSQL
DATABASE_URL=postgresql://trader:password@localhost:5432/trading_system

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
# Redis para caching
REDIS_URL=redis://localhost:6379/0

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
# Límites de riesgo (0.01 = 1%)
RISK_MAX_POSITION_SIZE=0.02
RISK_MAX_PORTFOLIO_RISK=0.10
RISK_MAX_DRAWDOWN_STOP=0.10

# =============================================================================
# TRADING LIMITS
# =============================================================================
# Límites de trading
TRADING_MIN_ORDER_SIZE=10.0
TRADING_MAX_ORDER_SIZE=1000.0
TRADING_MAX_ORDERS_PER_DAY=200

# =============================================================================
# MONITORING
# =============================================================================
# Logging
LOG_LEVEL=INFO

# Alertas
ALERT_EMAIL=your_email@example.com
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
ALERT_TELEGRAM_BOT_TOKEN=your_telegram_bot_token
ALERT_TELEGRAM_CHAT_ID=your_telegram_chat_id

# =============================================================================
# SECURITY
# =============================================================================
# Encryption
ENCRYPTION_KEY_ID=trading_system_v1

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# =============================================================================
# DEVELOPMENT
# =============================================================================
# Solo para desarrollo
DEBUG=false
TESTING=false
"""
        
        env_path = Path(".env.template")
        with open(env_path, 'w') as f:
            f.write(env_template)
        
        print(f"✅ Template de variables de entorno creado: {env_path}")
    
    def _create_setup_script(self):
        """Crea script de configuración inicial"""
        
        setup_script = '''#!/usr/bin/env python3
"""
SCRIPT DE CONFIGURACIÓN INICIAL
Configura el sistema de trading por primera vez
"""

import os
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from config.secure_config import SecureConfigManager
import keyring

def main():
    print("🚀 CONFIGURACIÓN INICIAL DEL SISTEMA DE TRADING")
    print("="*60)
    
    # Verificar variables de entorno
    check_environment()
    
    # Configurar secretos
    setup_secrets()
    
    # Validar configuración
    validate_config()
    
    print("\\n✅ CONFIGURACIÓN INICIAL COMPLETADA")
    print("="*60)
    print("📋 El sistema está listo para usar")

def check_environment():
    """Verifica variables de entorno"""
    print("\\n🔍 VERIFICANDO VARIABLES DE ENTORNO")
    print("-" * 40)
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]
    
    optional_vars = [
        'NEWS_API_KEY',
        'FRED_API_KEY',
        'DATABASE_URL',
        'REDIS_URL'
    ]
    
    missing_required = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Configurado")
        else:
            print(f"❌ {var}: No configurado")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var}: Configurado")
        else:
            print(f"⚠️  {var}: No configurado (opcional)")
    
    if missing_required:
        print(f"\\n❌ Variables requeridas faltantes: {', '.join(missing_required)}")
        print("📋 Configura las variables en .env y ejecuta nuevamente")
        sys.exit(1)

def setup_secrets():
    """Configura secretos en keyring"""
    print("\\n🔐 CONFIGURANDO SECRETOS")
    print("-" * 40)
    
    try:
        config_manager = SecureConfigManager()
        
        # Configurar API keys desde variables de entorno
        binance_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_API_SECRET')
        
        if binance_key and binance_secret:
            config_manager.store_secret('binance_api_key', binance_key)
            config_manager.store_secret('binance_api_secret', binance_secret)
            print("✅ Binance API credentials almacenados")
        
        # Configurar otros secretos
        news_key = os.getenv('NEWS_API_KEY')
        if news_key:
            config_manager.store_secret('news_api_key', news_key)
            print("✅ News API key almacenado")
        
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            config_manager.store_secret('fred_api_key', fred_key)
            print("✅ FRED API key almacenado")
        
    except Exception as e:
        print(f"❌ Error configurando secretos: {e}")
        sys.exit(1)

def validate_config():
    """Valida la configuración"""
    print("\\n🔍 VALIDANDO CONFIGURACIÓN")
    print("-" * 40)
    
    try:
        config_manager = SecureConfigManager()
        validation_report = config_manager.validate_configuration()
        
        if validation_report['valid']:
            print("✅ Configuración válida")
        else:
            print("❌ Configuración inválida")
            for error in validation_report['errors']:
                print(f"  - {error}")
        
        if validation_report['warnings']:
            print("\\n⚠️  Advertencias:")
            for warning in validation_report['warnings']:
                print(f"  - {warning}")
        
        print("\\n📊 Resumen de configuración:")
        summary = validation_report['configuration_summary']
        print(f"  - Entorno: {summary['environment']}")
        print(f"  - Max position size: {summary['risk_limits']['max_position_size']:.1%}")
        print(f"  - Max portfolio risk: {summary['risk_limits']['max_portfolio_risk']:.1%}")
        print(f"  - Log level: {summary['monitoring']['log_level']}")
        
    except Exception as e:
        print(f"❌ Error validando configuración: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        setup_path = Path("config/setup_config.py")
        with open(setup_path, 'w') as f:
            f.write(setup_script)
        
        # Hacer ejecutable
        setup_path.chmod(0o755)
        
        print(f"✅ Script de configuración creado: {setup_path}")
    
    def _validate_migration(self):
        """Valida la migración"""
        print("\n🔍 VALIDANDO MIGRACIÓN")
        print("-" * 40)
        
        # Verificar archivos creados
        files_to_check = [
            "config/production.json",
            "config/setup_config.py",
            ".env.template"
        ]
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
        
        # Verificar que se puede importar configuración
        try:
            config_manager = SecureConfigManager("config/production.json")
            print("✅ Configuración importable")
        except Exception as e:
            print(f"❌ Error importando configuración: {e}")

def main():
    """Función principal"""
    migrator = ConfigMigrator()
    migrator.migrate_to_secure_config()

if __name__ == "__main__":
    main() 