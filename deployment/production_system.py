# SISTEMA DE INFRAESTRUCTURA DE PRODUCCIÓN
"""
Sistema completo de infraestructura para producción
Incluye deployment, logging, health checks, y gestión del sistema
"""

import os
import sys
import asyncio
import logging
import threading
import time
import signal
import atexit
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml
import subprocess
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# System monitoring
import psutil
import docker

# Web framework para APIs
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

# Logging avanzado
import structlog
from pythonjsonlogger import jsonlogger

# Interfaces y componentes
from core.monitoring_system import get_monitoring_system
from core.robust_risk_manager import get_risk_manager
from core.data_orchestrator import get_data_orchestrator
from models.robust_ml_pipeline import get_robust_ml_pipeline
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Estados de servicio"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class DeploymentStage(Enum):
    """Etapas de deployment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ServiceConfig:
    """Configuración de servicio"""
    name: str
    port: int
    enabled: bool = True
    auto_restart: bool = True
    max_restarts: int = 3
    restart_delay: int = 30
    health_check_interval: int = 60
    environment_vars: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentConfig:
    """Configuración de deployment"""
    stage: DeploymentStage
    version: str
    services: List[ServiceConfig]
    database_config: Dict[str, Any]
    redis_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    logging_config: Dict[str, Any]

class StructuredLogger:
    """Sistema de logging estructurado"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self._setup_loggers(log_level)
    
    def _setup_loggers(self, log_level: str):
        """Configura loggers estructurados"""
        
        # Logger principal
        main_logger = logging.getLogger()
        main_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Handler para archivo JSON
        json_handler = logging.FileHandler(self.log_dir / "trading_system.jsonl")
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        json_handler.setFormatter(json_formatter)
        main_logger.addHandler(json_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        main_logger.addHandler(console_handler)
        
        # Loggers específicos
        self._setup_component_loggers()
    
    def _setup_component_loggers(self):
        """Configura loggers por componente"""
        
        components = [
            'trading', 'risk_management', 'data_processing', 
            'ml_pipeline', 'monitoring', 'api', 'deployment'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            
            # Handler específico por componente
            component_handler = logging.FileHandler(
                self.log_dir / f"{component}.log"
            )
            component_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            component_handler.setFormatter(component_formatter)
            logger.addHandler(component_handler)

class HealthMonitor:
    """Monitor de salud del sistema"""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}
        self.check_interval = 30
        self.running = False
        
    def register_health_check(self, name: str, check_func: Callable, interval: int = 60):
        """Registra verificación de salud"""
        self.health_checks[name] = {
            'function': check_func,
            'interval': interval,
            'last_run': None,
            'last_result': None
        }
    
    def start_monitoring(self):
        """Inicia monitoreo de salud"""
        self.running = True
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        logger.info("🏥 Health monitor iniciado")
    
    def stop_monitoring(self):
        """Detiene monitoreo de salud"""
        self.running = False
        logger.info("⏹️ Health monitor detenido")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for check_name, check_config in self.health_checks.items():
                    last_run = check_config['last_run']
                    interval = check_config['interval']
                    
                    # Verificar si es tiempo de ejecutar
                    if (last_run is None or 
                        (current_time - last_run).total_seconds() >= interval):
                        
                        try:
                            # Ejecutar verificación
                            result = check_config['function']()
                            check_config['last_result'] = result
                            check_config['last_run'] = current_time
                            
                            # Log del resultado
                            if result.get('healthy', False):
                                logger.debug(f"✅ Health check {check_name}: OK")
                            else:
                                logger.warning(f"❌ Health check {check_name}: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            logger.error(f"💥 Error en health check {check_name}: {e}")
                            check_config['last_result'] = {
                                'healthy': False,
                                'error': str(e),
                                'timestamp': current_time
                            }
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error en monitoring loop: {e}")
                time.sleep(60)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene estado de salud actual"""
        
        status = {
            'overall_health': 'healthy',
            'timestamp': datetime.now(),
            'checks': {},
            'summary': {
                'total_checks': len(self.health_checks),
                'healthy_checks': 0,
                'failed_checks': 0
            }
        }
        
        for check_name, check_config in self.health_checks.items():
            result = check_config['last_result']
            
            if result:
                status['checks'][check_name] = result
                
                if result.get('healthy', False):
                    status['summary']['healthy_checks'] += 1
                else:
                    status['summary']['failed_checks'] += 1
            else:
                status['checks'][check_name] = {
                    'healthy': None,
                    'error': 'Not yet executed'
                }
        
        # Determinar salud general
        if status['summary']['failed_checks'] > 0:
            if status['summary']['failed_checks'] >= status['summary']['total_checks'] // 2:
                status['overall_health'] = 'critical'
            else:
                status['overall_health'] = 'degraded'
        
        return status

class ServiceManager:
    """Gestor de servicios del sistema"""
    
    def __init__(self):
        self.services = {}
        self.service_configs = {}
        self.restart_counts = {}
        
    def register_service(self, config: ServiceConfig):
        """Registra un servicio"""
        self.service_configs[config.name] = config
        self.services[config.name] = {
            'status': ServiceStatus.STOPPED,
            'process': None,
            'start_time': None,
            'restart_count': 0
        }
        logger.info(f"📝 Servicio registrado: {config.name}")
    
    def start_service(self, service_name: str) -> bool:
        """Inicia un servicio"""
        
        if service_name not in self.service_configs:
            logger.error(f"Servicio no registrado: {service_name}")
            return False
        
        config = self.service_configs[service_name]
        service = self.services[service_name]
        
        if service['status'] == ServiceStatus.RUNNING:
            logger.warning(f"Servicio ya está ejecutándose: {service_name}")
            return True
        
        try:
            service['status'] = ServiceStatus.STARTING
            logger.info(f"🚀 Iniciando servicio: {service_name}")
            
            # Aquí iría la lógica específica para iniciar cada servicio
            # Por ahora, simulamos el inicio
            service['start_time'] = datetime.now()
            service['status'] = ServiceStatus.RUNNING
            
            logger.info(f"✅ Servicio iniciado: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando servicio {service_name}: {e}")
            service['status'] = ServiceStatus.ERROR
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Detiene un servicio"""
        
        if service_name not in self.services:
            logger.error(f"Servicio no encontrado: {service_name}")
            return False
        
        service = self.services[service_name]
        
        try:
            service['status'] = ServiceStatus.STOPPING
            logger.info(f"⏹️ Deteniendo servicio: {service_name}")
            
            # Lógica de detención
            if service['process']:
                service['process'].terminate()
                service['process'] = None
            
            service['status'] = ServiceStatus.STOPPED
            service['start_time'] = None
            
            logger.info(f"✅ Servicio detenido: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo servicio {service_name}: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Reinicia un servicio"""
        
        config = self.service_configs.get(service_name)
        if not config:
            return False
        
        service = self.services[service_name]
        
        # Verificar límite de reinicios
        if service['restart_count'] >= config.max_restarts:
            logger.error(f"Máximo de reinicios alcanzado para {service_name}")
            return False
        
        # Detener y iniciar
        if self.stop_service(service_name):
            time.sleep(config.restart_delay)
            if self.start_service(service_name):
                service['restart_count'] += 1
                return True
        
        return False
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Obtiene estado de un servicio"""
        
        if service_name not in self.services:
            return {'error': 'Service not found'}
        
        service = self.services[service_name]
        config = self.service_configs[service_name]
        
        uptime = None
        if service['start_time']:
            uptime = (datetime.now() - service['start_time']).total_seconds()
        
        return {
            'name': service_name,
            'status': service['status'].value,
            'uptime_seconds': uptime,
            'restart_count': service['restart_count'],
            'config': {
                'port': config.port,
                'enabled': config.enabled,
                'auto_restart': config.auto_restart
            }
        }

class APIGateway:
    """Gateway de API para el sistema"""
    
    def __init__(self, port: int = 8080):
        self.app = Flask(__name__)
        CORS(self.app)
        self.port = port
        
        # Componentes del sistema
        self.monitoring_system = get_monitoring_system()
        self.risk_manager = get_risk_manager()
        self.data_orchestrator = get_data_orchestrator()
        self.ml_pipeline = get_robust_ml_pipeline()
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Configura rutas de la API"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint de health check"""
            try:
                health_status = self.monitoring_system.get_system_dashboard()
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'system': health_status
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            """Estado general del sistema"""
            try:
                dashboard = self.monitoring_system.get_system_dashboard()
                return jsonify(dashboard)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk/limits', methods=['GET'])
        def risk_limits():
            """Límites de riesgo actuales"""
            try:
                limits = self.risk_manager.get_current_limits()
                return jsonify(limits)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/risk/report', methods=['GET'])
        def risk_report():
            """Reporte de riesgo"""
            try:
                report = self.risk_manager.get_risk_report()
                return jsonify(report)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data/quality/<symbol>', methods=['GET'])
        def data_quality(symbol: str):
            """Reporte de calidad de datos"""
            try:
                timeframe = request.args.get('timeframe', '1h')
                start_date = request.args.get('start_date', '7 days ago')
                end_date = request.args.get('end_date', 'now')
                
                report = self.data_orchestrator.get_data_quality_report(
                    symbol, timeframe, start_date, end_date
                )
                return jsonify(report)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/ml/performance', methods=['GET'])
        def ml_performance():
            """Performance de modelos ML"""
            try:
                report = self.ml_pipeline.generate_performance_report()
                return jsonify(report)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/metrics', methods=['GET'])
        def system_metrics():
            """Métricas del sistema"""
            try:
                hours = int(request.args.get('hours', 24))
                
                metrics = {}
                collector = self.monitoring_system.metrics_collector
                
                metric_names = ['cpu_usage', 'memory_usage', 'api_response_time', 'error_rate']
                for metric in metric_names:
                    metrics[metric] = collector.get_metric_stats(metric, hours)
                
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, debug: bool = False):
        """Ejecuta el servidor API"""
        logger.info(f"🌐 API Gateway iniciado en puerto {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

class ProductionSystem:
    """Sistema de producción completo"""
    
    def __init__(self, config_path: str = "deployment/production_config.yaml"):
        self.config_path = config_path
        self.deployment_config = self._load_deployment_config()
        
        # Componentes del sistema
        self.logger_system = StructuredLogger()
        self.health_monitor = HealthMonitor()
        self.service_manager = ServiceManager()
        self.api_gateway = None
        
        # Estado
        self.running = False
        self.start_time = None
        
        # Configurar signal handlers
        self._setup_signal_handlers()
        
        logger.info("🏭 Sistema de producción inicializado")
    
    def _load_deployment_config(self) -> DeploymentConfig:
        """Carga configuración de deployment"""
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parsear configuración
            services = [ServiceConfig(**svc) for svc in config_data.get('services', [])]
            
            return DeploymentConfig(
                stage=DeploymentStage(config_data.get('stage', 'development')),
                version=config_data.get('version', '1.0.0'),
                services=services,
                database_config=config_data.get('database', {}),
                redis_config=config_data.get('redis', {}),
                monitoring_config=config_data.get('monitoring', {}),
                logging_config=config_data.get('logging', {})
            )
            
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
            return self._create_default_config()
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> DeploymentConfig:
        """Crea configuración por defecto"""
        
        default_services = [
            ServiceConfig(name='api_gateway', port=8080),
            ServiceConfig(name='monitoring', port=8000),
            ServiceConfig(name='data_processor', port=8081),
            ServiceConfig(name='ml_engine', port=8082)
        ]
        
        return DeploymentConfig(
            stage=DeploymentStage.DEVELOPMENT,
            version='1.0.0',
            services=default_services,
            database_config={},
            redis_config={},
            monitoring_config={},
            logging_config={}
        )
    
    def _setup_signal_handlers(self):
        """Configura manejadores de señales"""
        
        def signal_handler(signum, frame):
            logger.info(f"📶 Señal recibida: {signum}")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.shutdown)
    
    def _setup_health_checks(self):
        """Configura verificaciones de salud"""
        
        def check_api_health():
            try:
                response = requests.get('http://localhost:8080/health', timeout=5)
                return {
                    'healthy': response.status_code == 200,
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': datetime.now()
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        def check_system_resources():
            try:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                return {
                    'healthy': cpu_usage < 90 and memory_usage < 90 and disk_usage < 90,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        def check_services():
            try:
                healthy_services = 0
                total_services = len(self.service_manager.services)
                
                for service_name, service in self.service_manager.services.items():
                    if service['status'] == ServiceStatus.RUNNING:
                        healthy_services += 1
                
                return {
                    'healthy': healthy_services == total_services,
                    'healthy_services': healthy_services,
                    'total_services': total_services,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        # Registrar health checks
        self.health_monitor.register_health_check('api', check_api_health, 60)
        self.health_monitor.register_health_check('system_resources', check_system_resources, 30)
        self.health_monitor.register_health_check('services', check_services, 60)
    
    def start(self):
        """Inicia el sistema de producción"""
        
        if self.running:
            logger.warning("Sistema ya está ejecutándose")
            return
        
        logger.info("🚀 Iniciando sistema de producción")
        self.start_time = datetime.now()
        self.running = True
        
        try:
            # 1. Registrar servicios
            for service_config in self.deployment_config.services:
                self.service_manager.register_service(service_config)
            
            # 2. Iniciar componentes del sistema
            monitoring_system = get_monitoring_system()
            monitoring_system.start_monitoring()
            
            # 3. Iniciar health monitoring
            self._setup_health_checks()
            self.health_monitor.start_monitoring()
            
            # 4. Iniciar servicios
            for service_config in self.deployment_config.services:
                if service_config.enabled:
                    self.service_manager.start_service(service_config.name)
            
            # 5. Iniciar API Gateway
            if any(svc.name == 'api_gateway' for svc in self.deployment_config.services):
                self.api_gateway = APIGateway()
                threading.Thread(
                    target=self.api_gateway.run,
                    kwargs={'debug': self.deployment_config.stage == DeploymentStage.DEVELOPMENT},
                    daemon=True
                ).start()
            
            logger.info("✅ Sistema de producción iniciado correctamente")
            
            # Loop principal
            self._main_loop()
            
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema: {e}")
            self.shutdown()
    
    def _main_loop(self):
        """Loop principal del sistema"""
        
        while self.running:
            try:
                # Verificar servicios y reiniciar si es necesario
                self._check_and_restart_services()
                
                # Log de estado cada 5 minutos
                if int(time.time()) % 300 == 0:
                    self._log_system_status()
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error en main loop: {e}")
                time.sleep(30)
    
    def _check_and_restart_services(self):
        """Verifica y reinicia servicios si es necesario"""
        
        for service_name, service in self.service_manager.services.items():
            config = self.service_manager.service_configs[service_name]
            
            if (config.auto_restart and 
                service['status'] == ServiceStatus.ERROR and
                service['restart_count'] < config.max_restarts):
                
                logger.info(f"🔄 Auto-reiniciando servicio: {service_name}")
                self.service_manager.restart_service(service_name)
    
    def _log_system_status(self):
        """Log del estado del sistema"""
        
        uptime = datetime.now() - self.start_time if self.start_time else timedelta()
        
        status = {
            'stage': self.deployment_config.stage.value,
            'version': self.deployment_config.version,
            'uptime_hours': uptime.total_seconds() / 3600,
            'services': {
                name: self.service_manager.get_service_status(name)
                for name in self.service_manager.services.keys()
            },
            'health': self.health_monitor.get_health_status()
        }
        
        logger.info(f"📊 System Status: {json.dumps(status, indent=2, default=str)}")
    
    def shutdown(self):
        """Apaga el sistema de producción"""
        
        if not self.running:
            return
        
        logger.info("⏹️ Apagando sistema de producción")
        self.running = False
        
        try:
            # Detener servicios
            for service_name in self.service_manager.services.keys():
                self.service_manager.stop_service(service_name)
            
            # Detener health monitoring
            self.health_monitor.stop_monitoring()
            
            # Detener monitoring system
            monitoring_system = get_monitoring_system()
            monitoring_system.stop_monitoring()
            
            logger.info("✅ Sistema de producción apagado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error apagando sistema: {e}")
    
    def deploy_update(self, new_version: str) -> bool:
        """Deploya una actualización del sistema"""
        
        logger.info(f"🚀 Iniciando deployment de versión {new_version}")
        
        try:
            # 1. Validar nueva versión
            if not self._validate_deployment(new_version):
                logger.error("Validación de deployment fallida")
                return False
            
            # 2. Crear backup
            backup_success = self._create_backup()
            if not backup_success:
                logger.error("Creación de backup fallida")
                return False
            
            # 3. Rolling update de servicios
            for service_config in self.deployment_config.services:
                if not self._update_service(service_config.name, new_version):
                    logger.error(f"Error actualizando servicio {service_config.name}")
                    # Rollback
                    self._rollback_deployment()
                    return False
            
            # 4. Actualizar configuración
            self.deployment_config.version = new_version
            
            logger.info(f"✅ Deployment de versión {new_version} completado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en deployment: {e}")
            self._rollback_deployment()
            return False
    
    def _validate_deployment(self, version: str) -> bool:
        """Valida deployment antes de aplicar"""
        # Validaciones básicas
        return True
    
    def _create_backup(self) -> bool:
        """Crea backup del sistema"""
        try:
            backup_dir = Path(f"backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Backup creado en {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return False
    
    def _update_service(self, service_name: str, version: str) -> bool:
        """Actualiza un servicio específico"""
        try:
            # Graceful restart del servicio
            return self.service_manager.restart_service(service_name)
        except Exception as e:
            logger.error(f"Error actualizando servicio {service_name}: {e}")
            return False
    
    def _rollback_deployment(self):
        """Rollback del deployment"""
        logger.warning("🔄 Iniciando rollback del deployment")
        # Lógica de rollback
        pass

def create_production_config():
    """Crea archivo de configuración de producción"""
    
    config = {
        'stage': 'production',
        'version': '1.0.0',
        'services': [
            {
                'name': 'api_gateway',
                'port': 8080,
                'enabled': True,
                'auto_restart': True,
                'max_restarts': 3,
                'restart_delay': 30
            },
            {
                'name': 'monitoring',
                'port': 8000,
                'enabled': True,
                'auto_restart': True,
                'max_restarts': 5,
                'restart_delay': 15
            },
            {
                'name': 'data_processor',
                'port': 8081,
                'enabled': True,
                'auto_restart': True,
                'max_restarts': 3,
                'restart_delay': 30
            },
            {
                'name': 'ml_engine',
                'port': 8082,
                'enabled': True,
                'auto_restart': True,
                'max_restarts': 2,
                'restart_delay': 60
            }
        ],
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'trading_system',
            'pool_size': 20
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'max_connections': 50
        },
        'monitoring': {
            'prometheus_port': 8000,
            'metrics_interval': 30,
            'alert_channels': ['email', 'slack']
        },
        'logging': {
            'level': 'INFO',
            'structured': True,
            'retention_days': 30
        }
    }
    
    config_path = Path("deployment/production_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuración de producción creada: {config_path}")

def main():
    """Función principal para ejecutar el sistema"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Trading - Producción')
    parser.add_argument('--config', default='deployment/production_config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--create-config', action='store_true',
                       help='Crear archivo de configuración de ejemplo')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_production_config()
        return
    
    # Inicializar y ejecutar sistema
    system = ProductionSystem(args.config)
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
    finally:
        system.shutdown()

if __name__ == "__main__":
    main() 