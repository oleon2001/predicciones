# SISTEMA AVANZADO DE MONITOREO Y ALERTAS
"""
Sistema completo de monitoreo, alertas y observabilidad para trading
Incluye métricas en tiempo real, health checks, y notificaciones
"""

import asyncio
import logging
import threading
import time
import psutil
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import warnings
warnings.filterwarnings('ignore')

# Métricas y visualización
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Interfaces
from config.secure_config import get_config_manager

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemStatus(Enum):
    """Estados del sistema"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Estructura de alerta"""
    id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    description: str
    component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

@dataclass
class HealthCheck:
    """Verificación de salud"""
    name: str
    status: SystemStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    api_response_time: float
    error_rate: float
    throughput: float

class MetricsCollector:
    """Recolector de métricas"""
    
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.max_metrics_history = 1000
        
        # Métricas Prometheus
        self.prometheus_metrics = {
            'api_requests_total': Counter('api_requests_total', 'Total API requests', ['method', 'endpoint']),
            'api_request_duration': Histogram('api_request_duration_seconds', 'API request duration'),
            'active_positions': Gauge('active_positions_total', 'Total active positions'),
            'portfolio_value': Gauge('portfolio_value_usd', 'Portfolio value in USD'),
            'system_cpu_usage': Gauge('system_cpu_usage_percent', 'System CPU usage'),
            'system_memory_usage': Gauge('system_memory_usage_percent', 'System memory usage'),
            'error_rate': Gauge('error_rate_percent', 'Error rate percentage'),
            'alerts_total': Counter('alerts_total', 'Total alerts', ['level']),
            'trades_executed': Counter('trades_executed_total', 'Total trades executed', ['symbol', 'side']),
            'ml_model_accuracy': Gauge('ml_model_accuracy_percent', 'ML model accuracy', ['model_name'])
        }
        
        logger.info("✅ Recolector de métricas inicializado")
    
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Registra una métrica"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[name].append((timestamp, value))
        
        # Limitar historial
        if len(self.metrics[name]) > self.max_metrics_history:
            self.metrics[name].popleft()
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Tuple[datetime, float]]:
        """Obtiene historial de métrica"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [(ts, val) for ts, val in self.metrics[name] if ts > cutoff_time]
    
    def get_metric_stats(self, name: str, hours: int = 24) -> Dict[str, float]:
        """Obtiene estadísticas de métrica"""
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {}
        
        values = [val for _, val in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'current': values[-1] if values else 0
        }
    
    def update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Actualiza métricas Prometheus"""
        try:
            for metric_name, value in metrics.items():
                if metric_name in self.prometheus_metrics:
                    prometheus_metric = self.prometheus_metrics[metric_name]
                    
                    if hasattr(prometheus_metric, 'set'):  # Gauge
                        prometheus_metric.set(value)
                    elif hasattr(prometheus_metric, 'observe'):  # Histogram
                        prometheus_metric.observe(value)
                    # Counter se maneja diferente, se incrementa explícitamente
                        
        except Exception as e:
            logger.error(f"Error actualizando métricas Prometheus: {e}")

class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.monitoring_config = config_manager.get_monitoring_config()
        self.alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = self._setup_notification_channels()
        
        # Reglas de alerta
        self.alert_rules = {
            'high_cpu': {
                'metric': 'cpu_usage',
                'threshold': 80,
                'operator': '>',
                'level': AlertLevel.WARNING,
                'description': 'Alto uso de CPU'
            },
            'high_memory': {
                'metric': 'memory_usage',
                'threshold': 85,
                'operator': '>',
                'level': AlertLevel.ERROR,
                'description': 'Alto uso de memoria'
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 5,
                'operator': '>',
                'level': AlertLevel.ERROR,
                'description': 'Alta tasa de errores'
            },
            'low_api_response': {
                'metric': 'api_response_time',
                'threshold': 5,
                'operator': '>',
                'level': AlertLevel.WARNING,
                'description': 'Tiempo de respuesta API lento'
            },
            'position_limit': {
                'metric': 'active_positions',
                'threshold': 8,
                'operator': '>',
                'level': AlertLevel.WARNING,
                'description': 'Muchas posiciones activas'
            }
        }
        
        logger.info("✅ Gestor de alertas inicializado")
    
    def _setup_notification_channels(self) -> Dict[str, Any]:
        """Configura canales de notificación"""
        channels = {}
        
        # Email
        email_config = self.monitoring_config.alert_channels.get('email')
        if email_config:
            channels['email'] = {
                'enabled': True,
                'config': email_config
            }
        
        # Slack
        slack_webhook = self.monitoring_config.alert_channels.get('slack')
        if slack_webhook:
            channels['slack'] = {
                'enabled': True,
                'webhook_url': slack_webhook
            }
        
        # Telegram
        telegram_config = self.monitoring_config.alert_channels.get('telegram')
        if telegram_config:
            channels['telegram'] = {
                'enabled': True,
                'bot_token': telegram_config.get('bot_token'),
                'chat_id': telegram_config.get('chat_id')
            }
        
        return channels
    
    def create_alert(self, alert_id: str, level: AlertLevel, title: str, 
                    description: str, component: str, metrics: Dict[str, Any] = None) -> Alert:
        """Crea una nueva alerta"""
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            title=title,
            description=description,
            component=component,
            metrics=metrics or {}
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Enviar notificación
        self._send_notification(alert)
        
        logger.warning(f"🚨 Alerta creada: {alert.title} - {alert.description}")
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resuelve una alerta"""
        
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolved_by = resolved_by
            
            # Enviar notificación de resolución
            self._send_resolution_notification(alert)
            
            logger.info(f"✅ Alerta resuelta: {alert.title}")
            
            # Remover de alertas activas
            del self.alerts[alert_id]
    
    def check_alert_rules(self, metrics: Dict[str, float]):
        """Verifica reglas de alerta"""
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric']
            threshold = rule['threshold']
            operator = rule['operator']
            level = rule['level']
            
            if metric_name in metrics:
                current_value = metrics[metric_name]
                
                triggered = False
                if operator == '>':
                    triggered = current_value > threshold
                elif operator == '<':
                    triggered = current_value < threshold
                elif operator == '>=':
                    triggered = current_value >= threshold
                elif operator == '<=':
                    triggered = current_value <= threshold
                elif operator == '==':
                    triggered = current_value == threshold
                
                if triggered:
                    alert_id = f"{rule_name}_{int(time.time())}"
                    
                    # Verificar si ya existe alerta similar reciente
                    if not self._has_recent_similar_alert(rule_name, 300):  # 5 minutos
                        self.create_alert(
                            alert_id=alert_id,
                            level=level,
                            title=f"Alerta: {rule['description']}",
                            description=f"{metric_name} = {current_value:.2f} (umbral: {threshold})",
                            component=metric_name,
                            metrics={metric_name: current_value, 'threshold': threshold}
                        )
                else:
                    # Resolver alerta si existe
                    self._resolve_similar_alerts(rule_name)
    
    def _has_recent_similar_alert(self, rule_name: str, seconds: int) -> bool:
        """Verifica si existe alerta similar reciente"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        for alert in list(self.alerts.values()):
            if rule_name in alert.id and alert.timestamp > cutoff_time:
                return True
        
        return False
    
    def _resolve_similar_alerts(self, rule_name: str):
        """Resuelve alertas similares"""
        alerts_to_resolve = []
        
        for alert_id, alert in self.alerts.items():
            if rule_name in alert_id:
                alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id, "auto_resolved")
    
    def _send_notification(self, alert: Alert):
        """Envía notificación de alerta"""
        
        for channel_name, channel_config in self.notification_channels.items():
            if not channel_config.get('enabled', False):
                continue
            
            try:
                if channel_name == 'email':
                    self._send_email_notification(alert, channel_config)
                elif channel_name == 'slack':
                    self._send_slack_notification(alert, channel_config)
                elif channel_name == 'telegram':
                    self._send_telegram_notification(alert, channel_config)
                    
            except Exception as e:
                logger.error(f"Error enviando notificación por {channel_name}: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Envía notificación por email"""
        
        # Configuración básica de email
        smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        to_email = config.get('to_email')
        
        if not all([username, password, to_email]):
            return
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to_email
        msg['Subject'] = f"🚨 Alerta Trading System: {alert.title}"
        
        body = f"""
        Alerta del Sistema de Trading
        
        Nivel: {alert.level.value.upper()}
        Título: {alert.title}
        Descripción: {alert.description}
        Componente: {alert.component}
        Timestamp: {alert.timestamp}
        
        Métricas:
        {json.dumps(alert.metrics, indent=2)}
        
        Este es un mensaje automático del sistema de monitoreo.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Enviar
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Envía notificación por Slack"""
        
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return
        
        # Mapear nivel a color
        color_map = {
            AlertLevel.INFO: 'good',
            AlertLevel.WARNING: 'warning',
            AlertLevel.ERROR: 'danger',
            AlertLevel.CRITICAL: 'danger'
        }
        
        payload = {
            'text': f"🚨 Alerta Trading System",
            'attachments': [
                {
                    'color': color_map.get(alert.level, 'warning'),
                    'title': alert.title,
                    'text': alert.description,
                    'fields': [
                        {'title': 'Nivel', 'value': alert.level.value.upper(), 'short': True},
                        {'title': 'Componente', 'value': alert.component, 'short': True},
                        {'title': 'Timestamp', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': False}
                    ]
                }
            ]
        }
        
        requests.post(webhook_url, json=payload)
    
    def _send_telegram_notification(self, alert: Alert, config: Dict[str, Any]):
        """Envía notificación por Telegram"""
        
        bot_token = config.get('bot_token')
        chat_id = config.get('chat_id')
        
        if not all([bot_token, chat_id]):
            return
        
        # Formatear mensaje
        message = f"""
🚨 *Alerta Trading System*

*Nivel:* {alert.level.value.upper()}
*Título:* {alert.title}
*Descripción:* {alert.description}
*Componente:* {alert.component}
*Timestamp:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        requests.post(url, json=payload)
    
    def _send_resolution_notification(self, alert: Alert):
        """Envía notificación de resolución"""
        
        # Mensaje de resolución más simple
        for channel_name, channel_config in self.notification_channels.items():
            if not channel_config.get('enabled', False):
                continue
            
            try:
                if channel_name == 'slack':
                    webhook_url = channel_config.get('webhook_url')
                    if webhook_url:
                        payload = {
                            'text': f"✅ Alerta resuelta: {alert.title}",
                            'attachments': [
                                {
                                    'color': 'good',
                                    'text': f"Resuelto por: {alert.resolved_by} a las {alert.resolved_at.strftime('%H:%M:%S')}"
                                }
                            ]
                        }
                        requests.post(webhook_url, json=payload)
                        
            except Exception as e:
                logger.error(f"Error enviando notificación de resolución: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtiene alertas activas"""
        return list(self.alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Obtiene historial de alertas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

class HealthChecker:
    """Verificador de salud del sistema"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.checks = {}
        self.last_system_check = None
        
        # Registrar verificaciones
        self._register_health_checks()
        
        logger.info("✅ Verificador de salud inicializado")
    
    def _register_health_checks(self):
        """Registra verificaciones de salud"""
        
        self.checks = {
            'system_resources': self._check_system_resources,
            'api_connectivity': self._check_api_connectivity,
            'database_connection': self._check_database_connection,
            'cache_system': self._check_cache_system,
            'ml_models': self._check_ml_models
        }
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Ejecuta todas las verificaciones de salud"""
        
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                status, metrics, error = check_func()
                response_time = time.time() - start_time
                
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=status,
                    last_check=datetime.now(),
                    response_time=response_time,
                    error_message=error,
                    metrics=metrics
                )
                
            except Exception as e:
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=SystemStatus.UNHEALTHY,
                    last_check=datetime.now(),
                    response_time=0,
                    error_message=str(e),
                    metrics={}
                )
        
        return results
    
    def _check_system_resources(self) -> Tuple[SystemStatus, Dict[str, Any], Optional[str]]:
        """Verifica recursos del sistema"""
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'memory_available': memory.available,
            'disk_free': disk.free
        }
        
        # Determinar estado
        if cpu_usage > 90 or memory.percent > 95 or disk.percent > 95:
            return SystemStatus.CRITICAL, metrics, "Recursos del sistema críticos"
        elif cpu_usage > 80 or memory.percent > 85 or disk.percent > 85:
            return SystemStatus.UNHEALTHY, metrics, "Recursos del sistema altos"
        elif cpu_usage > 70 or memory.percent > 75 or disk.percent > 75:
            return SystemStatus.DEGRADED, metrics, "Recursos del sistema elevados"
        else:
            return SystemStatus.HEALTHY, metrics, None
    
    def _check_api_connectivity(self) -> Tuple[SystemStatus, Dict[str, Any], Optional[str]]:
        """Verifica conectividad de APIs"""
        
        try:
            # Verificar Binance API
            import requests
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            
            if response.status_code == 200:
                return SystemStatus.HEALTHY, {'api_response_time': response.elapsed.total_seconds()}, None
            else:
                return SystemStatus.UNHEALTHY, {'status_code': response.status_code}, "API no responde correctamente"
                
        except Exception as e:
            return SystemStatus.UNHEALTHY, {}, f"Error de conectividad: {str(e)}"
    
    def _check_database_connection(self) -> Tuple[SystemStatus, Dict[str, Any], Optional[str]]:
        """Verifica conexión a base de datos"""
        
        # Simulación - implementar según base de datos real
        try:
            # Aquí iría la verificación real de DB
            return SystemStatus.HEALTHY, {'connection_time': 0.1}, None
        except Exception as e:
            return SystemStatus.UNHEALTHY, {}, f"Error de base de datos: {str(e)}"
    
    def _check_cache_system(self) -> Tuple[SystemStatus, Dict[str, Any], Optional[str]]:
        """Verifica sistema de cache"""
        
        # Simulación - implementar según cache real (Redis, etc.)
        try:
            # Aquí iría la verificación real de cache
            return SystemStatus.HEALTHY, {'cache_hit_rate': 0.85}, None
        except Exception as e:
            return SystemStatus.UNHEALTHY, {}, f"Error de cache: {str(e)}"
    
    def _check_ml_models(self) -> Tuple[SystemStatus, Dict[str, Any], Optional[str]]:
        """Verifica modelos de ML"""
        
        try:
            # Verificar que los modelos estén cargados y funcionando
            # Simulación - implementar según modelos reales
            return SystemStatus.HEALTHY, {'models_loaded': 3, 'avg_prediction_time': 0.05}, None
        except Exception as e:
            return SystemStatus.UNHEALTHY, {}, f"Error en modelos ML: {str(e)}"

class MonitoringSystem:
    """Sistema principal de monitoreo"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.config)
        self.health_checker = HealthChecker(self.metrics_collector)
        
        # Estado del sistema
        self.system_status = SystemStatus.HEALTHY
        self.start_time = datetime.now()
        
        # Threads de monitoreo
        self.monitoring_threads = []
        self.running = False
        
        # Inicializar servidor Prometheus
        self._start_prometheus_server()
        
        logger.info("✅ Sistema de monitoreo inicializado")
    
    def _start_prometheus_server(self):
        """Inicia servidor Prometheus"""
        try:
            start_http_server(8000)
            logger.info("🚀 Servidor Prometheus iniciado en puerto 8000")
        except Exception as e:
            logger.warning(f"No se pudo iniciar servidor Prometheus: {e}")
    
    def start_monitoring(self):
        """Inicia el monitoreo"""
        
        if self.running:
            return
        
        self.running = True
        
        # Thread para recolección de métricas
        metrics_thread = threading.Thread(target=self._metrics_collection_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
        self.monitoring_threads.append(metrics_thread)
        
        # Thread para verificaciones de salud
        health_thread = threading.Thread(target=self._health_check_loop)
        health_thread.daemon = True
        health_thread.start()
        self.monitoring_threads.append(health_thread)
        
        logger.info("🚀 Sistema de monitoreo iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        
        self.running = False
        
        # Esperar a que terminen los threads
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        
        logger.info("⏹️ Sistema de monitoreo detenido")
    
    def _metrics_collection_loop(self):
        """Loop de recolección de métricas"""
        
        while self.running:
            try:
                # Recolectar métricas del sistema
                system_metrics = self._collect_system_metrics()
                
                # Registrar métricas
                for metric_name, value in system_metrics.items():
                    self.metrics_collector.record_metric(metric_name, value)
                
                # Actualizar Prometheus
                self.metrics_collector.update_prometheus_metrics(system_metrics)
                
                # Verificar reglas de alerta
                self.alert_manager.check_alert_rules(system_metrics)
                
                # Dormir según intervalo configurado
                time.sleep(self.config.get_monitoring_config().metrics_interval)
                
            except Exception as e:
                logger.error(f"Error en recolección de métricas: {e}")
                time.sleep(30)  # Esperar más tiempo si hay error
    
    def _health_check_loop(self):
        """Loop de verificaciones de salud"""
        
        while self.running:
            try:
                # Ejecutar verificaciones de salud
                health_results = self.health_checker.run_health_checks()
                
                # Determinar estado general del sistema
                self._update_system_status(health_results)
                
                # Crear alertas si es necesario
                self._process_health_results(health_results)
                
                # Dormir según intervalo configurado
                time.sleep(self.config.get_monitoring_config().health_check_interval)
                
            except Exception as e:
                logger.error(f"Error en verificaciones de salud: {e}")
                time.sleep(60)  # Esperar más tiempo si hay error
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Recolecta métricas del sistema"""
        
        metrics = {}
        
        try:
            # Métricas del sistema
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            metrics.update({
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'active_connections': len(psutil.net_connections())
            })
            
            # Métricas de aplicación (simuladas)
            metrics.update({
                'api_response_time': np.random.uniform(0.1, 2.0),
                'error_rate': np.random.uniform(0, 5),
                'throughput': np.random.uniform(50, 200),
                'active_positions': np.random.randint(0, 10)
            })
            
        except Exception as e:
            logger.error(f"Error recolectando métricas: {e}")
        
        return metrics
    
    def _update_system_status(self, health_results: Dict[str, HealthCheck]):
        """Actualiza estado del sistema"""
        
        statuses = [check.status for check in health_results.values()]
        
        if SystemStatus.CRITICAL in statuses:
            self.system_status = SystemStatus.CRITICAL
        elif SystemStatus.UNHEALTHY in statuses:
            self.system_status = SystemStatus.UNHEALTHY
        elif SystemStatus.DEGRADED in statuses:
            self.system_status = SystemStatus.DEGRADED
        else:
            self.system_status = SystemStatus.HEALTHY
    
    def _process_health_results(self, health_results: Dict[str, HealthCheck]):
        """Procesa resultados de verificaciones de salud"""
        
        for check_name, check_result in health_results.items():
            if check_result.status != SystemStatus.HEALTHY:
                
                # Determinar nivel de alerta
                if check_result.status == SystemStatus.CRITICAL:
                    level = AlertLevel.CRITICAL
                elif check_result.status == SystemStatus.UNHEALTHY:
                    level = AlertLevel.ERROR
                else:
                    level = AlertLevel.WARNING
                
                # Crear alerta
                alert_id = f"health_check_{check_name}"
                self.alert_manager.create_alert(
                    alert_id=alert_id,
                    level=level,
                    title=f"Verificación de salud falló: {check_name}",
                    description=check_result.error_message or "Estado no saludable",
                    component=check_name,
                    metrics=check_result.metrics
                )
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Obtiene dashboard del sistema"""
        
        uptime = datetime.now() - self.start_time
        
        return {
            'system_status': self.system_status.value,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'recent_metrics': {
                name: self.metrics_collector.get_metric_stats(name, hours=1)
                for name in ['cpu_usage', 'memory_usage', 'api_response_time', 'error_rate']
            },
            'health_checks': {
                check_name: {
                    'status': check.status.value,
                    'last_check': check.last_check.isoformat(),
                    'response_time': check.response_time
                }
                for check_name, check in self.health_checker.run_health_checks().items()
            }
        }

# Singleton para acceso global
_monitoring_system = None

def get_monitoring_system() -> MonitoringSystem:
    """Obtiene el sistema de monitoreo (singleton)"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system 