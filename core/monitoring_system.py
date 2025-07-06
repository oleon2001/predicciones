# SISTEMA DE MONITOREO Y OBSERVABILIDAD
"""
Sistema completo de monitoreo, métricas y observabilidad
Incluye logging estructurado, métricas de performance, alertas y dashboards
"""

import logging
import time
import json
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from enum import Enum
import psutil
from contextlib import contextmanager
from functools import wraps
import queue
import threading

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricPoint:
    """Punto de métrica"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    timestamp: datetime

@dataclass
class ApplicationMetrics:
    """Métricas de la aplicación"""
    predictions_made: int
    models_trained: int
    api_calls_made: int
    cache_hit_rate: float
    active_sessions: int
    errors_count: int
    average_response_time: float
    timestamp: datetime

@dataclass
class AlertRule:
    """Regla de alerta"""
    name: str
    condition: str  # Expresión evaluable
    threshold: float
    severity: str  # low, medium, high, critical
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """Alerta generada"""
    rule_name: str
    message: str
    severity: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False

class StructuredLogger:
    """Logger estructurado con contexto"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter estructurado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Contexto adicional
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Establece contexto adicional"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Limpia contexto"""
        self.context.clear()
    
    def _log_with_context(self, level: LogLevel, message: str, **kwargs):
        """Log con contexto"""
        log_data = {
            'message': message,
            'context': self.context,
            'extra': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        
        formatted_message = json.dumps(log_data, default=str)
        
        if level == LogLevel.DEBUG:
            self.logger.debug(formatted_message)
        elif level == LogLevel.INFO:
            self.logger.info(formatted_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(formatted_message)
        elif level == LogLevel.ERROR:
            self.logger.error(formatted_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(formatted_message)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(LogLevel.CRITICAL, message, **kwargs)
    
    def log_exception(self, exception: Exception, **kwargs):
        """Log excepción con traceback"""
        self.error(
            f"Exception: {str(exception)}",
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            **kwargs
        )

class MetricsCollector:
    """Colector de métricas"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.metrics: List[MetricPoint] = []
        self.lock = Lock()
        
        # Métricas agregadas
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None):
        """Registra métrica"""
        with self.lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type=metric_type
            )
            self.metrics.append(point)
            
            # Actualizar métricas agregadas
            if metric_type == MetricType.COUNTER:
                self.counters[name] = self.counters.get(name, 0) + value
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                if name not in self.histograms:
                    self.histograms[name] = []
                self.histograms[name].append(value)
            elif metric_type == MetricType.TIMER:
                if name not in self.timers:
                    self.timers[name] = []
                self.timers[name].append(value)
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Dict[str, str] = None):
        """Incrementa contador"""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, 
                  labels: Dict[str, str] = None):
        """Establece gauge"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: float, 
                        labels: Dict[str, str] = None):
        """Registra valor en histograma"""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def record_timer(self, name: str, duration: float, 
                    labels: Dict[str, str] = None):
        """Registra tiempo"""
        self.record_metric(name, duration, MetricType.TIMER, labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de métricas"""
        with self.lock:
            return {
                'counters': self.counters.copy(),
                'gauges': self.gauges.copy(),
                'histograms': {
                    name: {
                        'count': len(values),
                        'sum': sum(values),
                        'avg': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
                    for name, values in self.histograms.items()
                },
                'timers': {
                    name: {
                        'count': len(values),
                        'avg_ms': sum(values) / len(values) if values else 0,
                        'min_ms': min(values) if values else 0,
                        'max_ms': max(values) if values else 0
                    }
                    for name, values in self.timers.items()
                }
            }
    
    def cleanup_old_metrics(self):
        """Limpia métricas antiguas"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        with self.lock:
            self.metrics = [
                m for m in self.metrics 
                if m.timestamp > cutoff_date
            ]

class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self.lock = Lock()
        
        # Configurar reglas por defecto
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configura reglas de alerta por defecto"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="system.cpu_percent > threshold",
                threshold=80.0,
                severity="high"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="system.memory_percent > threshold",
                threshold=85.0,
                severity="high"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                condition="app.cache_hit_rate < threshold",
                threshold=0.7,
                severity="medium"
            ),
            AlertRule(
                name="high_error_rate",
                condition="app.errors_count > threshold",
                threshold=10.0,
                severity="critical"
            )
        ]
        
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: AlertRule):
        """Añade regla de alerta"""
        with self.lock:
            self.rules.append(rule)
    
    def check_alerts(self, system_metrics: SystemMetrics, 
                    app_metrics: ApplicationMetrics):
        """Verifica condiciones de alerta"""
        with self.lock:
            current_time = datetime.now()
            
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                # Verificar cooldown
                if (rule.last_triggered and 
                    current_time - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                    continue
                
                # Evaluar condición
                if self._evaluate_condition(rule, system_metrics, app_metrics):
                    # Crear alerta
                    alert = Alert(
                        rule_name=rule.name,
                        message=f"Alert: {rule.name} triggered",
                        severity=rule.severity,
                        value=self._get_metric_value(rule, system_metrics, app_metrics),
                        threshold=rule.threshold,
                        timestamp=current_time
                    )
                    
                    self.alerts.append(alert)
                    rule.last_triggered = current_time
                    
                    # Log alerta
                    logger = StructuredLogger("alerts")
                    logger.warning(
                        f"Alert triggered: {rule.name}",
                        rule=rule.name,
                        severity=rule.severity,
                        value=alert.value,
                        threshold=rule.threshold
                    )
    
    def _evaluate_condition(self, rule: AlertRule, 
                           system_metrics: SystemMetrics,
                           app_metrics: ApplicationMetrics) -> bool:
        """Evalúa condición de alerta"""
        try:
            # Crear contexto para evaluación
            context = {
                'system': asdict(system_metrics),
                'app': asdict(app_metrics),
                'threshold': rule.threshold
            }
            
            # Evaluar condición
            return eval(rule.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger = StructuredLogger("alerts")
            logger.error(f"Error evaluating alert condition: {e}", rule=rule.name)
            return False
    
    def _get_metric_value(self, rule: AlertRule, 
                         system_metrics: SystemMetrics,
                         app_metrics: ApplicationMetrics) -> float:
        """Obtiene valor de métrica para alerta"""
        # Simplificado - en producción sería más sofisticado
        if "cpu_percent" in rule.condition:
            return system_metrics.cpu_percent
        elif "memory_percent" in rule.condition:
            return system_metrics.memory_percent
        elif "cache_hit_rate" in rule.condition:
            return app_metrics.cache_hit_rate
        elif "errors_count" in rule.condition:
            return app_metrics.errors_count
        return 0.0

class PerformanceMonitor:
    """Monitor de performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_timers: Dict[str, float] = {}
        self.lock = Lock()
    
    @contextmanager
    def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager para medir tiempo"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # ms
            self.metrics_collector.record_timer(name, duration, labels)
    
    def time_function(self, name: str = None, labels: Dict[str, str] = None):
        """Decorador para medir tiempo de función"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                timer_name = name or f"{func.__module__}.{func.__name__}"
                with self.timer(timer_name, labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

class MonitoringSystem:
    """Sistema principal de monitoreo"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Componentes
        self.logger = StructuredLogger("monitoring")
        self.metrics_collector = MetricsCollector(
            retention_days=self.config.get('retention_days', 30)
        )
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        
        # Estado
        self.running = False
        self.monitor_thread = None
        self.last_system_metrics = None
        self.last_app_metrics = None
    
    def start(self):
        """Inicia sistema de monitoreo"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Sistema de monitoreo iniciado")
    
    def stop(self):
        """Detiene sistema de monitoreo"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Sistema de monitoreo detenido")
    
    def _monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.running:
            try:
                # Recopilar métricas del sistema
                system_metrics = self._collect_system_metrics()
                app_metrics = self._collect_app_metrics()
                
                # Registrar métricas
                self._register_metrics(system_metrics, app_metrics)
                
                # Verificar alertas
                self.alert_manager.check_alerts(system_metrics, app_metrics)
                
                # Limpiar métricas antiguas
                self.metrics_collector.cleanup_old_metrics()
                
                # Guardar para referencia
                self.last_system_metrics = system_metrics
                self.last_app_metrics = app_metrics
                
                time.sleep(60)  # Monitorear cada minuto
                
            except Exception as e:
                self.logger.log_exception(e)
                time.sleep(60)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Recopila métricas del sistema"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            network_io=dict(psutil.net_io_counters()._asdict()),
            process_count=len(psutil.pids()),
            timestamp=datetime.now()
        )
    
    def _collect_app_metrics(self) -> ApplicationMetrics:
        """Recopila métricas de la aplicación"""
        # Estas métricas deberían ser actualizadas por la aplicación
        summary = self.metrics_collector.get_metrics_summary()
        
        return ApplicationMetrics(
            predictions_made=summary['counters'].get('predictions_made', 0),
            models_trained=summary['counters'].get('models_trained', 0),
            api_calls_made=summary['counters'].get('api_calls_made', 0),
            cache_hit_rate=summary['gauges'].get('cache_hit_rate', 0.0),
            active_sessions=summary['gauges'].get('active_sessions', 0),
            errors_count=summary['counters'].get('errors_count', 0),
            average_response_time=summary['timers'].get('response_time', {}).get('avg_ms', 0),
            timestamp=datetime.now()
        )
    
    def _register_metrics(self, system_metrics: SystemMetrics, 
                         app_metrics: ApplicationMetrics):
        """Registra métricas en collector"""
        # Métricas del sistema
        self.metrics_collector.set_gauge('system.cpu_percent', system_metrics.cpu_percent)
        self.metrics_collector.set_gauge('system.memory_percent', system_metrics.memory_percent)
        self.metrics_collector.set_gauge('system.disk_usage_percent', system_metrics.disk_usage_percent)
        self.metrics_collector.set_gauge('system.process_count', system_metrics.process_count)
        
        # Métricas de la aplicación
        self.metrics_collector.set_gauge('app.cache_hit_rate', app_metrics.cache_hit_rate)
        self.metrics_collector.set_gauge('app.active_sessions', app_metrics.active_sessions)
        self.metrics_collector.set_gauge('app.average_response_time', app_metrics.average_response_time)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard"""
        return {
            'system_metrics': asdict(self.last_system_metrics) if self.last_system_metrics else None,
            'app_metrics': asdict(self.last_app_metrics) if self.last_app_metrics else None,
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'recent_alerts': [asdict(alert) for alert in self.alert_manager.alerts[-10:]],
            'status': 'running' if self.running else 'stopped'
        }

# Instancia global
_monitoring_system: Optional[MonitoringSystem] = None

def get_monitoring_system() -> MonitoringSystem:
    """Obtiene instancia global del sistema de monitoreo"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system

def init_monitoring(config: Dict[str, Any] = None):
    """Inicializa sistema de monitoreo"""
    global _monitoring_system
    _monitoring_system = MonitoringSystem(config)
    _monitoring_system.start()

# Decoradores de conveniencia
def monitor_performance(name: str = None, labels: Dict[str, str] = None):
    """Decorador para monitorear performance"""
    return get_monitoring_system().performance_monitor.time_function(name, labels)

def log_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE):
    """Función de conveniencia para logging de métricas"""
    get_monitoring_system().metrics_collector.record_metric(name, value, metric_type) 