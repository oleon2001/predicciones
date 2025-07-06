# DEPENDENCY INJECTION CONTAINER
"""
Container de inyección de dependencias para el sistema de predicción
Implementa el patrón IoC (Inversion of Control) para gestionar dependencias
"""

from typing import Dict, Type, Any, Optional, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from functools import wraps
import inspect

from config.system_config import SystemConfig
from core.interfaces import *
from core.risk_manager import AdvancedRiskManager
from models.advanced_ml_models import ModelFactory, AdvancedEnsemble
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.macro_analyzer import MacroAnalyzer
from backtesting.advanced_backtester import AdvancedBacktester

logger = logging.getLogger(__name__)

@dataclass
class ServiceRegistration:
    """Registro de servicio"""
    service_type: Type
    implementation: Type
    lifetime: str = "singleton"  # singleton, transient, scoped
    factory: Optional[Callable] = None
    dependencies: Optional[Dict[str, str]] = None

class LifetimeManager:
    """Gestor de ciclo de vida de servicios"""
    
    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
    
    def get_singleton(self, service_type: Type, factory: Callable) -> Any:
        """Obtiene instancia singleton"""
        if service_type not in self._singletons:
            self._singletons[service_type] = factory()
        return self._singletons[service_type]
    
    def get_scoped(self, service_type: Type, factory: Callable) -> Any:
        """Obtiene instancia scoped"""
        if self._current_scope is None:
            raise RuntimeError("No hay scope activo")
        
        scope_instances = self._scoped_instances.get(self._current_scope, {})
        if service_type not in scope_instances:
            scope_instances[service_type] = factory()
            self._scoped_instances[self._current_scope] = scope_instances
        
        return scope_instances[service_type]
    
    def get_transient(self, factory: Callable) -> Any:
        """Obtiene instancia transient (nueva cada vez)"""
        return factory()
    
    def start_scope(self, scope_id: str):
        """Inicia nuevo scope"""
        self._current_scope = scope_id
        self._scoped_instances[scope_id] = {}
    
    def end_scope(self, scope_id: str):
        """Termina scope y limpia recursos"""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
        if self._current_scope == scope_id:
            self._current_scope = None

class DependencyContainer:
    """Container principal de inyección de dependencias"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._lifetime_manager = LifetimeManager()
        self._is_configured = False
    
    def register_singleton(self, service_type: Type, implementation: Type, 
                          dependencies: Optional[Dict[str, str]] = None) -> 'DependencyContainer':
        """Registra servicio como singleton"""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime="singleton",
            dependencies=dependencies
        )
        return self
    
    def register_transient(self, service_type: Type, implementation: Type,
                          dependencies: Optional[Dict[str, str]] = None) -> 'DependencyContainer':
        """Registra servicio como transient"""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime="transient",
            dependencies=dependencies
        )
        return self
    
    def register_scoped(self, service_type: Type, implementation: Type,
                       dependencies: Optional[Dict[str, str]] = None) -> 'DependencyContainer':
        """Registra servicio como scoped"""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime="scoped",
            dependencies=dependencies
        )
        return self
    
    def register_factory(self, service_type: Type, factory: Callable,
                        lifetime: str = "singleton") -> 'DependencyContainer':
        """Registra factory para servicio"""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=None,
            lifetime=lifetime,
            factory=factory
        )
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'DependencyContainer':
        """Registra instancia específica"""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation=None,
            lifetime="singleton",
            factory=lambda: instance
        )
        return self
    
    def resolve(self, service_type: Type) -> Any:
        """Resuelve dependencia"""
        if service_type not in self._services:
            raise ValueError(f"Servicio {service_type} no registrado")
        
        registration = self._services[service_type]
        
        # Crear factory
        if registration.factory:
            factory = registration.factory
        else:
            factory = lambda: self._create_instance(registration)
        
        # Resolver según lifetime
        if registration.lifetime == "singleton":
            return self._lifetime_manager.get_singleton(service_type, factory)
        elif registration.lifetime == "transient":
            return self._lifetime_manager.get_transient(factory)
        elif registration.lifetime == "scoped":
            return self._lifetime_manager.get_scoped(service_type, factory)
        else:
            raise ValueError(f"Lifetime desconocido: {registration.lifetime}")
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Crea instancia resolviendo dependencias"""
        implementation = registration.implementation
        
        # Obtener constructor
        constructor = implementation.__init__
        sig = inspect.signature(constructor)
        
        # Resolver dependencias
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Buscar tipo de dependencia
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                continue
            
            # Resolver dependencia
            try:
                dependency = self.resolve(param_type)
                kwargs[param_name] = dependency
            except ValueError:
                # Si no se puede resolver, usar default si existe
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                else:
                    logger.warning(f"No se pudo resolver dependencia {param_type} para {implementation}")
        
        return implementation(**kwargs)
    
    def configure_default_services(self, config: SystemConfig):
        """Configura servicios por defecto"""
        if self._is_configured:
            return
        
        # Configuración como singleton
        self.register_instance(SystemConfig, config)
        
        # Servicios principales
        self.register_singleton(IRiskManager, AdvancedRiskManager)
        self.register_singleton(ISentimentAnalyzer, SentimentAnalyzer)
        self.register_singleton(IMacroAnalyzer, MacroAnalyzer)
        self.register_singleton(IBacktester, AdvancedBacktester)
        
        # Factories
        self.register_factory(IEnsembleModel, lambda: ModelFactory.create_ensemble(config))
        
        # Cache Manager
        from core.cache_manager import CacheManager
        self.register_singleton(ICacheManager, CacheManager)
        
        self._is_configured = True
        logger.info("Container configurado con servicios por defecto")
    
    def start_scope(self, scope_id: str):
        """Inicia nuevo scope"""
        self._lifetime_manager.start_scope(scope_id)
    
    def end_scope(self, scope_id: str):
        """Termina scope"""
        self._lifetime_manager.end_scope(scope_id)

# Decorador para inyección automática
def inject(**dependencies):
    """Decorador para inyección automática de dependencias"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            
            # Resolver dependencias
            for param_name, service_type in dependencies.items():
                if param_name not in kwargs:
                    kwargs[param_name] = container.resolve(service_type)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Container global
_container: Optional[DependencyContainer] = None

def get_container() -> DependencyContainer:
    """Obtiene container global"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container

def configure_container(config: SystemConfig) -> DependencyContainer:
    """Configura container con configuración específica"""
    container = get_container()
    container.configure_default_services(config)
    return container

# Context manager para scopes
class DependencyScope:
    """Context manager para manejar scopes de dependencias"""
    
    def __init__(self, scope_id: str):
        self.scope_id = scope_id
        self.container = get_container()
    
    def __enter__(self):
        self.container.start_scope(self.scope_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.end_scope(self.scope_id)

# Utilities para testing
class TestContainer:
    """Container para testing con mocks"""
    
    def __init__(self, base_container: DependencyContainer):
        self.base_container = base_container
        self.overrides: Dict[Type, Any] = {}
    
    def override(self, service_type: Type, mock_instance: Any):
        """Override servicio con mock"""
        self.overrides[service_type] = mock_instance
    
    def resolve(self, service_type: Type) -> Any:
        """Resuelve con overrides"""
        if service_type in self.overrides:
            return self.overrides[service_type]
        return self.base_container.resolve(service_type)

def create_test_container(config: SystemConfig) -> TestContainer:
    """Crea container para testing"""
    base_container = DependencyContainer()
    base_container.configure_default_services(config)
    return TestContainer(base_container) 