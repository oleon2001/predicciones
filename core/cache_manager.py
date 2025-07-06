# SISTEMA DE CACHING AVANZADO
"""
Sistema de caching con TTL, invalidación inteligente y persistencia
Optimizado para datos financieros y predicciones
"""

import time
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from threading import RLock
import logging
from abc import ABC, abstractmethod
from enum import Enum

from core.interfaces import ICacheManager

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Estrategias de cache"""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # LRU + TTL

@dataclass
class CacheEntry:
    """Entrada del cache"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Verifica si la entrada está expirada"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def refresh_access(self):
        """Actualiza tiempo de acceso"""
        self.accessed_at = time.time()
        self.access_count += 1

class CacheStats:
    """Estadísticas del cache"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_mb = 0.0
        self.entries_count = 0
        self.start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def uptime_hours(self) -> float:
        """Tiempo de funcionamiento en horas"""
        return (time.time() - self.start_time) / 3600

class CacheManager(ICacheManager):
    """Gestor de cache avanzado"""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.HYBRID,
                 persist_to_disk: bool = True,
                 cache_dir: str = "cache"):
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.persist_to_disk = persist_to_disk
        self.cache_dir = Path(cache_dir)
        
        # Crear directorio de cache
        if self.persist_to_disk:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Almacenamiento en memoria
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._stats = CacheStats()
        
        # Índices para estrategias
        self._access_order: List[str] = []  # Para LRU
        self._expiration_times: Dict[str, float] = {}  # Para TTL
        
        # Cargar cache persistente
        if self.persist_to_disk:
            self._load_persistent_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Verificar expiración
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Actualizar acceso
            entry.refresh_access()
            self._update_access_order(key)
            
            self._stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Establece valor en cache"""
        with self._lock:
            # Usar TTL por defecto si no se especifica
            if ttl is None:
                ttl = self.default_ttl
            
            # Calcular tamaño
            size_bytes = self._calculate_size(value)
            
            # Verificar espacio disponible
            self._ensure_space(size_bytes)
            
            # Crear entrada
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remover entrada anterior si existe
            if key in self._cache:
                self._remove_entry(key)
            
            # Agregar nueva entrada
            self._cache[key] = entry
            self._update_access_order(key)
            
            if ttl:
                self._expiration_times[key] = time.time() + ttl
            
            # Persistir si está configurado
            if self.persist_to_disk:
                self._persist_entry(key, entry)
            
            self._update_stats()
    
    def delete(self, key: str) -> None:
        """Elimina entrada del cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._update_stats()
    
    def clear(self) -> None:
        """Limpia todo el cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._expiration_times.clear()
            
            if self.persist_to_disk:
                # Limpiar archivos persistentes
                for file_path in self.cache_dir.glob("*.cache"):
                    try:
                        file_path.unlink()
                    except OSError:
                        pass
            
            self._update_stats()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalida entradas que coinciden con el patrón"""
        with self._lock:
            keys_to_remove = []
            
            for key in self._cache:
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self._update_stats()
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        with self._lock:
            self._update_stats()
            return {
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'hit_rate': self._stats.hit_rate,
                'evictions': self._stats.evictions,
                'size_mb': self._stats.size_mb,
                'entries_count': self._stats.entries_count,
                'uptime_hours': self._stats.uptime_hours,
                'strategy': self.strategy.value,
                'max_size_mb': self.max_size_bytes / (1024 * 1024)
            }
    
    def cleanup_expired(self) -> int:
        """Limpia entradas expiradas"""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            self._update_stats()
            return len(expired_keys)
    
    def _calculate_size(self, value: Any) -> int:
        """Calcula tamaño aproximado del valor"""
        try:
            # Intentar con pickle para obtener tamaño serializado
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback a tamaño estimado
            if isinstance(value, (int, float)):
                return 8
            elif isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 1024  # Estimación por defecto
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Asegura espacio disponible en cache"""
        current_size = sum(entry.size_bytes for entry in self._cache.values())
        
        if current_size + required_bytes <= self.max_size_bytes:
            return
        
        # Necesitamos liberar espacio
        bytes_to_free = (current_size + required_bytes) - self.max_size_bytes
        
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(bytes_to_free)
        elif self.strategy == CacheStrategy.TTL:
            self._evict_expired_and_oldest(bytes_to_free)
        else:  # HYBRID
            self._evict_hybrid(bytes_to_free)
    
    def _evict_lru(self, bytes_to_free: int) -> None:
        """Evicts usando estrategia LRU"""
        freed_bytes = 0
        
        # Ordenar por último acceso
        sorted_keys = sorted(self._cache.keys(), 
                           key=lambda k: self._cache[k].accessed_at)
        
        for key in sorted_keys:
            if freed_bytes >= bytes_to_free:
                break
            
            freed_bytes += self._cache[key].size_bytes
            self._remove_entry(key)
            self._stats.evictions += 1
    
    def _evict_expired_and_oldest(self, bytes_to_free: int) -> None:
        """Evicts expirados y luego más antiguos"""
        freed_bytes = 0
        
        # Primero eliminar expirados
        expired_keys = [k for k, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            freed_bytes += self._cache[key].size_bytes
            self._remove_entry(key)
        
        # Si no es suficiente, eliminar más antiguos
        if freed_bytes < bytes_to_free:
            sorted_keys = sorted(self._cache.keys(), 
                               key=lambda k: self._cache[k].created_at)
            
            for key in sorted_keys:
                if freed_bytes >= bytes_to_free:
                    break
                
                freed_bytes += self._cache[key].size_bytes
                self._remove_entry(key)
                self._stats.evictions += 1
    
    def _evict_hybrid(self, bytes_to_free: int) -> None:
        """Evicts usando estrategia híbrida"""
        freed_bytes = 0
        
        # Calcular score para cada entrada (menor score = mayor prioridad de eviction)
        entries_with_scores = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Score basado en: age, access frequency, time since last access
            age_score = current_time - entry.created_at
            access_score = 1.0 / (entry.access_count + 1)
            recency_score = current_time - entry.accessed_at
            
            # Bonus por no estar expirado
            expiry_bonus = 0 if entry.is_expired() else -100
            
            total_score = age_score + access_score * 10 + recency_score + expiry_bonus
            entries_with_scores.append((key, total_score))
        
        # Ordenar por score (menor primero)
        entries_with_scores.sort(key=lambda x: x[1])
        
        # Evict entries with highest scores
        for key, _ in entries_with_scores:
            if freed_bytes >= bytes_to_free:
                break
            
            freed_bytes += self._cache[key].size_bytes
            self._remove_entry(key)
            self._stats.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remueve entrada del cache"""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        if key in self._expiration_times:
            del self._expiration_times[key]
        
        # Remover archivo persistente
        if self.persist_to_disk:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            try:
                cache_file.unlink()
            except OSError:
                pass
    
    def _update_access_order(self, key: str) -> None:
        """Actualiza orden de acceso para LRU"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _update_stats(self) -> None:
        """Actualiza estadísticas"""
        self._stats.entries_count = len(self._cache)
        self._stats.size_mb = sum(entry.size_bytes for entry in self._cache.values()) / (1024 * 1024)
    
    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persiste entrada a disco"""
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            
            # Crear estructura para persistir
            data = {
                'key': key,
                'value': entry.value,
                'created_at': entry.created_at,
                'ttl': entry.ttl,
                'metadata': {
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Error persistiendo entrada {key}: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Carga cache persistente desde disco"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Verificar si no ha expirado
                    if data.get('ttl'):
                        if time.time() - data['created_at'] > data['ttl']:
                            cache_file.unlink()
                            continue
                    
                    # Recrear entrada
                    entry = CacheEntry(
                        key=data['key'],
                        value=data['value'],
                        created_at=data['created_at'],
                        accessed_at=data['created_at'],
                        ttl=data.get('ttl'),
                        access_count=data.get('metadata', {}).get('access_count', 0),
                        size_bytes=data.get('metadata', {}).get('size_bytes', 0)
                    )
                    
                    self._cache[data['key']] = entry
                    self._access_order.append(data['key'])
                    
                except Exception as e:
                    logger.warning(f"Error cargando cache file {cache_file}: {e}")
                    try:
                        cache_file.unlink()
                    except:
                        pass
        
        except Exception as e:
            logger.warning(f"Error cargando cache persistente: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Genera hash para nombre de archivo"""
        return hashlib.md5(key.encode()).hexdigest()

# Decorador para cache automático
def cached(ttl: int = 3600, cache_manager: Optional[CacheManager] = None):
    """Decorador para cache automático de funciones"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Obtener cache manager
            if cache_manager is None:
                from core.dependency_container import get_container
                cm = get_container().resolve(ICacheManager)
            else:
                cm = cache_manager
            
            # Generar clave de cache
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Intentar obtener del cache
            result = cm.get(cache_key)
            if result is not None:
                return result
            
            # Ejecutar función y cachear resultado
            result = func(*args, **kwargs)
            cm.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator 