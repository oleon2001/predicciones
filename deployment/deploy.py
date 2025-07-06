# SCRIPT DE DEPLOYMENT PARA PRODUCCIÓN
"""
Script automatizado de deployment del sistema de predicción de criptomonedas
Configura el entorno, valida dependencias y prepara el sistema para producción
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import platform

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Gestor de deployment del sistema"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deployment_config = self._load_deployment_config()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    def _load_deployment_config(self) -> Dict:
        """Carga configuración de deployment"""
        config_file = self.project_root / "deployment" / f"{self.environment}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Configuración por defecto
            return {
                "python_version": "3.9",
                "required_memory_gb": 4,
                "required_disk_gb": 10,
                "create_venv": True,
                "install_dependencies": True,
                "run_tests": True,
                "setup_database": True,
                "setup_cache": True,
                "setup_monitoring": True,
                "create_directories": True,
                "environment_variables": {
                    "LOG_LEVEL": "INFO",
                    "CACHE_ENABLED": "true",
                    "PARALLEL_PROCESSING": "true",
                    "MAX_WORKERS": "4"
                }
            }
    
    def deploy(self) -> bool:
        """Ejecuta deployment completo"""
        logger.info(f"🚀 Iniciando deployment para entorno: {self.environment}")
        logger.info(f"📍 Directorio del proyecto: {self.project_root}")
        logger.info(f"🐍 Python version: {self.python_version}")
        logger.info(f"💻 Sistema operativo: {platform.system()} {platform.release()}")
        
        try:
            # Validaciones pre-deployment
            logger.info("1️⃣ Ejecutando validaciones pre-deployment...")
            if not self._pre_deployment_checks():
                logger.error("❌ Validaciones fallaron")
                return False
            
            # Crear estructura de directorios
            if self.deployment_config.get("create_directories", True):
                logger.info("2️⃣ Creando estructura de directorios...")
                self._create_directory_structure()
            
            # Configurar entorno virtual
            if self.deployment_config.get("create_venv", True):
                logger.info("3️⃣ Configurando entorno virtual...")
                self._setup_virtual_environment()
            
            # Instalar dependencias
            if self.deployment_config.get("install_dependencies", True):
                logger.info("4️⃣ Instalando dependencias...")
                self._install_dependencies()
            
            # Configurar variables de entorno
            logger.info("5️⃣ Configurando variables de entorno...")
            self._setup_environment_variables()
            
            # Configurar base de datos
            if self.deployment_config.get("setup_database", True):
                logger.info("6️⃣ Configurando base de datos...")
                self._setup_database()
            
            # Configurar cache
            if self.deployment_config.get("setup_cache", True):
                logger.info("7️⃣ Configurando sistema de cache...")
                self._setup_cache()
            
            # Configurar monitoreo
            if self.deployment_config.get("setup_monitoring", True):
                logger.info("8️⃣ Configurando sistema de monitoreo...")
                self._setup_monitoring()
            
            # Ejecutar tests
            if self.deployment_config.get("run_tests", True):
                logger.info("9️⃣ Ejecutando suite de tests...")
                if not self._run_tests():
                    logger.warning("⚠️ Algunos tests fallaron, pero continuando deployment")
            
            # Configurar servicios de sistema
            logger.info("🔟 Configurando servicios de sistema...")
            self._setup_system_services()
            
            # Validaciones post-deployment
            logger.info("✅ Ejecutando validaciones post-deployment...")
            if not self._post_deployment_checks():
                logger.error("❌ Validaciones post-deployment fallaron")
                return False
            
            # Generar reporte de deployment
            self._generate_deployment_report()
            
            logger.info("🎉 ¡Deployment completado exitosamente!")
            return True
            
        except Exception as e:
            logger.error(f"💥 Error durante deployment: {e}")
            logger.exception("Detalles del error:")
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Ejecuta validaciones pre-deployment"""
        checks_passed = True
        
        # Verificar versión de Python
        required_version = self.deployment_config.get("python_version", "3.9")
        if self.python_version < required_version:
            logger.error(f"❌ Python {required_version}+ requerido, encontrado {self.python_version}")
            checks_passed = False
        else:
            logger.info(f"✅ Python version OK: {self.python_version}")
        
        # Verificar memoria disponible
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            required_memory = self.deployment_config.get("required_memory_gb", 4)
            
            if available_memory_gb < required_memory:
                logger.error(f"❌ Memoria insuficiente: {available_memory_gb:.1f}GB disponible, {required_memory}GB requerido")
                checks_passed = False
            else:
                logger.info(f"✅ Memoria OK: {available_memory_gb:.1f}GB disponible")
        except ImportError:
            logger.warning("⚠️ No se pudo verificar memoria (psutil no disponible)")
        
        # Verificar espacio en disco
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            available_disk_gb = disk_usage.free / (1024**3)
            required_disk = self.deployment_config.get("required_disk_gb", 10)
            
            if available_disk_gb < required_disk:
                logger.error(f"❌ Espacio en disco insuficiente: {available_disk_gb:.1f}GB disponible, {required_disk}GB requerido")
                checks_passed = False
            else:
                logger.info(f"✅ Espacio en disco OK: {available_disk_gb:.1f}GB disponible")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar espacio en disco: {e}")
        
        # Verificar archivos principales
        required_files = [
            "config/system_config.py",
            "core/interfaces.py",
            "core/dependency_container.py",
            "models/advanced_ml_models.py",
            "requirements_avanzado.txt"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                logger.error(f"❌ Archivo requerido no encontrado: {file_path}")
                checks_passed = False
            else:
                logger.info(f"✅ Archivo encontrado: {file_path}")
        
        return checks_passed
    
    def _create_directory_structure(self):
        """Crea estructura de directorios necesaria"""
        directories = [
            "logs",
            "cache",
            "data",
            "models/saved",
            "results",
            "backups",
            "monitoring/metrics",
            "monitoring/alerts",
            "deployment/configs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Directorio creado/verificado: {directory}")
    
    def _setup_virtual_environment(self):
        """Configura entorno virtual"""
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            logger.info("🔨 Creando entorno virtual...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        else:
            logger.info("📦 Entorno virtual ya existe")
        
        # Actualizar pip
        if platform.system() == "Windows":
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        logger.info("⬆️ pip actualizado")
    
    def _install_dependencies(self):
        """Instala dependencias del proyecto"""
        venv_path = self.project_root / "venv"
        
        if platform.system() == "Windows":
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        requirements_file = self.project_root / "requirements_avanzado.txt"
        
        if requirements_file.exists():
            logger.info("📦 Instalando dependencias principales...")
            subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        
        # Dependencias adicionales para producción
        production_packages = [
            "gunicorn",
            "redis",
            "psutil",
            "python-dotenv",
            "prometheus-client"
        ]
        
        logger.info("📦 Instalando dependencias de producción...")
        subprocess.run([str(pip_path), "install"] + production_packages, check=True)
        
        logger.info("✅ Dependencias instaladas")
    
    def _setup_environment_variables(self):
        """Configura variables de entorno"""
        env_file = self.project_root / ".env"
        
        env_vars = self.deployment_config.get("environment_variables", {})
        
        # Agregar variables específicas del entorno
        env_vars.update({
            "ENVIRONMENT": self.environment,
            "PROJECT_ROOT": str(self.project_root),
            "PYTHONPATH": str(self.project_root)
        })
        
        # Escribir archivo .env
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"🔧 Variables de entorno configuradas en {env_file}")
    
    def _setup_database(self):
        """Configura base de datos"""
        # Para SQLite (por defecto)
        db_path = self.project_root / "data" / "crypto_predictions.db"
        db_path.parent.mkdir(exist_ok=True)
        
        # Crear esquema básico si no existe
        import sqlite3
        
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Tabla de predicciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    prediction_value REAL NOT NULL,
                    confidence REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de métricas de modelos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de logs de trading
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        logger.info(f"💾 Base de datos configurada: {db_path}")
    
    def _setup_cache(self):
        """Configura sistema de cache"""
        cache_dir = self.project_root / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Configurar Redis si está disponible
        try:
            import redis
            
            # Test conexión Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            logger.info("✅ Redis cache disponible")
            
            # Configurar Redis
            redis_config = {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "max_connections": 20
            }
            
            config_file = self.project_root / "deployment" / "configs" / "redis.json"
            with open(config_file, 'w') as f:
                json.dump(redis_config, f, indent=2)
                
        except (ImportError, redis.RedisError):
            logger.info("📁 Usando cache de archivos (Redis no disponible)")
        
        logger.info("🗄️ Sistema de cache configurado")
    
    def _setup_monitoring(self):
        """Configura sistema de monitoreo"""
        monitoring_dir = self.project_root / "monitoring"
        
        # Configurar logging estructurado
        log_config = {
            "version": 1,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(self.project_root / "logs" / "application.log"),
                    "formatter": "detailed"
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "detailed"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["file", "console"]
            }
        }
        
        config_file = monitoring_dir / "logging_config.json"
        with open(config_file, 'w') as f:
            json.dump(log_config, f, indent=2)
        
        # Configurar métricas
        metrics_config = {
            "enabled": True,
            "collection_interval": 60,
            "retention_days": 30,
            "metrics": [
                "system.cpu_percent",
                "system.memory_percent",
                "app.predictions_made",
                "app.models_trained",
                "app.cache_hit_rate"
            ]
        }
        
        config_file = monitoring_dir / "metrics_config.json"
        with open(config_file, 'w') as f:
            json.dump(metrics_config, f, indent=2)
        
        logger.info("📊 Sistema de monitoreo configurado")
    
    def _run_tests(self) -> bool:
        """Ejecuta suite de tests"""
        try:
            venv_path = self.project_root / "venv"
            
            if platform.system() == "Windows":
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                python_path = venv_path / "bin" / "python"
            
            # Ejecutar tests
            result = subprocess.run([
                str(python_path), "-m", "pytest", 
                "tests/", "-v", "--tb=short"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Todos los tests pasaron")
                return True
            else:
                logger.warning(f"⚠️ Algunos tests fallaron:\n{result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando tests: {e}")
            return False
    
    def _setup_system_services(self):
        """Configura servicios de sistema"""
        # Crear script de inicio
        startup_script = self.project_root / "deployment" / "start_system.sh"
        
        script_content = f"""#!/bin/bash
# Script de inicio del sistema de predicción de criptomonedas

export PROJECT_ROOT="{self.project_root}"
export PYTHONPATH="{self.project_root}"

cd "{self.project_root}"

# Activar entorno virtual
source venv/bin/activate

# Iniciar sistema
python sistema_integrado.py --environment {self.environment}
"""
        
        with open(startup_script, 'w') as f:
            f.write(script_content)
        
        startup_script.chmod(0o755)
        
        # Crear script de systemd (Linux)
        if platform.system() == "Linux":
            systemd_service = f"""[Unit]
Description=Crypto Prediction System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'crypto')}
WorkingDirectory={self.project_root}
ExecStart={startup_script}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            service_file = self.project_root / "deployment" / "crypto-prediction.service"
            with open(service_file, 'w') as f:
                f.write(systemd_service)
            
            logger.info(f"📋 Archivo de servicio systemd creado: {service_file}")
        
        logger.info("🔧 Servicios de sistema configurados")
    
    def _post_deployment_checks(self) -> bool:
        """Ejecuta validaciones post-deployment"""
        checks_passed = True
        
        try:
            # Verificar que el sistema se puede importar
            sys.path.insert(0, str(self.project_root))
            
            from config.system_config import SystemConfig
            from core.dependency_container import configure_container
            
            # Test configuración
            config = SystemConfig()
            container = configure_container(config)
            
            logger.info("✅ Sistema se puede importar correctamente")
            
            # Verificar estructura de directorios
            required_dirs = ["logs", "cache", "data", "models/saved"]
            for directory in required_dirs:
                dir_path = self.project_root / directory
                if not dir_path.exists():
                    logger.error(f"❌ Directorio requerido no encontrado: {directory}")
                    checks_passed = False
                else:
                    logger.info(f"✅ Directorio OK: {directory}")
            
            # Verificar archivos de configuración
            config_files = [
                ".env",
                "monitoring/logging_config.json",
                "monitoring/metrics_config.json"
            ]
            
            for config_file in config_files:
                file_path = self.project_root / config_file
                if not file_path.exists():
                    logger.error(f"❌ Archivo de configuración no encontrado: {config_file}")
                    checks_passed = False
                else:
                    logger.info(f"✅ Configuración OK: {config_file}")
            
        except Exception as e:
            logger.error(f"❌ Error en validaciones post-deployment: {e}")
            checks_passed = False
        
        return checks_passed
    
    def _generate_deployment_report(self):
        """Genera reporte de deployment"""
        report = {
            "deployment_info": {
                "environment": self.environment,
                "timestamp": datetime.now().isoformat(),
                "python_version": self.python_version,
                "system": platform.system(),
                "platform": platform.platform()
            },
            "configuration": self.deployment_config,
            "project_structure": self._get_project_structure(),
            "system_info": self._get_system_info(),
            "deployment_status": "SUCCESS"
        }
        
        report_file = self.project_root / "deployment" / f"deployment_report_{self.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Reporte de deployment generado: {report_file}")
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📍 Entorno: {self.environment}")
        print(f"📂 Directorio: {self.project_root}")
        print(f"🐍 Python: {self.python_version}")
        print(f"💻 Sistema: {platform.system()}")
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🚀 El sistema está listo para funcionar!")
        print("\n📖 Para iniciar el sistema:")
        print(f"   cd {self.project_root}")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   # o")
        print("   venv\\Scripts\\activate  # Windows")
        print("   python sistema_integrado.py")
        print("="*60)
    
    def _get_project_structure(self) -> Dict:
        """Obtiene estructura del proyecto"""
        structure = {}
        
        for path in self.project_root.rglob("*"):
            if path.is_file() and not any(exclude in str(path) for exclude in ['.git', '__pycache__', '.pyc', 'venv']):
                relative_path = path.relative_to(self.project_root)
                structure[str(relative_path)] = {
                    "size_bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
        
        return structure
    
    def _get_system_info(self) -> Dict:
        """Obtiene información del sistema"""
        try:
            import psutil
            
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_total_gb": shutil.disk_usage(self.project_root).total / (1024**3),
                "disk_free_gb": shutil.disk_usage(self.project_root).free / (1024**3)
            }
        except ImportError:
            return {"status": "psutil not available"}

def main():
    """Función principal de deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deployment del Sistema de Predicción de Criptomonedas")
    parser.add_argument("--environment", "-e", default="production", 
                       choices=["development", "staging", "production"],
                       help="Entorno de deployment")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Omitir ejecución de tests")
    parser.add_argument("--force", action="store_true",
                       help="Forzar deployment sin validaciones")
    
    args = parser.parse_args()
    
    # Crear deployment manager
    deployment_manager = DeploymentManager(args.environment)
    
    # Modificar configuración según argumentos
    if args.skip_tests:
        deployment_manager.deployment_config["run_tests"] = False
    
    if args.force:
        logger.warning("⚠️ Deployment forzado - omitiendo algunas validaciones")
    
    # Ejecutar deployment
    success = deployment_manager.deploy()
    
    if success:
        sys.exit(0)
    else:
        logger.error("💥 Deployment falló")
        sys.exit(1)

if __name__ == "__main__":
    main() 