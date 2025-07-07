# 🚀 GUÍA COMPLETA DE DESPLIEGUE - SISTEMA DE TRADING

## 📋 REQUISITOS PREVIOS

### Sistema Operativo
- **Linux**: Ubuntu 20.04+ (recomendado)
- **Windows**: Windows 10/11 con WSL2
- **macOS**: macOS 11+

### Software Requerido
```bash
# Python 3.8+ (recomendado 3.12)
python --version

# Git
git --version

# PostgreSQL (opcional pero recomendado)
psql --version

# Redis (opcional para cache)
redis-server --version
```

## 🔧 INSTALACIÓN PASO A PASO

### **PASO 1: Clonar y Configurar**

```bash
# 1. Clonar el repositorio
git clone <tu-repositorio>
cd predicciones

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### **PASO 2: Instalación Automática**

```bash
# Ejecutar instalación automática
python instalacion_automatica.py
```

Si prefieres instalar manualmente:

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements_avanzado.txt
```

### **PASO 3: Configuración de APIs**

```bash
# 1. Copiar template de configuración
cp config/config_template.json config/config.json

# 2. Crear archivo de variables de entorno
cp .env.template .env
```

Editar `.env` con tus API keys:
```env
# APIs de trading
BINANCE_API_KEY=tu_api_key_binance
BINANCE_API_SECRET=tu_api_secret_binance

# APIs de noticias
NEWS_API_KEY=tu_news_api_key

# APIs de datos macro (opcional)
ALPHA_VANTAGE_KEY=tu_alpha_vantage_key
FRED_API_KEY=tu_fred_api_key

# Configuración de sistema
LOG_LEVEL=INFO
CACHE_ENABLED=true
PARALLEL_PROCESSING=true
MAX_WORKERS=4
```

### **PASO 4: Configuración Segura**

```bash
# Configurar sistema de seguridad
python config/secure_config.py --setup

# Migrar configuración existente
python config/migrate_config.py
```

## 🎯 OPCIONES DE DESPLIEGUE

### **OPCIÓN 1: Despliegue Básico (Desarrollo)**

```bash
# 1. Ejecutar tests
python test_sistema.py

# 2. Lanzar sistema integrado
python lanzar_sistema.py

# 3. O usar el demo
python demo_sistema.py
```

### **OPCIÓN 2: Despliegue de Producción**

```bash
# 1. Ejecutar deployment automático
python deployment/deploy.py --environment=production

# 2. Iniciar sistema de producción
python deployment/production_system.py
```

### **OPCIÓN 3: Despliegue con Docker**

```bash
# 1. Construir imagen
docker build -t crypto-trading-system .

# 2. Ejecutar contenedor
docker run -d \
  --name trading-system \
  -p 8080:8080 \
  -e BINANCE_API_KEY=tu_key \
  -e BINANCE_API_SECRET=tu_secret \
  -v $(pwd)/config:/app/config \
  crypto-trading-system
```

## 📊 VERIFICACIÓN DEL DESPLIEGUE

### **Health Check**
```bash
# Verificar salud del sistema
curl http://localhost:8080/health

# Verificar métricas
curl http://localhost:8080/api/system/metrics
```

### **Tests de Integración**
```bash
# Ejecutar suite completa de tests
python tests/comprehensive_test_suite.py

# Tests específicos
python tests/test_integration.py
```

## 🔧 CONFIGURACIÓN AVANZADA

### **Base de Datos (PostgreSQL)**
```sql
-- Crear base de datos
CREATE DATABASE crypto_trading;
CREATE USER trading_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE crypto_trading TO trading_user;
```

### **Redis Cache**
```bash
# Configurar Redis
redis-server --port 6379 --daemonize yes

# Verificar
redis-cli ping
```

### **Monitoreo**
```bash
# Iniciar Prometheus (opcional)
prometheus --config.file=monitoring/prometheus.yml

# Iniciar Grafana (opcional)
grafana-server --config=monitoring/grafana.ini
```

## 🚀 COMANDOS DE EJECUCIÓN

### **Análisis Básico**
```bash
# Sistema integrado
python sistema_integrado.py

# Predicción avanzada
python prediccion_avanzada.py

# Solo predicción básica
python prediccion.py
```

### **Backtesting**
```bash
# Backtesting avanzado
python -c "
from backtesting.advanced_backtester import AdvancedBacktester
backtester = AdvancedBacktester()
# Ejecutar backtesting
"

# Backtesting realista
python -c "
from backtesting.realistic_backtester import RealisticBacktester
backtester = RealisticBacktester()
# Ejecutar backtesting
"
```

### **Gestión de Riesgo**
```bash
# Test de risk manager
python -c "
from core.robust_risk_manager import RobustRiskManager
risk_manager = RobustRiskManager()
# Ejecutar pruebas
"
```

## 📈 MONITOREO Y MANTENIMIENTO

### **Logs**
```bash
# Ver logs del sistema
tail -f logs/trading_system.jsonl

# Logs por componente
tail -f logs/risk_management.log
tail -f logs/ml_pipeline.log
```

### **Métricas**
- **API Health**: `http://localhost:8080/health`
- **System Status**: `http://localhost:8080/api/system/status`
- **Risk Metrics**: `http://localhost:8080/api/risk/report`
- **ML Performance**: `http://localhost:8080/api/ml/performance`

## 🔒 SEGURIDAD

### **Configuración Segura**
```bash
# Generar claves de encriptación
python config/secure_config.py --generate-keys

# Encriptar configuración
python config/secure_config.py --encrypt-config

# Verificar seguridad
python config/secure_config.py --security-check
```

### **Respaldo**
```bash
# Crear respaldo
python tools/backup_system.py

# Restaurar respaldo
python tools/restore_system.py --backup-file=backup_20240101.tar.gz
```

## 🐛 TROUBLESHOOTING

### **Problemas Comunes**

1. **Error de API Keys**
   ```bash
   # Verificar configuración
   python config/secure_config.py --verify-keys
   ```

2. **Problemas de Dependencias**
   ```bash
   # Reinstalar dependencias
   pip install -r requirements_avanzado.txt --force-reinstall
   ```

3. **Errores de Conexión**
   ```bash
   # Test de conectividad
   python -c "
   from binance.client import Client
   client = Client()
   print(client.get_server_time())
   "
   ```

4. **Problemas de Memoria**
   ```bash
   # Limpiar cache
   python limpiar_sistema.py
   ```

## 📱 INTERFACES DE USUARIO

### **Web Dashboard**
- URL: `http://localhost:8080`
- Autenticación: Configurar en `config/secure_config.py`

### **API REST**
- Base URL: `http://localhost:8080/api`
- Documentación: `http://localhost:8080/api/docs`

### **CLI Tools**
```bash
# Herramientas de línea de comandos
python tools/cli_manager.py --help
```

## 🔄 ACTUALIZACIONES

### **Actualización del Sistema**
```bash
# Actualizar código
git pull origin main

# Actualizar dependencias
pip install -r requirements_avanzado.txt --upgrade

# Migrar configuración
python config/migrate_config.py

# Reiniciar sistema
python deployment/production_system.py --restart
```

### **Rolling Update**
```bash
# Deployment sin downtime
python deployment/deploy.py --rolling-update
```

## 📋 CHECKLIST DE DESPLIEGUE

### **Pre-Despliegue**
- [ ] Python 3.8+ instalado
- [ ] Git configurado
- [ ] API keys obtenidas
- [ ] Configuración de entorno completada
- [ ] Dependencias instaladas

### **Despliegue**
- [ ] Tests pasados
- [ ] Configuración validada
- [ ] Servicios iniciados
- [ ] Health check OK
- [ ] Monitoreo activo

### **Post-Despliegue**
- [ ] Logs verificados
- [ ] Métricas disponibles
- [ ] Alertas configuradas
- [ ] Backup programado
- [ ] Documentación actualizada

## 🎯 PRÓXIMOS PASOS

1. **Configurar alertas** para monitoreo
2. **Implementar CI/CD** para automatización
3. **Configurar backup automático**
4. **Optimizar performance** según uso
5. **Escalar horizontalmente** si es necesario

¡Tu sistema está listo para producción! 🚀 