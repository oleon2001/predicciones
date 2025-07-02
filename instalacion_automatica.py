#!/usr/bin/env python3
"""
SCRIPT DE INSTALACIÓN AUTOMATICA
Sistema Integrado de Análisis y Predicción de Criptomonedas

Este script:
1. Verifica e instala todas las dependencias necesarias
2. Configura el sistema automáticamente
3. Ejecuta tests de conectividad
4. Proporciona guía de uso

Autor: AI Expert Developer & Economist
Versión: 2.0
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import importlib

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Imprime mensaje con color"""
    print(f"{color}{message}{Colors.END}")

def print_header(title):
    """Imprime header con estilo"""
    print("\n" + "="*80)
    print_colored(f"🚀 {title}", Colors.BOLD + Colors.CYAN)
    print("="*80)

def check_python_version():
    """Verifica versión de Python"""
    print_header("VERIFICANDO PYTHON")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored("❌ ERROR: Se requiere Python 3.8 o superior", Colors.RED)
        return False
    
    print_colored("✅ Versión de Python compatible", Colors.GREEN)
    return True

def install_package(package):
    """Instala un paquete específico"""
    try:
        print(f"   Instalando {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print_colored(f"   ✅ {package} instalado correctamente", Colors.GREEN)
            return True
        else:
            print_colored(f"   ❌ Error instalando {package}: {result.stderr}", Colors.RED)
            return False
            
    except subprocess.TimeoutExpired:
        print_colored(f"   ⏰ Timeout instalando {package}", Colors.YELLOW)
        return False
    except Exception as e:
        print_colored(f"   ❌ Error instalando {package}: {e}", Colors.RED)
        return False

def install_dependencies():
    """Instala todas las dependencias"""
    print_header("INSTALANDO DEPENDENCIAS")
    
    # Dependencias básicas
    basic_packages = [
        "python-binance==1.0.19",
        "pandas==2.1.4",
        "numpy==1.24.3",
        "requests==2.31.0",
        "matplotlib==3.8.2",
        "seaborn==0.13.0"
    ]
    
    # Dependencias de análisis técnico
    technical_packages = [
        "ta==0.10.2",
        "scipy==1.11.4"
    ]
    
    # Dependencias de ML
    ml_packages = [
        "scikit-learn==1.3.2",
        "xgboost==2.0.3"
    ]
    
    # Dependencias opcionales
    optional_packages = [
        "newsapi-python==0.2.6",
        "plotly==5.17.0",
        "statsmodels==0.14.1"
    ]
    
    # TensorFlow (separado por compatibilidad)
    tensorflow_packages = [
        "tensorflow==2.15.0"
    ]
    
    success_count = 0
    total_packages = 0
    
    # Actualizar pip primero
    print("📦 Actualizando pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                  capture_output=True)
    
    # Instalar por categorías
    categories = [
        ("Básicas", basic_packages),
        ("Análisis Técnico", technical_packages), 
        ("Machine Learning", ml_packages),
        ("Opcionales", optional_packages)
    ]
    
    for category_name, packages in categories:
        print(f"\n📚 Instalando librerías {category_name}...")
        for package in packages:
            total_packages += 1
            if install_package(package):
                success_count += 1
    
    # TensorFlow por separado (puede fallar en algunos sistemas)
    print(f"\n🧠 Instalando TensorFlow (Deep Learning)...")
    print("   Nota: Si falla, el sistema funcionará sin LSTM")
    
    for package in tensorflow_packages:
        total_packages += 1
        if install_package(package):
            success_count += 1
        else:
            print_colored("   ⚠️ TensorFlow falló, pero el sistema funcionará sin LSTM", Colors.YELLOW)
    
    print(f"\n📊 Resumen de instalación:")
    print(f"   • Exitosos: {success_count}/{total_packages}")
    print(f"   • Fallidos: {total_packages - success_count}/{total_packages}")
    
    if success_count >= total_packages * 0.8:  # 80% o más exitoso
        print_colored("✅ Instalación completada con éxito", Colors.GREEN)
        return True
    else:
        print_colored("⚠️ Instalación parcial. Algunas funciones pueden no estar disponibles", Colors.YELLOW)
        return False

def test_imports():
    """Verifica que las librerías se importan correctamente"""
    print_header("VERIFICANDO IMPORTACIONES")
    
    import_tests = [
        ("pandas", "Análisis de datos"),
        ("numpy", "Computación numérica"), 
        ("matplotlib", "Gráficos"),
        ("sklearn", "Machine Learning"),
        ("binance", "API de Binance"),
        ("ta", "Análisis técnico"),
        ("xgboost", "XGBoost ML")
    ]
    
    optional_imports = [
        ("tensorflow", "Deep Learning"),
        ("newsapi", "API de noticias"),
        ("plotly", "Gráficos interactivos")
    ]
    
    success_count = 0
    
    for module, description in import_tests:
        try:
            importlib.import_module(module)
            print_colored(f"   ✅ {module} ({description})", Colors.GREEN)
            success_count += 1
        except ImportError:
            print_colored(f"   ❌ {module} ({description})", Colors.RED)
    
    print(f"\n📚 Librerías opcionales:")
    for module, description in optional_imports:
        try:
            importlib.import_module(module)
            print_colored(f"   ✅ {module} ({description})", Colors.GREEN)
        except ImportError:
            print_colored(f"   ⚠️ {module} ({description}) - No disponible", Colors.YELLOW)
    
    print(f"\n📊 Librerías básicas: {success_count}/{len(import_tests)} funcionando")
    
    return success_count >= len(import_tests) * 0.8

def create_directory_structure():
    """Crea estructura de directorios"""
    print_header("CREANDO ESTRUCTURA DE DIRECTORIOS")
    
    directories = [
        "resultados",
        "modelos", 
        "logs",
        "config",
        "graficos"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_colored(f"   ✅ Directorio creado: {directory}/", Colors.GREEN)
        else:
            print_colored(f"   📁 Directorio ya existe: {directory}/", Colors.BLUE)

def create_config_file():
    """Crea archivo de configuración personalizado"""
    print_header("CONFIGURACIÓN PERSONALIZADA")
    
    config_file = Path("config.json")
    
    if config_file.exists():
        print_colored("📄 Archivo config.json ya existe", Colors.BLUE)
        response = input("¿Deseas sobrescribirlo? (s/N): ").lower()
        if response != 's':
            return True
    
    print("🔧 Configurando el sistema...")
    print("Puedes dejar campos vacíos presionando Enter (se usarán valores por defecto)")
    
    # Configuración básica
    config = {
        "api_key": "",
        "api_secret": "",
        "news_api_key": "",
        "pares_analizar": [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT",
            "DOGEUSDT", "SHIBUSDT", "PEPEUSDT"
        ],
        "intervalo_datos": "1h",
        "periodo_historico": "30 day ago UTC",
        "horizontes_prediccion": [1, 4, 12, 24],
        "modo_avanzado": True,
        "guardar_resultados": True,
        "mostrar_graficos": True
    }
    
    # Solicitar claves API
    print("\n🔑 CONFIGURACIÓN DE CLAVES API")
    print("Opcional: Puedes configurar esto después editando config.json")
    
    api_key = input("API Key de Binance (opcional): ").strip()
    if api_key:
        config["api_key"] = api_key
        
        api_secret = input("API Secret de Binance: ").strip()
        config["api_secret"] = api_secret
    
    news_key = input("NewsAPI Key (opcional): ").strip()
    if news_key:
        config["news_api_key"] = news_key
    
    # Configuración de análisis
    print("\n📊 CONFIGURACIÓN DE ANÁLISIS")
    
    intervalos_disponibles = ["15m", "1h", "4h", "1d"]
    print(f"Intervalos disponibles: {', '.join(intervalos_disponibles)}")
    intervalo = input(f"Intervalo de datos (default: {config['intervalo_datos']}): ").strip()
    if intervalo and intervalo in intervalos_disponibles:
        config["intervalo_datos"] = intervalo
    
    # Pares adicionales
    print(f"\nPares configurados: {', '.join(config['pares_analizar'])}")
    pares_extra = input("Pares adicionales (separados por coma): ").strip()
    if pares_extra:
        nuevos_pares = [p.strip().upper() for p in pares_extra.split(",")]
        config["pares_analizar"].extend(nuevos_pares)
    
    # Guardar configuración
    try:
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print_colored("✅ Archivo config.json creado exitosamente", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Error creando config.json: {e}", Colors.RED)
        return False

def test_binance_connection():
    """Verifica conexión con Binance"""
    print_header("VERIFICANDO CONECTIVIDAD")
    
    try:
        from binance.client import Client
        
        # Leer configuración si existe
        config_file = Path("config.json")
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                api_key = config.get("api_key", "")
                api_secret = config.get("api_secret", "")
        else:
            api_key = ""
            api_secret = ""
        
        # Probar conexión
        if api_key and api_secret:
            print("🔐 Probando conexión autenticada...")
            client = Client(api_key, api_secret)
            try:
                account = client.get_account()
                print_colored("✅ Conexión autenticada exitosa", Colors.GREEN)
                print(f"   Permisos: {', '.join(account.get('permissions', []))}")
            except:
                print_colored("⚠️ Conexión autenticada falló, probando modo público...", Colors.YELLOW)
                client = Client()
                client.ping()
                print_colored("✅ Conexión pública exitosa", Colors.GREEN)
        else:
            print("🌐 Probando conexión pública...")
            client = Client()
            client.ping()
            print_colored("✅ Conexión pública exitosa", Colors.GREEN)
        
        # Probar obtener datos
        print("📊 Probando obtención de datos...")
        ticker = client.get_ticker(symbol="BTCUSDT")
        price = float(ticker['price'])
        print_colored(f"✅ Datos obtenidos - BTC/USDT: ${price:,.2f}", Colors.GREEN)
        
        return True
        
    except Exception as e:
        print_colored(f"❌ Error de conectividad: {e}", Colors.RED)
        return False

def create_launcher_script():
    """Crea script de lanzamiento fácil"""
    print_header("CREANDO SCRIPT DE LANZAMIENTO")
    
    launcher_content = '''#!/usr/bin/env python3
"""
LANZADOR DEL SISTEMA INTEGRADO
Ejecuta: python lanzar_sistema.py
"""

import sys
import os

def main():
    print("🚀 LANZANDO SISTEMA INTEGRADO DE ANÁLISIS")
    print("="*60)
    
    try:
        # Intentar importar el sistema integrado
        import sistema_integrado
        
        # Ejecutar sistema
        sistema_integrado.main()
        
    except ImportError as e:
        print(f"❌ Error importando sistema: {e}")
        print("🔧 Ejecuta: python instalacion_automatica.py")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Sistema interrumpido por el usuario")
        
    except Exception as e:
        print(f"❌ Error ejecutando sistema: {e}")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("lanzar_sistema.py", "w") as f:
            f.write(launcher_content)
        
        print_colored("✅ Script de lanzamiento creado: lanzar_sistema.py", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Error creando lanzador: {e}", Colors.RED)
        return False

def show_usage_guide():
    """Muestra guía de uso"""
    print_header("GUÍA DE USO")
    
    guide = """
🎯 CÓMO USAR EL SISTEMA:

1️⃣ LANZAMIENTO RÁPIDO:
   python lanzar_sistema.py

2️⃣ LANZAMIENTO DIRECTO:
   python sistema_integrado.py

3️⃣ SOLO ANÁLISIS BÁSICO:
   python prediccion.py

4️⃣ SOLO ANÁLISIS AVANZADO:
   python prediccion_avanzada.py

📁 ARCHIVOS IMPORTANTES:
   • config.json - Configuración principal
   • resultados/ - Análisis guardados
   • modelos/ - Modelos de ML entrenados
   • graficos/ - Gráficos generados

⚙️ CONFIGURACIÓN:
   • Edita config.json para personalizar
   • Añade tus claves API para funcionalidad completa
   • Ajusta pares a analizar según tu interés

🔑 CLAVES API NECESARIAS:
   • Binance API (para datos de mercado)
   • NewsAPI (para análisis de sentimientos)

📊 FUNCIONALIDADES:
   ✅ Análisis técnico completo (30+ indicadores)
   ✅ Predicciones con Machine Learning
   ✅ Análisis de sentimientos
   ✅ Detección de patrones
   ✅ Gráficos interactivos
   ✅ Reportes detallados
   ✅ Sistema de scoring predictivo

⚠️ ADVERTENCIAS:
   • Solo para fines educativos
   • NO es asesoramiento financiero
   • Los mercados crypto son altamente volátiles
   • Siempre DYOR (Do Your Own Research)
"""
    
    print_colored(guide, Colors.CYAN)

def main():
    """Función principal de instalación"""
    print_colored("🔮 INSTALACIÓN AUTOMATICA DEL SISTEMA PREDICTIVO", Colors.BOLD + Colors.PURPLE)
    print_colored("   Desarrollado por AI Expert Developer & Economist", Colors.PURPLE)
    print_colored("   Versión 2.0 - Análisis Técnico + Machine Learning", Colors.PURPLE)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        print_colored("⚠️ Algunas dependencias fallaron, pero continuamos...", Colors.YELLOW)
    
    # Verificar importaciones
    if not test_imports():
        print_colored("❌ Faltan librerías críticas. Revisa la instalación.", Colors.RED)
        sys.exit(1)
    
    # Crear estructura
    create_directory_structure()
    
    # Configurar sistema
    create_config_file()
    
    # Verificar conectividad
    test_binance_connection()
    
    # Crear lanzador
    create_launcher_script()
    
    # Mostrar guía
    show_usage_guide()
    
    # Mensaje final
    print_header("INSTALACIÓN COMPLETADA")
    print_colored("🎉 ¡Sistema instalado exitosamente!", Colors.GREEN + Colors.BOLD)
    print_colored("🚀 Para empezar: python lanzar_sistema.py", Colors.CYAN + Colors.BOLD)
    print_colored("📖 Lee la guía de uso arriba para más detalles", Colors.BLUE)
    print()
    print_colored("⚠️ RECORDATORIO IMPORTANTE:", Colors.RED + Colors.BOLD)
    print_colored("Este sistema es para fines educativos únicamente.", Colors.RED)
    print_colored("NO constituye asesoramiento financiero.", Colors.RED)
    print_colored("Siempre realiza tu propia investigación antes de invertir.", Colors.RED)

if __name__ == "__main__":
    main() 