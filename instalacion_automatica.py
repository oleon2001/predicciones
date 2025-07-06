#!/usr/bin/env python3
"""
Script de instalación automática para el sistema de predicción de criptomonedas
Compatible con Python 3.12
"""

import subprocess
import sys
import os
import platform

def ejecutar_comando(comando, descripcion):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔧 {descripcion}...")
    try:
        resultado = subprocess.run(comando, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {descripcion} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {descripcion}: {e}")
        print(f"Salida de error: {e.stderr}")
        return False

def verificar_python():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Se requiere Python 3.8 o superior")
        return False
    
    if version.minor >= 12:
        print("✅ Python 3.12+ detectado - Compatible")
    else:
        print("⚠️ Versión de Python aceptable pero se recomienda 3.12+")
    
    return True

def actualizar_pip():
    """Actualiza pip a la última versión"""
    return ejecutar_comando(
        f"{sys.executable} -m pip install --upgrade pip",
        "Actualizando pip"
    )

def instalar_setuptools():
    """Instala setuptools actualizado"""
    return ejecutar_comando(
        f"{sys.executable} -m pip install --upgrade setuptools wheel",
        "Instalando setuptools y wheel"
    )

def instalar_dependencias_basicas():
    """Instala las dependencias básicas primero"""
    dependencias_basicas = [
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "requests>=2.31.0"
    ]
    
    for dep in dependencias_basicas:
        if not ejecutar_comando(
            f"{sys.executable} -m pip install {dep}",
            f"Instalando {dep}"
        ):
            return False
    return True

def instalar_ml_dependencias():
    """Instala dependencias de Machine Learning"""
    ml_deps = [
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "scipy>=1.11.0"
    ]
    
    for dep in ml_deps:
        if not ejecutar_comando(
            f"{sys.executable} -m pip install {dep}",
            f"Instalando {dep}"
        ):
            return False
    return True

def instalar_apis():
    """Instala las APIs necesarias"""
    apis = [
        "python-binance>=1.0.19",
        "newsapi-python>=0.2.6"
    ]
    
    for api in apis:
        if not ejecutar_comando(
            f"{sys.executable} -m pip install {api}",
            f"Instalando {api}"
        ):
            return False
    return True

def instalar_analisis_tecnico():
    """Instala librerías de análisis técnico"""
    return ejecutar_comando(
        f"{sys.executable} -m pip install ta>=0.10.0",
        "Instalando librería de análisis técnico"
    )

def instalar_visualizacion():
    """Instala librerías de visualización"""
    viz_deps = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ]
    
    for dep in viz_deps:
        if not ejecutar_comando(
            f"{sys.executable} -m pip install {dep}",
            f"Instalando {dep}"
        ):
            return False
    return True

def instalar_tensorflow():
    """Instala TensorFlow (opcional)"""
    print("\n🤖 ¿Instalar TensorFlow para Deep Learning? (s/n): ", end="")
    respuesta = input().lower().strip()
    
    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        # Intentar TensorFlow GPU primero
        if not ejecutar_comando(
            f"{sys.executable} -m pip install tensorflow>=2.15.0",
            "Instalando TensorFlow"
        ):
            print("⚠️ Falló TensorFlow GPU, intentando CPU...")
            return ejecutar_comando(
                f"{sys.executable} -m pip install tensorflow-cpu>=2.15.0",
                "Instalando TensorFlow CPU"
            )
    else:
        print("⏭️ Saltando TensorFlow")
        return True

def instalar_dependencias_adicionales():
    """Instala dependencias adicionales"""
    deps_adicionales = [
        "textblob>=0.17.0",
        "vaderSentiment>=3.3.0",
        "python-dateutil>=2.8.0",
        "pytz>=2023.3",
        "tqdm>=4.65.0",
        "statsmodels>=0.14.0",
        "arch>=6.2.0"
    ]
    
    for dep in deps_adicionales:
        if not ejecutar_comando(
            f"{sys.executable} -m pip install {dep}",
            f"Instalando {dep}"
        ):
            print(f"⚠️ Advertencia: No se pudo instalar {dep}")
    
    return True

def verificar_instalacion():
    """Verifica que las dependencias principales estén instaladas"""
    print("\n🔍 Verificando instalación...")
    
    modulos_importantes = [
        'numpy', 'pandas', 'requests', 'sklearn', 'xgboost',
        'matplotlib', 'seaborn', 'ta'
    ]
    
    faltantes = []
    for modulo in modulos_importantes:
        try:
            __import__(modulo)
            print(f"✅ {modulo}")
        except ImportError:
            print(f"❌ {modulo} - FALTANTE")
            faltantes.append(modulo)
    
    if faltantes:
        print(f"\n⚠️ Módulos faltantes: {', '.join(faltantes)}")
        return False
    else:
        print("\n🎉 ¡Todas las dependencias principales están instaladas!")
        return True

def main():
    """Función principal de instalación"""
    print("🚀 INSTALADOR AUTOMÁTICO - SISTEMA DE PREDICCIÓN DE CRIPTOMONEDAS")
    print("="*70)
    
    # Verificar Python
    if not verificar_python():
        return False
    
    # Actualizar herramientas básicas
    if not actualizar_pip():
        print("❌ Error actualizando pip")
        return False
    
    if not instalar_setuptools():
        print("❌ Error instalando setuptools")
        return False
    
    # Instalar dependencias en orden
    pasos = [
        ("Dependencias básicas", instalar_dependencias_basicas),
        ("APIs", instalar_apis),
        ("Machine Learning", instalar_ml_dependencias),
        ("Análisis técnico", instalar_analisis_tecnico),
        ("Visualización", instalar_visualizacion),
        ("TensorFlow", instalar_tensorflow),
        ("Dependencias adicionales", instalar_dependencias_adicionales)
    ]
    
    for nombre, funcion in pasos:
        print(f"\n📦 PASO: {nombre}")
        if not funcion():
            print(f"❌ Error en paso: {nombre}")
            return False
    
    # Verificar instalación
    if verificar_instalacion():
        print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Ejecuta: python prediccion_avanzada.py")
        print("2. Revisa los archivos de configuración")
        print("3. ¡Disfruta del sistema de predicción!")
        return True
    else:
        print("\n⚠️ Instalación completada con advertencias")
        print("Algunas dependencias pueden no estar disponibles")
        return False

if __name__ == "__main__":
    try:
        exito = main()
        sys.exit(0 if exito else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Instalación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1) 