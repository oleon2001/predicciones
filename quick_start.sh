#!/bin/bash

# =======================================================
# SCRIPT DE INICIO RÁPIDO - SISTEMA DE TRADING
# =======================================================

echo "🚀 SISTEMA DE TRADING DE CRIPTOMONEDAS - INICIO RÁPIDO"
echo "======================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
show_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

show_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar Python
check_python() {
    show_message "Verificando Python..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        show_message "Python ${PYTHON_VERSION} encontrado"
        return 0
    else
        show_error "Python 3 no encontrado. Instala Python 3.8+ primero."
        return 1
    fi
}

# Verificar si existe entorno virtual
check_venv() {
    if [ -d "venv" ]; then
        show_message "Entorno virtual encontrado"
        return 0
    else
        show_warning "Entorno virtual no encontrado. Creando..."
        python3 -m venv venv
        return $?
    fi
}

# Activar entorno virtual
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        show_message "Activando entorno virtual..."
        source venv/bin/activate
        return 0
    elif [ -f "venv/Scripts/activate" ]; then
        show_message "Activando entorno virtual (Windows)..."
        source venv/Scripts/activate
        return 0
    else
        show_error "No se pudo activar el entorno virtual"
        return 1
    fi
}

# Instalar dependencias
install_dependencies() {
    show_message "Instalando dependencias..."
    
    # Actualizar pip
    python -m pip install --upgrade pip
    
    # Instalar dependencias
    if [ -f "requirements_avanzado.txt" ]; then
        pip install -r requirements_avanzado.txt
    else
        show_error "requirements_avanzado.txt no encontrado"
        return 1
    fi
}

# Configurar variables de entorno
setup_env() {
    show_message "Configurando variables de entorno..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.template" ]; then
            cp env.template .env
            show_warning "Archivo .env creado desde template"
            show_warning "¡IMPORTANTE! Edita .env con tus API keys antes de continuar"
        else
            show_error "Template de configuración no encontrado"
            return 1
        fi
    else
        show_message "Archivo .env ya existe"
    fi
}

# Ejecutar tests
run_tests() {
    show_message "Ejecutando tests básicos..."
    
    if [ -f "test_sistema.py" ]; then
        python test_sistema.py
    else
        show_warning "test_sistema.py no encontrado, saltando tests"
    fi
}

# Función principal
main() {
    echo
    show_message "Iniciando configuración automática..."
    
    # Verificar Python
    if ! check_python; then
        exit 1
    fi
    
    # Verificar/crear entorno virtual
    if ! check_venv; then
        show_error "No se pudo crear el entorno virtual"
        exit 1
    fi
    
    # Activar entorno virtual
    if ! activate_venv; then
        show_error "No se pudo activar el entorno virtual"
        exit 1
    fi
    
    # Instalar dependencias
    if ! install_dependencies; then
        show_error "Error instalando dependencias"
        exit 1
    fi
    
    # Configurar entorno
    if ! setup_env; then
        show_error "Error configurando entorno"
        exit 1
    fi
    
    # Ejecutar tests
    run_tests
    
    echo
    show_message "¡Configuración completada!"
    echo
    echo "🎯 PRÓXIMOS PASOS:"
    echo "1. Edita el archivo .env con tus API keys"
    echo "2. Ejecuta: python lanzar_sistema.py"
    echo "3. O ejecuta: python demo_sistema.py (para modo demo)"
    echo "4. Para producción: python deployment/production_system.py"
    echo
    echo "📚 DOCUMENTACIÓN:"
    echo "- Guía completa: deploy_guide.md"
    echo "- README: README.md"
    echo "- Producción: README_PRODUCTION.md"
    echo
    echo "🔧 OPCIONES AVANZADAS:"
    echo "- Sistema integrado: python sistema_integrado.py"
    echo "- Predicción avanzada: python prediccion_avanzada.py"
    echo "- Tests completos: python tests/comprehensive_test_suite.py"
    echo "- Deployment automático: python deployment/deploy.py"
    echo
    echo "📊 MONITOREO:"
    echo "- Health check: http://localhost:8080/health"
    echo "- Métricas: http://localhost:8080/api/system/metrics"
    echo
    echo "🚀 ¡Sistema listo para usar!"
}

# Función para mostrar ayuda
show_help() {
    echo "SISTEMA DE TRADING DE CRIPTOMONEDAS - SCRIPT DE INICIO"
    echo "====================================================="
    echo
    echo "USO: $0 [OPCIÓN]"
    echo
    echo "OPCIONES:"
    echo "  -h, --help     Mostrar esta ayuda"
    echo "  -t, --test     Solo ejecutar tests"
    echo "  -i, --install  Solo instalar dependencias"
    echo "  -d, --demo     Ejecutar en modo demo"
    echo "  -p, --prod     Ejecutar en modo producción"
    echo
    echo "EJEMPLOS:"
    echo "  $0             Configuración completa"
    echo "  $0 --demo      Ejecutar demo después de configurar"
    echo "  $0 --prod      Ejecutar en producción"
    echo
}

# Procesar argumentos
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -t|--test)
        activate_venv
        run_tests
        exit 0
        ;;
    -i|--install)
        check_venv
        activate_venv
        install_dependencies
        exit 0
        ;;
    -d|--demo)
        main
        show_message "Ejecutando modo demo..."
        python demo_sistema.py
        exit 0
        ;;
    -p|--prod)
        main
        show_message "Ejecutando modo producción..."
        python deployment/production_system.py
        exit 0
        ;;
    "")
        main
        exit 0
        ;;
    *)
        show_error "Opción no reconocida: $1"
        show_help
        exit 1
        ;;
esac 