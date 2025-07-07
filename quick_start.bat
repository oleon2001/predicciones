@echo off
setlocal enabledelayedexpansion

REM =======================================================
REM SCRIPT DE INICIO RÁPIDO - SISTEMA DE TRADING (Windows)
REM =======================================================

echo.
echo ========================================================
echo 🚀 SISTEMA DE TRADING DE CRIPTOMONEDAS - INICIO RÁPIDO
echo ========================================================
echo.

REM Función para mostrar mensajes
:show_message
echo [INFO] %~1
exit /b

:show_warning
echo [WARNING] %~1
exit /b

:show_error
echo [ERROR] %~1
exit /b

REM Verificar Python
:check_python
call :show_message "Verificando Python..."
python --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    call :show_message "Python !PYTHON_VERSION! encontrado"
    exit /b 0
) else (
    call :show_error "Python no encontrado. Instala Python 3.8+ primero."
    exit /b 1
)

REM Verificar si existe entorno virtual
:check_venv
if exist "venv\" (
    call :show_message "Entorno virtual encontrado"
    exit /b 0
) else (
    call :show_warning "Entorno virtual no encontrado. Creando..."
    python -m venv venv
    exit /b !errorlevel!
)

REM Activar entorno virtual
:activate_venv
if exist "venv\Scripts\activate.bat" (
    call :show_message "Activando entorno virtual..."
    call venv\Scripts\activate.bat
    exit /b 0
) else (
    call :show_error "No se pudo activar el entorno virtual"
    exit /b 1
)

REM Instalar dependencias
:install_dependencies
call :show_message "Instalando dependencias..."

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar dependencias
if exist "requirements_avanzado.txt" (
    pip install -r requirements_avanzado.txt
    exit /b !errorlevel!
) else (
    call :show_error "requirements_avanzado.txt no encontrado"
    exit /b 1
)

REM Configurar variables de entorno
:setup_env
call :show_message "Configurando variables de entorno..."

if not exist ".env" (
    if exist "env.template" (
        copy env.template .env >nul
        call :show_warning "Archivo .env creado desde template"
        call :show_warning "¡IMPORTANTE! Edita .env con tus API keys antes de continuar"
    ) else (
        call :show_error "Template de configuración no encontrado"
        exit /b 1
    )
) else (
    call :show_message "Archivo .env ya existe"
)
exit /b 0

REM Ejecutar tests
:run_tests
call :show_message "Ejecutando tests básicos..."

if exist "test_sistema.py" (
    python test_sistema.py
) else (
    call :show_warning "test_sistema.py no encontrado, saltando tests"
)
exit /b 0

REM Función principal
:main
echo.
call :show_message "Iniciando configuración automática..."

REM Verificar Python
call :check_python
if !errorlevel! neq 0 exit /b 1

REM Verificar/crear entorno virtual
call :check_venv
if !errorlevel! neq 0 (
    call :show_error "No se pudo crear el entorno virtual"
    exit /b 1
)

REM Activar entorno virtual
call :activate_venv
if !errorlevel! neq 0 (
    call :show_error "No se pudo activar el entorno virtual"
    exit /b 1
)

REM Instalar dependencias
call :install_dependencies
if !errorlevel! neq 0 (
    call :show_error "Error instalando dependencias"
    exit /b 1
)

REM Configurar entorno
call :setup_env
if !errorlevel! neq 0 (
    call :show_error "Error configurando entorno"
    exit /b 1
)

REM Ejecutar tests
call :run_tests

echo.
call :show_message "¡Configuración completada!"
echo.
echo 🎯 PRÓXIMOS PASOS:
echo 1. Edita el archivo .env con tus API keys
echo 2. Ejecuta: python lanzar_sistema.py
echo 3. O ejecuta: python demo_sistema.py (para modo demo)
echo 4. Para producción: python deployment/production_system.py
echo.
echo 📚 DOCUMENTACIÓN:
echo - Guía completa: deploy_guide.md
echo - README: README.md
echo - Producción: README_PRODUCTION.md
echo.
echo 🔧 OPCIONES AVANZADAS:
echo - Sistema integrado: python sistema_integrado.py
echo - Predicción avanzada: python prediccion_avanzada.py
echo - Tests completos: python tests/comprehensive_test_suite.py
echo - Deployment automático: python deployment/deploy.py
echo.
echo 📊 MONITOREO:
echo - Health check: http://localhost:8080/health
echo - Métricas: http://localhost:8080/api/system/metrics
echo.
echo 🚀 ¡Sistema listo para usar!
exit /b 0

REM Función para mostrar ayuda
:show_help
echo SISTEMA DE TRADING DE CRIPTOMONEDAS - SCRIPT DE INICIO
echo =====================================================
echo.
echo USO: %~nx0 [OPCIÓN]
echo.
echo OPCIONES:
echo   help       Mostrar esta ayuda
echo   test       Solo ejecutar tests
echo   install    Solo instalar dependencias
echo   demo       Ejecutar en modo demo
echo   prod       Ejecutar en modo producción
echo.
echo EJEMPLOS:
echo   %~nx0             Configuración completa
echo   %~nx0 demo        Ejecutar demo después de configurar
echo   %~nx0 prod        Ejecutar en producción
echo.
exit /b 0

REM Procesar argumentos
if "%1"=="help" (
    call :show_help
    exit /b 0
)

if "%1"=="test" (
    call :activate_venv
    call :run_tests
    exit /b 0
)

if "%1"=="install" (
    call :check_venv
    call :activate_venv
    call :install_dependencies
    exit /b 0
)

if "%1"=="demo" (
    call :main
    call :show_message "Ejecutando modo demo..."
    python demo_sistema.py
    exit /b 0
)

if "%1"=="prod" (
    call :main
    call :show_message "Ejecutando modo producción..."
    python deployment/production_system.py
    exit /b 0
)

if "%1"=="" (
    call :main
    exit /b 0
)

call :show_error "Opción no reconocida: %1"
call :show_help
exit /b 1 