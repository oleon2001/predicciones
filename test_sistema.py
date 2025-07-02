#!/usr/bin/env python3
"""
SCRIPT DE PRUEBAS DEL SISTEMA
Verifica que todos los componentes funcionen correctamente

Autor: AI Expert Developer & Economist
Versión: 2.0
"""

import sys
import traceback
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.BLUE):
    """Imprime mensaje con color"""
    print(f"{color}{message}{Colors.END}")

def print_test_header(test_name):
    """Imprime header de test"""
    print(f"\n{'='*60}")
    print_colored(f"🧪 PRUEBA: {test_name}", Colors.BOLD + Colors.CYAN)
    print('='*60)

def print_result(success, message):
    """Imprime resultado de test"""
    if success:
        print_colored(f"✅ {message}", Colors.GREEN)
    else:
        print_colored(f"❌ {message}", Colors.RED)
    return success

class TestSistema:
    """Clase para ejecutar todas las pruebas del sistema"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = {}
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        
        print_colored("🚀 INICIANDO PRUEBAS DEL SISTEMA", Colors.BOLD + Colors.CYAN)
        print_colored("Desarrollado por AI Expert Developer & Economist", Colors.CYAN)
        print_colored("Versión 2.0 - Sistema Integrado de Predicción", Colors.CYAN)
        
        # Lista de tests
        tests = [
            ("Importaciones Básicas", self.test_basic_imports),
            ("Importaciones Avanzadas", self.test_advanced_imports),
            ("Estructura de Archivos", self.test_file_structure),
            ("Configuración", self.test_configuration),
            ("Generación de Datos Demo", self.test_demo_data_generation),
            ("Indicadores Técnicos", self.test_technical_indicators),
            ("Machine Learning", self.test_ml_components),
            ("Conectividad Binance", self.test_binance_connectivity),
            ("Sistema Demo Completo", self.test_demo_system),
            ("Visualizaciones", self.test_visualizations)
        ]
        
        # Ejecutar tests
        for test_name, test_function in tests:
            print_test_header(test_name)
            try:
                success = test_function()
                self.test_results[test_name] = success
                if success:
                    self.tests_passed += 1
                else:
                    self.tests_failed += 1
            except Exception as e:
                print_colored(f"❌ Error ejecutando test: {e}", Colors.RED)
                print_colored(f"Traceback: {traceback.format_exc()}", Colors.YELLOW)
                self.test_results[test_name] = False
                self.tests_failed += 1
        
        # Mostrar resultados finales
        self.show_final_results()
    
    def test_basic_imports(self):
        """Test de importaciones básicas"""
        
        imports_to_test = [
            ('pandas', 'import pandas as pd'),
            ('numpy', 'import numpy as np'),
            ('matplotlib', 'import matplotlib.pyplot as plt'),
            ('datetime', 'from datetime import datetime, timedelta'),
            ('json', 'import json'),
            ('pathlib', 'from pathlib import Path')
        ]
        
        all_success = True
        
        for lib_name, import_statement in imports_to_test:
            try:
                exec(import_statement)
                print_result(True, f"{lib_name} importado correctamente")
            except ImportError as e:
                print_result(False, f"Error importando {lib_name}: {e}")
                all_success = False
        
        return all_success
    
    def test_advanced_imports(self):
        """Test de importaciones avanzadas"""
        
        advanced_imports = [
            ('binance.client', 'from binance.client import Client'),
            ('sklearn', 'import sklearn'),
            ('ta', 'import ta'),
            ('seaborn', 'import seaborn as sns')
        ]
        
        optional_imports = [
            ('tensorflow', 'import tensorflow as tf'),
            ('xgboost', 'import xgboost as xgb'),
            ('newsapi', 'from newsapi import NewsApiClient')
        ]
        
        success_count = 0
        total_required = len(advanced_imports)
        
        # Importaciones requeridas
        print("📦 Librerías requeridas:")
        for lib_name, import_statement in advanced_imports:
            try:
                exec(import_statement)
                print_result(True, f"{lib_name}")
                success_count += 1
            except ImportError:
                print_result(False, f"{lib_name} no disponible")
        
        # Importaciones opcionales
        print("\n📦 Librerías opcionales:")
        for lib_name, import_statement in optional_imports:
            try:
                exec(import_statement)
                print_result(True, f"{lib_name}")
            except ImportError:
                print_colored(f"⚠️ {lib_name} no disponible (opcional)", Colors.YELLOW)
        
        return success_count >= total_required * 0.8  # 80% éxito mínimo
    
    def test_file_structure(self):
        """Test de estructura de archivos"""
        
        required_files = [
            'prediccion.py',
            'prediccion_avanzada.py', 
            'sistema_integrado.py',
            'demo_sistema.py',
            'instalacion_automatica.py',
            'config_ejemplo.json',
            'requirements_avanzado.txt'
        ]
        
        required_dirs = [
            'resultados',
            'modelos',
            'graficos',
            'logs'
        ]
        
        all_success = True
        
        # Verificar archivos
        print("📄 Verificando archivos:")
        for filename in required_files:
            file_path = Path(filename)
            if file_path.exists():
                print_result(True, f"{filename}")
            else:
                print_result(False, f"{filename} no encontrado")
                all_success = False
        
        # Verificar/crear directorios
        print("\n📁 Verificando directorios:")
        for dirname in required_dirs:
            dir_path = Path(dirname)
            if dir_path.exists():
                print_result(True, f"{dirname}/ existe")
            else:
                try:
                    dir_path.mkdir(exist_ok=True)
                    print_result(True, f"{dirname}/ creado")
                except Exception as e:
                    print_result(False, f"Error creando {dirname}/: {e}")
                    all_success = False
        
        return all_success
    
    def test_configuration(self):
        """Test de configuración"""
        
        # Verificar config_ejemplo.json
        config_ejemplo = Path("config_ejemplo.json")
        if not config_ejemplo.exists():
            return print_result(False, "config_ejemplo.json no encontrado")
        
        try:
            with open(config_ejemplo, 'r') as f:
                config_data = json.load(f)
            
            required_keys = [
                'api_key', 'api_secret', 'pares_analizar', 
                'intervalo_datos', 'modo_avanzado'
            ]
            
            all_keys_present = all(key in config_data for key in required_keys)
            
            if all_keys_present:
                print_result(True, "Estructura de configuración válida")
                
                # Verificar si config.json existe
                config_personal = Path("config.json")
                if config_personal.exists():
                    print_result(True, "config.json encontrado")
                else:
                    print_colored("⚠️ config.json no encontrado. Copia config_ejemplo.json", Colors.YELLOW)
                
                return True
            else:
                return print_result(False, "Estructura de configuración inválida")
                
        except json.JSONDecodeError:
            return print_result(False, "Error parseando config_ejemplo.json")
        except Exception as e:
            return print_result(False, f"Error leyendo configuración: {e}")
    
    def test_demo_data_generation(self):
        """Test de generación de datos demo"""
        
        try:
            # Importar generador demo
            from demo_sistema import GeneradorDatosDemo
            
            generador = GeneradorDatosDemo()
            
            # Generar datos de prueba
            df = generador.generar_datos_historicos('BTCUSDT', periodos=100)
            
            # Verificar estructura
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            columns_ok = all(col in df.columns for col in required_columns)
            
            # Verificar datos
            data_length_ok = len(df) == 100
            no_nulls = not df.isnull().any().any()
            positive_prices = (df[['open', 'high', 'low', 'close']] > 0).all().all()
            
            if columns_ok and data_length_ok and no_nulls and positive_prices:
                print_result(True, f"Datos generados correctamente: {len(df)} períodos")
                return True
            else:
                print_result(False, "Datos generados con problemas")
                return False
                
        except ImportError:
            return print_result(False, "No se puede importar GeneradorDatosDemo")
        except Exception as e:
            return print_result(False, f"Error generando datos demo: {e}")
    
    def test_technical_indicators(self):
        """Test de indicadores técnicos"""
        
        try:
            from demo_sistema import GeneradorDatosDemo, AnalizadorTecnicoDemo
            
            # Generar datos
            generador = GeneradorDatosDemo()
            df = generador.generar_datos_historicos('BTCUSDT', periodos=200)
            
            # Calcular indicadores
            analizador = AnalizadorTecnicoDemo()
            df_with_indicators = analizador.calcular_indicadores(df)
            
            # Verificar indicadores
            indicators_to_check = ['rsi', 'macd', 'sma_20', 'ema_12', 'bb_upper', 'bb_lower']
            
            indicators_ok = all(col in df_with_indicators.columns for col in indicators_to_check)
            
            if indicators_ok:
                # Verificar que no sean todos NaN
                valid_indicators = []
                for indicator in indicators_to_check:
                    if not df_with_indicators[indicator].isna().all():
                        valid_indicators.append(indicator)
                
                print_result(True, f"Indicadores calculados: {', '.join(valid_indicators)}")
                return len(valid_indicators) >= len(indicators_to_check) * 0.8
            else:
                return print_result(False, "Faltan indicadores técnicos")
                
        except Exception as e:
            return print_result(False, f"Error calculando indicadores: {e}")
    
    def test_ml_components(self):
        """Test de componentes ML"""
        
        try:
            from demo_sistema import PrediccionMLDemo, GeneradorDatosDemo
            
            # Generar datos
            generador = GeneradorDatosDemo()
            df = generador.generar_datos_historicos('BTCUSDT', periodos=300)
            
            # Test predictor ML
            predictor = PrediccionMLDemo()
            
            # Entrenar modelo
            resultado_entrenamiento = predictor.entrenar_modelo_demo(df, 'BTCUSDT')
            
            if resultado_entrenamiento.get('exito', False):
                print_result(True, f"Modelo ML entrenado: {resultado_entrenamiento['mensaje']}")
                
                # Test predicciones
                predicciones = predictor.predecir(df, 'BTCUSDT')
                
                if len(predicciones) > 0:
                    print_result(True, f"Predicciones generadas para {len(predicciones)} horizontes")
                    return True
                else:
                    return print_result(False, "No se generaron predicciones")
            else:
                # Fallback a predicción simple
                predicciones = predictor._generar_prediccion_simple(df)
                if len(predicciones) > 0:
                    print_result(True, "Predicción simple funcionando (fallback)")
                    return True
                else:
                    return print_result(False, "Predicción simple falló")
                    
        except Exception as e:
            return print_result(False, f"Error en componentes ML: {e}")
    
    def test_binance_connectivity(self):
        """Test de conectividad con Binance"""
        
        try:
            from binance.client import Client
            
            # Test conexión pública (sin API keys)
            client = Client()
            
            # Ping básico
            client.ping()
            print_result(True, "Ping a Binance API exitoso")
            
            # Test obtener datos
            ticker = client.get_ticker(symbol="BTCUSDT")
            price = float(ticker['price'])
            
            if price > 0:
                print_result(True, f"Datos obtenidos - BTC: ${price:,.2f}")
                return True
            else:
                return print_result(False, "Precio inválido obtenido")
                
        except Exception as e:
            return print_result(False, f"Error conectividad Binance: {e}")
    
    def test_demo_system(self):
        """Test del sistema demo completo"""
        
        try:
            from demo_sistema import SistemaDemoCompleto
            
            # Crear sistema demo
            sistema = SistemaDemoCompleto()
            
            # Test análisis completo (sin gráficos)
            resultado = sistema.ejecutar_demo('BTCUSDT', mostrar_graficos=False)
            
            # Verificar resultado
            required_keys = ['symbol', 'analisis', 'predicciones']
            keys_ok = all(key in resultado for key in required_keys)
            
            if keys_ok:
                symbol = resultado['symbol']
                precio = resultado['analisis']['precio_actual']
                num_predicciones = len(resultado['predicciones'])
                
                print_result(True, f"Demo completo: {symbol} - ${precio:,.2f} - {num_predicciones} predicciones")
                return True
            else:
                return print_result(False, "Resultado de demo incompleto")
                
        except Exception as e:
            return print_result(False, f"Error en sistema demo: {e}")
    
    def test_visualizations(self):
        """Test de visualizaciones"""
        
        try:
            import matplotlib.pyplot as plt
            
            # Test básico de matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
            ax.set_title("Test Plot")
            
            # Guardar en memoria (no mostrar)
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            # Verificar que se creó contenido
            buffer.seek(0)
            content_length = len(buffer.read())
            
            if content_length > 1000:  # PNG no vacío
                print_result(True, f"Gráfico generado ({content_length} bytes)")
                return True
            else:
                return print_result(False, "Gráfico vacío o muy pequeño")
                
        except Exception as e:
            return print_result(False, f"Error en visualizaciones: {e}")
    
    def show_final_results(self):
        """Muestra resultados finales"""
        
        print_test_header("RESULTADOS FINALES")
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 ESTADÍSTICAS:")
        print(f"   • Tests ejecutados: {total_tests}")
        print(f"   • Tests exitosos: {self.tests_passed}")
        print(f"   • Tests fallidos: {self.tests_failed}")
        print(f"   • Tasa de éxito: {success_rate:.1f}%")
        
        # Mostrar detalles por test
        print(f"\n📋 DETALLE POR TEST:")
        for test_name, success in self.test_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}")
        
        # Recomendaciones finales
        print(f"\n🎯 RECOMENDACIONES:")
        
        if success_rate >= 90:
            print_colored("🎉 ¡Excelente! El sistema está completamente funcional", Colors.GREEN)
            print_colored("🚀 Puedes usar todas las funcionalidades", Colors.GREEN)
        elif success_rate >= 70:
            print_colored("✅ Sistema funcional con limitaciones menores", Colors.YELLOW)
            print_colored("📝 Revisa los tests fallidos para funcionalidad completa", Colors.YELLOW)
        elif success_rate >= 50:
            print_colored("⚠️ Sistema parcialmente funcional", Colors.YELLOW)
            print_colored("🔧 Instala dependencias faltantes", Colors.YELLOW)
        else:
            print_colored("❌ Sistema con problemas graves", Colors.RED)
            print_colored("🆘 Ejecuta: python instalacion_automatica.py", Colors.RED)
        
        print(f"\n📚 PRÓXIMOS PASOS:")
        if success_rate >= 70:
            print("1. 🎮 Ejecutar: python demo_sistema.py")
            print("2. ⚙️ Configurar claves API en config.json")
            print("3. 🚀 Lanzar: python lanzar_sistema.py")
        else:
            print("1. 🔧 Instalar dependencias faltantes")
            print("2. 🔄 Ejecutar tests nuevamente")
            print("3. 📖 Revisar README.md para troubleshooting")

def main():
    """Función principal"""
    
    print("🧪 SISTEMA DE PRUEBAS AUTOMATIZADAS")
    print("=" * 60)
    
    # Crear y ejecutar tester
    tester = TestSistema()
    tester.run_all_tests()
    
    print(f"\n⚠️ RECORDATORIO:")
    print("Este sistema es para fines educativos únicamente.")
    print("NO constituye asesoramiento financiero.")

if __name__ == "__main__":
    main() 