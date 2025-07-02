#!/usr/bin/env python3
"""
SCRIPT DE LIMPIEZA DEL SISTEMA
Organiza, limpia y optimiza el espacio de trabajo

Funciones:
- Limpia archivos temporales
- Organiza resultados por fecha
- Comprime logs antiguos
- Libera espacio en disco
- Mantiene estructura organizada

Autor: AI Expert Developer & Economist
Versión: 2.0
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import json
import glob

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

def print_header(title):
    """Imprime header con estilo"""
    print(f"\n{'='*60}")
    print_colored(f"🧹 {title}", Colors.BOLD + Colors.CYAN)
    print('='*60)

class LimpiadorSistema:
    """Clase para limpiar y organizar el sistema"""
    
    def __init__(self):
        self.archivos_eliminados = 0
        self.espacio_liberado = 0
        self.archivos_organizados = 0
        
        # Directorios del sistema
        self.directorios = {
            'resultados': Path('resultados'),
            'graficos': Path('graficos'),
            'logs': Path('logs'),
            'modelos': Path('modelos'),
            'temp': Path('temp'),
            'cache': Path('__pycache__')
        }
        
        # Extensiones de archivos temporales
        self.temp_extensions = [
            '.tmp', '.temp', '.log', '.cache', 
            '.pyc', '.pyo', '.bak', '~'
        ]
    
    def ejecutar_limpieza_completa(self):
        """Ejecuta limpieza completa del sistema"""
        
        print_colored("🧹 INICIANDO LIMPIEZA COMPLETA DEL SISTEMA", Colors.BOLD + Colors.CYAN)
        print_colored("Desarrollado por AI Expert Developer & Economist", Colors.CYAN)
        
        # Menú de opciones
        self.mostrar_menu()
        
        # Resumen final
        self.mostrar_resumen()
    
    def mostrar_menu(self):
        """Muestra menú interactivo de limpieza"""
        
        opciones = [
            ("1", "🗑️ Limpiar archivos temporales", self.limpiar_temporales),
            ("2", "📁 Organizar resultados por fecha", self.organizar_resultados),
            ("3", "📦 Comprimir logs antiguos", self.comprimir_logs),
            ("4", "🧼 Limpiar cache Python", self.limpiar_cache_python),
            ("5", "🗂️ Organizar gráficos", self.organizar_graficos),
            ("6", "📊 Analizar uso de espacio", self.analizar_espacio),
            ("7", "🚀 Limpieza completa automática", self.limpieza_automatica),
            ("8", "❌ Salir", None)
        ]
        
        while True:
            print_header("OPCIONES DE LIMPIEZA")
            
            for codigo, descripcion, _ in opciones:
                print(f"{codigo}. {descripcion}")
            
            try:
                opcion = input(f"\nSelecciona una opción (1-8): ").strip()
                
                if opcion == "8":
                    break
                
                # Buscar y ejecutar opción
                for codigo, desc, funcion in opciones:
                    if opcion == codigo and funcion:
                        print_header(desc.split(" ", 1)[1])  # Quitar emoji
                        funcion()
                        break
                else:
                    print_colored("❌ Opción inválida", Colors.RED)
                    
            except KeyboardInterrupt:
                print_colored("\n👋 Limpieza interrumpida por el usuario", Colors.YELLOW)
                break
            except Exception as e:
                print_colored(f"❌ Error: {e}", Colors.RED)
    
    def limpiar_temporales(self):
        """Limpia archivos temporales"""
        
        print("🔍 Buscando archivos temporales...")
        
        archivos_temp = []
        
        # Buscar por extensiones
        for ext in self.temp_extensions:
            archivos_temp.extend(glob.glob(f"**/*{ext}", recursive=True))
        
        # Buscar archivos específicos
        patrones_temp = [
            "*.tmp", "*.temp", "*.swp", "*.swo",
            ".DS_Store", "Thumbs.db", "desktop.ini"
        ]
        
        for patron in patrones_temp:
            archivos_temp.extend(glob.glob(patron))
        
        if not archivos_temp:
            print_colored("✅ No se encontraron archivos temporales", Colors.GREEN)
            return
        
        print(f"📋 Encontrados {len(archivos_temp)} archivos temporales:")
        
        for archivo in archivos_temp[:10]:  # Mostrar solo primeros 10
            print(f"   📄 {archivo}")
        
        if len(archivos_temp) > 10:
            print(f"   ... y {len(archivos_temp) - 10} más")
        
        respuesta = input(f"\n¿Eliminar estos archivos? (s/N): ").lower()
        
        if respuesta == 's':
            eliminados = 0
            espacio = 0
            
            for archivo in archivos_temp:
                try:
                    path = Path(archivo)
                    if path.exists():
                        size = path.stat().st_size
                        path.unlink()
                        eliminados += 1
                        espacio += size
                        
                except Exception as e:
                    print_colored(f"⚠️ Error eliminando {archivo}: {e}", Colors.YELLOW)
            
            self.archivos_eliminados += eliminados
            self.espacio_liberado += espacio
            
            print_colored(f"✅ Eliminados {eliminados} archivos temporales", Colors.GREEN)
            print_colored(f"💾 Espacio liberado: {self.format_size(espacio)}", Colors.GREEN)
        else:
            print_colored("⏭️ Omitiendo limpieza de temporales", Colors.YELLOW)
    
    def organizar_resultados(self):
        """Organiza resultados por fecha"""
        
        resultados_dir = self.directorios['resultados']
        
        if not resultados_dir.exists():
            print_colored("📁 Directorio resultados/ no existe", Colors.YELLOW)
            return
        
        # Buscar archivos de resultados
        archivos_resultados = list(resultados_dir.glob("*.json"))
        
        if not archivos_resultados:
            print_colored("✅ No hay archivos de resultados para organizar", Colors.GREEN)
            return
        
        print(f"📊 Encontrados {len(archivos_resultados)} archivos de resultados")
        
        organizados = 0
        
        for archivo in archivos_resultados:
            try:
                # Obtener fecha del archivo
                fecha_modificacion = datetime.fromtimestamp(archivo.stat().st_mtime)
                
                # Crear directorio por año-mes
                mes_dir = resultados_dir / fecha_modificacion.strftime("%Y-%m")
                mes_dir.mkdir(exist_ok=True)
                
                # Mover archivo si no está ya en subdirectorio
                if archivo.parent == resultados_dir:
                    nuevo_path = mes_dir / archivo.name
                    archivo.rename(nuevo_path)
                    organizados += 1
                    
            except Exception as e:
                print_colored(f"⚠️ Error organizando {archivo.name}: {e}", Colors.YELLOW)
        
        self.archivos_organizados += organizados
        
        print_colored(f"✅ Organizados {organizados} archivos por fecha", Colors.GREEN)
    
    def comprimir_logs(self):
        """Comprime logs antiguos"""
        
        logs_dir = self.directorios['logs']
        
        if not logs_dir.exists():
            print_colored("📁 Directorio logs/ no existe", Colors.YELLOW)
            return
        
        # Buscar logs antiguos (más de 7 días)
        fecha_limite = datetime.now() - timedelta(days=7)
        logs_antiguos = []
        
        for log_file in logs_dir.glob("*.log"):
            fecha_mod = datetime.fromtimestamp(log_file.stat().st_mtime)
            if fecha_mod < fecha_limite:
                logs_antiguos.append(log_file)
        
        if not logs_antiguos:
            print_colored("✅ No hay logs antiguos para comprimir", Colors.GREEN)
            return
        
        print(f"📦 Encontrados {len(logs_antiguos)} logs antiguos")
        
        # Crear archivo comprimido
        fecha_str = datetime.now().strftime("%Y%m%d")
        zip_filename = logs_dir / f"logs_antiguos_{fecha_str}.zip"
        
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for log_file in logs_antiguos:
                    zipf.write(log_file, log_file.name)
            
            # Eliminar logs originales después de comprimir
            espacio_ahorrado = 0
            for log_file in logs_antiguos:
                espacio_ahorrado += log_file.stat().st_size
                log_file.unlink()
            
            self.espacio_liberado += espacio_ahorrado
            
            print_colored(f"✅ Logs comprimidos en {zip_filename.name}", Colors.GREEN)
            print_colored(f"💾 Espacio ahorrado: {self.format_size(espacio_ahorrado)}", Colors.GREEN)
            
        except Exception as e:
            print_colored(f"❌ Error comprimiendo logs: {e}", Colors.RED)
    
    def limpiar_cache_python(self):
        """Limpia cache de Python"""
        
        cache_dirs = []
        cache_files = []
        
        # Buscar directorios __pycache__
        for cache_dir in Path('.').rglob('__pycache__'):
            cache_dirs.append(cache_dir)
        
        # Buscar archivos .pyc
        for pyc_file in Path('.').rglob('*.pyc'):
            cache_files.append(pyc_file)
        
        total_items = len(cache_dirs) + len(cache_files)
        
        if total_items == 0:
            print_colored("✅ No hay cache de Python para limpiar", Colors.GREEN)
            return
        
        print(f"🐍 Encontrados:")
        print(f"   📁 {len(cache_dirs)} directorios __pycache__")
        print(f"   📄 {len(cache_files)} archivos .pyc")
        
        espacio_liberado = 0
        items_eliminados = 0
        
        # Eliminar directorios cache
        for cache_dir in cache_dirs:
            try:
                # Calcular tamaño antes de eliminar
                for file in cache_dir.rglob('*'):
                    if file.is_file():
                        espacio_liberado += file.stat().st_size
                
                shutil.rmtree(cache_dir)
                items_eliminados += 1
                
            except Exception as e:
                print_colored(f"⚠️ Error eliminando {cache_dir}: {e}", Colors.YELLOW)
        
        # Eliminar archivos .pyc
        for pyc_file in cache_files:
            try:
                espacio_liberado += pyc_file.stat().st_size
                pyc_file.unlink()
                items_eliminados += 1
                
            except Exception as e:
                print_colored(f"⚠️ Error eliminando {pyc_file}: {e}", Colors.YELLOW)
        
        self.archivos_eliminados += items_eliminados
        self.espacio_liberado += espacio_liberado
        
        print_colored(f"✅ Eliminados {items_eliminados} elementos de cache", Colors.GREEN)
        print_colored(f"💾 Espacio liberado: {self.format_size(espacio_liberado)}", Colors.GREEN)
    
    def organizar_graficos(self):
        """Organiza gráficos por tipo y fecha"""
        
        graficos_dir = self.directorios['graficos']
        
        if not graficos_dir.exists():
            print_colored("📁 Directorio graficos/ no existe", Colors.YELLOW)
            return
        
        # Buscar archivos de gráficos
        tipos_graficos = {
            'analisis': ['*analisis*'],
            'predicciones': ['*prediccion*'],
            'demo': ['*demo*'],
            'comparativo': ['*comparativo*', '*resumen*']
        }
        
        organizados = 0
        
        for tipo, patrones in tipos_graficos.items():
            tipo_dir = graficos_dir / tipo
            tipo_dir.mkdir(exist_ok=True)
            
            for patron in patrones:
                for archivo in graficos_dir.glob(f"{patron}.png"):
                    if archivo.parent == graficos_dir:  # Solo archivos en raíz
                        try:
                            nuevo_path = tipo_dir / archivo.name
                            archivo.rename(nuevo_path)
                            organizados += 1
                        except Exception as e:
                            print_colored(f"⚠️ Error moviendo {archivo.name}: {e}", Colors.YELLOW)
        
        self.archivos_organizados += organizados
        
        print_colored(f"✅ Organizados {organizados} gráficos por tipo", Colors.GREEN)
    
    def analizar_espacio(self):
        """Analiza uso de espacio en disco"""
        
        print("💾 Analizando uso de espacio...")
        
        stats = {}
        
        for nombre, directorio in self.directorios.items():
            if directorio.exists():
                tamaño = self.calcular_tamaño_directorio(directorio)
                stats[nombre] = tamaño
            else:
                stats[nombre] = 0
        
        # Mostrar estadísticas
        print(f"\n📊 USO DE ESPACIO POR DIRECTORIO:")
        print(f"{'Directorio':<15} {'Tamaño':<12} {'Archivos':<8}")
        print("-" * 40)
        
        total_size = 0
        for nombre, tamaño in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            if tamaño > 0:
                directorio = self.directorios[nombre]
                num_archivos = sum(1 for _ in directorio.rglob('*') if _.is_file())
                print(f"{nombre:<15} {self.format_size(tamaño):<12} {num_archivos:<8}")
                total_size += tamaño
        
        print("-" * 40)
        print(f"{'TOTAL':<15} {self.format_size(total_size):<12}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        
        if stats.get('logs', 0) > 10 * 1024 * 1024:  # 10MB
            print("   📦 Comprimir logs antiguos")
        
        if stats.get('resultados', 0) > 50 * 1024 * 1024:  # 50MB
            print("   📁 Organizar resultados antiguos")
        
        if stats.get('graficos', 0) > 20 * 1024 * 1024:  # 20MB
            print("   🖼️ Comprimir o eliminar gráficos antiguos")
    
    def limpieza_automatica(self):
        """Ejecuta limpieza completa automática"""
        
        print("🚀 Iniciando limpieza automática completa...")
        
        # Ejecutar todas las limpiezas
        self.limpiar_cache_python()
        self.limpiar_temporales_auto()
        self.organizar_resultados()
        self.organizar_graficos()
        self.comprimir_logs()
        
        print_colored("🎉 Limpieza automática completada", Colors.GREEN)
    
    def limpiar_temporales_auto(self):
        """Limpia temporales automáticamente (sin preguntar)"""
        
        archivos_temp = []
        
        # Buscar archivos temporales
        for ext in self.temp_extensions:
            archivos_temp.extend(glob.glob(f"**/*{ext}", recursive=True))
        
        if archivos_temp:
            eliminados = 0
            espacio = 0
            
            for archivo in archivos_temp:
                try:
                    path = Path(archivo)
                    if path.exists():
                        size = path.stat().st_size
                        path.unlink()
                        eliminados += 1
                        espacio += size
                except:
                    pass
            
            self.archivos_eliminados += eliminados
            self.espacio_liberado += espacio
            
            print_colored(f"✅ Auto-eliminados {eliminados} archivos temporales", Colors.GREEN)
    
    def calcular_tamaño_directorio(self, directorio):
        """Calcula tamaño total de un directorio"""
        
        total = 0
        try:
            for archivo in directorio.rglob('*'):
                if archivo.is_file():
                    total += archivo.stat().st_size
        except Exception:
            pass
        
        return total
    
    def format_size(self, bytes_size):
        """Formatea tamaño en bytes a formato legible"""
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def mostrar_resumen(self):
        """Muestra resumen de la limpieza"""
        
        print_header("RESUMEN DE LIMPIEZA")
        
        print(f"📊 ESTADÍSTICAS FINALES:")
        print(f"   🗑️ Archivos eliminados: {self.archivos_eliminados}")
        print(f"   📁 Archivos organizados: {self.archivos_organizados}")
        print(f"   💾 Espacio liberado: {self.format_size(self.espacio_liberado)}")
        
        if self.archivos_eliminados > 0 or self.archivos_organizados > 0:
            print_colored("🎉 ¡Limpieza completada exitosamente!", Colors.GREEN)
        else:
            print_colored("✨ Sistema ya estaba limpio y organizado", Colors.BLUE)
        
        print(f"\n📋 PRÓXIMOS PASOS:")
        print("   🧪 Ejecutar: python test_sistema.py")
        print("   🚀 Lanzar: python lanzar_sistema.py")
        print("   📊 Verificar: python demo_sistema.py")

def main():
    """Función principal"""
    
    print("🧹 SISTEMA DE LIMPIEZA Y ORGANIZACIÓN")
    print("Desarrollado por AI Expert Developer & Economist")
    print("=" * 60)
    
    # Crear y ejecutar limpiador
    limpiador = LimpiadorSistema()
    limpiador.ejecutar_limpieza_completa()
    
    print(f"\n✨ LIMPIEZA FINALIZADA")
    print("Sistema optimizado y listo para usar")

if __name__ == "__main__":
    main() 