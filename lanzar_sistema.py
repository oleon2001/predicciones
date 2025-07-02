#!/usr/bin/env python3
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
        print("\n⏹️ Sistema interrumpido por el usuario")
        
    except Exception as e:
        print(f"❌ Error ejecutando sistema: {e}")

if __name__ == "__main__":
    main()
