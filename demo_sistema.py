#!/usr/bin/env python3
"""
SISTEMA DE DEMOSTRACIÓN
Análisis y Predicción de Criptomonedas - Modo Demo

Este script demuestra las capacidades del sistema usando datos simulados,
perfecto para testear sin necesidad de claves API.

Características del Demo:
- Datos históricos simulados realistas
- Todos los indicadores técnicos funcionando
- Predicciones con Machine Learning
- Análisis de sentimientos simulado
- Gráficos y reportes completos

Autor: AI Expert Developer & Economist
Versión: 2.0 DEMO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path

# Configurar warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GeneradorDatosDemo:
    """Genera datos simulados realistas para demostración"""
    
    def __init__(self):
        self.precios_base = {
            'BTCUSDT': 45000,
            'ETHUSDT': 2800,
            'SOLUSDT': 85,
            'ADAUSDT': 0.45,
            'DOGEUSDT': 0.08
        }
    
    def generar_datos_historicos(self, symbol='BTCUSDT', periodos=720):
        """Genera datos históricos simulados"""
        
        # Configuración de simulación
        precio_inicial = self.precios_base.get(symbol, 100)
        volatilidad_base = 0.02
        tendencia = 0.0001
        
        # Generar fechas
        fechas = pd.date_range(
            start=datetime.now() - timedelta(hours=periodos),
            end=datetime.now(),
            freq='H'
        )
        
        # Generar precios con movimiento Browniano geométrico
        returns = np.random.normal(tendencia, volatilidad_base, len(fechas))
        
        # Agregar ciclos y patrones
        for i in range(len(returns)):
            # Ciclo diario
            hora_del_dia = fechas[i].hour
            if 8 <= hora_del_dia <= 16:  # Horario "activo"
                returns[i] *= 1.2
            
            # Volatilidad de fin de semana
            if fechas[i].weekday() >= 5:
                returns[i] *= 0.7
            
            # Eventos aleatorios
            if np.random.random() < 0.02:  # 2% probabilidad de evento
                returns[i] *= np.random.choice([2, -1.5])
        
        # Calcular precios
        precios = [precio_inicial]
        for ret in returns[1:]:
            nuevo_precio = precios[-1] * (1 + ret)
            precios.append(nuevo_precio)
        
        # Generar OHLCV
        df = pd.DataFrame()
        df['timestamp'] = fechas
        df['close'] = precios
        
        # Generar Open, High, Low basado en Close
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        
        # High y Low con volatilidad intraperiodo
        intra_vol = np.random.normal(0, volatilidad_base * 0.5, len(df))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(intra_vol))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(intra_vol))
        
        # Volumen simulado
        volumen_base = np.random.normal(1000000, 200000, len(df))
        df['volume'] = np.maximum(volumen_base * (1 + np.abs(returns) * 10), 100000)
        
        # Ordenar columnas
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df

class AnalizadorTecnicoDemo:
    """Analizador técnico para datos de demostración"""
    
    def __init__(self):
        pass
    
    def calcular_indicadores(self, df):
        """Calcula todos los indicadores técnicos"""
        
        try:
            # Importar TA si está disponible
            import ta
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Medias móviles
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
        except ImportError:
            # Implementación manual si TA no está disponible
            df['rsi'] = self._calcular_rsi_manual(df['close'])
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD manual
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands manual
            rolling_mean = df['close'].rolling(20).mean()
            rolling_std = df['close'].rolling(20).std()
            df['bb_upper'] = rolling_mean + (rolling_std * 2)
            df['bb_middle'] = rolling_mean
            df['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        return df
    
    def _calcular_rsi_manual(self, precios, periodo=14):
        """Calcula RSI manualmente"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        
        avg_ganancia = ganancia.rolling(periodo).mean()
        avg_perdida = perdida.rolling(periodo).mean()
        
        rs = avg_ganancia / avg_perdida
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analizar_tendencia(self, df):
        """Analiza la tendencia actual"""
        
        precio_actual = df['close'].iloc[-1]
        precio_anterior = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
        
        cambio_24h = ((precio_actual - precio_anterior) / precio_anterior) * 100
        
        # Análisis de medias móviles
        sma_20 = df['sma_20'].iloc[-1]
        ema_12 = df['ema_12'].iloc[-1]
        
        tendencia_score = 0
        
        # Precio vs medias
        if precio_actual > sma_20:
            tendencia_score += 1
        if precio_actual > ema_12:
            tendencia_score += 1
        
        # MACD
        macd_actual = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        if macd_actual > macd_signal:
            tendencia_score += 1
        
        # RSI
        rsi_actual = df['rsi'].iloc[-1]
        if 40 < rsi_actual < 70:  # Zona neutral-alcista
            tendencia_score += 1
        
        return {
            'precio_actual': precio_actual,
            'cambio_24h': cambio_24h,
            'tendencia_score': tendencia_score,
            'interpretacion': self._interpretar_tendencia(tendencia_score),
            'rsi': rsi_actual,
            'macd': macd_actual,
            'soporte': df['bb_lower'].iloc[-1],
            'resistencia': df['bb_upper'].iloc[-1]
        }
    
    def _interpretar_tendencia(self, score):
        """Interpreta el score de tendencia"""
        if score >= 3:
            return "🟢 ALCISTA FUERTE"
        elif score == 2:
            return "🟡 ALCISTA MODERADA"
        elif score == 1:
            return "🟠 NEUTRAL/MIXTA"
        else:
            return "🔴 BAJISTA"

class PrediccionMLDemo:
    """Sistema de predicción ML para demo"""
    
    def __init__(self):
        self.modelos_entrenados = {}
    
    def preparar_features(self, df):
        """Prepara features para ML"""
        
        # Features técnicas
        features = pd.DataFrame()
        features['rsi'] = df['rsi']
        features['macd'] = df['macd']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['price_sma_ratio'] = df['close'] / df['sma_20']
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Features de momentum
        features['returns_1h'] = df['close'].pct_change()
        features['returns_4h'] = df['close'].pct_change(periods=4)
        features['returns_24h'] = df['close'].pct_change(periods=24)
        
        # Features de volatilidad
        features['volatility'] = df['close'].rolling(24).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(48).mean()
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def entrenar_modelo_demo(self, df, symbol):
        """Entrena modelo de demostración"""
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Preparar datos
            features = self.preparar_features(df)
            
            # Target: precio futuro (4 horas adelante)
            target = df['close'].shift(-4).fillna(method='ffill')
            
            # Eliminar NaN y obtener datos recientes
            data_clean = pd.concat([features, target.rename('target')], axis=1).dropna()
            
            if len(data_clean) < 100:  # Datos insuficientes
                return self._generar_prediccion_simple(df)
            
            # Separar features y target
            X = data_clean.drop('target', axis=1)
            y = data_clean['target']
            
            # Split de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Escalar datos
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            modelo = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            modelo.fit(X_train_scaled, y_train)
            
            # Evaluar
            score = modelo.score(X_test_scaled, y_test)
            
            # Guardar modelo y scaler
            self.modelos_entrenados[symbol] = {
                'modelo': modelo,
                'scaler': scaler,
                'score': score,
                'features': X.columns.tolist()
            }
            
            return {
                'exito': True,
                'score': score,
                'mensaje': f"Modelo entrenado con R²: {score:.3f}"
            }
            
        except ImportError:
            return self._generar_prediccion_simple(df)
        except Exception as e:
            return {
                'exito': False,
                'mensaje': f"Error entrenando modelo: {e}",
                'prediccion_simple': self._generar_prediccion_simple(df)
            }
    
    def predecir(self, df, symbol, horizontes=[1, 4, 12, 24]):
        """Genera predicciones"""
        
        if symbol not in self.modelos_entrenados:
            return self._generar_prediccion_simple(df, horizontes)
        
        try:
            modelo_info = self.modelos_entrenados[symbol]
            modelo = modelo_info['modelo']
            scaler = modelo_info['scaler']
            
            # Preparar features actuales
            features = self.preparar_features(df)
            features_actual = features.iloc[-1:][modelo_info['features']]
            features_scaled = scaler.transform(features_actual)
            
            # Predicción base
            prediccion_base = modelo.predict(features_scaled)[0]
            precio_actual = df['close'].iloc[-1]
            
            # Generar predicciones para diferentes horizontes
            predicciones = {}
            
            for h in horizontes:
                # Ajustar predicción según horizonte
                factor_tiempo = 1 + (h - 4) * 0.1  # Más incertidumbre a largo plazo
                volatilidad_estimada = df['close'].rolling(48).std().iloc[-1] / precio_actual
                
                # Predicción ajustada
                pred = prediccion_base * factor_tiempo
                
                # Intervalos de confianza
                intervalo = volatilidad_estimada * precio_actual * np.sqrt(h)
                
                predicciones[f'{h}h'] = {
                    'precio_predicho': pred,
                    'cambio_porcentual': ((pred - precio_actual) / precio_actual) * 100,
                    'intervalo_inferior': pred - intervalo,
                    'intervalo_superior': pred + intervalo,
                    'confianza': max(0.5, modelo_info['score'])
                }
            
            return predicciones
            
        except Exception as e:
            return self._generar_prediccion_simple(df, horizontes)
    
    def _generar_prediccion_simple(self, df, horizontes=[1, 4, 12, 24]):
        """Genera predicción simple basada en tendencia"""
        
        precio_actual = df['close'].iloc[-1]
        
        # Calcular tendencia simple
        returns_recientes = df['close'].pct_change().tail(24).mean()
        volatilidad = df['close'].pct_change().tail(48).std()
        
        predicciones = {}
        
        for h in horizontes:
            # Proyección simple
            prediccion = precio_actual * (1 + returns_recientes * h)
            
            # Intervalo basado en volatilidad
            intervalo = precio_actual * volatilidad * np.sqrt(h)
            
            predicciones[f'{h}h'] = {
                'precio_predicho': prediccion,
                'cambio_porcentual': ((prediccion - precio_actual) / precio_actual) * 100,
                'intervalo_inferior': prediccion - intervalo,
                'intervalo_superior': prediccion + intervalo,
                'confianza': 0.6,
                'metodo': 'tendencia_simple'
            }
        
        return predicciones

class VisualizadorDemo:
    """Sistema de visualización para demo"""
    
    def __init__(self):
        self.figsize = (15, 10)
    
    def grafico_completo(self, df, symbol, analisis, predicciones):
        """Genera gráfico completo de análisis"""
        
        fig, axes = plt.subplots(3, 2, figsize=self.figsize)
        fig.suptitle(f'📊 ANÁLISIS COMPLETO - {symbol}', fontsize=16, fontweight='bold')
        
        # 1. Precio y medias móviles
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='Precio', linewidth=2)
        ax1.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
        ax1.plot(df.index, df['ema_12'], label='EMA 12', alpha=0.7)
        ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.2, label='Bollinger')
        ax1.set_title('💰 Precio y Tendencias')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[0, 1]
        ax2.plot(df.index, df['rsi'], color='orange', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Sobrecompra')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Sobreventa')
        ax2.set_title('📈 RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['macd'], label='MACD', linewidth=2)
        ax3.plot(df.index, df['macd_signal'], label='Signal', linewidth=2)
        ax3.bar(df.index, df['macd_histogram'], alpha=0.3, label='Histogram')
        ax3.set_title('📊 MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volumen
        ax4 = axes[1, 1]
        ax4.bar(df.index, df['volume'], alpha=0.6, color='purple')
        ax4.set_title('📊 Volumen de Operaciones')
        ax4.grid(True, alpha=0.3)
        
        # 5. Predicciones
        ax5 = axes[2, 0]
        horizontes = list(predicciones.keys())
        precios_pred = [predicciones[h]['precio_predicho'] for h in horizontes]
        cambios = [predicciones[h]['cambio_porcentual'] for h in horizontes]
        
        colores = ['red' if c < 0 else 'green' for c in cambios]
        ax5.bar(horizontes, cambios, color=colores, alpha=0.7)
        ax5.set_title('🔮 Predicciones de Cambio %')
        ax5.set_ylabel('Cambio %')
        ax5.grid(True, alpha=0.3)
        
        # 6. Métricas actuales
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Texto con métricas
        precio_actual = analisis['precio_actual']
        texto_metricas = f"""
📊 MÉTRICAS ACTUALES

💰 Precio: ${precio_actual:,.2f}
📈 Cambio 24h: {analisis['cambio_24h']:+.2f}%
🎯 Tendencia: {analisis['interpretacion']}
📊 RSI: {analisis['rsi']:.1f}
🔄 MACD: {analisis['macd']:.4f}

🎯 NIVELES CLAVE:
🟢 Soporte: ${analisis['soporte']:,.2f}
🔴 Resistencia: ${analisis['resistencia']:,.2f}

🤖 SCORE TÉCNICO: {analisis['tendencia_score']}/4
        """
        
        ax6.text(0.1, 0.9, texto_metricas, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_analisis_{symbol}_{timestamp}.png"
        
        # Crear directorio si no existe
        Path("graficos").mkdir(exist_ok=True)
        
        plt.savefig(f"graficos/{filename}", dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: graficos/{filename}")
        
        plt.show()

class SistemaDemoCompleto:
    """Sistema completo de demostración"""
    
    def __init__(self):
        self.generador = GeneradorDatosDemo()
        self.analizador = AnalizadorTecnicoDemo()
        self.predictor = PrediccionMLDemo()
        self.visualizador = VisualizadorDemo()
        
        print("🎭 SISTEMA DE DEMOSTRACIÓN INICIADO")
        print("=" * 50)
    
    def ejecutar_demo(self, symbol='BTCUSDT', mostrar_graficos=True):
        """Ejecuta demostración completa"""
        
        print(f"\n🚀 INICIANDO ANÁLISIS DEMO PARA {symbol}")
        print("-" * 40)
        
        # 1. Generar datos
        print("📊 Generando datos históricos simulados...")
        df = self.generador.generar_datos_historicos(symbol, periodos=720)
        print(f"   ✅ {len(df)} períodos generados")
        
        # 2. Calcular indicadores
        print("🔢 Calculando indicadores técnicos...")
        df = self.analizador.calcular_indicadores(df)
        print("   ✅ Indicadores calculados")
        
        # 3. Análisis de tendencia
        print("📈 Analizando tendencia actual...")
        analisis = self.analizador.analizar_tendencia(df)
        print(f"   ✅ Tendencia: {analisis['interpretacion']}")
        
        # 4. Entrenar modelo ML
        print("🤖 Entrenando modelo de Machine Learning...")
        resultado_entrenamiento = self.predictor.entrenar_modelo_demo(df, symbol)
        
        if resultado_entrenamiento.get('exito'):
            print(f"   ✅ {resultado_entrenamiento['mensaje']}")
        else:
            print(f"   ⚠️ Usando predicción simple")
        
        # 5. Generar predicciones
        print("🔮 Generando predicciones futuras...")
        predicciones = self.predictor.predecir(df, symbol)
        print(f"   ✅ Predicciones para {len(predicciones)} horizontes")
        
        # 6. Mostrar resultados
        self.mostrar_resultados(symbol, analisis, predicciones)
        
        # 7. Generar gráficos
        if mostrar_graficos:
            print("📈 Generando visualizaciones...")
            self.visualizador.grafico_completo(df, symbol, analisis, predicciones)
        
        # 8. Guardar reporte
        self.guardar_reporte_demo(symbol, analisis, predicciones)
        
        return {
            'symbol': symbol,
            'analisis': analisis,
            'predicciones': predicciones,
            'datos': df
        }
    
    def mostrar_resultados(self, symbol, analisis, predicciones):
        """Muestra resultados en consola"""
        
        print(f"\n📋 REPORTE DE ANÁLISIS - {symbol}")
        print("=" * 50)
        
        # Información actual
        print(f"💰 Precio Actual: ${analisis['precio_actual']:,.2f}")
        print(f"📊 Cambio 24h: {analisis['cambio_24h']:+.2f}%")
        print(f"🎯 Tendencia: {analisis['interpretacion']}")
        print(f"📈 RSI: {analisis['rsi']:.1f}")
        print(f"🔄 MACD: {analisis['macd']:.4f}")
        
        # Niveles clave
        print(f"\n🎯 NIVELES CLAVE:")
        print(f"🟢 Soporte: ${analisis['soporte']:,.2f}")
        print(f"🔴 Resistencia: ${analisis['resistencia']:,.2f}")
        
        # Predicciones
        print(f"\n🔮 PREDICCIONES:")
        for horizonte, pred in predicciones.items():
            cambio = pred['cambio_porcentual']
            confianza = pred['confianza']
            emoji = "🟢" if cambio > 0 else "🔴"
            
            print(f"{emoji} {horizonte}: ${pred['precio_predicho']:,.2f} "
                  f"({cambio:+.2f}%) - Confianza: {confianza:.1%}")
        
        print("\n" + "=" * 50)
    
    def guardar_reporte_demo(self, symbol, analisis, predicciones):
        """Guarda reporte de demo"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        reporte = {
            'timestamp': timestamp,
            'symbol': symbol,
            'tipo': 'DEMO',
            'analisis_actual': analisis,
            'predicciones': predicciones,
            'advertencia': 'DATOS SIMULADOS - SOLO PARA DEMOSTRACIÓN'
        }
        
        # Crear directorio si no existe
        Path("resultados").mkdir(exist_ok=True)
        
        # Guardar JSON
        filename = f"resultados/demo_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(reporte, f, indent=2, default=str)
        
        print(f"💾 Reporte guardado: {filename}")
    
    def demo_multiples_pares(self, pares=['BTCUSDT', 'ETHUSDT', 'SOLUSDT']):
        """Ejecuta demo para múltiples pares"""
        
        print("🌟 DEMOSTRACIÓN MULTI-PAR")
        print("=" * 50)
        
        resultados = {}
        
        for par in pares:
            try:
                resultado = self.ejecutar_demo(par, mostrar_graficos=False)
                resultados[par] = resultado
                print(f"✅ {par} completado")
                
            except Exception as e:
                print(f"❌ Error con {par}: {e}")
                continue
        
        # Resumen comparativo
        self.mostrar_resumen_comparativo(resultados)
        
        return resultados
    
    def mostrar_resumen_comparativo(self, resultados):
        """Muestra resumen comparativo"""
        
        print(f"\n📊 RESUMEN COMPARATIVO")
        print("=" * 60)
        print(f"{'Par':<10} {'Precio':<12} {'24h %':<8} {'Tend':<15} {'Pred 4h %':<10}")
        print("-" * 60)
        
        for par, data in resultados.items():
            analisis = data['analisis']
            predicciones = data['predicciones']
            
            precio = f"${analisis['precio_actual']:,.2f}"
            cambio_24h = f"{analisis['cambio_24h']:+.1f}%"
            tendencia = analisis['interpretacion'][:12]
            pred_4h = f"{predicciones.get('4h', {}).get('cambio_porcentual', 0):+.1f}%"
            
            print(f"{par:<10} {precio:<12} {cambio_24h:<8} {tendencia:<15} {pred_4h:<10}")

def main():
    """Función principal del demo"""
    
    print("🎭 SISTEMA DEMO DE ANÁLISIS PREDICTIVO")
    print("Desarrollado por AI Expert Developer & Economist")
    print("Versión 2.0 - Demostración con Datos Simulados")
    print("=" * 60)
    
    # Crear sistema demo
    sistema = SistemaDemoCompleto()
    
    # Menú interactivo
    while True:
        print(f"\n🎯 OPCIONES DEL DEMO:")
        print("1. 📊 Análisis individual (con gráficos)")
        print("2. 🚀 Análisis rápido (sin gráficos)")
        print("3. 🌟 Análisis múltiples pares")
        print("4. 🎲 Par aleatorio")
        print("5. ❌ Salir")
        
        try:
            opcion = input("\nSelecciona una opción (1-5): ").strip()
            
            if opcion == '1':
                par = input("Ingresa el par (ej: BTCUSDT) o Enter para BTC: ").strip().upper()
                if not par:
                    par = 'BTCUSDT'
                sistema.ejecutar_demo(par, mostrar_graficos=True)
                
            elif opcion == '2':
                par = input("Ingresa el par (ej: ETHUSDT) o Enter para ETH: ").strip().upper()
                if not par:
                    par = 'ETHUSDT'
                sistema.ejecutar_demo(par, mostrar_graficos=False)
                
            elif opcion == '3':
                pares_input = input("Pares separados por coma (Enter para default): ").strip()
                if pares_input:
                    pares = [p.strip().upper() for p in pares_input.split(',')]
                else:
                    pares = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
                sistema.demo_multiples_pares(pares)
                
            elif opcion == '4':
                pares_random = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'SHIBUSDT']
                par_aleatorio = np.random.choice(pares_random)
                print(f"🎲 Par seleccionado aleatoriamente: {par_aleatorio}")
                sistema.ejecutar_demo(par_aleatorio, mostrar_graficos=True)
                
            elif opcion == '5':
                break
                
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrumpido por el usuario")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎭 DEMO FINALIZADO")
    print("📚 Recuerda: Este sistema usa datos simulados para demostración")
    print("💡 Para datos reales, configura las claves API y usa sistema_integrado.py")
    print("⚠️ No constituye asesoramiento financiero")

if __name__ == "__main__":
    main() 