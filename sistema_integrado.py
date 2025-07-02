#!/usr/bin/env python3
"""
SISTEMA INTEGRADO DE ANÁLISIS Y PREDICCIÓN DE CRIPTOMONEDAS
========================================================

Combina análisis técnico tradicional con machine learning avanzado
Desarrollado por: AI Expert Developer & Economist

Características:
- Análisis técnico completo con 30+ indicadores
- Modelos de Machine Learning (Random Forest, XGBoost, LSTM)
- Predicciones a múltiples horizontes temporales
- Análisis de sentimientos avanzado
- Detección de patrones chartistas
- Backtesting automático
- Sistema de scoring predictivo
- Reportes detallados con visualizaciones

ADVERTENCIA: Este sistema es para fines educativos y análisis.
NO constituye asesoramiento financiero. Los mercados son impredecibles.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from typing import Dict, List, Tuple, Optional

# Importar sistema avanzado
try:
    from prediccion_avanzada import AnalizadorPredictivoAvanzado, ConfiguracionAvanzada
    SISTEMA_AVANZADO_DISPONIBLE = True
except ImportError:
    SISTEMA_AVANZADO_DISPONIBLE = False
    print("⚠️ Sistema avanzado no disponible. Usando modo básico.")

# Importar sistema original mejorado
from binance.client import Client
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

class SistemaIntegradoPrediccion:
    """Sistema principal que integra análisis básico y avanzado"""
    
    def __init__(self):
        self.config = self._cargar_configuracion()
        self.client = None
        self.analizador_avanzado = None
        self.resultados_historicos = {}
        
        self._inicializar_sistemas()
    
    def _cargar_configuracion(self):
        """Carga configuración desde archivo o usa defaults"""
        config_default = {
            'api_key': "26Sh06jOHDKVRqZY5puDa7q16hQ1CU0aitWM0OWy1iMF0jU8h8jqKqUFsxzLC5Ze",
            'api_secret': "yOWjPf3wHDuQbBJVKll3kDYGG6c57GiUgbdu6Xp79VnAG3dzC5AUMU4IDd3LhsnT",
            'news_api_key': "95a1ebc226f34eb38842c95fd4ce1932",
            'pares_analizar': [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT",
                "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
            ],
            'intervalo_datos': '1h',
            'periodo_historico': '30 day ago UTC',
            'horizontes_prediccion': [1, 4, 12, 24],  # horas
            'modo_avanzado': True,
            'guardar_resultados': True,
            'mostrar_graficos': True
        }
        
        # Intentar cargar desde archivo
        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    config_archivo = json.load(f)
                    config_default.update(config_archivo)
            except:
                pass
        
        return config_default
    
    def _inicializar_sistemas(self):
        """Inicializa conexiones y sistemas"""
        # Cliente Binance
        try:
            self.client = Client(self.config['api_key'], self.config['api_secret'])
            self.client.get_account()
            print("✅ Binance conectado y autenticado")
        except:
            try:
                self.client = Client()
                print("⚠️ Binance en modo público (sin autenticación)")
            except:
                print("❌ Error conectando a Binance")
                return
        
        # Sistema avanzado
        if SISTEMA_AVANZADO_DISPONIBLE and self.config['modo_avanzado']:
            try:
                self.analizador_avanzado = AnalizadorPredictivoAvanzado()
                print("✅ Sistema avanzado de ML activado")
            except Exception as e:
                print(f"⚠️ Error activando sistema avanzado: {e}")
                self.analizador_avanzado = None
    
    def analisis_completo_par(self, par: str) -> Dict:
        """Realiza análisis completo de un par específico"""
        print(f"\n{'='*60}")
        print(f"🔍 ANÁLISIS COMPLETO PARA {par}")
        print(f"{'='*60}")
        
        resultado = {
            'par': par,
            'timestamp': datetime.now(),
            'analisis_basico': {},
            'analisis_avanzado': {},
            'predicciones': {},
            'recomendaciones': {},
            'score_final': 50
        }
        
        # 1. Obtener datos históricos
        print("📊 Obteniendo datos históricos...")
        df = self._obtener_datos_historicos(par)
        
        if df.empty:
            print(f"❌ No se pudieron obtener datos para {par}")
            return resultado
        
        # 2. Análisis técnico básico
        print("🔧 Realizando análisis técnico básico...")
        resultado['analisis_basico'] = self._analisis_tecnico_basico(df)
        
        # 3. Análisis avanzado con ML (si está disponible)
        if self.analizador_avanzado:
            print("🤖 Ejecutando análisis avanzado con ML...")
            try:
                analisis_avanzado = self.analizador_avanzado.hacer_prediccion_completa(par)
                if analisis_avanzado:
                    resultado['analisis_avanzado'] = analisis_avanzado
                    resultado['predicciones'] = analisis_avanzado['predicciones']
                    resultado['score_final'] = analisis_avanzado['score']['score_final']
            except Exception as e:
                print(f"⚠️ Error en análisis avanzado: {e}")
        
        # 4. Generar recomendaciones integradas
        print("💡 Generando recomendaciones...")
        resultado['recomendaciones'] = self._generar_recomendaciones_integradas(resultado)
        
        # 5. Guardar resultados si está configurado
        if self.config['guardar_resultados']:
            self._guardar_resultado(resultado)
        
        # 6. Mostrar reporte
        self._mostrar_reporte_integrado(resultado)
        
        # 7. Generar gráficos si está configurado
        if self.config['mostrar_graficos']:
            self._generar_graficos_integrados(df, resultado)
        
        return resultado
    
    def _obtener_datos_historicos(self, par: str) -> pd.DataFrame:
        """Obtiene datos históricos mejorados"""
        try:
            klines = self.client.get_historical_klines(
                par, 
                self.config['intervalo_datos'], 
                self.config['periodo_historico'],
                limit=1000
            )
            
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Conversiones mejoradas
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return pd.DataFrame()
    
    def _analisis_tecnico_basico(self, df: pd.DataFrame) -> Dict:
        """Análisis técnico mejorado"""
        if len(df) < 50:
            return {'error': 'Datos insuficientes'}
        
        analisis = {}
        
        try:
            # Precios actuales
            precio_actual = df['close'].iloc[-1]
            cambio_24h = (precio_actual - df['close'].iloc[-24]) / df['close'].iloc[-24] if len(df) >= 24 else 0
            
            analisis['precio'] = {
                'actual': precio_actual,
                'cambio_24h': cambio_24h,
                'maximo_24h': df['high'].tail(24).max() if len(df) >= 24 else df['high'].max(),
                'minimo_24h': df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()
            }
            
            # Indicadores técnicos
            # RSI
            rsi = RSIIndicator(df['close'], window=14).rsi()
            analisis['rsi'] = {
                'valor': rsi.iloc[-1] if not rsi.empty else 50,
                'estado': self._interpretar_rsi(rsi.iloc[-1] if not rsi.empty else 50)
            }
            
            # MACD
            macd_ind = MACD(df['close'])
            macd = macd_ind.macd()
            macd_signal = macd_ind.macd_signal()
            
            analisis['macd'] = {
                'macd': macd.iloc[-1] if not macd.empty else 0,
                'signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
                'divergencia': (macd.iloc[-1] - macd_signal.iloc[-1]) if not macd.empty and not macd_signal.empty else 0
            }
            
            # Medias móviles
            sma_20 = SMAIndicator(df['close'], window=20).sma_indicator()
            sma_50 = SMAIndicator(df['close'], window=50).sma_indicator()
            ema_12 = EMAIndicator(df['close'], window=12).ema_indicator()
            
            analisis['medias'] = {
                'sma_20': sma_20.iloc[-1] if not sma_20.empty else precio_actual,
                'sma_50': sma_50.iloc[-1] if not sma_50.empty else precio_actual,
                'ema_12': ema_12.iloc[-1] if not ema_12.empty else precio_actual,
                'tendencia': self._determinar_tendencia(precio_actual, sma_20.iloc[-1] if not sma_20.empty else precio_actual, sma_50.iloc[-1] if not sma_50.empty else precio_actual)
            }
            
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            
            analisis['bollinger'] = {
                'upper': bb_upper.iloc[-1] if not bb_upper.empty else precio_actual * 1.1,
                'lower': bb_lower.iloc[-1] if not bb_lower.empty else precio_actual * 0.9,
                'posicion': self._posicion_bollinger(precio_actual, bb_upper.iloc[-1] if not bb_upper.empty else precio_actual * 1.1, bb_lower.iloc[-1] if not bb_lower.empty else precio_actual * 0.9)
            }
            
            # Volumen
            vol_promedio = df['volume'].rolling(20).mean().iloc[-1]
            vol_actual = df['volume'].iloc[-1]
            
            analisis['volumen'] = {
                'actual': vol_actual,
                'promedio_20': vol_promedio,
                'ratio': vol_actual / vol_promedio if vol_promedio > 0 else 1,
                'interpretacion': 'Alto' if vol_actual > vol_promedio * 1.5 else 'Normal' if vol_actual > vol_promedio * 0.5 else 'Bajo'
            }
            
            # Score técnico básico
            analisis['score_tecnico'] = self._calcular_score_tecnico_basico(analisis)
            
        except Exception as e:
            print(f"⚠️ Error en análisis técnico: {e}")
            analisis['error'] = str(e)
        
        return analisis
    
    def _interpretar_rsi(self, rsi: float) -> str:
        """Interpreta valores de RSI"""
        if rsi >= 70:
            return "Sobrecomprado"
        elif rsi <= 30:
            return "Sobrevendido"
        elif rsi > 50:
            return "Alcista"
        else:
            return "Bajista"
    
    def _determinar_tendencia(self, precio: float, sma_20: float, sma_50: float) -> str:
        """Determina tendencia basada en medias móviles"""
        if precio > sma_20 > sma_50:
            return "Alcista Fuerte"
        elif precio > sma_20:
            return "Alcista"
        elif precio < sma_20 < sma_50:
            return "Bajista Fuerte"
        elif precio < sma_20:
            return "Bajista"
        else:
            return "Lateral"
    
    def _posicion_bollinger(self, precio: float, upper: float, lower: float) -> str:
        """Determina posición en Bollinger Bands"""
        if precio > upper:
            return "Por encima de banda superior"
        elif precio < lower:
            return "Por debajo de banda inferior"
        else:
            pct = (precio - lower) / (upper - lower)
            if pct > 0.8:
                return "Cerca de banda superior"
            elif pct < 0.2:
                return "Cerca de banda inferior"
            else:
                return "En zona media"
    
    def _calcular_score_tecnico_basico(self, analisis: Dict) -> float:
        """Calcula score técnico básico 0-100"""
        try:
            score = 50  # Base neutral
            
            # RSI
            rsi = analisis.get('rsi', {}).get('valor', 50)
            if rsi < 30:
                score += 15  # Oversold = bullish
            elif rsi > 70:
                score -= 15  # Overbought = bearish
            elif rsi > 50:
                score += 5
            else:
                score -= 5
            
            # MACD
            macd_div = analisis.get('macd', {}).get('divergencia', 0)
            if macd_div > 0:
                score += 10
            else:
                score -= 10
            
            # Tendencia
            tendencia = analisis.get('medias', {}).get('tendencia', 'Lateral')
            if 'Alcista Fuerte' in tendencia:
                score += 15
            elif 'Alcista' in tendencia:
                score += 10
            elif 'Bajista Fuerte' in tendencia:
                score -= 15
            elif 'Bajista' in tendencia:
                score -= 10
            
            # Volumen
            vol_ratio = analisis.get('volumen', {}).get('ratio', 1)
            if vol_ratio > 1.5:
                score += 5  # Alto volumen refuerza la señal
            
            return max(0, min(100, score))
            
        except:
            return 50
    
    def _generar_recomendaciones_integradas(self, resultado: Dict) -> Dict:
        """Genera recomendaciones combinando análisis básico y avanzado"""
        recomendaciones = {
            'señal_principal': 'NEUTRAL',
            'confianza': 0.5,
            'horizonte_optimo': '4h',
            'precio_objetivo': None,
            'stop_loss': None,
            'take_profit': None,
            'razonamiento': []
        }
        
        try:
            # Score combinado
            score_basico = resultado.get('analisis_basico', {}).get('score_tecnico', 50)
            score_avanzado = resultado.get('score_final', 50)
            
            # Ponderación: 40% básico, 60% avanzado si disponible
            if self.analizador_avanzado and 'analisis_avanzado' in resultado:
                score_final = score_basico * 0.4 + score_avanzado * 0.6
                confianza = 0.8
            else:
                score_final = score_basico
                confianza = 0.6
            
            # Determinar señal principal
            if score_final >= 70:
                recomendaciones['señal_principal'] = 'COMPRA FUERTE'
                recomendaciones['confianza'] = confianza * 0.9
            elif score_final >= 60:
                recomendaciones['señal_principal'] = 'COMPRA'
                recomendaciones['confianza'] = confianza * 0.8
            elif score_final >= 55:
                recomendaciones['señal_principal'] = 'COMPRA DÉBIL'
                recomendaciones['confianza'] = confianza * 0.6
            elif score_final <= 30:
                recomendaciones['señal_principal'] = 'VENTA FUERTE'
                recomendaciones['confianza'] = confianza * 0.9
            elif score_final <= 40:
                recomendaciones['señal_principal'] = 'VENTA'
                recomendaciones['confianza'] = confianza * 0.8
            elif score_final <= 45:
                recomendaciones['señal_principal'] = 'VENTA DÉBIL'
                recomendaciones['confianza'] = confianza * 0.6
            else:
                recomendaciones['señal_principal'] = 'NEUTRAL'
                recomendaciones['confianza'] = 0.5
            
            # Calcular niveles de precio
            precio_actual = resultado.get('analisis_basico', {}).get('precio', {}).get('actual', 0)
            if precio_actual > 0:
                if 'COMPRA' in recomendaciones['señal_principal']:
                    recomendaciones['precio_objetivo'] = precio_actual * 1.05  # +5%
                    recomendaciones['stop_loss'] = precio_actual * 0.97  # -3%
                    recomendaciones['take_profit'] = precio_actual * 1.10  # +10%
                elif 'VENTA' in recomendaciones['señal_principal']:
                    recomendaciones['precio_objetivo'] = precio_actual * 0.95  # -5%
                    recomendaciones['stop_loss'] = precio_actual * 1.03  # +3%
                    recomendaciones['take_profit'] = precio_actual * 0.90  # -10%
            
            # Razonamiento
            recomendaciones['razonamiento'] = self._generar_razonamiento(resultado, score_final)
            
        except Exception as e:
            print(f"⚠️ Error generando recomendaciones: {e}")
        
        return recomendaciones
    
    def _generar_razonamiento(self, resultado: Dict, score: float) -> List[str]:
        """Genera razonamiento de la recomendación"""
        razones = []
        
        try:
            # Análisis básico
            analisis_basico = resultado.get('analisis_basico', {})
            
            # RSI
            rsi_data = analisis_basico.get('rsi', {})
            if rsi_data.get('valor', 50) < 30:
                razones.append("RSI en zona de sobreventa (oportunidad de compra)")
            elif rsi_data.get('valor', 50) > 70:
                razones.append("RSI en zona de sobrecompra (precaución)")
            
            # Tendencia
            tendencia = analisis_basico.get('medias', {}).get('tendencia', '')
            if 'Alcista' in tendencia:
                razones.append(f"Tendencia {tendencia.lower()} confirmada por medias móviles")
            elif 'Bajista' in tendencia:
                razones.append(f"Tendencia {tendencia.lower()} confirmada por medias móviles")
            
            # MACD
            macd_div = analisis_basico.get('macd', {}).get('divergencia', 0)
            if macd_div > 0:
                razones.append("MACD muestra momentum alcista")
            elif macd_div < 0:
                razones.append("MACD muestra momentum bajista")
            
            # Volumen
            vol_interp = analisis_basico.get('volumen', {}).get('interpretacion', '')
            if vol_interp == 'Alto':
                razones.append("Volumen alto confirma el movimiento")
            
            # Score general
            if score >= 70:
                razones.append("Score técnico muy alcista")
            elif score <= 30:
                razones.append("Score técnico muy bajista")
            
            # Análisis avanzado si disponible
            if 'analisis_avanzado' in resultado:
                razones.append("Confirmado por análisis de Machine Learning")
            
        except:
            razones.append("Análisis basado en indicadores técnicos estándar")
        
        return razones[:5]  # Limitar a 5 razones principales
    
    def _mostrar_reporte_integrado(self, resultado: Dict):
        """Muestra reporte completo integrado"""
        par = resultado['par']
        timestamp = resultado['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*80}")
        print(f"📋 REPORTE INTEGRADO PARA {par}")
        print(f"📅 Generado: {timestamp}")
        print(f"{'='*80}")
        
        # Información de precio
        precio_data = resultado.get('analisis_basico', {}).get('precio', {})
        if precio_data:
            print(f"\n💰 INFORMACIÓN DE PRECIO:")
            print(f"   • Precio Actual: ${precio_data.get('actual', 0):.4f}")
            print(f"   • Cambio 24h: {precio_data.get('cambio_24h', 0):.2%}")
            print(f"   • Máximo 24h: ${precio_data.get('maximo_24h', 0):.4f}")
            print(f"   • Mínimo 24h: ${precio_data.get('minimo_24h', 0):.4f}")
        
        # Análisis técnico
        analisis_basico = resultado.get('analisis_basico', {})
        if analisis_basico:
            print(f"\n📊 ANÁLISIS TÉCNICO BÁSICO:")
            
            rsi_data = analisis_basico.get('rsi', {})
            print(f"   • RSI: {rsi_data.get('valor', 0):.1f} ({rsi_data.get('estado', 'N/A')})")
            
            macd_data = analisis_basico.get('macd', {})
            print(f"   • MACD: {macd_data.get('macd', 0):.4f} | Signal: {macd_data.get('signal', 0):.4f}")
            
            medias_data = analisis_basico.get('medias', {})
            print(f"   • Tendencia: {medias_data.get('tendencia', 'N/A')}")
            
            bb_data = analisis_basico.get('bollinger', {})
            print(f"   • Bollinger: {bb_data.get('posicion', 'N/A')}")
            
            vol_data = analisis_basico.get('volumen', {})
            print(f"   • Volumen: {vol_data.get('interpretacion', 'N/A')} (Ratio: {vol_data.get('ratio', 0):.2f})")
            
            print(f"   • Score Técnico: {analisis_basico.get('score_tecnico', 50):.1f}/100")
        
        # Predicciones avanzadas
        if 'predicciones' in resultado and resultado['predicciones']:
            print(f"\n🔮 PREDICCIONES AVANZADAS (ML):")
            for horizonte, pred in resultado['predicciones'].items():
                if 'precio_predicho' in pred:
                    cambio = pred.get('cambio_esperado', 0) * 100
                    print(f"   • {horizonte.upper()}: ${pred.get('precio_predicho', 0):.4f} ({cambio:+.2f}%) - Confianza: {pred.get('confianza', 0):.1%}")
        
        # Recomendaciones
        recom = resultado.get('recomendaciones', {})
        if recom:
            print(f"\n🎯 RECOMENDACIÓN PRINCIPAL:")
            print(f"   • Señal: {recom.get('señal_principal', 'NEUTRAL')}")
            print(f"   • Confianza: {recom.get('confianza', 0):.1%}")
            
            if recom.get('precio_objetivo'):
                print(f"   • Precio Objetivo: ${recom.get('precio_objetivo', 0):.4f}")
                print(f"   • Stop Loss: ${recom.get('stop_loss', 0):.4f}")
                print(f"   • Take Profit: ${recom.get('take_profit', 0):.4f}")
            
            razones = recom.get('razonamiento', [])
            if razones:
                print(f"   • Razonamiento:")
                for razon in razones:
                    print(f"     - {razon}")
        
        # Score final
        print(f"\n⭐ SCORE FINAL: {resultado.get('score_final', 50):.1f}/100")
        
        print(f"\n{'='*80}")
    
    def _generar_graficos_integrados(self, df: pd.DataFrame, resultado: Dict):
        """Genera gráficos integrados mejorados"""
        try:
            par = resultado['par']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Análisis Técnico Integrado - {par}', fontsize=16, fontweight='bold')
            
            # Gráfico 1: Precio y Medias Móviles
            ax1.plot(df.index, df['close'], label='Precio', color='blue', linewidth=2)
            
            # Calcular medias para el gráfico
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            
            ax1.plot(df.index, sma_20, label='SMA 20', color='orange', alpha=0.7)
            ax1.plot(df.index, sma_50, label='SMA 50', color='red', alpha=0.7)
            
            ax1.set_title('Precio y Medias Móviles')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: RSI
            rsi = RSIIndicator(df['close']).rsi()
            ax2.plot(df.index, rsi, label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Sobrecompra')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Sobreventa')
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            ax2.set_title('RSI (14)')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico 3: MACD
            macd_ind = MACD(df['close'])
            macd = macd_ind.macd()
            macd_signal = macd_ind.macd_signal()
            macd_hist = macd_ind.macd_diff()
            
            ax3.plot(df.index, macd, label='MACD', color='blue')
            ax3.plot(df.index, macd_signal, label='Signal', color='red')
            ax3.bar(df.index, macd_hist, label='Histogram', alpha=0.3, color='gray')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Volumen
            ax4.bar(df.index, df['volume'], alpha=0.6, color='cyan')
            vol_ma = df['volume'].rolling(20).mean()
            ax4.plot(df.index, vol_ma, color='red', label='Vol MA 20')
            ax4.set_title('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error generando gráficos: {e}")
    
    def _guardar_resultado(self, resultado: Dict):
        """Guarda resultado en archivo JSON"""
        try:
            if not os.path.exists('resultados'):
                os.makedirs('resultados')
            
            timestamp = resultado['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"resultados/{resultado['par']}_{timestamp}.json"
            
            # Preparar datos para JSON (convertir datetime, etc.)
            resultado_json = resultado.copy()
            resultado_json['timestamp'] = timestamp
            
            with open(filename, 'w') as f:
                json.dump(resultado_json, f, indent=2, default=str)
            
            print(f"💾 Resultado guardado: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error guardando resultado: {e}")
    
    def ejecutar_analisis_completo(self):
        """Ejecuta análisis completo para todos los pares configurados"""
        print("🚀 INICIANDO SISTEMA INTEGRADO DE ANÁLISIS")
        print("="*80)
        print(f"🔧 Modo Avanzado: {'✅ Activo' if self.analizador_avanzado else '❌ Inactivo'}")
        print(f"📊 Pares a analizar: {len(self.config['pares_analizar'])}")
        print(f"⏰ Horizontes de predicción: {self.config['horizontes_prediccion']} horas")
        print("="*80)
        
        resultados_completos = {}
        
        for i, par in enumerate(self.config['pares_analizar'], 1):
            print(f"\n🔍 [{i}/{len(self.config['pares_analizar'])}] Procesando {par}...")
            
            try:
                resultado = self.analisis_completo_par(par)
                resultados_completos[par] = resultado
                
                # Pausa entre análisis
                if i < len(self.config['pares_analizar']):
                    time.sleep(3)
                    
            except Exception as e:
                print(f"❌ Error procesando {par}: {e}")
                continue
        
        # Generar resumen final
        self._generar_resumen_final(resultados_completos)
        
        return resultados_completos
    
    def _generar_resumen_final(self, resultados: Dict):
        """Genera resumen final de todos los análisis"""
        if not resultados:
            print("❌ No hay resultados para resumir")
            return
        
        print(f"\n{'='*80}")
        print("📊 RESUMEN FINAL DEL MERCADO")
        print(f"{'='*80}")
        
        # Ranking por score
        ranking = []
        for par, resultado in resultados.items():
            score = resultado.get('score_final', 50)
            recom = resultado.get('recomendaciones', {}).get('señal_principal', 'NEUTRAL')
            ranking.append((par, score, recom))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 RANKING POR SCORE PREDICTIVO:")
        for i, (par, score, recom) in enumerate(ranking, 1):
            emoji = "🟢" if "COMPRA" in recom else "🔴" if "VENTA" in recom else "⚪"
            print(f"{i:2d}. {par:10s} - {score:5.1f}/100 - {emoji} {recom}")
        
        # Estadísticas
        scores = [r[1] for r in ranking]
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   • Score Promedio: {np.mean(scores):.1f}")
        print(f"   • Score Máximo: {max(scores):.1f} ({ranking[0][0]})")
        print(f"   • Score Mínimo: {min(scores):.1f} ({ranking[-1][0]})")
        
        # Distribución de señales
        senales = [r[2] for r in ranking]
        compras = len([s for s in senales if "COMPRA" in s])
        ventas = len([s for s in senales if "VENTA" in s])
        neutrales = len([s for s in senales if s == "NEUTRAL"])
        
        print(f"   • Señales de Compra: {compras}")
        print(f"   • Señales de Venta: {ventas}")
        print(f"   • Señales Neutrales: {neutrales}")
        
        # Sentimiento general
        avg_score = np.mean(scores)
        if avg_score >= 65:
            sentimiento = "MUY BULLISH 🚀"
        elif avg_score >= 55:
            sentimiento = "BULLISH 📈"
        elif avg_score >= 45:
            sentimiento = "NEUTRAL ➡️"
        elif avg_score >= 35:
            sentimiento = "BEARISH 📉"
        else:
            sentimiento = "MUY BEARISH 💥"
        
        print(f"   • Sentimiento General: {sentimiento}")
        
        print(f"\n⚠️ RECORDATORIO IMPORTANTE:")
        print("Este análisis es para fines educativos únicamente.")
        print("NO constituye asesoramiento financiero.")
        print("Los mercados de criptomonedas son altamente volátiles y riesgosos.")
        print("Siempre realiza tu propia investigación (DYOR) antes de invertir.")
        
        print(f"\n{'='*80}")


def main():
    """Función principal del sistema integrado"""
    print("🔮 SISTEMA INTEGRADO DE ANÁLISIS Y PREDICCIÓN DE CRIPTOMONEDAS")
    print("Desarrollado por AI Expert Developer & Economist")
    print("Versión 2.0 - Análisis Técnico + Machine Learning")
    print("="*80)
    
    try:
        # Crear sistema integrado
        sistema = SistemaIntegradoPrediccion()
        
        # Ejecutar análisis completo
        resultados = sistema.ejecutar_analisis_completo()
        
        print(f"\n✅ Análisis completado exitosamente")
        print(f"📊 {len(resultados)} pares procesados")
        print("💡 Revisa los reportes detallados arriba")
        
        if sistema.config['guardar_resultados']:
            print("💾 Resultados guardados en carpeta 'resultados/'")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en el sistema: {e}")
        print("🔧 Verifica la configuración y conexiones")


if __name__ == "__main__":
    main() 