# 🚀 MÉTODOS ANALÍTICOS DE TRADING PARA INTEGRAR

## 📊 Métodos Adicionales Recomendados (Excluyendo Fibonacci y Medias Móviles ya implementados)

---

## 🎯 **CATEGORÍA 1: INDICADORES TÉCNICOS AVANZADOS**

### **1. ICHIMOKU CLOUD (Kumo)**
**Prioridad:** ⭐⭐⭐⭐⭐ Alta

**Componentes:**
- **Tenkan-sen** (Línea de Conversión): (9-period high + 9-period low) / 2
- **Kijun-sen** (Línea Base): (26-period high + 26-period low) / 2
- **Senkou Span A** (Leading Span A): (Tenkan + Kijun) / 2, proyectado 26 períodos adelante
- **Senkou Span B** (Leading Span B): (52-period high + 52-period low) / 2, proyectado 26 períodos adelante
- **Chikou Span** (Lagging Span): Precio de cierre proyectado 26 períodos atrás

**Señales:**
- **Bullish**: Precio arriba de la nube (Kumo)
- **Bearish**: Precio debajo de la nube
- **Neutral**: Precio dentro de la nube
- **Kumo Twist**: Cambio de color de la nube (muy fuerte)

**Uso:**
- Determinar tendencia principal
- Identificar soportes/resistencias dinámicos
- Confirmar señales de otros indicadores

**Librería Python:** `ta` tiene `IchimokuIndicator`

---

### **2. VWAP (Volume Weighted Average Price)**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**Cálculo:**
```python
VWAP = Σ(Precio Típico × Volumen) / Σ(Volumen)
Precio Típico = (High + Low + Close) / 3
```

**Señales:**
- Precio > VWAP: Sesgo alcista
- Precio < VWAP: Sesgo bajista
- VWAP como soporte/resistencia dinámico
- VWAP del día vs VWAP histórico

**Uso:**
- Confirmar tendencia intradiaria
- Detectar divergencias precio-VWAP
- Nivel de referencia institucional

**Librería Python:** `pandas_ta` o cálculo manual

---

### **3. SUPERTREND**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
ATR = Average True Range
Upper Band = (High + Low) / 2 + (Multiplier × ATR)
Lower Band = (High + Low) / 2 - (Multiplier × ATR)

Si precio > Upper Band → Señal COMPRA
Si precio < Lower Band → Señal VENTA
```

**Parámetros típicos:**
- Período ATR: 10
- Multiplier: 2.0 o 3.0

**Uso:**
- Señal de tendencia clara
- Filtro de ruido
- Stop-loss dinámico

**Librería Python:** `pandas_ta` o `talib`

---

### **4. PARABOLIC SAR**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
SAR(t) = SAR(t-1) + AF × [EP(t-1) - SAR(t-1)]
AF = Acceleration Factor (incrementa con nuevos extremos)
EP = Extreme Point (máximo en uptrend, mínimo en downtrend)
```

**Señales:**
- SAR debajo del precio → Tendencia alcista
- SAR encima del precio → Tendencia bajista
- Cambio de posición → Posible reversión

**Uso:**
- Seguimiento de tendencia
- Stop-loss dinámico
- Confirmación de reversiones

**Librería Python:** `talib.SAR()` o `pandas_ta`

---

### **5. DONCHIAN CHANNELS**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
Upper Band = Max(High, período N)
Lower Band = Min(Low, período N)
Middle Band = (Upper + Lower) / 2
```

**Señales:**
- Ruptura de banda superior → Posible continuación alcista
- Ruptura de banda inferior → Posible continuación bajista
- Canal estrecho → Acumulación (breakout inminente)

**Uso:**
- Detectar breakouts
- Identificar volatilidad
- Sistema de seguimiento de tendencia

**Librería Python:** `pandas_ta` o cálculo manual

---

### **6. KELTNER CHANNELS**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
Middle Line = EMA(Close, período)
Upper Band = Middle + (Multiplier × ATR)
Lower Band = Middle - (Multiplier × ATR)
```

**Ventaja sobre Bollinger:**
- Usa ATR en lugar de desviación estándar
- Menos sensible a outliers
- Mejor para mercados trending

**Librería Python:** `pandas_ta` o cálculo manual

---

### **7. COMMODITY CHANNEL INDEX (CCI)**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
CCI = (Precio Típico - SMA(Precio Típico, N)) / (0.015 × MAD)
MAD = Mean Absolute Deviation
```

**Señales:**
- CCI > +100: Sobrecompra (pero puede continuar)
- CCI < -100: Sobreventa (pero puede continuar)
- CCI cruza +100 desde abajo: Señal alcista
- CCI cruza -100 desde arriba: Señal bajista

**Uso:**
- Identificar sobrecompra/sobreventa en trending markets
- Detectar reversiones en mercados laterales

**Librería Python:** `talib.CCI()` o `pandas_ta`

---

### **8. AROON INDICATOR**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
Aroon Up = ((N - Períodos desde máximo) / N) × 100
Aroon Down = ((N - Períodos desde mínimo) / N) × 100
Aroon Oscillator = Aroon Up - Aroon Down
```

**Señales:**
- Aroon Up > 70: Tendencia alcista fuerte
- Aroon Down > 70: Tendencia bajista fuerte
- Aroon Oscillator > 0: Momentum alcista

**Uso:**
- Identificar fuerza de tendencia
- Detectar cambios de tendencia tempranos

**Librería Python:** `talib.AROON()` o `pandas_ta`

---

### **9. MONEY FLOW INDEX (MFI)**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
Raw Money Flow = Precio Típico × Volumen
Positive Money Flow = Suma de Raw Money Flow cuando precio sube
Negative Money Flow = Suma de Raw Money Flow cuando precio baja
MFI = 100 - (100 / (1 + Money Flow Ratio))
Money Flow Ratio = Positive Money Flow / Negative Money Flow
```

**Señales:**
- MFI > 80: Sobrecompra
- MFI < 20: Sobreventa
- Divergencias: Muy importantes

**Ventaja sobre RSI:**
- Considera volumen (más confiable)

**Librería Python:** `talib.MFI()` o `pandas_ta`

---

### **10. TRIPLE EXPONENTIAL MOVING AVERAGE (TEMA)**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
EMA1 = EMA(Close, período)
EMA2 = EMA(EMA1, período)
EMA3 = EMA(EMA2, período)
TEMA = 3×EMA1 - 3×EMA2 + EMA3
```

**Ventaja:**
- Menor lag que EMA tradicional
- Más sensible a cambios de precio

**Uso:**
- Reemplazar o complementar EMAs tradicionales
- Señales más tempranas

**Librería Python:** `pandas_ta` o cálculo manual

---

## 📈 **CATEGORÍA 2: ANÁLISIS DE VOLUMEN AVANZADO**

### **11. VOLUME PROFILE (VP) / POINT OF CONTROL (POC)**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**Concepto:**
- Distribución de volumen por nivel de precio
- Identifica zonas donde más trading ocurrió

**Niveles Clave:**
- **POC (Point of Control)**: Precio con mayor volumen
- **Value Area High (VAH)**: 70% del volumen por arriba del POC
- **Value Area Low (VAL)**: 70% del volumen por debajo del POC

**Uso:**
- Soportes/resistencias muy fuertes
- Zonas de acumulación/distribución
- Objetivos de precio

**Librería Python:** `volume_profile` o implementación manual

---

### **12. VOLUME WEIGHTED MACD (VW-MACD)**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Concepto:**
- MACD ponderado por volumen
- Más confiable que MACD tradicional

**Uso:**
- Confirmar señales MACD con volumen
- Filtrar señales falsas

**Librería Python:** Implementación manual

---

### **13. ACCUMULATION/DISTRIBUTION LINE (A/D)**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = Money Flow Multiplier × Volume
A/D Line = Suma acumulada de Money Flow Volume
```

**Señales:**
- A/D sube mientras precio baja → Divergencia alcista (acumulación)
- A/D baja mientras precio sube → Divergencia bajista (distribución)

**Uso:**
- Detectar acumulación/distribución institucional
- Confirmar tendencias de precio

**Librería Python:** `talib.AD()` o `pandas_ta`

---

### **14. CHAIKIN MONEY FLOW (CMF)**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
CMF = Sum(Money Flow Volume, N) / Sum(Volume, N)
Money Flow Volume = ((Close - Low) - (High - Close)) / (High - Low) × Volume
```

**Señales:**
- CMF > 0: Compra de presión
- CMF < 0: Presión de venta
- CMF > 0.1: Compra fuerte
- CMF < -0.1: Venta fuerte

**Uso:**
- Confirmar señales de precio
- Detectar acumulación/distribución

**Librería Python:** `pandas_ta` o implementación manual

---

## 🎨 **CATEGORÍA 3: PATRONES CHARTISTAS AVANZADOS**

### **15. PATRONES DE VELAS JAPONESAS COMPLETOS**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**Patrones Adicionales a Implementar:**

**A. Patrones de Reversión:**
- **Three White Soldiers** (3 soldados blancos)
- **Three Black Crows** (3 cuervos negros)
- **Evening Star** (Estrella vespertina)
- **Morning Star** (Estrella matutina)
- **Dark Cloud Cover** (Cobertura de nube oscura)
- **Piercing Pattern** (Patrón penetrante)
- **Harami** (Embrión)
- **Three Inside Up/Down**

**B. Patrones de Continuación:**
- **Three Line Strike**
- **Upside/Downside Gap Three Methods**
- **Rising/Falling Three Methods**

**C. Patrones Únicos:**
- **Hanging Man** (Hombre colgado)
- **Inverted Hammer** (Martillo invertido)
- **Shooting Star** (ya implementado parcialmente)
- **Spinning Top**

**Librería Python:** `ta-lib` tiene `CDL*` funciones para patrones

---

### **16. PATRONES DE PRECIO (Price Patterns)**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**A. Triángulos:**
- **Ascending Triangle** (Triángulo ascendente)
- **Descending Triangle** (Triángulo descendente)
- **Symmetrical Triangle** (Triángulo simétrico)
- **Descending Wedge** (Cuña descendente)
- **Ascending Wedge** (Cuña ascendente)

**B. Rectángulos/Canales:**
- **Channel** (Canal)
- **Flag** (Bandera)
- **Pennant** (Bandera de gallardete)

**C. Reversión:**
- **Head and Shoulders** (Cabeza y hombros)
- **Inverse Head and Shoulders**
- **Double Top** (Doble techo)
- **Double Bottom** (Doble suelo)
- **Triple Top/Bottom**
- **Cup and Handle** (Taza y mango)
- **Rounding Top/Bottom**

**D. Continuación:**
- **Bullish/Bearish Flag**
- **Bullish/Bearish Pennant**
- **Ascending/Descending Triangle**

**Librería Python:** `chartpattern` o implementación con detección de líneas de tendencia

---

### **17. ELLIOTT WAVE THEORY**
**Prioridad:** ⭐⭐⭐ Media-Alta (complejo)

**Concepto:**
- Mercado se mueve en ondas de 5 (impulsivas) y 3 (correctivas)
- Ondas 1-2-3-4-5 (tendencia) y A-B-C (corrección)

**Uso:**
- Identificar ciclos de mercado
- Predecir objetivos de precio
- Determinar posición en el ciclo

**Librería Python:** `py-elliott-wave` o implementación manual (complejo)

---

## 📊 **CATEGORÍA 4: ANÁLISIS DE MOMENTUM AVANZADO**

### **18. RELATIVE STRENGTH (RS) / RELATIVE MOMENTUM**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
RS = Precio Actual / Precio Hace N períodos
O comparar con índice (ej: BTC vs mercado)
```

**Uso:**
- Identificar activos que superan al mercado
- Comparar criptomonedas entre sí
- Detectar rotación sectorial

---

### **19. MOMENTUM OSCILLATOR**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
Momentum = Close(t) - Close(t-N)
```

**Uso:**
- Detectar aceleración/desaceleración
- Identificar divergencias

**Librería Python:** `talib.MOM()` o `pandas_ta`

---

### **20. RATE OF CHANGE (ROC)**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
ROC = ((Close(t) - Close(t-N)) / Close(t-N)) × 100
```

**Uso:**
- Similar a Momentum pero normalizado
- Mejor para comparar diferentes activos

**Librería Python:** `talib.ROC()` o `pandas_ta`

---

### **21. WILLIAMS %R MEJORADO**
**Prioridad:** ⭐⭐⭐ Media (ya está implementado básico)

**Mejoras:**
- Suavizado con EMA
- Señales de divergencia
- Combinación con otros indicadores

---

## 🎯 **CATEGORÍA 5: SOPORTES Y RESISTENCIAS AVANZADOS**

### **22. PIVOT POINTS (CLÁSICO)**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
PP = (High + Low + Close) / 3
R1 = 2×PP - Low
S1 = 2×PP - High
R2 = PP + (High - Low)
S2 = PP - (High - Low)
R3 = High + 2×(PP - Low)
S3 = Low - 2×(High - PP)
```

**Variantes:**
- **Fibonacci Pivots**: Usa ratios de Fibonacci
- **Woodie Pivots**: Fórmula alternativa
- **Camarilla Pivots**: Niveles muy cercanos al precio
- **DeMark Pivots**: Basado en condiciones de mercado

**Uso:**
- Niveles de soporte/resistencia para el día
- Objetivos de precio
- Stop-loss dinámicos

**Librería Python:** `pandas_ta.pivot_points()` o cálculo manual

---

### **23. FLOOR TRADERS' PIVOTS (MÉTODO DE FLOOR)**
**Prioridad:** ⭐⭐⭐ Media

Similar a Pivot Points pero con diferentes cálculos utilizados por traders de piso.

---

### **24. ANCHORED VWAP**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Concepto:**
- VWAP anclado a un punto específico (ej: inicio de tendencia, evento importante)
- Más relevante que VWAP del día para análisis multi-día

**Uso:**
- Análisis de tendencias largas
- Identificar puntos de entrada basados en VWAP histórico

---

## 📐 **CATEGORÍA 6: OSCILADORES Y DIVERGENCIAS**

### **25. DIVERGENCIAS (RSI, MACD, Price)**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**Tipos:**
- **Divergencia Alcista**: Precio hace menor bajo, RSI hace mayor bajo
- **Divergencia Bajista**: Precio hace mayor alto, RSI hace menor alto
- **Divergencia Oculta Alcista**: Precio hace mayor bajo, RSI hace menor bajo
- **Divergencia Oculta Bajista**: Precio hace menor alto, RSI hace mayor alto

**Implementación:**
- Detectar máximos/mínimos locales
- Comparar con indicadores
- Clasificar tipo de divergencia

**Uso:**
- Señales muy fuertes de reversión
- Confirmación de tendencias

---

### **26. MOMENTUM DIVERGENCE INDICATOR (MDI)**
**Prioridad:** ⭐⭐⭐ Media

Combinación de múltiples divergencias para señal más robusta.

---

## 🔄 **CATEGORÍA 7: CICLOS Y ONDAS**

### **27. FOURIER TRANSFORM / SPECTRAL ANALYSIS**
**Prioridad:** ⭐⭐⭐ Media-Alta

**Concepto:**
- Identifica ciclos dominantes en el precio
- Predice posibles reversiones basadas en ciclos

**Uso:**
- Identificar ciclos estacionales
- Predicción de puntos de reversión

**Librería Python:** `numpy.fft` o `scipy.signal`

---

### **28. HURST EXPONENT**
**Prioridad:** ⭐⭐⭐ Media

**Concepto:**
- Mide tendencia a revertir (mean-reverting) vs trending
- H > 0.5: Trending (persistente)
- H < 0.5: Mean-reverting
- H = 0.5: Random walk

**Uso:**
- Determinar tipo de estrategia adecuada
- Identificar cambios de régimen

**Librería Python:** Implementación con `numpy` o librerías especializadas

---

## 📊 **CATEGORÍA 8: ANÁLISIS MULTI-TIMEFRAME**

### **29. MULTI-TIMEFRAME ANALYSIS**
**Prioridad:** ⭐⭐⭐⭐⭐ Muy Alta

**Concepto:**
- Analizar mismo activo en múltiples timeframes simultáneamente
- Timeframes sugeridos: 15m, 1h, 4h, 1d, 1w

**Reglas:**
- Tendencia mayor determina dirección general
- Timeframe menor para entrada
- Alineación de señales en múltiples timeframes = señal fuerte

**Uso:**
- Confirmación de señales
- Identificar mejores puntos de entrada
- Evitar señales falsas

---

### **30. MARKET STRUCTURE ANALYSIS**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Concepto:**
- Identificar Higher Highs (HH) / Higher Lows (HL) → Uptrend
- Identificar Lower Highs (LH) / Lower Lows (LL) → Downtrend
- Cambio de estructura = cambio de tendencia

**Uso:**
- Análisis objetivo de tendencia
- Identificar reversiones tempranas

---

## 🎯 **CATEGORÍA 9: INDICADORES INSTITUCIONALES**

### **31. ORDER FLOW / ORDER BOOK ANALYSIS**
**Prioridad:** ⭐⭐⭐⭐ Alta (requiere datos de order book)

**Métricas:**
- Bid-Ask Spread
- Order Book Imbalance
- Market Depth
- Liquidity Zones

**Uso:**
- Detectar acumulación/distribución institucional
- Identificar soportes/resistencias basados en liquidez

**Fuente de Datos:** Binance API tiene order book data

---

### **32. FUNDING RATE ANALYSIS (Para Futuros)**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Concepto:**
- Funding rate en contratos perpetuos
- Funding rate positivo: Más longs que shorts
- Funding rate negativo: Más shorts que longs

**Señales:**
- Funding rate extremo: Posible reversión
- Divergencia precio-funding: Señal de reversión

**Fuente de Datos:** Binance API para funding rates

---

## 📈 **CATEGORÍA 10: ANÁLISIS ESTADÍSTICO AVANZADO**

### **33. BOLLINGER %B**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
%B = (Price - Lower Band) / (Upper Band - Lower Band)
```

**Uso:**
- Normalizado 0-1
- Mejor para comparar entre activos
- Señales cuando %B sale de 0-1

---

### **34. BOLLINGER BANDWIDTH**
**Prioridad:** ⭐⭐⭐ Media

**Cálculo:**
```python
Bandwidth = (Upper Band - Lower Band) / Middle Band
```

**Uso:**
- Medir volatilidad relativa
- Detectar Squeeze (bajo bandwidth = breakout inminente)

---

### **35. STANDARD DEVIATION BANDS**
**Prioridad:** ⭐⭐⭐ Media

Similar a Bollinger pero usando múltiplos de desviación estándar.

---

### **36. Z-SCORE / PRICE RELATIVE TO MEAN**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Cálculo:**
```python
Z-Score = (Price - Mean) / StdDev
```

**Uso:**
- Identificar cuando precio está estadísticamente extremo
- Señales cuando |Z| > 2 (2 desviaciones estándar)

---

## 🔗 **CATEGORÍA 11: ANÁLISIS DE CORRELACIONES**

### **37. DYNAMIC CORRELATION ANALYSIS**
**Prioridad:** ⭐⭐⭐⭐ Alta

**Concepto:**
- Correlación entre activos a lo largo del tiempo
- Detectar cuando correlaciones se rompen (oportunidades)
- Correlación con mercado general (BTC, ETH)

**Uso:**
- Diversificación de portfolio
- Detectar desacoplamientos
- Rotación sectorial

---

### **38. BETA RELATIVE TO INDEX**
**Prioridad:** ⭐⭐⭐ Media

**Concepto:**
- Beta vs BTC o vs mercado crypto general
- Activos con beta > 1: Más volátiles
- Activos con beta < 1: Menos volátiles

---

## 📊 **PRIORIZACIÓN DE IMPLEMENTACIÓN**

### **FASE 1 (Implementar Primero) - Impacto Alto, Complejidad Media:**
1. ✅ **VWAP** - Crítico para análisis intradiario
2. ✅ **Volume Profile / POC** - Soporte/resistencia muy fuerte
3. ✅ **Ichimoku Cloud** - Sistema completo de análisis
4. ✅ **Multi-Timeframe Analysis** - Mejora significativa de señales
5. ✅ **Divergencias** - Señales muy confiables
6. ✅ **Pivot Points** - Niveles diarios estándar
7. ✅ **Patrones de Velas Completos** - Expandir detección actual
8. ✅ **Market Structure (HH/HL/LH/LL)** - Análisis objetivo

### **FASE 2 (Segunda Prioridad) - Impacto Alto, Complejidad Media-Alta:**
9. ✅ **Supertrend** - Señal de tendencia clara
10. ✅ **Donchian Channels** - Detección de breakouts
11. ✅ **Keltner Channels** - Alternativa a Bollinger
12. ✅ **Money Flow Index (MFI)** - RSI con volumen
13. ✅ **Chaikin Money Flow** - Análisis de acumulación
14. ✅ **Order Flow Analysis** - Si hay acceso a order book
15. ✅ **Funding Rate Analysis** - Para futuros

### **FASE 3 (Tercera Prioridad) - Impacto Medio, Complejidad Variable:**
16. ✅ **Patrones de Precio** (Triángulos, H&S, etc.) - Complejo pero valioso
17. ✅ **Parabolic SAR** - Seguimiento de tendencia
18. ✅ **Commodity Channel Index** - Oscilador adicional
19. ✅ **Aroon Indicator** - Fuerza de tendencia
20. ✅ **Anchored VWAP** - VWAP para análisis largo plazo

### **FASE 4 (Implementación Avanzada) - Impacto Variable, Complejidad Alta:**
21. ✅ **Elliott Wave Theory** - Muy complejo
22. ✅ **Fourier Transform** - Análisis de ciclos
23. ✅ **Hurst Exponent** - Tipo de mercado
24. ✅ **Dynamic Correlation** - Análisis multi-asset avanzado

---

## 🛠️ **LIBRERÍAS PYTHON RECOMENDADAS**

```python
# Indicadores técnicos
pip install ta-lib          # TA-Lib (librería estándar)
pip install pandas-ta       # Pandas Technical Analysis
pip install ta              # Ya está instalada

# Análisis avanzado
pip install finta           # Financial Technical Analysis
pip install stockstats      # Estadísticas de mercado
pip install technical       # Indicadores técnicos

# Análisis de patrones
pip install pattern-recognition  # Si existe
pip install chartpattern    # Para patrones chartistas

# Análisis de volumen
pip install volume_profile  # Volume Profile

# Análisis estadístico
pip install scipy          # Ya probablemente instalado
pip install statsmodels    # Modelos estadísticos
```

---

## 📝 **NOTAS DE IMPLEMENTACIÓN**

1. **Priorizar métodos con mayor impacto**: VWAP, Volume Profile, Ichimoku
2. **Combinar métodos**: No usar en aislamiento, combinarlos aumenta robustez
3. **Validar señales**: Cada nuevo método debe validarse con backtesting
4. **Evitar redundancia**: No implementar múltiples métodos que miden lo mismo
5. **Integración gradual**: Agregar métodos uno por uno y validar antes del siguiente

---

**Fecha**: 2025-01-XX  
**Versión**: Recomendaciones para Sistema de Predicción v2.0+
