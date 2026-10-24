# 📊 MÉTODOS DE ANÁLISIS TÉCNICO UTILIZADOS

## ✅ **SÍ, EL SISTEMA USA FIBONACCI Y MUCHOS OTROS MÉTODOS**

---

## 🔢 **1. MÉTODO DE FIBONACCI**

### **Ubicación en el código:**
- Archivo: `prediccion.py`, función `calcular_niveles_fibonacci()` (líneas 226-303)

### **Cómo Funciona:**

**A. Ratios de Fibonacci Utilizados:**
```python
# Retrocesos de Fibonacci:
FIBO_RATIOS_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]

# Extensiones de Fibonacci:
FIBO_RATIOS_EXTENSION = [0.236, 0.382, 0.618, 1.0, 1.618]
```

**B. Cálculo:**
1. Identifica el **máximo** y **mínimo** del período analizado
2. Calcula el **rango total**: `diff = highest_high - lowest_low`
3. Determina la **dirección del movimiento** (alcista o bajista)
4. Calcula niveles de **retroceso** y **extensión**

**C. Retrocesos de Fibonacci:**
```python
# Para movimiento alcista:
nivel = highest_high - (diff * ratio)
# Ejemplo: 23.6%, 38.2%, 50%, 61.8%, 78.6%

# Para movimiento bajista:
nivel = lowest_low + (diff * ratio)
```

**D. Extensiones de Fibonacci:**
```python
# Proyecciones más allá del rango:
nivel = highest_high + (diff * ratio)  # Alcista
nivel = lowest_low - (diff * ratio)    # Bajista
```

**E. Uso en el Sistema de Scoring:**
- Detecta si el precio está **cerca de un nivel Fibonacci** (dentro del 1.5%)
- Si hay **rebote desde un soporte Fibonacci**: +0.75 puntos alcistas
- Si hay **rechazo desde una resistencia Fibonacci**: +0.75 puntos bajistas
- Visualiza los niveles en los gráficos con colores diferenciados

**F. Visualización:**
- Los niveles se muestran como líneas horizontales en el gráfico
- Retrocesos: líneas discontinuas verdes
- Extensiones: líneas punteadas rojas
- Marca el punto máximo y mínimo usado para el cálculo

---

## 📈 **2. MEDIAS MÓVILES**

### **Tipos Implementados:**

**A. SMA (Simple Moving Average):**
- **SMA Corta**: 10 períodos
- **SMA Larga**: 50 períodos
- **Uso**: Identificar cruces dorados/muerte, determinar tendencia

**B. EMA (Exponential Moving Average):**
- **EMA 9**: Para momentum rápido
- **EMA 21**: Para tendencia media
- **EMA 50**: Para tendencia larga

**C. Señales:**
- **Cruce Dorado**: SMA/EMA corta cruza por encima de la larga → Señal alcista
- **Cruce de la Muerte**: SMA/EMA corta cruza por debajo de la larga → Señal bajista
- **Tendencia**: Precio por encima/debajo de las medias móviles

---

## 📊 **3. MOMENTUM INDICATORS**

### **A. RSI (Relative Strength Index)**
- **Período**: 14 (básico), 10 (avanzado)
- **Rango**: 0-100
- **Interpretación**:
  - >70: Sobrecompra (posible corrección bajista)
  - <30: Sobreventa (posible rebote alcista)
  - 50: Neutral
- **Uso en Scoring**: RSI <30 → +15 puntos, RSI >70 → -15 puntos

### **B. MACD (Moving Average Convergence Divergence)**
- **Parámetros**: 
  - Fast EMA: 12 períodos
  - Slow EMA: 26 períodos
  - Signal: 9 períodos
- **Componentes**:
  - Línea MACD
  - Línea de Señal
  - Histograma (diferencia entre ambas)
- **Señales**:
  - Cruce alcista: MACD cruza por encima de Signal
  - Cruce bajista: MACD cruza por debajo de Signal
- **Uso en Scoring**: MACD > Signal → +10 puntos

### **C. Stochastic Oscillator**
- **Componentes**: %K y %D
- **Uso**: Identificar condiciones de sobrecompra/sobreventa
- **Combinado**: Forma parte del `momentum_score` combinado

### **D. Williams %R**
- **Rango**: -100 a 0
- **Uso**: Similar a Stochastic, detecta extremos de mercado
- **Combinado**: Forma parte del `momentum_score` combinado

---

## 📉 **4. VOLATILIDAD**

### **A. Bollinger Bands**
- **Configuración**: 
  - Media móvil: 20 períodos
  - Desviaciones: 2
- **Componentes**:
  - Banda superior
  - Banda media (SMA 20)
  - Banda inferior
- **Interpretación**:
  - Precio toca banda superior → Posible corrección bajista
  - Precio toca banda inferior → Posible rebote alcista
  - Banda ancha → Alta volatilidad
  - Banda estrecha → Baja volatilidad
- **Uso en Scoring**: Posición del precio dentro de las bandas afecta el score

### **B. ATR (Average True Range)**
- **Ventana**: 10 períodos
- **Uso**: Medir volatilidad real del mercado
- **Aplicación**: Stop-loss dinámicos, ajuste de posición

---

## 🕯️ **5. PATRONES DE VELAS JAPONESAS**

### **Patrones Detectados:**

**A. Doji:**
```python
condición: abs(apertura - cierre) / (alto - bajo) < 0.1
```
- **Significado**: Indecisión del mercado
- **Uso**: Señal de posible reversión

**B. Hammer:**
```python
condición: (min(apertura, cierre) - bajo) > 2 * abs(apertura - cierre)
         AND (alto - max(apertura, cierre)) < abs(apertura - cierre)
```
- **Significado**: Posible reversión alcista
- **Uso en Scoring**: Detecta patrones alcistas

**C. Shooting Star:**
- **Significado**: Posible reversión bajista
- **Uso en Scoring**: Detecta patrones bajistas

**D. Bullish Engulfing:**
```python
condición: cierre > apertura_anterior AND apertura < cierre_anterior
         AND abs(cierre - apertura) > abs(cierre_anterior - apertura_anterior)
```
- **Significado**: Patrón alcista fuerte
- **Uso en Scoring**: +5 puntos al score de patrones

**E. Bearish Engulfing:**
```python
condición: cierre < apertura_anterior AND apertura > cierre_anterior
         AND abs(apertura - cierre) > abs(apertura_anterior - cierre_anterior)
```
- **Significado**: Patrón bajista fuerte
- **Uso en Scoring**: -5 puntos al score de patrones

---

## 🎯 **6. SOPORTES Y RESISTENCIAS**

### **Método: Puntos Pivote**

**A. Pivot High:**
```python
# Un precio es pivot high si:
for j in range(1, 6):
    precio[i] >= precio[i-j]  # Más alto que los 5 anteriores
    precio[i] >= precio[i+j]  # Más alto que los 5 siguientes
```
- **Uso**: Identifica resistencias (niveles donde el precio puede ser rechazado)

**B. Pivot Low:**
```python
# Un precio es pivot low si:
for j in range(1, 6):
    precio[i] <= precio[i-j]  # Más bajo que los 5 anteriores
    precio[i] <= precio[i+j]  # Más bajo que los 5 siguientes
```
- **Uso**: Identifica soportes (niveles donde el precio puede rebotar)

**C. Implementación:**
- Analiza los últimos 50 períodos
- Mantiene los 5 pivot highs y 5 pivot lows más recientes
- Usa estos niveles junto con Fibonacci para confirmación

---

## 📊 **7. ANÁLISIS DE VOLUMEN**

### **Indicadores de Volumen:**

**A. OBV (On-Balance Volume):**
- **Concepto**: Volumen acumulado según dirección del precio
- **Uso**: Confirmar tendencias de precio

**B. VPT (Volume Price Trend):**
- **Concepto**: Correlación entre volumen y cambio de precio
- **Uso**: Identificar divergencias precio-volumen

**C. Volume Ratio:**
```python
ratio = volumen_actual / volumen_promedio_20
```
- **Interpretación**:
  - >1.5: Alto volumen (confirma movimiento)
  - 0.5-1.5: Volumen normal
  - <0.5: Bajo volumen (precaución)
- **Uso en Scoring**: Alto volumen refuerza señales (+5 puntos)

---

## 📐 **8. ADX (Average Directional Index)**

### **Fuerza de Tendencia:**
- **Rango**: 0-100
- **Interpretación**:
  - >25: Tendencia fuerte
  - <20: Tendencia débil/lateral
- **Uso**: Determinar si el mercado está en tendencia o rango lateral

---

## 🎨 **9. FEATURES DERIVADAS**

### **A. Precios Derivados:**
- **Precio Medio**: `(high + low) / 2`
- **Precio Típico**: `(high + low + close) / 3`
- **Precio Ponderado**: `(high + low + 2*close) / 4`

### **B. Returns Múltiples Períodos:**
- Returns de 1h, 4h, 8h
- Volatilidad asociada a cada return

### **C. Lags (Rezagos):**
- Precio de cierre hace 1, 2, 3, 6 períodos
- Volumen hace 1, 2, 3, 6 períodos
- RSI hace 1, 2, 3, 6 períodos

### **D. Rolling Statistics:**
- Media móvil (5, 10 períodos)
- Desviación estándar (5, 10 períodos)
- Mínimo (5, 10 períodos)
- Máximo (5, 10 períodos)

### **E. Diferencias:**
- Diferencia entre EMA 9 y EMA 21
- Diferencia entre EMA 21 y EMA 50
- Posición relativa en Bollinger Bands (0-1)

---

## 🤖 **10. MÉTODOS DE MACHINE LEARNING**

### **A. Random Forest:**
- **Tipo**: Regresión y Clasificación
- **Parámetros**: 100 árboles, profundidad 10
- **Uso**: Predicción de precio y dirección

### **B. XGBoost:**
- **Tipo**: Gradient Boosting
- **Parámetros**: 100 estimadores, profundidad 6, learning rate 0.1
- **Uso**: Predicción de precio y dirección

### **C. Gradient Boosting:**
- **Tipo**: Regresión
- **Parámetros**: Similar a XGBoost
- **Uso**: Predicción de precio

### **D. LSTM (Long Short-Term Memory):**
- **Tipo**: Red Neuronal Recurrente
- **Arquitectura**: 2 capas LSTM (50 unidades cada una)
- **Sequence Length**: 60 períodos
- **Uso**: Capturar patrones temporales complejos
- **Parámetros**: 50 epochs, batch size 32, early stopping

### **E. Ensemble:**
- **Método**: Promedio ponderado de predicciones
- **Modelos**: Random Forest + XGBoost + Gradient Boosting
- **Ventaja**: Reduce overfitting, aumenta robustez

---

## 🔄 **11. MÉTODOS DE VALIDACIÓN**

### **A. Time Series Split:**
- **Importante**: NO usa división aleatoria (evita data leakage)
- **División**: 80% entrenamiento, 20% prueba
- **Orden**: Temporal (primero entrenamiento, después prueba)

### **B. Walk-Forward Analysis:**
- **Implementado en**: `robust_ml_pipeline.py`
- **Concepto**: Entrenar con datos pasados, probar con datos futuros
- **Ventaja**: Más realista para trading

---

## 📊 **12. ANÁLISIS DE SENTIMIENTOS**

### **Método: Análisis de Palabras Clave**

**A. Fuentes:**
- Noticias de NewsAPI
- Análisis de títulos y descripciones

**B. Palabras Positivas:**
- 'surge', 'rally', 'bull', 'pump', 'moon', 'breakthrough', 'adoption', 'partnership'
- **Score**: +2 puntos por cada palabra

**C. Palabras Negativas:**
- 'crash', 'dump', 'bear', 'decline', 'fall', 'regulatory', 'ban', 'hack'
- **Score**: -2 puntos por cada palabra

**D. Fear & Greed Index:**
- **Rango**: 0-100
- **Interpretación**: 
  - 0-25: Miedo extremo
  - 75-100: Codicia extrema
- **Nota**: Actualmente simulado (no conectado a API real)

---

## 📈 **13. SISTEMA DE SCORING COMBINADO**

### **Ponderación de Métodos:**

**A. Score ML (40%):**
- Predicciones de Random Forest
- Predicciones de XGBoost
- Predicciones de LSTM (si disponible)
- Combinación por promedio

**B. Score Técnico (20%):**
- RSI: ±15 puntos
- MACD: ±10 puntos
- EMAs: ±5 puntos
- Volumen: +5 si es alto

**C. Score Patrones (20%):**
- Patrones de velas: ±5 puntos
- Soportes/resistencias: Contribución al score
- Fibonacci: +0.75 si hay rebote/rechazo

**D. Score Sentimientos (20%):**
- Noticias: Normalizado a -10 a +10
- Fear & Greed: Normalizado
- Combinado: 60% noticias + 40% Fear & Greed

---

## 🎯 **14. DETECCIÓN DE REGÍMENES DE MERCADO**

### **Implementado en**: `analysis/regime_detector.py`

**Regímenes Detectados:**
- **Bull Market**: Mercado alcista
- **Bear Market**: Mercado bajista
- **Sideways**: Mercado lateral
- **Volatile**: Alta volatilidad

**Método**: Análisis combinado de múltiples indicadores

---

## 📋 **RESUMEN DE MÉTODOS**

### ✅ **Métodos Clásicos de Análisis Técnico:**
1. ✅ **Fibonacci** (Retrocesos y Extensiones)
2. ✅ **Medias Móviles** (SMA, EMA)
3. ✅ **RSI** (Relative Strength Index)
4. ✅ **MACD** (Moving Average Convergence Divergence)
5. ✅ **Bollinger Bands**
6. ✅ **Stochastic Oscillator**
7. ✅ **Williams %R**
8. ✅ **ATR** (Average True Range)
9. ✅ **ADX** (Average Directional Index)
10. ✅ **OBV** (On-Balance Volume)
11. ✅ **VPT** (Volume Price Trend)

### ✅ **Análisis Chartista:**
1. ✅ **Patrones de Velas Japonesas** (Doji, Hammer, Engulfing, Shooting Star)
2. ✅ **Soportes y Resistencias** (Puntos Pivote)
3. ✅ **Niveles de Fibonacci** (Soportes/Resistencias teóricas)

### ✅ **Machine Learning:**
1. ✅ **Random Forest**
2. ✅ **XGBoost**
3. ✅ **Gradient Boosting**
4. ✅ **LSTM** (Deep Learning)
5. ✅ **Ensemble Methods**

### ✅ **Análisis Fundamental:**
1. ✅ **Análisis de Sentimientos** (Noticias)
2. ✅ **Fear & Greed Index**

### ✅ **Feature Engineering:**
1. ✅ **Lags** (Rezagos temporales)
2. ✅ **Rolling Statistics** (Estadísticas móviles)
3. ✅ **Diferencias entre indicadores**
4. ✅ **Características temporales** (hora, día de semana)

---

## 🎯 **CONCLUSIÓN**

El sistema utiliza una **combinación robusta** de:
- ✅ **Análisis técnico tradicional** (incluyendo Fibonacci)
- ✅ **Análisis chartista** (patrones de velas, soportes/resistencias)
- ✅ **Machine Learning avanzado** (múltiples modelos ensemble)
- ✅ **Análisis de sentimientos**

**Fortaleza del Sistema**: La combinación de múltiples métodos independientes aumenta la confiabilidad de las predicciones, ya que si varios métodos coinciden, la señal es más fuerte.

**⚠️ Importante**: Ningún método por sí solo es infalible. La combinación de múltiples señales es lo que hace robusto al sistema.

---

**Fecha**: 2025-01-XX  
**Versión del Sistema**: 2.0
